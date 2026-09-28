"""Fairness, accountability and transparency tools for DRUP.

All of them exploit one structural fact: after the idempotent walk
correction, DRUP is a *multilinear* polynomial of the edge estimates in which
no edge appears twice, and for a fixed user it is *affine* in that user's own
edge estimates. This gives closed-form explanations, closed-form
counterfactuals, bounded per-interaction influence and a bounded-sensitivity
public operator.
"""

import math

import numpy as np
import torch

from .propagation import item_gram


# --------------------------------------------------------------------------
# Transparency: exact attributions and minimal counterfactual explanations
# --------------------------------------------------------------------------

def user_contribution(w_u, c_u):
    """User u's additive contribution to the public operator G_corr."""
    wt = c_u * w_u
    return torch.outer(wt, wt) - torch.diag(wt * wt) + torch.diag(c_u * c_u * w_u)


class LeaveOneOut:
    """Implicit leave-one-out operator G^(-u) = G - contrib(w_u^orig).

    Never materialises an (n, n) matrix: products with G^(-u) cost one
    mat-vec with the public G plus O(n) corrections, which makes exact
    explanations cheap even with ~10^4 items. ``self[:, i]`` returns a column.
    """

    def __init__(self, G, w_u, c_u):
        self.G, self.c = G, c_u
        self.wt = c_u * w_u
        self.d = c_u * c_u * w_u

    def rmatvec(self, x):                  # x @ G^(-u)
        wt = self.wt
        return x @ self.G - ((x @ wt) * wt - x * wt * wt + x * self.d)

    def column(self, i):
        col = self.G[:, i] - self.wt * self.wt[i]
        col = col.clone()
        col[i] = col[i] + self.wt[i] ** 2 - self.d[i]
        return col

    def __getitem__(self, idx):
        _, i = idx
        return self.column(i)


def leave_one_out_operator(G, w_u, c_u):
    return LeaveOneOut(G, w_u, c_u)


def exact_three_hop(w_u, c_u, G_loo):
    """Corrected 3-hop scores of user u (all items) for an arbitrary row w_u,
    re-inserting the user's own contribution into the operator. Equals a full
    recomputation of drup.propagation.three_hop with that row (C fixed)."""
    wt = c_u * w_u
    d = c_u * c_u * w_u
    delta = wt * wt - d
    base = G_loo.rmatvec(wt) if isinstance(G_loo, LeaveOneOut) else wt @ G_loo
    # wt @ contrib(w_u) = (wt.wt) wt - wt*wt^2 + wt*d
    own = (wt @ wt) * wt - wt * wt * wt + wt * d
    return base + own - wt * (delta.sum() - 2 * delta) - (wt ** 3 - c_u ** 3 * w_u)


def affine_coefficients(i, w_u, c_u, G_loo):
    """a_j such that s_ui = const + sum_{j != i} a_j w_uj exactly
    (Prop. 5 in docs/THEORY.md):  a_j = c_j G_loo[j, i] + c_i w_ui c_j^2."""
    a = c_u * G_loo[:, i] + c_u[i] * w_u[i] * c_u * c_u
    a = a.clone()
    a[i] = 0.0
    return a


def contributions(i, w1, w0, c_u, G_loo, logged):
    """Exact attribution of s_ui to each logged interaction j of the user:
    phi_j = a_j (w1_j - w0_j), where w1 is the user's DR row and w0 the row
    with interaction j unlogged (W_uj -> Yhat_uj). Because s_ui is affine in
    the row, phi_j is simultaneously the leave-one-out effect, the Shapley
    value of j, and removal effects add up exactly over any subset."""
    phi = affine_coefficients(i, w1, c_u, G_loo) * (w1 - w0)
    return phi * logged


def minimal_counterfactual(i, k, w1, w0, c_u, G_loo, logged, margin=None, scale=1.0):
    """Minimum number of the user's logged interactions whose removal makes
    item k outrank item i (i, k unexposed candidates). Since the margin
    s_ui - s_uk decreases by exactly sum_{j in S} (phi_ij - phi_kj), picking
    the largest gains first is optimal (unit-cost selection).

    ``margin`` is the margin of the full score and ``scale`` the weight of the
    three-hop term in it (the one-hop term of an unexposed candidate does not
    depend on the user's other interactions). Without them the three-hop
    score alone is used.
    Returns (list of removed items or None if impossible, margin)."""
    if margin is None:
        s = exact_three_hop(w1, c_u, G_loo)
        margin = float(s[i] - s[k])
    if margin < 0:
        return [], margin
    gain = scale * (contributions(i, w1, w0, c_u, G_loo, logged) - contributions(k, w1, w0, c_u, G_loo, logged))
    gain[i] = gain[k] = 0.0
    order = torch.argsort(gain, descending=True)
    acc, removed = 0.0, []
    for j in order.tolist():
        if gain[j] <= 0:
            break
        acc += float(gain[j])
        removed.append(j)
        if acc > margin:
            return removed, margin
    return None, margin


# --------------------------------------------------------------------------
# Accountability: per-interaction influence bound and certified robustness
# --------------------------------------------------------------------------

def logged_weight_range(yh, tau, kind):
    """Range of the edge estimate of a logged (fake) interaction with any label
    and any propensity p >= tau.  DR: [yh - yh/tau, yh + (1 - yh)/tau];
    IPS: [0, 1/tau]; Obs: [0, 1]."""
    if kind == "DR":
        return yh - yh / tau, yh + (1.0 - yh) / tau
    if kind == "IPS":
        return torch.zeros_like(yh), torch.full_like(yh, 1.0 / tau)
    return torch.zeros_like(yh), torch.ones_like(yh)


def fake_user_effect_bounds(wt_u, c_lo, c_hi, yh_v, tau, n_logged, kind, corrected=True,
                            return_unlogged=False):
    """Box-exact bounds on the change of the un-normalised 3-hop scores of the
    users in ``wt_u`` (rows, n) caused by ONE injected user who logs at most
    ``n_logged`` interactions with arbitrary items, labels and propensities
    >= tau (Theorem 7).

    With frozen nuisances the injected row v changes s_u by
        Delta_k = x_k (D + a_k c_vk) - a_k x_k^2,  x = c_v w_v, D = <a, x>,
    where a = wt_u. We bound D using the n_logged most favourable slots and
    then maximise / minimise Delta_k exactly over the box (D, x_k): it is
    linear in D and a quadratic in x_k. Unlogged items keep w_vk = yh_vk
    (DR) or 0 (IPS / Obs); the fake user's normalisation c_v is only known to
    lie in [c_lo, c_hi] (exactly known when degrees come from the imputation).
    Without the walk correction (Obs / IPS / DR adjacency) the injected row
    enters as Delta_k = D x_k (set corrected=False).
    With return_unlogged=True, also returns the bounds for items the fake user
    does not log (x_k pinned), which tighten certificates because a profile
    can log at most n_logged items.
    """
    wl, wh = logged_weight_range(yh_v, tau, kind)
    unl = yh_v if kind == "DR" else torch.zeros_like(yh_v)
    cands_u = torch.stack([c_lo * unl, c_hi * unl])
    ux_lo, ux_hi = cands_u.min(0).values, cands_u.max(0).values
    cands_l = torch.stack([c_lo * wl, c_lo * wh, c_hi * wl, c_hi * wh])
    lx_lo, lx_hi = cands_l.min(0).values, cands_l.max(0).values
    a = wt_u

    def prod_range(xlo, xhi):
        p1, p2 = a * xlo[None, :], a * xhi[None, :]
        return torch.minimum(p1, p2), torch.maximum(p1, p2)
    u_lo, u_hi = prod_range(ux_lo, ux_hi)
    l_lo, l_hi = prod_range(lx_lo, lx_hi)
    kk = min(n_logged, a.shape[1])
    inc = torch.topk((l_hi - u_hi).clamp_min(0), kk, dim=1).values.sum(1, keepdim=True)
    dec = torch.topk((u_lo - l_lo).clamp_min(0), kk, dim=1).values.sum(1, keepdim=True)
    D_lo = u_lo.sum(1, keepdim=True) - dec
    D_hi = u_hi.sum(1, keepdim=True) + inc
    if return_unlogged:
        # item k NOT logged by the fake user: x_k is pinned to its unlogged value
        ulo, uhi = _box_extremes(a, D_lo, D_hi, ux_lo[None, :].expand_as(a), ux_hi[None, :].expand_as(a),
                                 c_lo, c_hi, corrected)
    x_lo = torch.minimum(ux_lo, lx_lo)[None, :].expand_as(a)
    x_hi = torch.maximum(ux_hi, lx_hi)[None, :].expand_as(a)
    best_lo, best_hi = _box_extremes(a, D_lo, D_hi, x_lo, x_hi, c_lo, c_hi, corrected)
    if return_unlogged:
        return best_lo, best_hi, ulo, uhi
    return best_lo, best_hi


def _box_extremes(a, D_lo, D_hi, x_lo, x_hi, c_lo, c_hi, corrected):
    """Exact extremes of Delta = x (D + a c) - a x^2 (or x D) over the box."""
    best_hi = torch.full_like(a, -float("inf"))
    best_lo = torch.full_like(a, float("inf"))
    for D in (D_lo.expand_as(a), D_hi.expand_as(a)):
        for cc in (c_lo[None, :].expand_as(a), c_hi[None, :].expand_as(a)):
            cands = (x_lo, x_hi)
            if corrected:
                # vertex of the quadratic; where a_k == 0 the function is linear
                nz = a.abs() > torch.finfo(a.dtype).tiny
                vert = torch.where(nz, (D + a * cc) / (2 * torch.where(nz, a, torch.ones_like(a))), x_lo)
                cands = (x_lo, x_hi, torch.maximum(torch.minimum(vert, x_hi), x_lo))
            for x in cands:
                val = x * (D + a * cc) - a * x * x if corrected else x * D
                best_hi = torch.maximum(best_hi, val)
                best_lo = torch.minimum(best_lo, val)
    return best_lo, best_hi


# --------------------------------------------------------------------------
# Privacy: differentially private public item operator
# --------------------------------------------------------------------------

def analytic_gaussian_sigma(epsilon, delta, sens):
    """Smallest sigma for which the Gaussian mechanism with L2 sensitivity
    ``sens`` is (epsilon, delta)-DP, for any epsilon > 0 (analytic Gaussian
    mechanism, Balle & Wang 2018, Theorem 8): the privacy profile
        Phi(sens/(2 sigma) - epsilon sigma/sens)
        - exp(epsilon) Phi(-sens/(2 sigma) - epsilon sigma/sens) <= delta
    is decreasing in sigma, so sigma is found by bisection."""
    from math import erf, exp, sqrt

    def Phi(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def profile(sig):
        a, b = sens / (2.0 * sig), epsilon * sig / sens
        return Phi(a - b) - exp(epsilon) * Phi(-a - b)
    lo, hi = 1e-6 * sens, sens
    while profile(hi) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if profile(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def dp_item_operator(W, C, epsilon, delta, R, generator=None, users=None):
    """(epsilon, delta)-DP release of G_corr under add/remove-one-user,
    conditional on public nuisances (propensities, imputation, C and R fixed
    independently of the private rows).

    Each user's weighted row is scaled so that ||wt_v||_2 <= R and
    ||c_v^2 w_v||_2 <= R^2. User v contributes
        wt_v^T wt_v - diag(wt_v^2) + diag(c_v^2 w_v)
    whose Frobenius norm is <= ||wt_v||^2 + ||c_v^2 w_v|| <= 2 R^2, so the
    L2 sensitivity of the upper triangle is 2 R^2. The noise scale is the
    analytic Gaussian calibration, valid for every epsilon > 0.
    ``users``: optional index of the (private) rows that form G.
    """
    if users is not None:
        W, C = W[users], C[users]
    wt = C * W
    d2 = C * C * W
    s1 = (R / wt.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
    s2 = (R * R / d2.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
    k = torch.minimum(s1, s2)[:, None]
    Wc = W * k
    G = item_gram(Wc, C, correct=True)
    if epsilon == float("inf"):
        return G, k.squeeze(1), 0.0
    sens = 2.0 * R * R
    sigma = analytic_gaussian_sigma(epsilon, delta, sens)
    n = G.shape[0]
    N = torch.randn(n, n, generator=generator, dtype=G.dtype) * sigma
    N = torch.triu(N)
    N = N + torch.triu(N, 1).T
    return G + N, k.squeeze(1), sigma


def spectral_components(G, max_rank=128):
    """Leading eigen-pairs (by |eigenvalue|) of a symmetric matrix. Exact
    eigh for small n, randomized (Halko et al.) for large n."""
    n = G.shape[0]
    if n <= 2000:
        evals, evecs = torch.linalg.eigh(G)
        idx = torch.argsort(evals.abs(), descending=True)[:max_rank]
        return evals[idx], evecs[:, idx]
    U, S, V = torch.svd_lowrank(G, q=max_rank + 10, niter=4)
    sign = torch.sign((U * V).sum(0))          # symmetric: eigval = sign * sigma
    idx = torch.argsort(S, descending=True)[:max_rank]
    return (S * sign)[idx], U[:, idx]


def low_rank_denoise(G, rank, comps=None):
    """Post-processing (free under DP): keep the ``rank`` eigen-components of
    the symmetric noisy operator with the largest |eigenvalue|."""
    if rank is None or rank >= G.shape[0]:
        return G
    evals, V = comps if comps is not None else spectral_components(G, rank)
    return (V[:, :rank] * evals[:rank]) @ V[:, :rank].T


# --------------------------------------------------------------------------
# Fairness metrics
# --------------------------------------------------------------------------

def topk_lists(scores, test, K, row_of=None):
    out = {}
    S = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    for u, items, rel in test:
        if len(items) == 0:
            continue
        r = u if row_of is None else row_of[u]
        s = S[r, items]
        out[u] = items[np.argsort(-s, kind="stable")[:K]]
    return out


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def popularity_rank_correlation(scores, test, item_pop, row_of=None):
    """PRU (Zhu et al., 2021): mean over users of the Spearman correlation
    between an item's training popularity and its predicted score among the
    user's candidates. 0 = no popularity bias."""
    from scipy.stats import spearmanr
    S = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    vals = []
    for u, items, rel in test:
        if len(items) < 3:
            continue
        r = u if row_of is None else row_of[u]
        rho = spearmanr(S[r, items], item_pop[items]).correlation
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals))


def exposure_conditional_bias(scores, test, item_pop, item_quality, row_of=None):
    """Causal item-side fairness: partial Spearman correlation between an
    item's mean predicted score (rank-normalised within each user) and its
    log training exposure, *controlling for its true quality* (like rate on
    the unbiased data). Under exposure invariance (Theorem 4) it is ~0."""
    from scipy.stats import rankdata
    S = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    n = len(item_pop)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for u, items, rel in test:
        if len(items) < 2:
            continue
        r = u if row_of is None else row_of[u]
        pr = (rankdata(S[r, items]) - 1) / (len(items) - 1)
        acc[items] += pr
        cnt[items] += 1
    ok = cnt > 0
    ms = acc[ok] / cnt[ok]
    a = rankdata(ms)
    b = rankdata(np.log1p(item_pop[ok]))
    c = rankdata(item_quality[ok])

    def resid(x, z):
        z1 = np.c_[np.ones_like(z), z]
        return x - z1 @ np.linalg.lstsq(z1, x, rcond=None)[0]
    ra, rb = resid(a, c), resid(b, c)
    return float(np.corrcoef(ra, rb)[0, 1])


def group_gap(per_user_metric, groups):
    """Absolute gap of the mean metric between two user groups (dict u->0/1)."""
    g0 = [v for u, v in per_user_metric.items() if groups.get(u) == 0]
    g1 = [v for u, v in per_user_metric.items() if groups.get(u) == 1]
    return float(abs(np.mean(g0) - np.mean(g1))), float(np.mean(g0)), float(np.mean(g1))
