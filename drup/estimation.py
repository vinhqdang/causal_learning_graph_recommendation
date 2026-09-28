"""Nuisance models: exposure propensities and outcome imputation."""

import torch


def _logit(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


def popularity_propensity(O, iters=50, ridge=1e-6):
    """Logistic exposure model p_ui = sigmoid(a + b*logit(r_u) + c*logit(r_i)).

    r_u / r_i are the user / item exposure rates. Fitted by Newton's method on
    all m*n pairs using row/column reductions (no (m, n, k) design tensor).
    A feature with no variation (e.g. every Coat user rated 24 items) is
    dropped, since it is collinear with the intercept.
    """
    dt = O.dtype
    feats = []  # (vector, axis) with axis 0 = user feature, 1 = item feature
    for vec, ax in ((_logit(O.mean(1)), 0), (_logit(O.mean(0)), 1)):
        if vec.std() > 1e-9:
            feats.append(((vec - vec.mean()) / vec.std(), ax))
    k = 1 + len(feats)
    theta = torch.zeros(k, dtype=torch.float64)
    theta[0] = _logit(O.mean()).item()

    def linpred(th):
        z = torch.full(O.shape, th[0].item(), dtype=dt)
        for c, (v, ax) in enumerate(feats, start=1):
            z = z + th[c].to(dt) * (v[:, None] if ax == 0 else v[None, :])
        return z

    for _ in range(iters):
        p = torch.sigmoid(linpred(theta))
        r = O - p
        w = p * (1 - p)
        red = {0: (r.sum(1), w.sum(1)), 1: (r.sum(0), w.sum(0))}
        g = torch.zeros(k, dtype=torch.float64)
        H = torch.zeros(k, k, dtype=torch.float64)
        g[0], H[0, 0] = r.sum(), w.sum()
        for a_, (va, axa) in enumerate(feats, start=1):
            g[a_] = (red[axa][0] * va).sum()
            H[0, a_] = H[a_, 0] = (red[axa][1] * va).sum()
            for b_, (vb, axb) in enumerate(feats, start=1):
                if axa == axb:
                    H[a_, b_] = (red[axa][1] * va * vb).sum()
                else:
                    vu = va if axa == 0 else vb
                    vi = vb if axa == 0 else va
                    H[a_, b_] = (vu[:, None] * w * vi[None, :]).sum()
        step = torch.linalg.solve(H + ridge * torch.eye(k, dtype=torch.float64), g)
        theta = theta + step
        if step.abs().max() < 1e-8:
            break
    return torch.sigmoid(linpred(theta)), theta


def baseline_imputation(O, Y, P, lam=5.0, iters=20, ips=True):
    """Additive imputation Yhat_ui = mu + b_u + b_i fitted on exposed pairs.

    Optionally inverse-propensity weighted so that the fit targets the
    full-population outcome surface. Closed-form alternating ridge updates.
    """
    Wt = O / P if ips else O.clone()
    Yo = O * Y
    mu = (Wt * Yo).sum() / Wt.sum()
    bu = torch.zeros(O.shape[0], dtype=O.dtype)
    bi = torch.zeros(O.shape[1], dtype=O.dtype)
    for _ in range(iters):
        res = Yo - mu - bi[None, :]
        bu = (Wt * res).sum(1) / (Wt.sum(1) + lam)
        res = Yo - mu - bu[:, None]
        bi = (Wt * res).sum(0) / (Wt.sum(0) + lam)
    return (mu + bu[:, None] + bi[None, :]).clamp(0.0, 1.0)


def clip_propensity(P, floor):
    return P.clamp(min=floor, max=1.0)


def lowrank_imputation(O, Y, P, rank=32, lam=10.0, ridge=5.0, iters=8, chunk=512, seed=0):
    """Personalised imputation Yhat = clip(mu + b_u + b_i + U V^T, 0, 1).

    The additive part is baseline_imputation; the low-rank residual is fitted
    by inverse-propensity-weighted alternating least squares on the logged
    pairs (weights O / P), processed in user / item chunks so that no
    (m, n, rank) tensor is ever materialised. Any fixed imputation keeps the
    DR edge estimates unbiased (Theorem 1); a more accurate one lowers their
    variance (Theorem 3), and a personalised one makes the 1-hop DR term
    personalised.
    """
    g = torch.Generator().manual_seed(seed)
    base = baseline_imputation(O, Y, P, lam=ridge)
    Wt = O / P
    Rz = O * (Y - base)                               # residual on logged pairs
    m, n = O.shape
    dt = O.dtype
    U = 0.01 * torch.randn(m, rank, generator=g, dtype=torch.float64).to(dt)
    V = 0.01 * torch.randn(n, rank, generator=g, dtype=torch.float64).to(dt)
    eye = torch.eye(rank, dtype=dt)

    def solve(Wm, Rm, F, out):
        for s in range(0, Wm.shape[0], chunk):
            w = Wm[s:s + chunk]                       # (c, n)
            A = torch.einsum("cn,nk,nl->ckl", w, F, F) + lam * eye
            b = (w * Rm[s:s + chunk]) @ F             # (c, k)
            out[s:s + chunk] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    for _ in range(iters):
        solve(Wt, Rz, V, U)
        solve(Wt.T, Rz.T, U, V)
    return (base + U @ V.T).clamp(0.0, 1.0)


def naive_bayes_propensity(base, O, Y, mar_rate):
    """Outcome-dependent (MNAR-on-Y) propensity in the spirit of Schnabel et
    al. (2016):
        P(O=1 | u, i, y) = P(O=1 | u, i) * P(Y=y | O=1) / P(Y=y | MAR),
    with P(O=1 | u, i) from the exposure model ``base`` and P(Y=1 | MAR)
    estimated on a small held-out random sample. Only entries of exposed
    pairs are used downstream (their Y is known); others keep ``base``.
    """
    p_obs = float((O * Y).sum() / O.sum())
    r1 = p_obs / mar_rate
    r0 = (1.0 - p_obs) / (1.0 - mar_rate)
    ratio = torch.where(Y > 0, torch.full_like(base, r1), torch.full_like(base, r0))
    return torch.where(O > 0, (base * ratio).clamp(max=1.0), base)


def get_propensity(d, O, Y, prop):
    """'given' (shipped with the data), 'pop' (logistic exposure model) or
    'nb' (exposure model x outcome-dependent Naive-Bayes factor)."""
    if prop == "given" and d.get("P_given") is not None:
        return d["P_given"].to(O.dtype)
    base = popularity_propensity(O)[0]
    if prop == "nb":
        return naive_bayes_propensity(base, O, Y, d["mar_rate"])
    return base
