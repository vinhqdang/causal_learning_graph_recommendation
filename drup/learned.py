"""Trained baselines: MF / LightGCN with pointwise (naive, IPS, DR, DR-JL,
MRDR) or pairwise (BPR, PDA, MACR-style) objectives, NAVIP (inverse-propensity
weighted neighbour aggregation), r-AdjNorm (LightGCN with normalisation
exponent r), SimGCL (LightGCN with noise-based contrastive views) and iALS
(weighted matrix factorisation by alternating least squares)."""

import math

import torch
import torch.nn.functional as F


class MF(torch.nn.Module):
    def __init__(self, m, n, d):
        super().__init__()
        self.U = torch.nn.Embedding(m, d)
        self.V = torch.nn.Embedding(n, d)
        self.bu = torch.nn.Embedding(m, 1)
        self.bi = torch.nn.Embedding(n, 1)
        for e in (self.U, self.V):
            torch.nn.init.normal_(e.weight, std=0.1)
        for e in (self.bu, self.bi):
            torch.nn.init.zeros_(e.weight)
        self.b0 = torch.nn.Parameter(torch.zeros(1))
        self.prior, self.pbeta = None, 0.0      # fixed debiased offset (standardised logit of the imputation)

    def embeddings(self):
        return self.U.weight, self.V.weight

    def forward(self, u, i, emb=None):
        Ue, Ve = emb if emb is not None else self.embeddings()
        out = (Ue[u] * Ve[i]).sum(-1) + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.b0
        return out + self.pbeta * self.prior[u, i] if self.pbeta else out

    def full_scores(self, rows):
        Ue, Ve = self.embeddings()
        out = Ue[rows] @ Ve.T + self.bu.weight[rows] + self.bi.weight.T + self.b0
        return out + self.pbeta * self.prior[rows] if self.pbeta else out


class LightGCN(MF):
    """LightGCN over a (possibly weighted) user-item graph of positive edges.
    Normalisation D^-r A D^-(1-r); r = 0.5 is LightGCN, other r is r-AdjNorm."""

    def __init__(self, m, n, d, edges_u, edges_i, weights, layers=2, r=0.5):
        super().__init__(m, n, d)
        self.m, self.n, self.layers = m, n, layers
        w = weights.float()
        du = torch.zeros(m).index_add_(0, edges_u, w).clamp_min(1e-8)
        di = torch.zeros(n).index_add_(0, edges_i, w).clamp_min(1e-8)
        v_ui = w * du[edges_u].pow(-r) * di[edges_i].pow(-(1 - r))     # item -> user message
        v_iu = w * di[edges_i].pow(-r) * du[edges_u].pow(-(1 - r))     # user -> item message
        idx = torch.stack([torch.cat([edges_u, edges_i + m]), torch.cat([edges_i + m, edges_u])])
        self.A = torch.sparse_coo_tensor(idx, torch.cat([v_ui, v_iu]), (m + n, m + n)).coalesce()

    def embeddings(self):
        E = torch.cat([self.U.weight, self.V.weight])
        acc = E
        for _ in range(self.layers):
            E = torch.sparse.mm(self.A, E)
            acc = acc + E
        acc = acc / (self.layers + 1)
        return acc[: self.m], acc[self.m:]


class LGNCos(LightGCN):
    """LightGCN whose ranking score is the cosine of the propagated embeddings, the score an
    InfoNCE loss with normalised embeddings optimises."""

    def full_scores(self, rows):
        Ue, Ve = self.embeddings()
        return F.normalize(Ue[rows], dim=-1) @ F.normalize(Ve, dim=-1).T


class AdvLGN(LightGCN):
    def __init__(self, *args, w_dim=64, tau=0.2, **kw):
        super().__init__(*args, **kw)
        self.Pu = torch.nn.Embedding(self.m, w_dim); self.Pi = torch.nn.Embedding(self.n, w_dim)
        torch.nn.init.xavier_normal_(self.Pu.weight); torch.nn.init.xavier_normal_(self.Pi.weight)
        self.tau = tau
    def full_scores(self, rows):
        Ue, Ve = self.embeddings()
        s = F.normalize(Ue[rows], dim=-1) @ F.normalize(Ve, dim=-1).T / self.tau
        return s + self.pbeta * self.prior[rows] if self.pbeta else s


class SimGCL(LightGCN):
    """SimGCL (Yu et al., SIGIR 2022): LightGCN propagation (without the
    layer-0 term) with optional uniform noise of norm eps added at every
    layer; two noisy views give an InfoNCE loss (see train)."""

    def __init__(self, *args, eps=0.1, **kw):
        super().__init__(*args, **kw)
        self.eps = eps

    def embeddings(self, perturb=False):
        E = torch.cat([self.U.weight, self.V.weight])
        acc = 0.0
        for _ in range(self.layers):
            E = torch.sparse.mm(self.A, E)
            if perturb:
                E = E + torch.sign(E) * F.normalize(torch.rand_like(E), dim=-1) * self.eps
            acc = acc + E
        acc = acc / self.layers
        return acc[: self.m], acc[self.m:]


class MACR(MF):
    """MACR-style counterfactual popularity removal (Wei et al., KDD 2021) on an
    MF backbone: y = sigma(f) * sigma(y_u) * sigma(y_i) with linear user and item
    branches; ranking by (sigma(f) - c) * sigma(y_u) * sigma(y_i)."""

    def __init__(self, m, n, d, c=0.3):
        super().__init__(m, n, d)
        self.wu = torch.nn.Linear(d, 1)
        self.wi = torch.nn.Linear(d, 1)
        self.c = c

    def branches(self, u, i, emb):
        Ue, Ve = emb
        return self.wu(Ue[u]).squeeze(-1), self.wi(Ve[i]).squeeze(-1)

    def full_scores(self, rows):
        Ue, Ve = self.embeddings()
        f = torch.sigmoid(Ue[rows] @ Ve.T + self.bu.weight[rows] + self.bi.weight.T + self.b0)
        su = torch.sigmoid(self.wu(Ue[rows]))            # (rows, 1)
        si = torch.sigmoid(self.wi(Ve)).T                # (1, n)
        return (f - self.c) * su * si


class IALS:
    """Weighted matrix factorisation for implicit feedback (Hu, Koren and
    Volinsky, 2008): confidence 1 + alpha on logged positives, 1 elsewhere,
    squared loss with L2 penalty, solved by alternating least squares."""

    def __init__(self, m, n, d, eu, ei, alpha=10.0, reg=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.U = 0.1 * torch.randn(m, d, generator=g)
        self.V = 0.1 * torch.randn(n, d, generator=g)
        self.eu, self.ei, self.alpha, self.reg = eu, ei, alpha, reg

    def _solve(self, X, Fx, ru, ri, nrows, chunk=2048):
        d = Fx.shape[1]
        G = Fx.T @ Fx + self.reg * torch.eye(d)
        order = torch.argsort(ru)
        ru, ri = ru[order], ri[order]
        bounds = torch.searchsorted(ru, torch.arange(0, nrows + chunk, chunk).clamp(max=nrows))
        for c, s in enumerate(range(0, nrows, chunk)):
            e = min(s + chunk, nrows)
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            r, f = ru[lo:hi] - s, Fx[ri[lo:hi]]
            A = G.expand(e - s, d, d).clone()
            A.index_add_(0, r, self.alpha * f[:, :, None] * f[:, None, :])
            b = torch.zeros(e - s, d).index_add_(0, r, (1.0 + self.alpha) * f)
            X[s:e] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def step(self):
        self._solve(self.U, self.V, self.eu, self.ei, self.U.shape[0])
        self._solve(self.V, self.U, self.ei, self.eu, self.V.shape[0])

    def full_scores(self, rows):
        return self.U[rows] @ self.V.T


class IALSW:
    """iALS with a dense confidence matrix: logged positives 1 + alpha (target 1), every other
    pair 1 - fn * Q_ui (target 0), where Q_ui is the probability that a pair that is not a logged
    positive is a positive after all. Exact per-user and per-item ridge solves (small matrices)."""

    def __init__(self, m, n, d, eu, ei, Q, alpha=10.0, reg=1.0, fn=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.U = 0.1 * torch.randn(m, d, generator=g)
        self.V = 0.1 * torch.randn(n, d, generator=g)
        self.reg = reg
        pos = torch.zeros(m, n, dtype=torch.bool)
        pos[eu, ei] = True
        self.C = torch.where(pos, torch.full((m, n), 1.0 + alpha), (1.0 - fn * Q).clamp_min(1e-3)).float()
        self.T = pos.float()

    def _solve(self, X, Fx, C, T):
        d = Fx.shape[1]
        for s in range(0, X.shape[0], 256):
            c, t = C[s:s + 256], T[s:s + 256]
            A = torch.einsum("un,nd,ne->ude", c, Fx, Fx) + self.reg * torch.eye(d)
            b = torch.einsum("un,nd->ud", c * t, Fx)
            X[s:s + 256] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def step(self):
        self._solve(self.U, self.V, self.C, self.T)
        self._solve(self.V, self.U, self.C.T, self.T.T)

    def full_scores(self, rows):
        return self.U[rows] @ self.V.T


class IALSLR:
    """DR-iALS with the confidence of the unlogged pairs, w = 1 - fn * Q, replaced by its rank-K
    truncated SVD sum_k s_k a_k b_k^T. The weighted normal equations then need only K small Gram
    matrices per half-step, G_k = F^T diag(b_k) F, plus a sparse correction on the logged positives
    (confidence 1 + alpha instead of w): O(n d^2 K + nnz d^2) per step, no dense m x n solve."""

    def __init__(self, m, n, d, eu, ei, Q, alpha=10.0, reg=1.0, fn=1.0, K=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.U = 0.1 * torch.randn(m, d, generator=g)
        self.V = 0.1 * torch.randn(n, d, generator=g)
        self.eu, self.ei, self.alpha, self.reg = eu, ei, alpha, reg
        Wm = (1.0 - fn * Q).clamp_min(1e-3).float()
        a, sv, b = torch.svd_lowrank(Wm, q=K, niter=4)
        self.L = a * sv                      # (m, K): left factors times singular values
        self.R = b                           # (n, K)

    def _solve(self, X, Fx, L, Rf, ru, ri, nrows, chunk=2048):
        d = Fx.shape[1]
        G = torch.einsum("nd,ne,nk->kde", Fx, Fx, Rf)                 # (K, d, d)
        order = torch.argsort(ru)
        ru, ri = ru[order], ri[order]
        bounds = torch.searchsorted(ru, torch.arange(0, nrows + chunk, chunk).clamp(max=nrows))
        eye = self.reg * torch.eye(d)
        for c, s in enumerate(range(0, nrows, chunk)):
            e = min(s + chunk, nrows)
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            r, f = ru[lo:hi] - s, Fx[ri[lo:hi]]
            A = torch.einsum("uk,kde->ude", L[s:e], G) + eye
            wl = (L[ru[lo:hi]] * Rf[ri[lo:hi]]).sum(-1)               # low-rank weight at positives
            coef = (1.0 + self.alpha - wl)
            A.index_add_(0, r, coef[:, None, None] * f[:, :, None] * f[:, None, :])
            b = torch.zeros(e - s, d).index_add_(0, r, (1.0 + self.alpha) * f)
            X[s:e] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def step(self):
        self._solve(self.U, self.V, self.L, self.R, self.eu, self.ei, self.U.shape[0])
        self._solve(self.V, self.U, self.R, self.L, self.ei, self.eu, self.V.shape[0])

    def full_scores(self, rows):
        return self.U[rows] @ self.V.T


def train_ials(model, iters, on_epoch):
    for ep in range(iters):
        model.step()
        if on_epoch(model, ep) is True:
            break


def _infonce(a, b, temp=0.2):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temp
    return F.cross_entropy(logits, torch.arange(len(a)))


def _elu1(x):
    return F.elu(x) + 1.0


def train(model, O, Y, P, Yhat, loss_type, lr, wd, epochs, batch, on_epoch,
          patience=5, pairs_per_epoch=None, seed=0, item_pop=None, gamma=0.1, log=None,
          cl_weight=0.1, aux_weight=1e-3, tau=0.2, fn=0.0, ipsw=0, n_neg=32, clip_w=5.0, fn_w=None, qmode="dr",
          adv_lr=1e-4, adv_interval=5, adv_eta=0, k_neg=64.0, adv_min_steps=100):
    """Pointwise losses: naive / IPS iterate over exposed pairs; DR samples
    pairs uniformly from all (u, i) (pseudo-label DR with a fixed imputation).
    Pairwise losses: 'bpr' (logged positives vs uniformly sampled items that
    are not logged positives) and 'pda' (BPR on elu'(f) * pop_i^gamma,
    Zhang et al., 2021; ranking uses f alone, i.e. popularity is removed at
    inference), 'macr' (pointwise on positives and sampled negatives with user
    and item branches) and 'simgcl' (BPR plus InfoNCE between two noisy views).
    'drjl' / 'mrdr' learn the imputation jointly (Wang et al., 2019; Guo et al.,
    2021): an imputation model predicts the label, the imputed error is the loss
    against it, and the imputation is fitted to the observed errors with weights
    1/p (DR-JL) or (1-p)/p^2 (MRDR).
    on_epoch(model, ep) evaluates the model and returns either the early-
    stopping criterion (higher is better; ``patience`` applies) or a bool
    (True: stop now), which lets the caller run its own per-split stopping."""
    fn_w = fn if fn_w is None else fn_w
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m, n = O.shape
    ADV = loss_type in ("advinfonce", "dradvinfonce")
    if ADV:
        advp = [model.Pu.weight, model.Pi.weight]
        opt = torch.optim.Adam([p for p in model.parameters() if not any(p is q for q in advp)], lr=lr)
        opt_adv = torch.optim.Adam(advp, lr=adv_lr); n_adv = 0
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = O * Y > 0
    imp = None
    if loss_type in ("drjl", "mrdr"):
        imp = MF(m, n, model.U.weight.shape[1])
        opt_imp = torch.optim.Adam(imp.parameters(), lr=lr)
        ou, oi = torch.nonzero(O > 0, as_tuple=True)
    if loss_type in ("infonce", "drinfonce", "drbpr", "drpn", "dradvinfonce"):
        # probability that a pair that is not a logged positive is a positive after all,
        # under Y independent of O given the imputation: Yhat (1 - p) / (1 - p Yhat)
        Q = (Yhat * (1.0 - P) / (1.0 - P * Yhat).clamp_min(1e-6)).clamp(0.0, 1.0)
        if qmode == "yhat":                     # ablation: imputation only, no propensity correction
            Q = Yhat.clamp(0.0, 1.0)
        elif qmode == "pop":                    # ablation: item popularity only, same mean as the DR weight
            qp = item_pop[None, :].expand_as(Q)
            Q = (qp * (Q.mean() / qp.mean())).clamp(0.0, 1.0)
        elif qmode == "shuffle":                # control: same values, no information
            gq = torch.Generator().manual_seed(seed + 7)
            Q = Q.flatten()[torch.randperm(Q.numel(), generator=gq)].reshape(Q.shape)
    if loss_type in ("bpr", "pda", "simgcl", "macr", "infonce", "drinfonce", "drbpr", "pn", "drpn",
                     "advinfonce", "dradvinfonce"):
        eu, ei = torch.nonzero(pos, as_tuple=True)
    else:
        eu, ei = torch.nonzero(O > 0, as_tuple=True)

    def _negs(u):
        J = torch.randint(0, n, (len(u), n_neg), generator=g)
        return torch.where(pos[u[:, None], J], torch.randint(0, n, (len(u), n_neg), generator=g), J)

    def _logits(u, i, J, Ue, Ve):
        ue = F.normalize(Ue[u], dim=-1); pe = F.normalize(Ve[i], dim=-1); ne = F.normalize(Ve[J], dim=-1)
        lp = (ue * pe).sum(-1) / tau; ln = (ue[:, None, :] * ne).sum(-1) / tau
        if model.pbeta:
            lp = lp + model.pbeta * model.prior[u, i]; ln = ln + model.pbeta * model.prior[u[:, None], J]
        if loss_type == "dradvinfonce":
            ln = ln + torch.log((1.0 - fn * Q[u[:, None], J]).clamp_min(1e-3))
        return lp, ln

    def _nll(lp, ln, logp):
        return torch.logaddexp(lp, torch.logsumexp(ln + logp, 1) + math.log(k_neg * n_neg)) - lp

    def _logp(u, J):
        return torch.log_softmax(torch.einsum('bd,bnd->bn', model.Pu.weight[u], model.Pi.weight[J]), 1)

    best, bad = -1e9, 0
    for ep in range(epochs):
        model.train()
        if ADV and ep > 0 and (adv_interval <= 1 or (ep + 3) % adv_interval == 0) and n_adv < adv_eta:
            with torch.no_grad():
                Ue_, Ve_ = model.embeddings()
            n_pos = min(len(eu), pairs_per_epoch) if pairs_per_epoch else len(eu)
            n_pass = max(1, math.ceil(n_pos / batch))
            S_round = max(n_pass, adv_min_steps)   # a round: one full pass, at least adv_min_steps steps
            kl = []; steps = 0
            while steps < S_round and n_pos > 0:   # cycle through reshuffled passes over the positives
                pa = torch.randperm(len(eu), generator=g)
                if pairs_per_epoch: pa = pa[:pairs_per_epoch]
                for s in range(0, len(pa), batch):
                    if steps >= S_round: break
                    u, i = eu[pa[s:s+batch]], ei[pa[s:s+batch]]; J = _negs(u)
                    with torch.no_grad(): lp, ln = _logits(u, i, J, Ue_, Ve_)
                    logp = _logp(u, J); la = -_nll(lp, ln, logp).mean()
                    opt_adv.zero_grad(); la.backward(); opt_adv.step()
                    with torch.no_grad():
                        kl.append(float((logp.detach().exp() * (logp.detach() + math.log(n_neg))).sum(1).mean()))
                    steps += 1
            kl_tail = kl[-10:]
            n_adv += 1
            print(f"    adv round {n_adv} ep {ep} steps {steps} KL {sum(kl_tail)/max(1, len(kl_tail)):.4f}", flush=True)
        if loss_type in ("dr", "drjl", "mrdr"):
            N = pairs_per_epoch or len(eu)
            su = torch.randint(0, m, (N,), generator=g)
            si = torch.randint(0, n, (N,), generator=g)
        else:
            perm = torch.randperm(len(eu), generator=g)
            if pairs_per_epoch:
                perm = perm[:pairs_per_epoch]
            su, si = eu[perm], ei[perm]
        for s in range(0, len(su), batch):
            u, i = su[s:s + batch], si[s:s + batch]
            emb = model.embeddings()
            if loss_type == "macr":
                j = torch.randint(0, n, (len(u),), generator=g)
                uu, ii = torch.cat([u, u]), torch.cat([i, j])
                y = torch.cat([torch.ones(len(u)), pos[u, j].float()])
                f = model(uu, ii, emb)
                bu_, bi_ = model.branches(uu, ii, emb)
                prob = (torch.sigmoid(f) * torch.sigmoid(bu_) * torch.sigmoid(bi_)).clamp(1e-6, 1 - 1e-6)
                loss = (F.binary_cross_entropy(prob, y) + aux_weight * F.binary_cross_entropy_with_logits(bu_, y)
                        + aux_weight * F.binary_cross_entropy_with_logits(bi_, y))
                reg = wd * sum((e ** 2).sum() for e in (emb[0][uu], emb[1][ii])) / len(uu)
            elif loss_type in ("infonce", "drinfonce"):
                J = torch.randint(0, n, (len(u), n_neg), generator=g)
                Jb = pos[u[:, None], J]
                J = torch.where(Jb, torch.randint(0, n, (len(u), n_neg), generator=g), J)
                ue = F.normalize(emb[0][u], dim=-1)
                pe = F.normalize(emb[1][i], dim=-1)
                ne = F.normalize(emb[1][J], dim=-1)
                lp = (ue * pe).sum(-1) / tau
                ln = (ue[:, None, :] * ne).sum(-1) / tau
                if loss_type == "drinfonce" and fn > 0:
                    ln = ln + torch.log((1.0 - fn * Q[u[:, None], J]).clamp_min(1e-3))
                nll = torch.logsumexp(torch.cat([lp[:, None], ln], 1), 1) - lp
                if loss_type == "drinfonce" and ipsw:
                    w = 1.0 / P[u, i]
                    w = (w / w.mean()).clamp_max(clip_w)
                    nll = nll * w / w.mean()
                loss = nll.mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][J])) / len(u)
            elif loss_type in ("advinfonce", "dradvinfonce"):
                J = _negs(u); lp, ln = _logits(u, i, J, emb[0], emb[1])
                with torch.no_grad(): logp = _logp(u, J)
                loss = _nll(lp, ln, logp).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][J])) / len(u)
            elif loss_type in ("pn", "drpn"):
                j = torch.randint(0, n, (len(u),), generator=g)
                j = torch.where(pos[u, j], torch.randint(0, n, (len(u),), generator=g), j)
                fp, fneg = model(u, i, emb), model(u, j, emb)
                wn = (1.0 - fn_w * Q[u, j]).clamp_min(1e-3) if loss_type == "drpn" else torch.ones(len(u))
                loss = (F.softplus(-fp) + wn * F.softplus(fneg)).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][j])) / len(u)
            elif loss_type in ("bpr", "pda", "simgcl", "drbpr"):
                j = torch.randint(0, n, (len(u),), generator=g)
                bad_neg = pos[u, j]
                j = torch.where(bad_neg, torch.randint(0, n, (len(u),), generator=g), j)
                fp, fneg = model(u, i, emb), model(u, j, emb)
                if loss_type == "pda":
                    fp = _elu1(fp) * item_pop[i].pow(gamma)
                    fneg = _elu1(fneg) * item_pop[j].pow(gamma)
                if loss_type == "drbpr":
                    wn = (1.0 - fn_w * Q[u, j]).clamp_min(1e-3)
                    loss = (-(wn * F.logsigmoid(fp - fneg)).mean()) / wn.mean()
                else:
                    loss = -F.logsigmoid(fp - fneg).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][j])) / len(u)
                if loss_type == "simgcl":
                    v1, v2 = model.embeddings(perturb=True), model.embeddings(perturb=True)
                    uq, iq = torch.unique(u), torch.unique(i)
                    # InfoNCE over at most 2,048 users and items of the batch
                    uq = uq[torch.randperm(len(uq), generator=g)[:2048]]
                    iq = iq[torch.randperm(len(iq), generator=g)[:2048]]
                    loss = loss + cl_weight * (_infonce(v1[0][uq], v2[0][uq]) + _infonce(v1[1][iq], v2[1][iq]))
            elif loss_type in ("drjl", "mrdr"):
                # 1) imputation step on logged pairs
                k = torch.randint(0, len(ou), (len(u),), generator=g)
                a, b = ou[k], oi[k]
                with torch.no_grad():
                    fa = model(a, b)
                e_obs = F.binary_cross_entropy_with_logits(fa, Y[a, b].float(), reduction="none")
                e_imp = F.binary_cross_entropy_with_logits(fa, torch.sigmoid(imp(a, b)), reduction="none")
                pa = P[a, b].float()
                w = 1.0 / pa if loss_type == "drjl" else (1.0 - pa) / pa ** 2
                l_imp = (w * (e_imp - e_obs) ** 2).sum() / w.sum()
                opt_imp.zero_grad()
                l_imp.backward()
                opt_imp.step()
                # 2) prediction step on uniformly sampled pairs
                logit = model(u, i, emb)
                with torch.no_grad():
                    yt = torch.sigmoid(imp(u, i))
                o = O[u, i].float()
                p = P[u, i].float()
                l_hat = F.binary_cross_entropy_with_logits(logit, yt, reduction="none")
                l_obs = F.binary_cross_entropy_with_logits(logit, Y[u, i].float(), reduction="none")
                loss = (l_hat + o / p * (l_obs - l_hat)).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i])) / len(u)
            else:
                logit = model(u, i, emb)
                y = Y[u, i].float()
                if loss_type == "naive":
                    loss = F.binary_cross_entropy_with_logits(logit, y)
                elif loss_type == "ips":
                    w = 1.0 / P[u, i].float()
                    loss = (w * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).sum() / w.sum()
                else:
                    o = O[u, i].float()
                    p = P[u, i].float()
                    yh = Yhat[u, i].float()
                    l_imp = F.binary_cross_entropy_with_logits(logit, yh, reduction="none")
                    l_obs = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                    loss = (l_imp + o / p * (l_obs - l_imp)).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i])) / len(u)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = on_epoch(model, ep)
        if isinstance(v, bool):
            if v:
                break
            continue
        if log:
            log(f"  ep {ep} crit {v:.4f}")
        if v > best:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if ADV:
        print(f"    adv rounds done {n_adv}/{adv_eta}", flush=True)
    return best
