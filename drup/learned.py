"""Trained baselines: MF / LightGCN with pointwise (naive, IPS, DR, DR-JL,
MRDR) or pairwise (BPR, PDA, MACR-style) objectives, NAVIP (inverse-propensity
weighted neighbour aggregation), r-AdjNorm (LightGCN with normalisation
exponent r), SimGCL (LightGCN with noise-based contrastive views) and iALS
(weighted matrix factorisation by alternating least squares)."""

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

    def embeddings(self):
        return self.U.weight, self.V.weight

    def forward(self, u, i, emb=None):
        Ue, Ve = emb if emb is not None else self.embeddings()
        return (Ue[u] * Ve[i]).sum(-1) + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.b0

    def full_scores(self, rows):
        Ue, Ve = self.embeddings()
        return Ue[rows] @ Ve.T + self.bu.weight[rows] + self.bi.weight.T + self.b0


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
          cl_weight=0.1, aux_weight=1e-3):
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
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    m, n = O.shape
    pos = O * Y > 0
    imp = None
    if loss_type in ("drjl", "mrdr"):
        imp = MF(m, n, model.U.weight.shape[1])
        opt_imp = torch.optim.Adam(imp.parameters(), lr=lr)
        ou, oi = torch.nonzero(O > 0, as_tuple=True)
    if loss_type in ("bpr", "pda", "simgcl", "macr"):
        eu, ei = torch.nonzero(pos, as_tuple=True)
    else:
        eu, ei = torch.nonzero(O > 0, as_tuple=True)
    best, bad = -1e9, 0
    for ep in range(epochs):
        model.train()
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
            elif loss_type in ("bpr", "pda", "simgcl"):
                j = torch.randint(0, n, (len(u),), generator=g)
                bad_neg = pos[u, j]
                j = torch.where(bad_neg, torch.randint(0, n, (len(u),), generator=g), j)
                fp, fn = model(u, i, emb), model(u, j, emb)
                if loss_type == "pda":
                    fp = _elu1(fp) * item_pop[i].pow(gamma)
                    fn = _elu1(fn) * item_pop[j].pow(gamma)
                loss = -F.logsigmoid(fp - fn).mean()
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
    return best
