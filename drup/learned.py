"""Trained baselines: MF / LightGCN with pointwise (naive, IPS, DR) or
pairwise (BPR, PDA) objectives, NAVIP (inverse-propensity weighted neighbour
aggregation) and r-AdjNorm (LightGCN with normalisation exponent r)."""

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


def _elu1(x):
    return F.elu(x) + 1.0


def train(model, O, Y, P, Yhat, loss_type, lr, wd, epochs, batch, on_epoch,
          patience=5, pairs_per_epoch=None, seed=0, item_pop=None, gamma=0.1, log=None):
    """Pointwise losses: naive / IPS iterate over exposed pairs; DR samples
    pairs uniformly from all (u, i) (pseudo-label DR with a fixed imputation).
    Pairwise losses: 'bpr' (logged positives vs uniformly sampled items that
    are not logged positives) and 'pda' (BPR on elu'(f) * pop_i^gamma,
    Zhang et al., 2021; ranking uses f alone, i.e. popularity is removed at
    inference). on_epoch(model, ep) evaluates the model and returns the
    early-stopping criterion (higher is better)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    m, n = O.shape
    pos = O * Y > 0
    if loss_type in ("bpr", "pda"):
        eu, ei = torch.nonzero(pos, as_tuple=True)
    else:
        eu, ei = torch.nonzero(O > 0, as_tuple=True)
    best, bad = -1e9, 0
    for ep in range(epochs):
        model.train()
        if loss_type == "dr":
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
            if loss_type in ("bpr", "pda"):
                j = torch.randint(0, n, (len(u),), generator=g)
                bad_neg = pos[u, j]
                j = torch.where(bad_neg, torch.randint(0, n, (len(u),), generator=g), j)
                fp, fn = model(u, i, emb), model(u, j, emb)
                if loss_type == "pda":
                    fp = _elu1(fp) * item_pop[i].pow(gamma)
                    fn = _elu1(fn) * item_pop[j].pow(gamma)
                loss = -F.logsigmoid(fp - fn).mean()
                reg = wd * sum((e ** 2).sum() for e in (emb[0][u], emb[1][i], emb[1][j])) / len(u)
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
        if log:
            log(f"  ep {ep} crit {v:.4f}")
        if v > best:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best
