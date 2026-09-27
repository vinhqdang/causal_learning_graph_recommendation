"""Trained baselines: MF / LightGCN with naive, IPS or DR objectives, and
NAVIP (inverse-propensity weighted neighbour aggregation for LightGCN)."""

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
    """LightGCN over a (possibly weighted) user-item graph of positive edges."""

    def __init__(self, m, n, d, edges_u, edges_i, weights, layers=2):
        super().__init__(m, n, d)
        self.m, self.n, self.layers = m, n, layers
        w = weights.float()
        du = torch.zeros(m).index_add_(0, edges_u, w).clamp_min(1e-8)
        di = torch.zeros(n).index_add_(0, edges_i, w).clamp_min(1e-8)
        val = w / (du[edges_u].sqrt() * di[edges_i].sqrt())
        idx = torch.stack([torch.cat([edges_u, edges_i + m]), torch.cat([edges_i + m, edges_u])])
        self.A = torch.sparse_coo_tensor(idx, torch.cat([val, val]), (m + n, m + n)).coalesce()

    def embeddings(self):
        E = torch.cat([self.U.weight, self.V.weight])
        acc = E
        for _ in range(self.layers):
            E = torch.sparse.mm(self.A, E)
            acc = acc + E
        acc = acc / (self.layers + 1)
        return acc[: self.m], acc[self.m:]


def train(model, O, Y, P, Yhat, loss_type, lr, wd, epochs, batch, val_fn,
          patience=5, pairs_per_epoch=None, seed=0, log=None):
    """Pointwise training. naive/IPS losses iterate over exposed pairs; the
    DR loss samples pairs uniformly from all (u, i) (pseudo-label DR with a
    fixed imputation Yhat). Early stopping on val_fn (higher is better)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    m, n = O.shape
    eu, ei = torch.nonzero(O > 0, as_tuple=True)
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        if loss_type == "dr":
            N = pairs_per_epoch or len(eu)
            su = torch.randint(0, m, (N,), generator=g)
            si = torch.randint(0, n, (N,), generator=g)
        else:
            perm = torch.randperm(len(eu), generator=g)
            su, si = eu[perm], ei[perm]
        for s in range(0, len(su), batch):
            u, i = su[s:s + batch], si[s:s + batch]
            emb = model.embeddings()
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
            v = val_fn(model)
        if log:
            log(f"  ep {ep} val {v:.4f}")
        if v > best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best
