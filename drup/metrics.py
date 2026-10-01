"""Ranking metrics on unbiased candidate sets."""

import numpy as np
import torch
from scipy.stats import rankdata


def evaluate(scores, test, ks=(5, 10), row_of=None, per_user=None):
    """scores: (rows, n) tensor; test: list of (user, items, relevance).

    row_of maps a user id to its row in ``scores`` (identity if None).
    Users without any relevant candidate are skipped (NDCG undefined).
    per_user: optional metric name; if given, also returns {user: value}.
    """
    out = {f"ndcg@{k}": [] for k in ks}
    users = []
    out.update({f"recall@{k}": [] for k in ks})
    out["auc"] = []
    S = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    for u, items, rel in test:
        if len(items) == 0 or rel.sum() == 0:
            continue
        users.append(u)
        r = u if row_of is None else row_of[u]
        s = S[r, items]
        order = np.argsort(-s, kind="stable")
        rs = rel[order]
        npos = rel.sum()
        for k in ks:
            top = rs[:k]
            disc = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = (top * disc).sum()
            ideal = disc[: int(min(npos, k))].sum()
            out[f"ndcg@{k}"].append(dcg / ideal)
            out[f"recall@{k}"].append(top.sum() / npos)
        # AUC of the candidate list (ties count one half); 1 if no negative
        nneg = len(rel) - npos
        if nneg > 0:
            rk = rankdata(s)
            out["auc"].append(float((rk[rel > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg)))
        else:
            out["auc"].append(1.0)
    agg = {k: float(np.mean(v)) for k, v in out.items()}
    if per_user is not None:
        return agg, dict(zip(users, [float(x) for x in out[per_user]]))
    return agg


class UserMetrics:
    """All ranking metrics of every test user at once (same values as
    ``evaluate``), vectorised over users on the device of the score matrix.

    Built once from the test list; ``compute(scores)`` returns {metric: (U,)
    float64 tensor} for the users that have a relevant candidate, in the order
    of ``self.users``. Aggregates over a split are means over its users.
    """

    def __init__(self, test, ks, row_of, device):
        keep = [(u, it, rel) for u, it, rel in test if len(it) > 0 and rel.sum() > 0]
        self.users = [u for u, _, _ in keep]
        self.pos = {u: k for k, u in enumerate(self.users)}
        self.ks = tuple(ks)
        U, L = len(keep), max(len(it) for _, it, _ in keep)
        items = np.zeros((U, L), dtype=np.int64)
        rel = np.zeros((U, L))
        valid = np.zeros((U, L), dtype=bool)
        for k, (_, it, r) in enumerate(keep):
            items[k, :len(it)], rel[k, :len(it)], valid[k, :len(it)] = it, r, True
        self.rows = torch.tensor([row_of[u] if row_of is not None else u for u in self.users], device=device)
        self.items = torch.tensor(items, device=device)
        self.rel = torch.tensor(rel, device=device, dtype=torch.float64)
        self.valid = torch.tensor(valid, device=device)
        self.npos = self.rel.sum(1)
        self.nneg = self.valid.sum(1) - self.npos
        self.disc = 1.0 / torch.log2(torch.arange(2, L + 2, device=device, dtype=torch.float64))

    def compute(self, scores):
        s = scores[self.rows[:, None], self.items].double()
        s = torch.where(self.valid, s, torch.full_like(s, float("inf")))     # pads sort last
        order = torch.sort(s, dim=1, stable=True)[1]                          # ascending: ties as in rankdata
        out = {}
        desc = torch.sort(torch.where(self.valid, -s, torch.full_like(s, float("inf"))), dim=1, stable=True)[1]
        rs = torch.gather(self.rel, 1, desc)
        for k in self.ks:
            top = rs[:, :k]
            d = self.disc[: top.shape[1]]
            dcg = (top * d).sum(1)
            cum = torch.cumsum(d, 0)
            ideal = cum[(torch.clamp(self.npos, max=k).long() - 1)]
            out[f"ndcg@{k}"] = dcg / ideal
            out[f"recall@{k}"] = top.sum(1) / self.npos
        # AUC with ties counted one half = tie-averaged rank sum
        ss = torch.gather(s, 1, order)
        L = ss.shape[1]
        pos = torch.arange(L, device=ss.device).expand_as(ss)
        new = torch.ones_like(ss, dtype=torch.bool)
        new[:, 1:] = ss[:, 1:] != ss[:, :-1]
        first = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), dim=1)[0]
        last_flag = torch.ones_like(new)
        last_flag[:, :-1] = new[:, 1:]
        last = torch.flip(torch.cummin(torch.flip(torch.where(last_flag, pos, torch.full_like(pos, L)), [1]), dim=1)[0], [1])
        avg = (first + last).double() / 2 + 1
        relsort = torch.gather(self.rel, 1, order)
        rank_sum = (avg * relsort).sum(1)
        auc = (rank_sum - self.npos * (self.npos + 1) / 2) / (self.npos * self.nneg.clamp(min=1))
        out["auc"] = torch.where(self.nneg > 0, auc, torch.ones_like(auc))
        return out
