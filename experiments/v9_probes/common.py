"""Shared pieces of the v9 probes: data (drup/data2.py, stored KuaiRand setting), splits, validation
selection (per split, best validation configuration; test read only for it) and paired comparison with the
stored KuaiRand result files. Re-uses experiments/v8_probes/harness.py for Selector / compare."""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "v8_probes"))
from harness import Selector, report, std  # noqa: E402,F401
from drup.data import split_test  # noqa: E402
from drup.data2 import load_kuairand_rich  # noqa: E402


def load(extra_users=False, label="is_click", train_until=None):
    d = load_kuairand_rich(version="pure", n_users=10000, seed=0, label=label, extra_users=extra_users,
                           train_until=train_until)
    rows_users = sorted({u for u, _, _ in d["test"]})
    D = dict(ds="kuairand", ks=(5, 10, 20), key="ndcg@10", by="user", S=5, d=d,
             O=d["O"].float(), Y=d["Y"].float(), rows=torch.tensor(rows_users),
             row_of={u: k for k, u in enumerate(rows_users)})
    D["splits"] = [split_test(d["test"], 0.3, s, by="user") for s in range(5)]
    return D


def ease(X, lam):
    G = X.T @ X
    B = torch.linalg.inv(G + lam * torch.eye(G.shape[0]))
    B = -B / torch.diag(B)[None, :]
    B.fill_diagonal_(0.0)
    return B


def ease_gram(G, lam):
    B = torch.linalg.inv(G + lam * torch.eye(G.shape[0]))
    B = -B / torch.diag(B)[None, :]
    B.fill_diagonal_(0.0)
    return B


def rowstd(S):
    S = S - S.mean(1, keepdim=True)
    return S / S.std(1, keepdim=True).clamp_min(1e-9)


class FastSelector:
    """Selector with the vectorised drup.metrics.UserMetrics (same values as evaluate); same result format.
    Scores passed to add() are (len(rows), n) matrices indexed by D["row_of"]."""

    def __init__(self, D, name):
        from drup.metrics import UserMetrics
        self.D, self.name, self.n = D, name, 0
        self.um = [(UserMetrics(v, D["ks"], D["row_of"], "cpu"), UserMetrics(t, D["ks"], D["row_of"], "cpu"))
                   for v, t in D["splits"]]
        self.best = [(-1.0, None, None, None) for _ in D["splits"]]

    def add(self, S, cfg):
        self.n += 1
        vals = []
        for s, (uv, ut) in enumerate(self.um):
            v = float(uv.compute(S)[self.D["key"]].mean())
            vals.append(v)
            if v > self.best[s][0]:
                pu = ut.compute(S)[self.D["key"]]
                self.best[s] = (v, dict(cfg), float(pu.mean()), dict(zip(ut.users, pu.tolist())))
        return float(np.mean(vals))

    def result(self):
        return {"test_mean": float(np.mean([b[2] for b in self.best])),
                "test_sd": float(np.std([b[2] for b in self.best])),
                "val_mean": float(np.mean([b[0] for b in self.best])),
                "chosen": [b[1] for b in self.best],
                "per_user": [{str(u): v for u, v in b[3].items()} for b in self.best],
                "n_configs": self.n}


class BucketSelector(FastSelector):
    """FastSelector with users bucketed by candidate-list length (less padding; same values)."""

    EDGES = (32, 64, 128, 256, 100000)

    def __init__(self, D, name):
        from drup.metrics import UserMetrics
        self.D, self.name, self.n = D, name, 0

        def buckets(lst):
            out, lo = [], 0
            for hi in self.EDGES:
                part = [t for t in lst if lo < len(t[1]) <= hi]
                if any(len(it) > 0 and r.sum() > 0 for _, it, r in part):
                    out.append(UserMetrics(part, D["ks"], D["row_of"], "cpu"))
                lo = hi
            return out
        self.um = [(buckets(v), buckets(t)) for v, t in D["splits"]]
        self.best = [(-1.0, None, None, None) for _ in D["splits"]]

    def add(self, S, cfg):
        self.n += 1
        vals = []
        key = self.D["key"]
        for s, (bv, bt) in enumerate(self.um):
            tot = sum(float(u.compute(S)[key].sum()) for u in bv)
            v = tot / sum(len(u.users) for u in bv)
            vals.append(v)
            if v > self.best[s][0]:
                pu, us = [], []
                for u in bt:
                    pu += u.compute(S)[key].tolist()
                    us += u.users
                self.best[s] = (v, dict(cfg), float(np.mean(pu)), dict(zip(us, pu)))
        return float(np.mean(vals))
