"""Shared pieces of the v11 probes and diagnostics (paper/v7/DESIGN.md).

Setting: the temporal all-user KuaiRand setting of results/v10 (drup/data2.py, version "pure", n_users=None,
train_until = 2022-04-22 00:00 UTC+8), the candidate layout and the five user-level validation / test splits of
experiments/run_leaf.py (protocol_split, frac_val 0.3, seeds 0..4), the bucketed per-split validation selector of
drup/leaf.py, and paired comparison with the stored results/v10 JSONs (per-user keys = dataset row ids).

bundle()   data dict (cached under $V11_CACHE, default /tmp/v11_bundle.pt; rebuilt when missing)
Ctx        layout, splits, candidate masks, helpers to turn item vectors / user-item tables into bucketed scores
stored()   the stored methods of Stage 1 (LEAF-LGN, LEAF-tied, QP family, r-AdjNorm, LightGCN, iALS)
pair()     per-user paired difference (x1e-3, 95% half-width, t p) and the Nadeau-Bengio split-level p
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from drup.leaf import CUT_MS, BucketSelector, Layout, protocol_split  # noqa: E402
from significance import nadeau_bengio  # noqa: E402

KEY = "ndcg@10"
FRAC_VAL = 0.3
CACHE = os.environ.get("V11_CACHE", "/tmp/v11_bundle.pt")
V10 = os.path.join(ROOT, "results", "v10")
OUTDIR = os.path.join(ROOT, "results", "v11_probes")
DAY_MS = 86400000
T_UPLOAD0 = 1649433600000      # 2022-04-09 00:00 UTC+8, first upload day of the pool videos


def bundle():
    """Temporal all-user data without the dense matrices (rebuilt from the log by dense_OY)."""
    if os.path.exists(CACHE):
        return torch.load(CACHE, weights_only=False)
    from drup.data2 import load_kuairand_rich
    d = load_kuairand_rich(version="pure", n_users=None, train_until=CUT_MS, dense_dtype=torch.float32,
                           with_signals=False)
    keep = {k: d[k] for k in ("test", "log", "test_log", "item_feat", "users_raw", "items_raw", "user_feat",
                              "user_feat_cols", "meta")}
    keep["shape"] = tuple(d["O"].shape)
    torch.save(keep, CACHE)
    return keep


def dense_OY(B):
    """O (impressed before the cut) and Y (clicked) as dense float32, as the loader builds them."""
    m, n = B["shape"]
    lg = B["log"]
    O = torch.zeros(m, n)
    O[torch.as_tensor(lg.u.values), torch.as_tensor(lg.i.values)] = 1.0
    g = lg.groupby(["u", "i"]).is_click.max()
    Y = torch.zeros(m, n)
    Y[torch.as_tensor(g.index.get_level_values(0).values), torch.as_tensor(g.index.get_level_values(1).values)] = \
        torch.as_tensor(g.values.astype(np.float32))
    return O, Y * O


class Ctx:
    def __init__(self, B, seeds=5):
        self.B = B
        self.test = B["test"]
        self.m, self.n = B["shape"]
        self.layout = Layout(self.test)
        ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(self.test)]
        self.splits = [protocol_split(ctest, s, FRAC_VAL) for s in range(seeds)]
        self.row_ids = np.array([u for u, _, _ in self.test], dtype=np.int64)
        self.len = [torch.as_tensor([len(self.test[k][1]) for k in bk["k"]]) for bk in self.layout.buckets]
        self.mask = [torch.arange(bk["cand"].shape[1])[None, :] < ln[:, None]
                     for bk, ln in zip(self.layout.buckets, self.len)]
        self.rel = []
        for bk in self.layout.buckets:
            R = torch.zeros(bk["cand"].shape)
            for r, k in enumerate(bk["k"]):
                R[r, :len(self.test[k][2])] = torch.as_tensor(self.test[k][2])
            self.rel.append(R)
        # test-time position: time of the user's random impression of each candidate (first one), ms
        tl = B["test_log"]
        first = tl.groupby(["u", "i"]).time_ms.min()
        self.cand_time = []
        for bk, mk in zip(self.layout.buckets, self.mask):
            rows = np.repeat(bk["rows"].numpy(), bk["cand"].shape[1])
            idx = pd.MultiIndex.from_arrays([rows, bk["cand"].numpy().ravel()])
            t = first.reindex(idx).values.reshape(bk["cand"].shape)
            self.cand_time.append(torch.as_tensor(np.nan_to_num(t, nan=0).astype(np.float64)))

    def selector(self):
        return BucketSelector(self.layout, self.splits, (10,), KEY, self.row_ids)

    def item(self, v):
        """Bucketed scores of an item vector v (n,)."""
        v = torch.as_tensor(v, dtype=torch.float32)
        return [v[bk["cand"]] for bk in self.layout.buckets]

    def table(self, fn):
        """Bucketed scores fn(rows, cand) -> (U_b, L_b)."""
        return [fn(bk["rows"], bk["cand"]).float() for bk in self.layout.buckets]

    def rowstd(self, S):
        """Per-user standardisation over the real candidates (padding excluded, set to 0)."""
        out = []
        for s, mk in zip(S, self.mask):
            s = torch.where(mk, s, torch.zeros_like(s))
            cnt = mk.sum(1, keepdim=True).clamp_min(1)
            mu = s.sum(1, keepdim=True) / cnt
            var = (torch.where(mk, (s - mu) ** 2, torch.zeros_like(s))).sum(1, keepdim=True) / cnt
            out.append(torch.where(mk, (s - mu) / var.sqrt().clamp_min(1e-9), torch.zeros_like(s)))
        return out

    def lin(self, parts, ws):
        out = None
        for P, w in zip(parts, ws):
            if w == 0:
                continue
            out = [w * p for p in P] if out is None else [o + w * p for o, p in zip(out, P)]
        return out if out is not None else [torch.zeros_like(p) for p in parts[0]]

    def evaluate_fixed(self, S):
        """Per-user nDCG@10 of one score (no selection), as a stored-format record (all splits same config)."""
        sel = self.selector()
        sel.add(S, {"fixed": True})
        r = sel.result()
        return {"per_user": r["per_user"], "per_split": r["per_split"], "val_best": r["val_best"],
                "chosen": r["chosen"], "test": r["test_mean"]}


def record(sel):
    r = sel.result()
    return {"per_user": r["per_user"], "per_split": r["per_split"], "val_best": r["val_best"],
            "chosen": r["chosen"], "test": r["test_mean"], "n_configs": r["n_configs"]}


def _combine(recs):
    pu, ps, ch, vb = [], [], [], []
    for s in range(len(recs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in recs]))
        pu.append(recs[j]["per_user"][s])
        ps.append(recs[j]["per_split"][s])
        ch.append(recs[j]["chosen"][s])
        vb.append(recs[j]["val_best"][s])
    return {"per_user": pu, "per_split": [{KEY: p[KEY]} for p in ps], "chosen": ch, "val_best": vb,
            "test": float(np.mean([p[KEY] for p in ps]))}


def stored():
    res = {}
    lg = json.load(open(os.path.join(V10, "leaf_lgn.json")))["results"]
    res["LEAF-LGN"] = _combine([lg["LEAF-LGN"]])
    lt = json.load(open(os.path.join(V10, "leaf_tied.json")))["results"]
    res["LEAF-tied"] = _combine([lt["LEAF-tied"]])
    base = json.load(open(os.path.join(V10, "learned_kuairand_t_base.json")))["results"]
    for m in ("LightGCN", "r-AdjNorm", "iALS"):
        if m in base:
            res[m] = _combine([base[m]])
    qp = []
    for f in sorted(glob.glob(os.path.join(V10, "learned_kuairand_t_qp*.json"))):
        r = json.load(open(f))["results"]
        qp += [r[m] for m in ("DR-BPR-LGN-1", "DR-BPR-LGN-p2", "DR-BPR-LGN-p") if m in r]
    res["QP family"] = _combine(qp)
    return res


def peru(per_user):
    acc = {}
    for q in per_user:
        for u, v in q.items():
            acc.setdefault(u, []).append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def pair(a, b):
    """a - b: per-user paired difference x1e-3 [95% half-width], t p, Nadeau-Bengio p over split means."""
    x, y = peru(a["per_user"]), peru(b["per_user"])
    us = sorted(set(x) & set(y))
    d = np.array([x[u] - y[u] for u in us])
    n = len(d)
    if np.abs(d).max() < 1e-9:
        return {"n": n, "diff": 0.0, "half": 0.0, "p": 1.0, "nb": 1.0}
    half = 1000 * stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    nb = nadeau_bengio({"per_split": a["per_split"]}, {"per_split": b["per_split"]}, KEY, FRAC_VAL)
    ds = [1000 * (p[KEY] - q[KEY]) for p, q in zip(a["per_split"], b["per_split"])]
    return {"n": n, "diff": 1000 * float(d.mean()), "half": float(half),
            "p": float(stats.ttest_1samp(d, 0.0).pvalue), "nb": float(nb), "split_diffs": ds}


def fmt(r):
    sd = " ".join(f"{x:+.1f}" for x in r.get("split_diffs", []))
    return f"{r['diff']:+7.2f} [{r['half']:4.2f}] p={r['p']:.1e} NB p={r['nb']:.3f} (splits {sd})"


class Out:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
