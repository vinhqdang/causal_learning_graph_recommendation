"""Probe L27-LGN: LightGCN (the stored KuaiRand configuration: lr 1e-2, wd 1e-4, dim 64, 2 layers, BPR) trained
by the unchanged experiments/run_learned.py code on the temporal all-user setting (data/raw/kuairand_t.pt),
alone and with the LEAF lifelong channels added to its row-standardised score. Per split, (epoch, channel
weights, popularity offset) are selected on validation; the LightGCN-alone arm selects (epoch) only.
python3 experiments/v9_probes/probe_lgn_leaf.py OUT.txt
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO_)
_load = torch.load
torch.load = lambda p, *a, **k: _load("data/raw/kuairand_t.pt" if p == "data/raw/kuairand.pt" else p, *a, **k)
sys.path.insert(0, os.path.join(REPO_, "experiments"))
sys.path.insert(0, HERE)
import learned_ctx as LC  # noqa: E402
from common import BucketSelector, rowstd, std  # noqa: E402
from drup.data import split_test  # noqa: E402

out = os.path.abspath(sys.argv[1]) if os.path.isabs(sys.argv[1]) else os.path.join(REPO_, sys.argv[1])
ctx = LC.prepare("kuairand", "pop", 0.05, "add", 16, 5, 0.3, 3)
cache = _load("data/raw/kuairand_t.pt", weights_only=False)
test, users_raw = cache["test"], cache["users_raw"]
assert [u for u, _, _ in test] == ctx["rows"].tolist()
U = len(test)
Lmax = max(len(it) for _, it, _ in test)
cand = torch.zeros(U, Lmax, dtype=torch.long)
for k, (_, it, _) in enumerate(test):
    cand[k, :len(it)] = torch.as_tensor(it)
ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
D = dict(ds="kuairand", ks=(5, 10, 20), key="ndcg@10", by="user", S=5, rows=torch.arange(U),
         row_of={k: k for k in range(U)})
D["splits"] = [split_test(ctest, 0.3, s, by="user") for s in range(5)]
X = ctx["O"] * ctx["Y"]
pop = std(torch.log1p(X.sum(0)))[cand]
del X

# lifelong channels (identical construction to probe_27k.py, window "pre")
L = np.load("data/raw/lifelong_27k.npz", allow_pickle=True)
tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"])
base = (tot.c + 1.0) / (tot.n + 2.0)
itf = pd.read_csv("data/raw/KuaiRand-Pure/data/video_features_basic_pure.csv").set_index("video_id")
items = np.union1d(*[pd.read_csv(f"data/raw/KuaiRand-Pure/data/{f}", usecols=["video_id"]).video_id.unique()
                     for f in ("log_standard_4_08_to_4_21_pure.csv", "log_random_4_22_to_5_08_pure.csv")])
itf = itf.reindex(items).reset_index(drop=True)
assert len(items) == ctx["n"]


def keystr(col, k):
    if k == "tag":
        return col.astype(str).str.split(",").str[0]
    return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)


tu = np.concatenate([np.full(len(it), u) for u, it, _ in test])
ti = np.concatenate([it for _, it, _ in test])
t_pos = np.concatenate([np.arange(len(it)) for _, it, _ in test])
t_row = np.concatenate([np.full(len(it), k) for k, (_, it, _) in enumerate(test)])
chan = {}
for k, what in [("tag", "lift"), ("author_id", "cnt"), ("author_id", "lift"), ("music_id", "cnt")]:
    vals = pd.Index(L[f"{k}_values"])
    code_i = vals.get_indexer(keystr(itf[k], k))
    df = pd.DataFrame({"u_raw": L[f"{k}_pre_u"], "k": L[f"{k}_pre_k"]})
    df[["n", "c", "l"]] = L[f"{k}_pre_ncl"]
    b = df.u_raw.map(base).fillna(base.mean()).values
    df["v"] = (df.c + 5 * b) / (df.n + 5) - b if what == "lift" else np.log1p(df.n)
    q = pd.DataFrame({"u_raw": users_raw[tu], "k": code_i[ti]}).merge(df[["u_raw", "k", "v"]], on=["u_raw", "k"], how="left")
    v = q.v.fillna(0.0).values.astype(np.float32)
    S = torch.zeros(U, Lmax)
    S[torch.as_tensor(t_row), torch.as_tensor(t_pos)] = torch.as_tensor(v / (v[v != 0].std() if (v != 0).any() else 1.0))
    chan[f"{k}_{what}"] = S
del L

alone = BucketSelector(D, "LightGCN[t]")
plus = BucketSelector(D, "LightGCN+LEAF-channels[t]")
wgrid = {"tag_lift": (0.0, 0.5, 1.0), "author_id_cnt": (0.5, 1.0, 2.0), "author_id_lift": (0.0, 0.5)}
hist = []


def record(S, ep):
    Sc = torch.gather(rowstd(S.float()), 1, cand)
    v = alone.add(Sc, {"epoch": ep})
    for bta in (0.0, 0.25):
        for gs in itertools.product(*wgrid.values()):
            T = Sc + bta * pop
            for c, g in zip(wgrid, gs):
                if g:
                    T = T + g * chan[c]
            plus.add(T, dict(epoch=ep, beta=bta, **dict(zip(wgrid, gs))))
    hist.append(v)
    print(f"epoch {ep} val(LightGCN) {v:.4f} best plus val {np.mean([b[0] for b in plus.best]):.4f}", flush=True)
    best = int(np.argmax(hist))
    return len(hist) - 1 - best >= 3


ja = {"epochs": 30, "batch": 8192, "patience": 3, "threads": 3}
LC.train_cfg(ctx, "LightGCN", {"lr": 1e-2, "wd": 1e-4, "dim": 64, "layers": 2}, 0, ja, record)


def peru(pul):
    a = {}
    for q_ in pul:
        for u, v_ in q_.items():
            a.setdefault(u, []).append(v_)
    return {u: np.mean(v_) for u, v_ in a.items()}


res = {"alone": alone.result(), "plus": plus.result()}
x, y = peru(res["plus"]["per_user"]), peru(res["alone"]["per_user"])
us = sorted(set(x) & set(y))
dd = np.array([x[u] - y[u] for u in us])
hw = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
lines = [f"LightGCN[t] alone: test {res['alone']['test_mean']:.4f} +- {res['alone']['test_sd']:.4f} chosen {res['alone']['chosen']}",
         f"LightGCN[t] + LEAF channels: test {res['plus']['test_mean']:.4f} +- {res['plus']['test_sd']:.4f} chosen {res['plus']['chosen']}",
         f"  plus - alone: {1000*dd.mean():+.2f} [{1000*hw:.2f}] p={stats.ttest_1samp(dd, 0).pvalue:.3g} n={len(dd)}"]
print("\n".join(lines), flush=True)
with open(out, "a") as f:
    f.write("\n".join(lines) + "\n")
for k_, r in res.items():
    json.dump(r, open(out.replace(".txt", f"_{k_}.json"), "w"))
