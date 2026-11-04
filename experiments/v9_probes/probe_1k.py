"""Probe L (KuaiRand-1K): does the users' lifelong history OUTSIDE the random pool add to a tuned in-pool
recommender on the random-exposure test?  All weights selected per split on validation (by user, 0.3,
seeds 0..4); paired per-user comparison of the selected models.

Backbone (in-pool): item-item EASE fitted on the standard logs of all 27,285 KuaiRand-Pure users (pool
video ids coincide between Pure and 1K), applied to each 1K user's in-pool clicks, plus a popularity offset.
Channels from the out-of-pool log (training data only):
  tag lift   : smoothed exposure-conditional click rate of the user on the video's first tag minus the
               user's base rate (impressions without a click are the denominator)
  author cnt : log impressions of the user on the video's author's other videos
python3 experiments/v9_probes/probe_1k.py OUT.txt
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import REPO, Selector, ease_gram, rowstd, std
from drup.data import split_test
from drup.data2 import load_kuairand_rich

torch.set_num_threads(2)
out = sys.argv[1]
# optional temporal protocol: "--until" keeps only training impressions strictly before 2022-04-22 00:00
# (UTC+8), the first day of the random-exposure period, for every channel (in-pool, out-of-pool, Pure Gram)
UNTIL = 1650556800000 if "--until" in sys.argv else None
d = load_kuairand_rich(version="1k", n_users=None, train_until=UNTIL)
O, Y = d["O"].float(), d["Y"].float()
m, n = O.shape
items = d["items_raw"]
itf = d["item_feat"]
op = d["outpool"]
test = d["test"]
rows_users = sorted({u for u, _, _ in test})
D = dict(ds="kuairand1k", ks=(5, 10, 20), key="ndcg@10", by="user", S=5, rows=torch.tensor(rows_users),
         row_of={u: k for k, u in enumerate(rows_users)})
D["splits"] = [split_test(test, 0.3, s, by="user") for s in range(5)]
rows = D["rows"]

# in-pool item-item Gram from all Pure users' standard logs (clicks), on the 1K item index
pdir = f"{REPO}/data/raw/KuaiRand-Pure/data/"
ps = pd.concat([pd.read_csv(pdir + f, usecols=["user_id", "video_id", "is_click", "time_ms"]) for f in
                ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")])
if UNTIL is not None:
    ps = ps[ps.time_ms < UNTIL]
ps = ps[(ps.is_click > 0) & ps.video_id.isin(items)].drop_duplicates(["user_id", "video_id"])
pu = np.searchsorted(np.sort(ps.user_id.unique()), ps.user_id.values)
pi = np.searchsorted(items, ps.video_id.values)
Xp = torch.zeros(int(pu.max()) + 1, n)
Xp[torch.as_tensor(pu), torch.as_tensor(pi)] = 1.0
G = Xp.T @ Xp
popv = std(torch.log1p(Xp.sum(0)))[None, :].expand(len(rows), -1)
del Xp, ps
X = O * Y

# out-of-pool channels
cols = ["video_id", "author_id", "tag"]
want = pd.Index(op.video_id.unique())
vf = pd.concat([c[c.video_id.isin(want)] for c in pd.read_csv(d["outpool_item_feat_path"], usecols=cols,
                                                            chunksize=1_000_000)])
vf = vf.drop_duplicates("video_id").set_index("video_id")
vf["tag"] = vf["tag"].astype(str).str.split(",").str[0]
op = op.join(vf, on="video_id")
base = op.groupby("u").is_click.mean()
tu = np.concatenate([np.full(len(it), u) for u, it, _ in test])
ti = np.concatenate([it for _, it, _ in test])


def channel(key, what):
    g = op.groupby(["u", key]).is_click.agg(["sum", "count"]).reset_index()
    if what == "lift":
        g["v"] = (g["sum"] + 5 * g.u.map(base)) / (g["count"] + 5) - g.u.map(base)
    else:
        g["v"] = np.log1p(g["count"])
    vals = itf[key].astype(str).values if key == "tag" else itf[key].values
    if key == "tag":
        g[key] = g[key].astype(str)
    q = pd.DataFrame({"u": tu, "i": ti, key: vals[ti]}).merge(g[["u", key, "v"]], on=["u", key], how="left")
    S = torch.zeros(m, n)
    S[torch.as_tensor(q.u.values), torch.as_tensor(q.i.values)] = torch.as_tensor(q.v.fillna(0.0).values.astype(np.float32))
    S = S[rows]
    return S / S[S != 0].std().clamp_min(1e-9)


tag_lift = channel("tag", "lift")
auth_cnt = channel("author_id", "count")


def write(name, res):
    lines = [f"{name}: test {res['test_mean']:.4f} +- {res['test_sd']:.4f} (val {res['val_mean']:.4f}, "
             f"{res['n_configs']} cfgs) chosen {res['chosen']}"]
    print(lines[0], flush=True)
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n")
    json.dump(res, open(out.replace(".txt", f"_{name}.json"), "w"))


def peru(pul):
    a = {}
    for q in pul:
        for u, v in q.items():
            a.setdefault(u, []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


def paired(r1, r2, label):
    x, y = peru(r1["per_user"]), peru(r2["per_user"])
    us = sorted(set(x) & set(y))
    dd = np.array([x[u] - y[u] for u in us])
    hw = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
    s = f"  {label}: {1000*dd.mean():+.2f} [{1000*hw:.2f}] p={stats.ttest_1samp(dd, 0).pvalue:.3g} n={len(dd)}"
    print(s, flush=True)
    with open(out, "a") as f:
        f.write(s + "\n")


Ss = {lam: rowstd(X[rows] @ ease_gram(G, lam)) for lam in (300.0, 1000.0, 3000.0)}
res = {}
sel = Selector(D, "pop")
sel.add(popv, {})
res["pop"] = sel.result()
write("pop", res["pop"])
for name, chans in [("EASE+pop", []), ("EASE+pop+taglift", ["tag"]), ("EASE+pop+taglift+authcnt", ["tag", "auth"]),
                    ("pop+taglift+authcnt", ["tag", "auth", "noease"])]:
    sel = Selector(D, name)
    gt = (0.0, 0.25, 0.5, 1.0) if "tag" in chans else (0.0,)
    ga = (0.0, 0.25, 0.5) if "auth" in chans else (0.0,)
    for lam, b, g1, g2 in itertools.product(Ss, (0.0, 0.25, 0.5, 1.0), gt, ga):
        base_s = 0 * popv if "noease" in chans else Ss[lam]
        if "noease" in chans and lam != 300.0:
            continue
        sel.add(base_s + b * popv + g1 * tag_lift + g2 * auth_cnt, dict(lam=lam, beta=b, g_tag=g1, g_auth=g2))
    res[name] = sel.result()
    write(name, res[name])
paired(res["EASE+pop"], res["pop"], "EASE+pop - pop")
paired(res["EASE+pop+taglift"], res["EASE+pop"], "EASE+pop+taglift - EASE+pop")
paired(res["EASE+pop+taglift+authcnt"], res["EASE+pop"], "EASE+pop+taglift+authcnt - EASE+pop")
paired(res["pop+taglift+authcnt"], res["pop"], "pop+taglift+authcnt - pop")
