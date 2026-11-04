"""Diagnostic D3 (KuaiRand-1K): does a user's history OUTSIDE the random pool (11.7K impressions per user on
4.4M other videos) predict which random-pool videos the user clicks? Content bridges: tag, author, music.
Scores are fitted on training logs only; the test is the random log (all 993 users, no selection; a
diagnostic of signal, not a method result).
python3 experiments/v9_probes/diag_outpool.py OUT.txt"""
import sys

import numpy as np
import pandas as pd
import torch

from common import REPO  # noqa: F401
from drup.data2 import load_kuairand_rich
from drup.metrics import evaluate

out = open(sys.argv[1], "a")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")
    out.flush()


d = load_kuairand_rich(version="1k", n_users=None)
O, Y = d["O"].float(), d["Y"].float()
m, n = O.shape
op = d["outpool"]
itf = d["item_feat"]
# features of the out-of-pool videos (chunked read of the 4.4M-row file)
cols = ["video_id", "author_id", "music_id", "tag"]
want = pd.Index(op.video_id.unique())
vf = pd.concat([c[c.video_id.isin(want)] for c in pd.read_csv(d["outpool_item_feat_path"], usecols=cols, chunksize=1_000_000)])
vf = vf.drop_duplicates("video_id").set_index("video_id")
vf["tag"] = vf["tag"].astype(str).str.split(",").str[0]
op = op.join(vf, on="video_id")
P(f"D3 out-of-pool impressions {len(op)}, with features {op.author_id.notna().mean():.3f}")
pool_auth = set(itf.author_id.dropna().tolist())
P(f"D3 share of out-of-pool impressions whose author also has a pool video {op.author_id.isin(pool_auth).mean():.4f}")
pop = (O * Y).sum(0)
test = d["test"]


def ev(S):
    return round(evaluate(S, test, (10,))["ndcg@10"], 4)


P("D3 random", ev(torch.rand(m, n)), " pop(in-pool clicks)", ev(pop[None, :].expand(m, -1)))
tu = np.concatenate([np.full(len(it), u) for u, it, _ in test])
ti = np.concatenate([it for _, it, _ in test])
for key in ["tag", "author_id", "music_id"]:
    # smoothed user click rate on the key value, from out-of-pool history (minus the user's base rate)
    g = op.groupby(["u", key]).is_click.agg(["sum", "count"]).reset_index()
    base = op.groupby("u").is_click.mean()
    g["aff"] = (g["sum"] + 5 * g.u.map(base)) / (g["count"] + 5) - g.u.map(base)
    g["lcount"] = np.log1p(g["count"])
    vals = itf[key].astype(str).values if key == "tag" else itf[key].values
    if key == "tag":
        g[key] = g[key].astype(str)
    q = pd.DataFrame({"u": tu, "i": ti, key: vals[ti]}).merge(g[["u", key, "aff", "lcount"]], on=["u", key], how="left")
    for what in ["aff", "lcount"]:
        S = torch.zeros(m, n)
        S[torch.as_tensor(q.u.values), torch.as_tensor(q.i.values)] = torch.as_tensor(q[what].fillna(0.0).values.astype(np.float32))
        P(f"D3 {key:9s} {what:6s} coverage {q[what].notna().mean():.3f} alone", ev(S + 1e-6 * torch.rand(m, n)), " + log pop (std)",
          ev(S / S[S != 0].std().clamp_min(1e-9) * 0.5 + (torch.log1p(pop) / torch.log1p(pop).std())[None, :]))

# Item level: does the author's / tag's response on OTHER videos (out-of-pool log) predict the uniform
# click rate of a pool video? Uniform rates from the full KuaiRand-Pure random log (diagnostic only).
rp = pd.read_csv(f"{REPO}/data/raw/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv", usecols=["video_id", "is_click"])
ur = rp.groupby("video_id").is_click.agg(["mean", "size"])
ur = ur[ur["size"] >= 50]
pool = pd.DataFrame({"video_id": d["items_raw"], "author_id": itf.author_id.values, "tag": itf.tag.values,
                     "music_id": itf.music_id.values})
mu = op.is_click.mean()
for key in ["author_id", "tag", "music_id"]:
    g = op.groupby(key).is_click.agg(["sum", "count"])
    g["ctr"] = (g["sum"] + 20 * mu) / (g["count"] + 20)
    g["lclk"] = np.log1p(g["sum"])
    j = pool.join(g, on=key).set_index("video_id").join(ur, how="inner")
    cov = j["ctr"].notna().mean()
    for c in ["ctr", "lclk"]:
        jj = j[j[c].notna()]
        P(f"D3 item-level {key:9s} {c:5s} coverage {cov:.3f} spearman with uniform pool rate "
          f"{jj[[c, 'mean']].corr('spearman').iloc[0, 1]:+.3f} (n={len(jj)})")
