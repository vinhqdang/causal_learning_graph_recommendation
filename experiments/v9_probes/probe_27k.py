"""Probe L27 (KuaiRand-Pure test, all users, lifelong histories from KuaiRand-27K): temporal protocol
(every training signal strictly before 2022-04-22 00:00 UTC+8); per-split validation selection (by user,
frac_val 0.3, seeds 0..4); paired per-user comparisons.

Arms (all with a popularity offset beta):
  EASE(click)                 item-item EASE on in-pool clicks
  EASE(click + w*shown)       impressions without click enter the input with weight w (policy channel)
  + lifelong channels         from data/raw/lifelong_27k.npz ("pre" window): per (user, key value) on videos
                              outside the pool: exposure-conditional click lift and log impressions, for
                              key in {tag, author, music}
python3 experiments/v9_probes/probe_27k.py OUT.txt [--window pre|all] [--userkey identity]
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import REPO, ease, rowstd, std
from common import FastSelector as Selector
from drup.data import split_test
from drup.data2 import load_kuairand_rich

torch.set_num_threads(3)
out = sys.argv[1]
WIN = "all" if "--window" in sys.argv and sys.argv[sys.argv.index("--window") + 1] == "all" else "pre"
CUT = 1650556800000
d = load_kuairand_rich(version="pure", n_users=None, train_until=(CUT if WIN == "pre" else None),
                       dense_dtype=torch.float32, with_signals=False)
O, Y = d["O"], d["Y"]
m, n = O.shape
users_raw = d["users_raw"]
test = d["test"]
rows_users = sorted({u for u, _, _ in test})
D = dict(ds="kuairand", ks=(5, 10, 20), key="ndcg@10", by="user", S=5, rows=torch.tensor(rows_users),
         row_of={u: k for k, u in enumerate(rows_users)})
D["splits"] = [split_test(test, 0.3, s, by="user") for s in range(5)]
rows = D["rows"]
print(f"users {m} items {n} test users {len(test)} window {WIN}", flush=True)
X = O * Y
pop = std(torch.log1p(X.sum(0)))[None, :].expand(len(rows), -1)
NC = O - X

NPZ = sys.argv[sys.argv.index("--npz") + 1] if "--npz" in sys.argv else f"{REPO}/data/raw/lifelong_27k.npz"
L = np.load(NPZ, allow_pickle=True)
itf = d["item_feat"]


def keystr(col, k):
    if k == "tag":
        return col.astype(str).str.split(",").str[0]
    return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)


tu = np.concatenate([np.full(len(it), u) for u, it, _ in test])
ti = np.concatenate([it for _, it, _ in test])
tot = pd.DataFrame(L[f"tot_{WIN}"], index=L[f"tot_{WIN}_user"], columns=["n", "c", "l"])
base = (tot.c + 1.0) / (tot.n + 2.0)
print(f"users with lifelong history: {np.isin(users_raw, tot.index).mean():.3f}", flush=True)
chan = {}
NEED = ("tag_lift", "author_id_cnt", "author_id_lift", "music_id_cnt")
for k in ("tag", "author_id", "music_id"):
    vals = pd.Index(L[f"{k}_values"])
    code_i = vals.get_indexer(keystr(itf[k], k))                     # pool item -> key code (-1 none)
    df = pd.DataFrame({"u_raw": L[f"{k}_{WIN}_u"], "k": L[f"{k}_{WIN}_k"]})
    df[["n", "c", "l"]] = L[f"{k}_{WIN}_ncl"]
    b = df.u_raw.map(base).fillna(base.mean()).values
    df["lift"] = (df.c + 5 * b) / (df.n + 5) - b
    df["cnt"] = np.log1p(df.n)
    q = pd.DataFrame({"u": tu, "i": ti, "u_raw": users_raw[tu], "k": code_i[ti]})
    q = q.merge(df[["u_raw", "k", "lift", "cnt"]], on=["u_raw", "k"], how="left")
    for what in ("lift", "cnt"):
        if f"{k}_{what}" not in NEED:
            continue
        S = torch.zeros(m, n)
        S[torch.as_tensor(q.u.values), torch.as_tensor(q.i.values)] = torch.as_tensor(q[what].fillna(0.0).values.astype(np.float32))
        S = S[rows]
        nz = S[S != 0]
        chan[f"{k}_{what}"] = S / (nz.std().clamp_min(1e-9) if len(nz) else 1.0)
        print(f"channel {k}_{what}: coverage of test candidates {q[what].notna().mean():.3f}", flush=True)
del L


def write(name, res):
    s = (f"{name}: test {res['test_mean']:.4f} +- {res['test_sd']:.4f} (val {res['val_mean']:.4f}, "
         f"{res['n_configs']} cfgs) chosen {res['chosen']}")
    print(s, flush=True)
    with open(out, "a") as f:
        f.write(s + "\n")
    json.dump(res, open(out.replace(".txt", f"_{name}.json"), "w"))


def peru(pul):
    a = {}
    for q_ in pul:
        for u, v in q_.items():
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


with open(out, "a") as f:
    f.write(f"== window {WIN}; users {m}; test users {len(test)}\n")
res = {}
sel = Selector(D, "pop")
sel.add(pop, {})
res["pop"] = sel.result()
write("pop", res["pop"])
base_s = {}
for lam, w in itertools.product((3000.0, 10000.0), (0.0, 0.5)):
    Z = X + w * NC
    base_s[(lam, w)] = rowstd(Z[rows] @ ease(Z, lam))
    del Z
del X, NC, O, Y, d
for name, wset, chs in [("EASE(click)+pop", (0.0,), []),
                        ("EASE(click+0.5shown)+pop", (0.5,), []),
                        ("EASE(click+0.5shown)+pop+lifelong", (0.5,), ["tag_lift", "author_id_cnt", "author_id_lift",
                                                                       "music_id_cnt"]),
                        ("EASE(click)+pop+lifelong", (0.0,), ["tag_lift", "author_id_cnt", "author_id_lift",
                                                             "music_id_cnt"]),
                        ("pop+lifelong", None, ["tag_lift", "author_id_cnt", "author_id_lift", "music_id_cnt"])]:
    sel = Selector(D, name)
    grids = [(0.0, 0.5)] * len(chs) if len(chs) > 2 else [(0.0, 0.25, 0.5, 1.0)] * len(chs)
    for (lam, w), bta in itertools.product(base_s if wset else [(None, None)], (0.0, 0.25, 0.5)):
        if wset is not None and w not in wset:
            continue
        S0 = base_s[(lam, w)] if wset is not None else 0 * pop
        for gs in itertools.product(*[(0.0, 0.5, 1.0)] * len(chs)):
            S = S0 + bta * pop
            for c, g in zip(chs, gs):
                if g:
                    S = S + g * chan[c]
            sel.add(S, dict(lam=lam, w=w, beta=bta, **dict(zip(chs, gs))))
    res[name] = sel.result()
    write(name, res[name])
paired(res["EASE(click+0.5shown)+pop"], res["EASE(click)+pop"], "policy channel: EASE(click+0.5shown) - EASE(click)")
paired(res["EASE(click+0.5shown)+pop+lifelong"], res["EASE(click+0.5shown)+pop"], "lifelong channels on top")
paired(res["EASE(click)+pop+lifelong"], res["EASE(click)+pop"], "lifelong channels on EASE(click)")
paired(res["EASE(click+0.5shown)+pop+lifelong"], res["EASE(click)+pop"], "full - EASE(click)")
paired(res["pop+lifelong"], res["pop"], "pop+lifelong - pop")
