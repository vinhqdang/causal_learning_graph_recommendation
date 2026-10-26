"""User-level comparison of DR-BPR-LightGCN (negatives down-weighted by the imputed probability of
being positive) with the trained baselines, per dataset. Learning-rate jobs are combined by
validation score per split; paired t-tests over users, Holm-adjusted within the dataset family."""
import glob, json, numpy as np
from scipy import stats


def combine(files, name, key):
    rs = [json.load(open(f))["results"][name] for f in files]
    per_user, per_split, chosen = [], [], []
    for s in range(len(rs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in rs]))
        per_user.append(rs[j]["per_user"][s]); per_split.append(rs[j]["per_split"][s][key]); chosen.append(rs[j]["chosen"][s])
    return {"per_user": per_user, "per_split": per_split, "chosen": chosen}


def pu(r):
    a = {}
    for q in r["per_user"]:
        for u, v in q.items():
            a.setdefault(u, []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


def load(pats, name):
    for p in pats:
        for g in glob.glob(p):
            r = json.load(open(g))["results"]
            if name in r:
                return r[name]


SETS = [("yahoo", "ndcg@5", "results/v4/learned_yahoo_drbpr_lr*.json", "results/v3/learned_yahoo_pop*.json",
         ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "DR-LightGCN", "SimGCL"]),
        ("kuairec", "ndcg@20", "results/v4/learned_kuairec_drbpr_lr*.json", "results/v3/learned_kuairec_pop*.json",
         ["LightGCN", "r-AdjNorm", "NAVIP", "DR-LightGCN", "iALS", "DR-JL", "MRDR", "LightGCN-pt"])]
for ds, key, pat, bpat, names in SETS:
    files = [f for f in sorted(glob.glob(pat)) if "bank" not in f]
    dr = combine(files, "DR-BPR-LGN-1", key)
    print(f"\n{ds}: DR-BPR-LGN over {len(files)} learning-rate jobs, mean {key} {np.mean(dr['per_split']):.4f}")
    rows = []
    for n in names:
        b = load([bpat], n)
        if b is None:
            continue
        x, y = pu(dr), pu(b)
        us = sorted(set(x) & set(y)); d = np.array([x[u] - y[u] for u in us])
        rows.append((n, len(us), 1000 * d.mean(), 1000 * stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
                     stats.ttest_1samp(d, 0).pvalue, b["test"][key][0], b["n_configs"]))
    ps = np.array([r[4] for r in rows]); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    for r, pa in zip(rows, adj):
        print(f"  vs {r[0]:12s} baseline {r[5]:.4f} ({r[6]} cfgs)  diff {r[2]:+7.2f}e-3 [{r[3]:.2f}] users {r[1]}  p {r[4]:.4f}  p_Holm {pa:.4f}")
