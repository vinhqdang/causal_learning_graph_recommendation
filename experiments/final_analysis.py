"""DR-BPR-LightGCN with an imputation offset: full grid (beta in {0, 0.5, 1, 2}) per dataset.
Jobs (one per beta set and learning rate) are combined by validation score per split; paired t-tests
over users against the trained baselines, Holm-adjusted within the dataset."""
import glob, json, os, sys
import collections
import numpy as np
from scipy import stats

ld = lambda f: json.load(open(f))


def combine(jobs, key):
    rs = [ld(f)["results"][n] for f, n in jobs if os.path.exists(f)]
    pu, ps, ch = [], [], []
    for s in range(len(rs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in rs]))
        pu.append(rs[j]["per_user"][s]); ps.append(rs[j]["per_split"][s][key]); ch.append(rs[j]["chosen"][s])
    return {"per_user": pu, "per_split": ps, "chosen": ch, "n_jobs": len(rs)}


def peru(r):
    a = {}
    for q in r["per_user"]:
        for u, v in q.items():
            a.setdefault(u, []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


def cmp(a, b):
    x, y = peru(a), peru(b)
    us = sorted(set(x) & set(y)); d = np.array([x[u] - y[u] for u in us])
    return 1000 * d.mean(), 1000 * stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)), stats.ttest_1samp(d, 0).pvalue


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


V4, V3 = "results/v4/", "results/v3/"
SETS = {
    "Coat": ("ndcg@5", [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN"), (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p"),
                        (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")],
             [V3 + "learned_coat_given.json"], ["LightGCN", "r-AdjNorm", "NAVIP", "DR-LightGCN", "iALS", "SimGCL", "MACR"]),
    "Yahoo": ("ndcg@5", [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] +
              [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")],
              glob.glob(V3 + "learned_yahoo_pop*.json"), ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "SimGCL"]),
    "KuaiRand": ("ndcg@10", [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                             (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")],
                 [V3 + "learned_kuairand_pop.json"], ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "BPR-MF", "SimGCL"]),
    "KuaiRec": ("ndcg@20", [(V4 + f"learned_kuairec_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3")] +
                [(V4 + f"learned_kuairec_p2_lr{lr}.json", "DR-BPR-LGN-p2") for lr in ("1e-2", "3e-3", "1e-3")] +
                [(V4 + "learned_kuairec_p_lr3e-3.json", "DR-BPR-LGN-p")],
                glob.glob(V3 + "learned_kuairec_pop*.json"), ["DR-JL", "MRDR", "LightGCN-pt", "DR-LightGCN", "NAVIP", "iALS", "LightGCN", "r-AdjNorm"]),
}
for ds, (key, jobs, bfiles, names) in SETS.items():
    jobs = [(f, n) for f, n in jobs if os.path.exists(f)]
    mine = combine(jobs, key)
    base = {}
    for f in bfiles:
        base.update(ld(f)["results"])
    print(f"\n{ds}: DR-BPR-LGN + offset, {mine['n_jobs']} jobs, mean {key} = {np.mean(mine['per_split']):.4f}; "
          f"beta chosen {dict(collections.Counter(c.get('pbeta', 0.0) for c in mine['chosen']))}")
    rows = []
    for n in names:
        if n in base:
            m, h, p = cmp(mine, base[n]); rows.append((n, base[n]["test"][key][0], base[n]["n_configs"], m, h, p))
    for r, a in zip(rows, holm([r[5] for r in rows])):
        print(f"  vs {r[0]:12s} {r[1]:.4f} ({r[2]:3d} cfgs)  diff {r[3]:+7.2f}e-3 [{r[4]:5.2f}]  p_Holm {a:.4f}")
