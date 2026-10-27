"""Summary of the training-seed runs of experiments/run_seeds.py.

(a) dataset mean over splits for each seed (seed 0 = stored results, final method via combine),
    mean +- std over seeds 1-3 and over seeds 0-3 (np.std, ddof 0).
(b) per-user average over seeds 1-3 and over the splits in which the user is a test user; paired
    t-tests of the final method against each baseline, Holm-adjusted within the dataset.
Writes results/v4/seeds_summary.txt.
"""
import json
import os

import numpy as np
from scipy import stats

ld = lambda f: json.load(open(f))


# copied from experiments/final_analysis.py (that file runs on import)
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


# same job lists as run_ensemble.FINAL_JOBS (copied from final_analysis SETS)
V4 = "results/v4/"
FINAL_JOBS = {
    "coat": [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN"), (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p"),
             (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")],
    "yahoo": [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] +
             [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")],
    "kuairand": [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                 (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")],
    "kuairec": [(V4 + f"learned_kuairec_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3")] +
               [(V4 + f"learned_kuairec_p2_lr{lr}.json", "DR-BPR-LGN-p2") for lr in ("1e-2", "3e-3", "1e-3")] +
               [(V4 + "learned_kuairec_p_lr3e-3.json", "DR-BPR-LGN-p")],
}
KEY = {"coat": "ndcg@5", "yahoo": "ndcg@5", "kuairand": "ndcg@10", "kuairec": "ndcg@20"}
# baselines and their part files (as in run_seeds.SRC)
SRC = {
    "coat": {"iALS": "results/v3/learned_coat_given.json", "DR-LightGCN": "results/v3/learned_coat_given.json"},
    "yahoo": {"r-AdjNorm": "results/v3/learned_yahoo_pop_A.json", "LightGCN": "results/v3/learned_yahoo_pop_B.json"},
    "kuairand": {"r-AdjNorm": "results/v3/learned_kuairand_pop.json", "LightGCN": "results/v3/learned_kuairand_pop.json"},
    "kuairec": {"DR-JL": "results/v3/learned_kuairec_pop_A.json", "MRDR": "results/v3/learned_kuairec_pop_A.json"},
}


def stored_final(ds):
    """Seed 0 for the final method: per split values and per-user values of combine()."""
    c = combine(FINAL_JOBS[ds], KEY[ds])
    return {"vals": c["per_split"], "per_user": c["per_user"]}


def stored_base(ds, name, path):
    r = ld(path)["results"][name]
    return {"vals": [x[KEY[ds]] for x in r["per_split"]], "per_user": r["per_user"]}


def main():
    lines = []
    out = lambda *x: (print(*x), lines.append(" ".join(str(v) for v in x)))
    for ds in SRC:
        path = f"results/v4/seeds_{ds}.json"
        if not os.path.exists(path):
            continue
        key = KEY[ds]
        S = ld(path)["results"]
        # seed -> {"vals": per-split values, "per_user": list}
        series = {}
        series["final"] = {0: stored_final(ds)}
        for name, p in SRC[ds].items():
            series[name] = {0: stored_base(ds, name, p)}
        for name in series:
            for sd in ("1", "2", "3"):
                e = S[name]["seeds"][sd]
                series[name][int(sd)] = {"vals": [x[key] for x in e["per_split"]], "per_user": e["per_user"]}
        out(f"\n{ds} (metric {key}); seeds: final method via combine, baselines from part files")
        out(f"  {'method':14s} " + " ".join(f"seed{s:d}".rjust(9) for s in range(4)) +
            "   mean+-std(1-3)     mean+-std(0-3)")
        for name in series:
            per_seed = [np.mean(series[name][s]["vals"]) for s in range(4)]
            m13, s13 = np.mean(per_seed[1:]), np.std(per_seed[1:])
            m03, s03 = np.mean(per_seed), np.std(per_seed)
            out(f"  {name:14s} " + " ".join(f"{v:9.4f}" for v in per_seed) +
                f"   {m13:.4f}+-{s13:.4f}    {m03:.4f}+-{s03:.4f}")
        # (b) per-user: all per_user entries of seeds 1-3 (every split), averaged per user
        pu = {name: {"per_user": [q for sd in (1, 2, 3) for q in series[name][sd]["per_user"]]} for name in series}
        rows = []
        for name in series:
            if name == "final":
                continue
            m, h, p = cmp(pu["final"], pu[name])
            rows.append((name, m, h, p))
        adj = holm([r[3] for r in rows])
        out(f"  per-user paired tests, final vs baseline (seeds 1-3, all splits), Holm within dataset:")
        for (name, m, h, p), pa in zip(rows, adj):
            out(f"    vs {name:12s} diff {m:+8.2f}e-3 [{h:6.2f}]  p {p:.4f}  p_Holm {pa:.4f}")
    open("results/v4/seeds_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
