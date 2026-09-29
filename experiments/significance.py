"""User-level significance tests with Holm correction.

For methods A and B, each test user's metric difference is averaged over the
splits in which the user is in the test part (the training log is the same
for every split, so users are the independent units). We report the mean
difference, a 95% t-interval, the paired t-test and Wilcoxon signed-rank
p-values over users, and the Holm-adjusted t-test p-value within the family
of comparisons of one dataset. As a split-level check we also report the
corrected resampled t-test of Nadeau & Bengio (2003) over split means.
"""

import argparse
import json
import os

import numpy as np
from scipy import stats


def load(paths):
    res = {}
    for p in paths:
        if os.path.exists(p):
            res.update(json.load(open(p))["results"])
    return res


def user_diffs(ra, rb):
    acc = {}
    for pa, pb in zip(ra["per_user"], rb["per_user"]):
        for u, va in pa.items():
            if u in pb:
                acc.setdefault(u, []).append(va - pb[u])
    return np.array([np.mean(v) for v in acc.values()])


def nadeau_bengio(ra, rb, key, frac_val):
    d = np.array([a[key] - b[key] for a, b in zip(ra["per_split"], rb["per_split"])])
    J = len(d)
    var = d.var(ddof=1) * (1.0 / J + (1 - frac_val) / frac_val)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    return float(2 * stats.t.sf(abs(t), J - 1))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--key", default=None)
    ap.add_argument("--ref", nargs="+", default=["DRUP"])
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--others", nargs="+", default=None, help="compare only with these methods")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="methods left out of the default family (reported with --others)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key = a.key or {"kuairec": "ndcg@20", "kuairand": "ndcg@10"}.get(a.dataset, "ndcg@5")
    res = load([f"results/v2/filters_{a.dataset}_{a.prop}.json", f"results/v2/learned_{a.dataset}_{a.prop}.json"])
    rows = []
    for ref in a.ref:
        if ref not in res:
            continue
        for m, r in res.items():
            if m == ref or "per_user" not in r:
                continue
            if (a.others is not None and m not in a.others) or (a.others is None and m in a.exclude):
                continue
            d = user_diffs(res[ref], r)
            if len(d) < 3:
                continue
            if np.abs(d).max() < 1e-5:
                # numerically identical rankings (e.g. both select the same
                # IPS-end configuration); floating-point noise is not a difference
                rows.append({"ref": ref, "other": m, "n_users": int(len(d)), "diff": 0.0,
                             "ci95": [0.0, 0.0], "p_t": 1.0, "p_wilcoxon": 1.0, "p_nb_splits": 1.0,
                             "identical": True})
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            p_t = float(stats.ttest_1samp(d, 0.0).pvalue)
            try:
                p_w = float(stats.wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() > 10 else 1.0
            except ValueError:
                p_w = 1.0
            rows.append({"ref": ref, "other": m, "n_users": int(len(d)), "diff": float(d.mean()),
                         "ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
                         "p_t": p_t, "p_wilcoxon": p_w,
                         "p_nb_splits": nadeau_bengio(res[ref], r, key, a.frac_val)})
    # Holm over every comparison run for the dataset (all references together),
    # for the t-test and for the Wilcoxon test; the within-reference
    # adjustment is kept for information only.
    for r, q in zip(rows, holm([r["p_t"] for r in rows])):
        r["p_holm"] = float(q)
    for r, q in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
        r["p_wilcoxon_holm"] = float(q)
    for ref in a.ref:
        fam = [r for r in rows if r["ref"] == ref]
        for r, q in zip(fam, holm([r["p_t"] for r in fam])):
            r["p_holm_within_ref"] = float(q)
    for r in rows:
        print(f"{r['ref']:10s} vs {r['other']:12s} n={r['n_users']:5d} diff={r['diff']:+.4f} "
              f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p_t']:.2g} holm={r['p_holm']:.2g} "
              f"wilc={r['p_wilcoxon']:.2g} wholm={r['p_wilcoxon_holm']:.2g} nb={r['p_nb_splits']:.2g}")
    out = a.out or f"results/v2/significance_{a.dataset}_{a.prop}.json"
    with open(out, "w") as f:
        json.dump({"key": key, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
