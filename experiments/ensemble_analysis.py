"""Ensemble (A + DRUP) versus A-repro, the original final method, DRUP and the trained baselines.

Per dataset with results/v4/ensemble_{ds}.json: paired user-level t-tests (users averaged over
splits), Holm-adjusted within the dataset; prints the w distribution and the repro gaps;
writes results/v4/ensemble_summary.txt.
"""
import collections
import json
import os

import numpy as np
from scipy import stats

from final_config import DS_NAME, FINAL_JOBS, PROP, SETS

ld = lambda f: json.load(open(f))


# copied from experiments/final_analysis.py (that file runs on import); job lists and SETS come from final_config
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


def main():
    lines = []
    out = lambda *x: (print(*x), lines.append(" ".join(str(v) for v in x)))
    for ds, sname in DS_NAME.items():
        path = f"results/v4/ensemble_{ds}.json"
        if not os.path.exists(path):
            continue
        key, bfiles, names = SETS[sname]
        jobs = FINAL_JOBS[ds]
        E = ld(path)
        R = E["results"]
        ens = {"per_user": R["Ens-A+DRUP"]["per_user"]}
        comps = [("A-repro (w=0)", {"per_user": R["A-repro"]["per_user"]}),
                 ("original final method", combine(jobs, key)),
                 ("DRUP (w=1, stored)", {"per_user": ld(f"results/v3/filters_{ds}_{PROP[ds]}.json")["results"]["DRUP"]["per_user"]}),
                 ("Ens-fixed-0.5", {"per_user": R["Ens-fixed-0.5"]["per_user"]})]
        base = {}
        for f in bfiles:
            if os.path.exists(f):
                base.update(ld(f)["results"])
        compared = [n for n in names if n in base]
        for n in compared:
            comps.append((n, {"per_user": base[n]["per_user"]}))
        # comparisons whose per-user values are identical to an earlier one are the same test: counted once
        family, dups = [], []
        for label, obj in comps:
            same = [l for l, o in family if o["per_user"] == obj["per_user"]]
            if same:
                dups.append((label, same[0]))
            else:
                family.append((label, obj))
        rows = []
        for label, obj in family:
            m, h, p = cmp(ens, obj)
            rows.append((label, m, h, p))
        adj = holm([r[3] for r in rows])
        out(f"\n{sname}: Ens-A+DRUP vs others, metric {key}; users averaged over splits, Holm within dataset")
        out(f"  Ens-A+DRUP test {R['Ens-A+DRUP']['test'][key][0]:.4f} (+-{R['Ens-A+DRUP']['test'][key][1]:.4f})")
        out(f"  Holm family: {len(rows)} tests; compared baselines: {', '.join(compared)}")
        for (label, m, h, p), pa in zip(rows, adj):
            out(f"  vs {label:24s} diff {m:+8.2f}e-3 [{h:6.2f}]  p {p:.4f}  p_Holm {pa:.4f}")
        for label, same in dups:
            out(f"  vs {label:24s} per-user values identical to '{same}'; not counted again in the Holm family")
        wc = collections.Counter(c["w"] for c in R["Ens-A+DRUP"]["chosen"])
        out(f"  w distribution over splits: {dict(sorted(wc.items()))}")
        for name in ("A-repro", "DRUP-repro", "Ens-fixed-0.5"):
            out(f"  {name:14s} test {R[name]['test'][key][0]:.4f} (+-{R[name]['test'][key][1]:.4f})")
        if "repro" in E:
            rp = E["repro"]
            for who in ("A", "B"):
                gaps = [r["stored"] - r["repro"] for r in rp[who]]
                out(f"  repro {who}: max |stored-repro| validation gap {max(abs(g) for g in gaps):.6f}; "
                    f"per split {[round(g, 6) for g in gaps]}")
            if "repro_ok" in rp:
                out(f"  repro_ok {rp['repro_ok']} (tolerance {rp['tolerance']})")
    open("results/v4/ensemble_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
