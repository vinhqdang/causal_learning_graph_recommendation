"""DRGP results (v6): final method per dataset, per-backbone rows, paired user-level tests.

Final method: the combine-over-jobs of the DRGP family of a dataset (the DRGP prior alone, DRGP-iALS
and DRGP-LGN-*), chosen per split by validation score, as in final_analysis.py. Each backbone family
is also combined on its own. Paired t-tests over users against the baselines, Holm-adjusted within
the dataset for the final method. Missing files are reported and skipped.
"""
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_config import FINAL_JOBS  # noqa: E402

ld = lambda f: json.load(open(f))
V5, V4, V3 = "results/v5/", "results/v4/", "results/v3/"


def has(f, n):
    """True if the job file exists and already holds the named method (a running job writes a
    method's result only when that method has finished)."""
    return os.path.exists(f) and n in ld(f).get("results", {})


def combine(jobs, key):
    """Per-split choice among the jobs (file, method) that hold their method. Returns None if no job
    holds its method yet. n_jobs is the selection pool (candidate results per split); n_configs is
    the number of configurations searched by those jobs."""
    rs = [ld(f)["results"][n] for f, n in jobs if has(f, n)]
    if not rs:
        return None
    pu, ps, ch = [], [], []
    for s in range(len(rs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in rs]))
        pu.append(rs[j]["per_user"][s]); ps.append(rs[j]["per_split"][s][key]); ch.append(rs[j]["chosen"][s])
    return {"per_user": pu, "per_split": ps, "chosen": ch, "n_jobs": len(rs),
            "n_configs": int(sum(r.get("n_configs", 0) for r in rs))}


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


def mean_of(res, key):
    return float(np.mean([p[key] if isinstance(p, dict) else p for p in res["per_split"]]))


def stored(files, names=None):
    """Stored results of the baseline files, keyed by method name (missing files are reported)."""
    base = {}
    for f in files:
        if os.path.exists(f):
            base.update(ld(f)["results"])
        else:
            print(f"  missing: {f}")
    return {n: base[n] for n in (names or base) if n in base}


def v2_final(ds, key):
    """QP-LightGCN (v2 final): the combine of the v2 jobs of FINAL_JOBS for the dataset."""
    return combine([(f, n) for f, n in FINAL_JOBS[ds] if os.path.exists(f)], key)


KEY = {"coat": "ndcg@5", "yahoo": "ndcg@5", "kuairand": "ndcg@10", "kuairec": "ndcg@20"}
NAME = {"coat": "Coat", "yahoo": "Yahoo", "kuairand": "KuaiRand", "kuairec": "KuaiRec"}


def family(ds):
    """DRGP jobs per backbone family (file, method name)."""
    if ds == "coat":
        return {"prior": [(V5 + f"learned_coat_drgp_f{f}_b{b}.json", "DRGP") for f in ("0.05", "0.1", "0.2") for b in (1, 2)],
                "iALS-p": [(V5 + "learned_coat_drgp_ials.json", "iALS-p")],
                "LGN": [(V5 + "learned_coat_drgp_lgn.json", n) for n in ("DRGP-LGN-bpr", "DRGP-LGN-dr", "DRGP-LGN-hyb")]}
    if ds == "kuairec":
        return {"prior": [(V5 + f"learned_kuairec_drgp_f{f}.json", "DRGP") for f in ("0.1", "0.2")],
                "iALS-p": [(V5 + "learned_kuairec_drgp_ials.json", "iALS-p")],
                "LGN": [(V5 + "learned_kuairec_drgp_lgn.json", n) for n in ("DRGP-LGN-dr", "DRGP-LGN-bpr")]}
    lgn = ["DRGP-LGN-bpr", "DRGP-LGN-hyb"]
    return {"prior": [(V5 + f"learned_{ds}_drgp_DRGP.json", "DRGP")],
            "iALS-p": [(V5 + f"learned_{ds}_drgp_iALS-p.json", "iALS-p")],
            "LGN": [(V5 + f"learned_{ds}_drgp_{n}.json", n) for n in lgn]}


# baselines per dataset: (stored part files, method names) and the v2 final (QP-LightGCN) is added as a combine
BASE = {
    "coat": ([V3 + "learned_coat_given.json", V4 + "learned_coat_adv.json", V3 + "filters_coat_given.json"],
             ["iALS", "LightGCN", "DR-LightGCN", "r-AdjNorm", "NAVIP", "SimGCL", "MACR", "AdvInfoNCE-LGN", "DRUP", "BSPM", "GF-CF"]),
    "yahoo": ([V3 + "learned_yahoo_pop.json", V3 + "learned_yahoo_pop_A.json", V3 + "learned_yahoo_pop_B.json",
               V3 + "learned_yahoo_pop_C.json", V3 + "learned_yahoo_pop_D.json", V3 + "filters_yahoo_pop.json",
               V4 + "learned_yahoo_adv.json"],
              ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "SimGCL", "AdvInfoNCE-LGN", "DRUP", "GF-CF"]),
    "kuairand": ([V3 + "learned_kuairand_pop.json", V3 + "filters_kuairand_pop.json", V4 + "learned_kuairand_adv.json"],
                 ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "BPR-MF", "SimGCL", "AdvInfoNCE-LGN", "DRUP", "GF-CF"]),
    "kuairec": ([V3 + "learned_kuairec_pop.json", V3 + "learned_kuairec_pop_A.json", V3 + "learned_kuairec_pop_B.json",
                 V3 + "filters_kuairec_pop.json", V4 + "learned_kuairec_adv.json"],
                ["DRUP", "DR-JL", "MRDR", "LightGCN-pt", "DR-LightGCN", "NAVIP", "iALS", "LightGCN", "r-AdjNorm",
                 "AdvInfoNCE-LGN"]),
}
# the matched-grid iALS baselines of v5 (same alpha/reg grid as DRGP-iALS)
MATCHED = {
    "coat": [(V5 + "learned_coat_add_ials.json", "iALS-p", "iALS + additive prior"),
             (V5 + "learned_coat_ialswide.json", "iALS-wide", "iALS-wide")],
    "kuairec": [(V5 + "learned_kuairec_add_ials.json", "iALS-p", "iALS + additive prior")],
}

for ds in ("coat", "yahoo", "kuairand", "kuairec"):
    key = KEY[ds]
    fam = family(ds)
    jobs = [j for f in fam.values() for j in f]
    present = [j for j in jobs if has(*j)]
    print(f"\n{NAME[ds]} ({key}): DRGP jobs found {len(present)}/{len(jobs)} (file present and method done)")
    if not present:
        continue
    mine = combine(present, key)
    print(f"  final (DRGP family, per-split validation choice): mean {mean_of(mine, key):.4f}  "
          f"chosen pbeta {sorted(set(str(c.get('pbeta')) for c in mine['chosen']))}")
    print(f"  DRGP family: {mine['n_configs']} configurations in {len(present)} job(s); "
          f"selection pool per split: {mine['n_jobs']} candidate results")
    for name, jl in fam.items():
        jl = [j for j in jl if has(*j)]
        if jl:
            r = combine(jl, key)
            print(f"  backbone {name:7s}: mean {mean_of(r, key):.4f}  ({len(jl)} job file(s), "
                  f"{r['n_configs']} configurations)")
    base = stored(BASE[ds][0], BASE[ds][1])
    base_rows = [(n, base[n]) for n in BASE[ds][1] if n in base]
    v2 = v2_final(ds, key)
    if v2 is not None:
        base_rows.append(("QP-LightGCN (v2 final)", v2))
    else:
        print("  missing: QP-LightGCN (v2 final) jobs")
    for f, n, label in MATCHED.get(ds, []):
        if os.path.exists(f) and n in ld(f)["results"]:
            base_rows.append((label, ld(f)["results"][n]))
        else:
            print(f"  missing: {f} [{n}]")
    rows = []
    for n, r in base_rows:
        m, h, p = cmp(mine, r)
        rows.append((n, mean_of(r, key), r.get("n_configs", 0), m, h, p))
    adj = holm([r[5] for r in rows])
    print(f"  {'vs baseline':24s} {'mean':>7s} {'cfgs':>5s}  {'diff(e-3)':>9s} {'+-CI':>6s}  p_Holm")
    for r, a in zip(rows, adj):
        print(f"  {r[0]:24s} {r[1]:7.4f} {r[2]:5d}  {r[3]:+9.2f} {r[4]:6.2f}  {a:.4f}")
    for name, jl in fam.items():
        jl = [j for j in jl if has(*j)]
        if not jl:
            continue
        r = combine(jl, key)
        out = [(n, *cmp(r, rr)) for n, rr in base_rows]
        print(f"  backbone {name:7s} vs same baselines (unadjusted): " +
              ", ".join(f"{n} {m:+.1f}e-3 p={p:.3f}" for n, m, _, p in out[:3]) + " ...")
