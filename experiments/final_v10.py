"""Stage 1 analysis for LEAF on the temporal all-user KuaiRand-27K setting (v10).

Gate (paper/v6/DESIGN.md, sections 5 to 7): LEAF-LGN must beat the strongest tuned baseline by
>= 10e-3 nDCG@10 with Holm-adjusted p < 0.05. The primary family holds LEAF-LGN and LEAF-tied
against each available baseline. Ablations are compared with the full LEAF variant (differences are
full minus ablation, so a positive value means the component helps). Inputs that are not present yet
are reported as notes; the analysis runs on whatever exists.

Statistics follow experiments/final_analysis.py and experiments/v7_common.py (copied helpers, not
imported). The Nadeau-Bengio split-level p-value comes from experiments/significance.py.
Output: results/v10/stage1.txt (also printed).
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from significance import nadeau_bengio  # importing significance.py runs nothing

V10 = "results/v10/"
KEY = "ndcg@10"
FRAC_VAL = 0.3
GATE_MARGIN = 10.0   # x1e-3 nDCG@10
GATE_ALPHA = 0.05
OUT = V10 + "stage1.txt"

BASE = V10 + "learned_kuairand_t_base.json"          # LightGCN, r-AdjNorm, iALS, iALS-p
BASE_PARTIAL = BASE + ".iALS-p.partial.json"         # not a result file; never read here
POP = V10 + "learned_kuairand_t_base_pop.json"       # LightGCN-p
QP = V10 + "learned_kuairand_t_qp*.json"             # QP family (DR-BPR-LGN-1 / -p2 / -p)
LEAF_TIED = V10 + "leaf_tied.json"
LEAF_TIED_ABL = V10 + "leaf_tied_ablate.json"
LEAF_LGN = V10 + "leaf_lgn.json"
LEAF_LGN_ABL = V10 + "leaf_lgn_ablate.json"

LEAF_SPECS = [
    ("LEAF-LGN", [(LEAF_LGN, "LEAF-LGN")]),
    ("LEAF-tied", [(LEAF_TIED, "LEAF-tied")]),
]
BASELINE_SPECS = [
    ("LightGCN", [(BASE, "LightGCN")]),
    ("r-AdjNorm", [(BASE, "r-AdjNorm")]),
    ("iALS", [(BASE, "iALS")]),
    ("iALS-p", [(BASE, "iALS-p")]),
    ("LightGCN-p", [(POP, "LightGCN-p")]),
    ("QP family", [(QP, m) for m in ("DR-BPR-LGN-1", "DR-BPR-LGN-p2", "DR-BPR-LGN-p")]),
]

body, notes = [], []


def out(s=""):
    body.append(s)


def note(s):
    if s not in notes:
        notes.append(s)


_cache = {}


def results_of(path):
    if path not in _cache:
        _cache[path] = json.load(open(path))["results"]
    return _cache[path]


def jobs_for(spec):
    """(file, record) for every existing job of a method; missing files and methods become notes."""
    jobs = []
    for pat, m in spec:
        files = sorted(glob.glob(pat))
        if not files:
            note(f"missing: {pat}")
        for f in files:
            res = results_of(f)
            if m in res:
                jobs.append((f, res[m]))
            else:
                note(f"{f} has no method {m}")
    return jobs


def combine(recs):
    """Per split, the job with the largest stored val_best (final_analysis.combine, dict-valued per_split)."""
    pu, ps, ch = [], [], []
    for s in range(len(recs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in recs]))
        pu.append(recs[j]["per_user"][s])
        ps.append(recs[j]["per_split"][s])
        ch.append(recs[j]["chosen"][s])
    return {"per_user": pu, "per_split": ps, "chosen": ch}


def method(name, spec):
    jobs = jobs_for(spec)
    if not jobs:
        note(f"no result for {name}")
        return None
    recs = [r for _, r in jobs]
    m = combine(recs)
    m.update(name=name, n_configs=int(sum(r["n_configs"] for r in recs)),
             files=sorted({f for f, _ in jobs}), n_jobs=len(recs))
    return m


def mean_sd(m):
    v = np.array([d[KEY] for d in m["per_split"]])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def peru(per_user):
    """Mean over splits of each user's metric (final_analysis.peru)."""
    acc = {}
    for q in per_user:
        for u, v in q.items():
            acc.setdefault(u, []).append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def compare(a, b):
    """Paired per-user difference a - b (final_analysis.cmp, users intersected, x1e-3)."""
    x, y = peru(a["per_user"]), peru(b["per_user"])
    us = sorted(set(x) & set(y))
    d = np.array([x[u] - y[u] for u in us])
    n = len(d)
    if np.abs(d).max() < 1e-5:
        return {"n": n, "diff": 0.0, "half": 0.0, "p": 1.0, "identical": True}
    half = 1000 * stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"n": n, "diff": 1000 * float(d.mean()), "half": float(half),
            "p": float(stats.ttest_1samp(d, 0.0).pvalue), "identical": False}


def pair(a, b):
    """compare plus Nadeau-Bengio split-level p (identical rankings: p = 1)."""
    r = compare(a, b)
    if r["identical"]:
        r["nb"] = 1.0
    else:
        r["nb"] = float(nadeau_bengio({"per_split": a["per_split"]}, {"per_split": b["per_split"]}, KEY, FRAC_VAL))
    return r


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    if len(ps) == 0:
        return ps
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def row(label, r, pH, nbH):
    if r["identical"]:
        return f"  {label:<40s} identical rankings in every split (diff 0, p = 1)"
    return (f"  {label:<40s} n={r['n']:6d}  {r['diff']:+8.2f} [{r['half']:5.2f}]  p={r['p']:.2e}  "
            f"p_Holm={pH:.2e}  NB p={r['nb']:.2e}  NB p_Holm={nbH:.2e}")


# ---------------------------------------------------------------------------------------------
# Inputs and methods
# ---------------------------------------------------------------------------------------------
methods = {}
for name, spec in LEAF_SPECS + BASELINE_SPECS:
    m = method(name, spec)
    if m is not None:
        methods[name] = m

leaf_names = [n for n, _ in LEAF_SPECS if n in methods]
base_names = [n for n, _ in BASELINE_SPECS if n in methods]

if os.path.exists(BASE_PARTIAL):
    note(f"partial, not used: {BASE_PARTIAL} (iALS-p is taken from the final file only)")

out("1. Methods: mean +- sd over the 5 splits of nDCG@10 (sd with ddof=1)")
out(f"  {'method':<14s} {'mean':>8s} {'+- sd':>8s} {'n_configs':>10s} {'jobs':>5s}  files")
for name in leaf_names + base_names:
    m = methods[name]
    mu, sd = mean_sd(m)
    files = ", ".join(os.path.basename(f) for f in m["files"])
    out(f"  {name:<14s} {mu:8.4f} {sd:8.4f} {m['n_configs']:10d} {m['n_jobs']:5d}  {files}")
out("  n_configs = configurations searched per split (pooled over the jobs of the method).")
out()

# users: intersection across all methods of this section
common = None
for name in leaf_names + base_names:
    keys = set(peru(methods[name]["per_user"]))
    common = keys if common is None else common & keys
out(f"2. Users: {len(common) if common is not None else 0} users are common to all methods in section 1 "
    "(per-user keys are dataset row ids; each paired comparison intersects its own two methods).")
out()

# ---------------------------------------------------------------------------------------------
# Primary family: LEAF variants vs each baseline
# ---------------------------------------------------------------------------------------------
primary = []
for L in leaf_names:
    for B in base_names:
        primary.append((L, B, pair(methods[L], methods[B])))
pH = holm([r["p"] for _, _, r in primary])
nbH = holm([r["nb"] for _, _, r in primary])

out("3. Primary family: per-user paired difference A - B, x1e-3 nDCG@10 [95% CI half-width], "
    "two-sided t p, Holm within this family (all LEAF variants x all baselines), "
    "Nadeau-Bengio p over split means and its Holm value in the same family")
for (L, B, r), p_h, nb_h in zip(primary, pH, holm([r["nb"] for _, _, r in primary])):
    out(row(f"{L} - {B}", r, p_h, nb_h))
if not primary:
    out("  no LEAF result or no baseline result is present; family is empty")
out()

# ---------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------
out("4. Gate (LEAF-LGN must beat the strongest tuned baseline by >= 10e-3 nDCG@10, p_Holm < 0.05)")
if "LEAF-LGN" not in methods or not base_names:
    gate_line = "  GATE: NOT EVALUABLE (LEAF-LGN or the baselines are missing)"
    out(gate_line)
else:
    strongest = max(base_names, key=lambda n: mean_sd(methods[n])[0])
    mu_s, _ = mean_sd(methods[strongest])
    mu_L, _ = mean_sd(methods["LEAF-LGN"])
    idx = [i for i, (L, B, _) in enumerate(primary) if L == "LEAF-LGN" and B == strongest][0]
    r = primary[idx][2]
    ph = pH[idx]
    passed = (r["diff"] >= GATE_MARGIN) and (ph < GATE_ALPHA)
    out(f"  strongest tuned baseline: {strongest} (mean nDCG@10 {mu_s:.4f}); LEAF-LGN mean {mu_L:.4f}")
    out(row(f"LEAF-LGN - {strongest}", r, ph, nbH[idx]))
    out(f"  GATE: {'PASS' if passed else 'FAIL'}  (diff {r['diff']:+.2f}e-3 vs required >= {GATE_MARGIN:.0f}e-3; "
        f"p_Holm {ph:.2e} vs required < {GATE_ALPHA})")
    each = []
    for L, B, rr in primary:
        if L == "LEAF-LGN":
            j = [i for i, (LL, BB, _) in enumerate(primary) if LL == L and BB == B][0]
            ok = rr["diff"] >= GATE_MARGIN and pH[j] < GATE_ALPHA
            each.append(f"    vs {B:<12s} {rr['diff']:+8.2f}e-3  p_Holm {pH[j]:.2e}  {'PASS' if ok else 'FAIL'}")
    out("  reading 'beats each baseline' (design section 7), same thresholds:")
    for e in each:
        out(e)
    if "LEAF-tied" in methods:
        jt = [i for i, (LL, BB, _) in enumerate(primary) if LL == "LEAF-tied" and BB == strongest][0]
        out(f"  info: LEAF-tied - {strongest}: {primary[jt][2]['diff']:+.2f}e-3, p_Holm {pH[jt]:.2e}")
out()

# ---------------------------------------------------------------------------------------------
# Ablations vs the full LEAF variant
# ---------------------------------------------------------------------------------------------
out("5. Ablations: full minus ablated arm, x1e-3 nDCG@10 [95% CI half-width] (positive = the component helps)")
abl_rows = []        # (label, pair result, arm name, full name)
arm_methods = {}
for full_name, path, prefix in [("LEAF-tied", LEAF_TIED_ABL, "LEAF-tied["),
                                ("LEAF-LGN", LEAF_LGN_ABL, "LEAF-LGN[")]:
    if not os.path.exists(path):
        note(f"missing: {path}")
        continue
    if full_name not in methods:
        note(f"ablations of {full_name} present but the full variant is missing")
        continue
    for arm in results_of(path):
        if arm.startswith(prefix):
            am = method(arm, [(path, arm)])
            if am is not None:
                arm_methods[arm] = am
                abl_rows.append((full_name, arm, pair(methods[full_name], am)))
abl_p = holm([r["p"] for _, _, r in abl_rows])
abl_nb = holm([r["nb"] for _, _, r in abl_rows])
out(f"  {'arm':<40s} {'n_configs':>9s}  mean    full-arm diff (x1e-3)")
for (full_name, arm, r), p_h, nb_h in zip(abl_rows, abl_p, abl_nb):
    out(f"  {'[' + full_name + '] ' + arm.split('[', 1)[1][:-1]:<40s} {arm_methods[arm]['n_configs']:9d}  "
        f"{mean_sd(arm_methods[arm])[0]:.4f}")
    out(row(f"{full_name} - {arm}", r, p_h, nb_h))
if not abl_rows:
    out("  no ablation result is present")
out("  Holm within this ablation family. Two-sided t p, NB p over split means.")
if not os.path.exists(LEAF_LGN_ABL):
    out("  LEAF-LGN ablations: not available yet (leaf_lgn_ablate.json missing), so the design "
        "check LEAF-LGN - LEAF-LGN(no lifelong) >= 8e-3 is not evaluable.")
if "LEAF-tied[shuffle_users]" in arm_methods and "LEAF-tied[no_lifelong]" in arm_methods:
    r = pair(arm_methods["LEAF-tied[shuffle_users]"], arm_methods["LEAF-tied[no_lifelong]"])
    if r["identical"]:
        ok, val = True, 0.0
    else:
        val = r["diff"]
        ok = abs(val) <= 2.0
    out(f"  design check (LEAF-tied shuffle_users vs no_lifelong, must be within +-2e-3): "
        f"{val:+.2f}e-3 -> {'PASS' if ok else 'FAIL'}")
out()

# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------
header = ["Stage 1 (v10): LEAF on the temporal all-user KuaiRand-27K setting, nDCG@10",
          "Inputs read: " + ", ".join(sorted(os.path.basename(p) for p in
                                             [LEAF_TIED, LEAF_TIED_ABL, LEAF_LGN, BASE, POP]
                                             + glob.glob(QP) + glob.glob(LEAF_LGN_ABL)
                                             if os.path.exists(p))),
          "Inputs missing or not used:"]
header += [f"  - {n}" for n in notes] if notes else ["  (none)"]
header.append("")
text = "\n".join(header + body) + "\n"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(text)
print(text)
