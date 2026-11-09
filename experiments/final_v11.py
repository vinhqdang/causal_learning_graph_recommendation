"""Stage B / C analysis for DAX on the temporal all-user KuaiRand-27K setting (v11), paper/v7/DESIGN.md Section 7.

Gate B1 (post-hoc, joint selection): LEAF-LGN + DAX minus the QP family >= 15e-3 nDCG@10 per user, with Holm-adjusted
  two-sided t p < 0.05 AND Holm-adjusted Nadeau-Bengio split-level p < 0.05. The primary family (Holm, both p) is
  LEAF-LGN+DAX against: QP family, QP family + DAX (the fair baseline), LEAF-LGN, r-AdjNorm, LightGCN.
  QP family + DAX - QP family and the LEAF-LGN increment are reported with it (expected about +6 to +9e-3).
Gate B2 (ablations, gains relative to the backbone of the same run, LEAF-LGN in dax_ablate.json):
  shuffle_tags (alpha permuted across items) removes >= 90% of the DAX gain;
  shuffle_items (permuted within tag) drops the gain by >= 30%;
  additive (MRP-style logistic with tab effects) stays >= 4e-3 below DAX.
Stage C (robustness: KuaiRand-1K, concurrent protocol, long-view label, seeds 1-3): DESIGN.md gives no numeric
  threshold for Stage C. The script reports, per present setting, the per-user difference of LEAF-LGN + DAX against
  QP family and against QP family + DAX, and their direction; it prints no pass or fail for C.

Inputs: results/v10/ (stored: LEAF-LGN, QP family, LightGCN, r-AdjNorm), results/v11/ (dax_lgn_post.json,
dax_lgn_joint_small.json, dax_lgn_joint.json, dax_qp.json, dax_lightgcn.json, dax_ablate.json, dax_ease.json,
c_*.json). Inputs that are absent are reported as notes. Environment variable V11_OUT overrides results/v11/.
Output: results/v11/stageB.txt (also printed).

Statistics follow experiments/final_v10.py (copied helpers): paired per-user differences, the t interval and p over
users, Holm within each family, Nadeau-Bengio over split means (experiments/significance.py).
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
V11 = os.environ.get("V11_OUT", "results/v11/").rstrip("/") + "/"
KEY = "ndcg@10"
FRAC_VAL = 0.3
B1_MARGIN = 15.0      # x1e-3 nDCG@10 per user
B1_ALPHA = 0.05
B2_SHUFFLE_TAGS = 0.90
B2_SHUFFLE_ITEMS = 0.30
B2_ADDITIVE = 4.0     # x1e-3
OUT = V11 + "stageB.txt"

STORED_LGN = V10 + "leaf_lgn.json"
STORED_BASE = V10 + "learned_kuairand_t_base.json"        # LightGCN, r-AdjNorm
STORED_QP = V10 + "learned_kuairand_t_qp.json"            # DR-BPR-LGN-1 / -p2 / -p
POST = V11 + "dax_lgn_post.json"
JOINT_SMALL = V11 + "dax_lgn_joint_small.json"
JOINT = V11 + "dax_lgn_joint.json"
QP_DAX = V11 + "dax_qp.json"
LGN_GCN = V11 + "dax_lightgcn.json"
ABLATE = V11 + "dax_ablate.json"
EASE = V11 + "dax_ease.json"
ROBUST = sorted(glob.glob(V11 + "c_*.json"))

body, notes = [], []
_cache = {}


def out(s=""):
    body.append(s)


def note(s):
    if s not in notes:
        notes.append(s)


def results_of(path):
    if path not in _cache:
        _cache[path] = json.load(open(path))["results"]
    return _cache[path]


def method_from(path, name, label=None):
    """A method of one result file (None with a note when the file or the method is missing)."""
    if not os.path.exists(path):
        note(f"missing: {path}")
        return None
    res = results_of(path)
    if name not in res:
        note(f"{path} has no method {name}")
        return None
    r = res[name]
    return {"name": label or name, "per_user": r["per_user"], "per_split": r["per_split"], "chosen": r["chosen"],
            "val_best": r["val_best"], "n_configs": int(r["n_configs"]), "file": os.path.basename(path)}


def combine(recs, name):
    """Per split, the job with the largest stored validation value (for stored methods with several jobs)."""
    pu, ps, ch, vb = [], [], [], []
    for s in range(len(recs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in recs]))
        pu.append(recs[j]["per_user"][s])
        ps.append(recs[j]["per_split"][s])
        ch.append(recs[j]["chosen"][s])
        vb.append(recs[j]["val_best"][s])
    return {"name": name, "per_user": pu, "per_split": ps, "chosen": ch, "val_best": vb,
            "n_configs": int(sum(r["n_configs"] for r in recs)), "file": "+".join(sorted({r["file"] for r in recs}))}


def qp_stored():
    if not os.path.exists(STORED_QP):
        note(f"missing: {STORED_QP}")
        return None
    recs = [method_from(STORED_QP, m, m) for m in ("DR-BPR-LGN-1", "DR-BPR-LGN-p2", "DR-BPR-LGN-p")]
    recs = [r for r in recs if r is not None]
    return combine(recs, "QP family") if recs else None


def peru(per_user):
    acc = {}
    for q in per_user:
        for u, v in q.items():
            acc.setdefault(u, []).append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def compare(a, b):
    """Paired per-user difference a - b (users intersected), x1e-3: mean, 95% t half-width, two-sided t p."""
    x, y = peru(a["per_user"]), peru(b["per_user"])
    us = sorted(set(x) & set(y))
    d = np.array([x[u] - y[u] for u in us])
    n = len(d)
    if n < 2 or np.abs(d).max() < 1e-9:
        return {"n": n, "diff": 0.0, "half": 0.0, "p": 1.0, "identical": True}
    half = 1000 * stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"n": n, "diff": 1000 * float(d.mean()), "half": float(half),
            "p": float(stats.ttest_1samp(d, 0.0).pvalue), "identical": False}


def pair(a, b):
    """compare plus the Nadeau-Bengio split-level p over split means (identical rankings: 1)."""
    r = compare(a, b)
    if r["identical"]:
        r["nb"] = 1.0
    else:
        r["nb"] = float(nadeau_bengio({"per_split": a["per_split"]}, {"per_split": b["per_split"]}, KEY, FRAC_VAL))
    r["split_diffs"] = [1000 * (p[KEY] - q[KEY]) for p, q in zip(a["per_split"], b["per_split"])]
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


def mean_sd(m):
    v = np.array([d[KEY] for d in m["per_split"]])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def line(label, r, pH=None, nbH=None):
    if r["identical"]:
        return f"  {label:<44s} identical rankings in every split (diff 0, p = 1)"
    ph = "" if pH is None else f"  p_Holm={pH:.2e}"
    nh = "" if nbH is None else f"  NB p_Holm={nbH:.2e}"
    sd = " ".join(f"{x:+.1f}" for x in r["split_diffs"])
    return (f"  {label:<44s} {r['diff']:+8.2f} [{r['half']:5.2f}]  p={r['p']:.2e}{ph}  NB p={r['nb']:.3f}{nh}"
            f"  (splits {sd})")


# ---------------------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------------------
M = {}
M["LEAF-LGN+DAX"] = method_from(POST, "LEAF-LGN+DAX")
M["LEAF-LGN"] = method_from(POST, "LEAF-LGN", "LEAF-LGN (posthoc run)")
M["LEAF-LGN(stored)"] = method_from(STORED_LGN, "LEAF-LGN", "LEAF-LGN (stored)")
M["LEAF-LGN+DAX (joint)"] = method_from(JOINT, "LEAF-LGN+DAX (joint)")
if M["LEAF-LGN+DAX (joint)"] is None:
    M["LEAF-LGN+DAX (joint)"] = method_from(JOINT_SMALL, "LEAF-LGN+DAX (joint)")
M["QP family"] = qp_stored()
M["QP family+DAX"] = method_from(QP_DAX, "QP family+DAX", "QP family + DAX")
M["QP family (retrained)"] = method_from(QP_DAX, "QP family", "QP family (retrained)")
M["r-AdjNorm"] = method_from(STORED_BASE, "r-AdjNorm")
M["LightGCN"] = method_from(STORED_BASE, "LightGCN")
M["LightGCN+DAX"] = method_from(LGN_GCN, "LightGCN+DAX", "LightGCN + DAX")
M["EASE"] = method_from(EASE, "EASE")
M["EASE+DAX"] = method_from(EASE, "EASE+DAX", "EASE + DAX")
M = {k: v for k, v in M.items() if v is not None}

out("1. Methods: mean +- sd over the splits of nDCG@10, configurations searched, source file")
out(f"  {'method':<24s} {'mean':>8s} {'+- sd':>8s} {'n_configs':>10s}  file")
for k, m in M.items():
    mu, sd = mean_sd(m)
    out(f"  {k:<24s} {mu:8.4f} {sd:8.4f} {m['n_configs']:10d}  {m['file']}")
out("  n_configs: posthoc runs count backbone trainings x DAX configurations x gammas; LEAF-LGN (stored) keeps the")
out("  count written by experiments/run_leaf.py, which is not comparable (it counts another unit).")
out()

# ---------------------------------------------------------------------------------------------
# Gate B1
# ---------------------------------------------------------------------------------------------
out("2. Gate B1 (post-hoc, joint selection): primary family = LEAF-LGN+DAX against each comparator")
out("   per-user paired difference x1e-3 nDCG@10 [95% half-width], two-sided t p, Nadeau-Bengio p over split means;")
out("   Holm over the family for both p values")
family = []
if "LEAF-LGN+DAX" not in M:
    note("LEAF-LGN+DAX (dax_lgn_post.json) is missing: gate B1 is not evaluable")
else:
    comparators = (("QP family", "QP family"), ("QP family+DAX", "QP family+DAX"),
                   ("LEAF-LGN", "LEAF-LGN(stored)"), ("r-AdjNorm", "r-AdjNorm"), ("LightGCN", "LightGCN"))
    for label, key in comparators:
        if key in M:
            family.append((label, pair(M["LEAF-LGN+DAX"], M[key])))
        else:
            note(f"comparator {label} is missing: its member of the primary family is not evaluable")
pH = holm([r["p"] for _, r in family])
nbH = holm([r["nb"] for _, r in family])
for (comp, r), ph, nh in zip(family, pH, nbH):
    out(line(f"LEAF-LGN+DAX - {comp}", r, ph, nh))
if not family:
    out("  family is empty")
out()

out("3. Gate B1 decision")
b1 = None
if family and "QP family" in M and "LEAF-LGN+DAX" in M:
    idx = [i for i, (c, _) in enumerate(family) if c == "QP family"][0]
    r = family[idx][1]
    ph, nh = pH[idx], nbH[idx]
    passed = r["diff"] >= B1_MARGIN and ph < B1_ALPHA and nh < B1_ALPHA
    b1 = passed
    out(f"  LEAF-LGN+DAX - QP family: {r['diff']:+.2f}e-3 (required >= {B1_MARGIN:.0f}e-3), "
        f"t p_Holm {ph:.2e} and NB p_Holm {nh:.3f} (both required < {B1_ALPHA})")
    out(f"  GATE B1: {'PASS' if passed else 'FAIL'}")
    if "QP family+DAX" in M:
        j = [i for i, (c, _) in enumerate(family) if c == "QP family+DAX"]
        if j:
            rr = family[j[0]][1]
            out(f"  reading against the fair baseline: LEAF-LGN+DAX - (QP family + DAX) = {rr['diff']:+.2f}e-3 "
                f"[{rr['half']:.2f}], p_Holm {pH[j[0]]:.2e}, NB p_Holm {nbH[j[0]]:.3f}")
    if "QP family+DAX" in M and "QP family" in M:
        rq = pair(M["QP family+DAX"], M["QP family"])
        out(f"  QP family + DAX - QP family: {rq['diff']:+.2f}e-3 [{rq['half']:.2f}], NB p {rq['nb']:.3f} "
            "(DESIGN expects about +6 to +9e-3 from the LEAF increment)")
else:
    out("  GATE B1: NOT EVALUABLE (LEAF-LGN+DAX, QP family or the comparator is missing)")
out()

# ---------------------------------------------------------------------------------------------
# Gate B2
# ---------------------------------------------------------------------------------------------
out("4. Gate B2 (ablations, relative to the backbone of the same run): gains are paired differences of the arm")
out("   against LEAF-LGN (dax_ablate.json), x1e-3 nDCG@10 [95% half-width]")
abl = {}
for ab in ("full", "shuffle_tags", "shuffle_items", "additive", "no_tab", "no_life_prior", "no_tag_prior"):
    name = "LEAF-LGN+DAX" if ab == "full" else f"LEAF-LGN+DAX[{ab}]"
    m = method_from(ABLATE, name, name)
    if m is not None:
        abl[ab] = m
base_abl = method_from(ABLATE, "LEAF-LGN", "LEAF-LGN (ablation run)")
if base_abl is None:
    note("LEAF-LGN base of dax_ablate.json is missing (gate B2 not evaluable)")
if "full" not in abl or base_abl is None:
    out("  GATE B2: NOT EVALUABLE (dax_ablate.json is missing the full arm or the base)")
    b2 = None
else:
    gain_full = pair(abl["full"], base_abl)
    out(line("full DAX gain (LEAF-LGN+DAX - LEAF-LGN)", gain_full))
    gains = {}
    for ab in ("shuffle_tags", "shuffle_items", "no_tab", "no_life_prior", "no_tag_prior"):
        if ab in abl:
            gains[ab] = pair(abl[ab], base_abl)
            out(line(f"{ab} gain", gains[ab]))
    if "additive" in abl:
        out(line("additive gain (MRP-style, tab effects)", pair(abl["additive"], base_abl)))
    g0 = gain_full["diff"]
    checks = []
    if "shuffle_tags" in gains and g0 > 0:
        removed = 1.0 - gains["shuffle_tags"]["diff"] / g0
        checks.append(("shuffle_tags removes >= 90% of the DAX gain", removed >= B2_SHUFFLE_TAGS,
                       f"removed {100 * removed:.1f}%"))
    elif "shuffle_tags" in gains:
        checks.append(("shuffle_tags removes >= 90% of the DAX gain", False, "full DAX gain not positive"))
    else:
        checks.append(("shuffle_tags removes >= 90% of the DAX gain", None, "not evaluable"))
    if "shuffle_items" in gains and g0 > 0:
        drop = 1.0 - gains["shuffle_items"]["diff"] / g0
        checks.append(("shuffle_items (within tag) drops the gain by >= 30%", drop >= B2_SHUFFLE_ITEMS,
                       f"drop {100 * drop:.1f}%"))
    elif "shuffle_items" in gains:
        checks.append(("shuffle_items (within tag) drops the gain by >= 30%", False, "full DAX gain not positive"))
    else:
        checks.append(("shuffle_items (within tag) drops the gain by >= 30%", None, "not evaluable"))
    if "additive" in abl:
        below = pair(abl["full"], abl["additive"])["diff"]
        checks.append(("additive stays >= 4e-3 below DAX", below >= B2_ADDITIVE, f"DAX - additive {below:+.2f}e-3"))
    else:
        checks.append(("additive stays >= 4e-3 below DAX", None, "not evaluable"))
    out("  checks:")
    for label, ok, txt in checks:
        status = "NOT EVALUABLE" if ok is None else ("PASS" if ok else "FAIL")
        out(f"    {label:<52s} {status:<14s} {txt}")
    if all(c[1] is not None for c in checks):
        b2 = all(c[1] for c in checks)
    elif any(c[1] is False for c in checks):
        b2 = False
    else:
        b2 = None
    out(f"  GATE B2: {'NOT EVALUABLE' if b2 is None else ('PASS' if b2 else 'FAIL')}")
out()

# ---------------------------------------------------------------------------------------------
# Stage C (robustness; no numeric threshold in DESIGN.md)
# ---------------------------------------------------------------------------------------------
out("5. Stage C (robustness): reported, not a pass or fail (DESIGN.md Section 7 defines the stage without a threshold)")
out("   per setting: LEAF-LGN+DAX - QP family and LEAF-LGN+DAX - (QP family + DAX), x1e-3 nDCG@10")
if not ROBUST:
    out("  not run: no c_*.json present (KuaiRand-1K, concurrent protocol, long-view label, seeds 1-3)")
    note("Stage C inputs missing (results/v11/c_*.json)")
for path in ROBUST:
    res = results_of(path)
    tag = os.path.basename(path)
    if "LEAF-LGN+DAX" not in res:
        note(f"{path} has no LEAF-LGN+DAX")
        continue
    lg = method_from(path, "LEAF-LGN+DAX", "LEAF-LGN+DAX")
    qp = method_from(path, "QP family", "QP family")
    qd = method_from(path, "QP family+DAX", "QP family+DAX")
    if qp is not None:
        out(line(f"{tag}: vs QP family", pair(lg, qp)))
    if qd is not None:
        out(line(f"{tag}: vs QP family + DAX", pair(lg, qd)))
out()

# ---------------------------------------------------------------------------------------------
# Reproduction checks and information (not gates)
# ---------------------------------------------------------------------------------------------
out("6. Checks and information (not gates)")
if "LEAF-LGN" in M and "LEAF-LGN(stored)" in M:
    out(line("LEAF-LGN (posthoc run) - LEAF-LGN (stored)", pair(M["LEAF-LGN"], M["LEAF-LGN(stored)"])))
if "QP family (retrained)" in M and "QP family" in M:
    out(line("QP family (retrained) - QP family (stored)", pair(M["QP family (retrained)"], M["QP family"])))
if "LEAF-LGN+DAX (joint)" in M and "LEAF-LGN+DAX" in M:
    out(line("LEAF-LGN+DAX (joint) - LEAF-LGN+DAX (posthoc)", pair(M["LEAF-LGN+DAX (joint)"], M["LEAF-LGN+DAX"])))
    out("  joint variant (DESIGN Section 5.1, main instantiation): reported beside the post-hoc gate, not in its family")
if "EASE+DAX" in M and "EASE" in M:
    out(line("EASE + DAX - EASE (E1 - E0)", pair(M["EASE+DAX"], M["EASE"])))
if "LightGCN+DAX" in M and "LightGCN" in M:
    out(line("LightGCN + DAX - LightGCN (retrained)", pair(M["LightGCN+DAX"], M["LightGCN"])))
out()

header = ["Stage B / C (v11): DAX on the temporal all-user KuaiRand-27K setting, nDCG@10",
          "Gate summary: B1 " + ("PASS" if b1 else "FAIL" if b1 is not None else "NOT EVALUABLE")
          + "; B2 " + ("NOT EVALUABLE" if b2 is None else ("PASS" if b2 else "FAIL"))
          + "; C reported (no threshold)",
          "Inputs read: " + ", ".join(sorted(os.path.basename(p) for p in
                                             [STORED_LGN, STORED_BASE, STORED_QP, POST, JOINT, JOINT_SMALL, QP_DAX,
                                              LGN_GCN, ABLATE, EASE] + ROBUST if os.path.exists(p))),
          "Inputs missing or not used:"]
header += [f"  - {n}" for n in notes] if notes else ["  (none)"]
header.append("")
text = "\n".join(header + body) + "\n"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(text)
print(text)
