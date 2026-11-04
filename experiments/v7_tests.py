"""A2: paired tests for the primary comparisons (DESIGN.md Section 2.3), the walk correction, the refinement
family (T6) and every method against the backbone.

For each comparison A minus B:
  - per-user difference: mean over the splits in which the user is a test user (significance.user_diffs);
  - paired t-test over users, 95% t half-width, MDE at 80% power (two-sided 5%) = (t_.975 + t_.80) * se;
  - Holm within the family (the primary family of a dataset; the walk-correction family; the refinement family;
    the backbone family), also for the Nadeau-Bengio p over split means (significance.nadeau_bengio);
  - TOST at the margin of DESIGN.md Section 2.3 (one-sided 5% each way); "equivalent" if TOST p < 0.05;
  - seed robustness (seeds 1-3, results/v4/seeds_<dataset>.json) where the file holds both methods.
Inputs that do not exist yet are skipped with a printed note.
Output: results/v6/tests.txt, results/v6/tests.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import (KEY, MARGIN, NAME, OUT, Report, holm, load_all, nadeau_bengio_p, paired_stats,  # noqa: E402
                       primary_pairs, user_diffs, val_mean)

SEED_NAME = {"QP family (pooled)": "final", "iALS": "iALS", "DR-LightGCN": "DR-LightGCN", "r-AdjNorm": "r-AdjNorm",
             "LightGCN": "LightGCN", "DR-JL": "DR-JL", "MRDR": "MRDR"}
REFINEMENT_BASE = {"coat": "LightGCN-p (add prior)", "kuairec": "LightGCN-p (add prior)", "yahoo": "LightGCN-p (add prior)",
                   "kuairand": "LightGCN-p (add prior)"}


def seed_diff(ds, a, b):
    """Mean over seeds 1-3 and splits of the per-user difference, with the paired t p-value; None if absent."""
    path = f"results/v4/seeds_{ds}.json"
    if not os.path.exists(path):
        return None
    res = json.load(open(path))["results"]
    if a not in res or b not in res:
        return None
    sa, sb = res[a]["seeds"], res[b]["seeds"]
    acc_a, acc_b = {}, {}
    for sd in ("1", "2", "3"):
        if sd not in sa or sd not in sb:
            return None
        for pa, pb in zip(sa[sd]["per_user"], sb[sd]["per_user"]):
            for u, v in pa.items():
                if u in pb:
                    acc_a.setdefault(u, []).append(v)
                    acc_b.setdefault(u, []).append(pb[u])
    us = sorted(acc_a)
    d = np.array([np.mean(acc_a[u]) - np.mean(acc_b[u]) for u in us])
    st = paired_stats(d, MARGIN[ds])
    return {"diff": st["diff"], "p": st["p"], "n_users": st["n_users"]}


def compare(ds, A, B, key, margin):
    d = user_diffs(A.per_user, B.per_user)
    st = paired_stats(d, margin)
    nb = nadeau_bengio_p(A.per_split, B.per_split, key)
    return {"A": A.name, "B": B.name, "n_users": st["n_users"], "diff": st["diff"], "half": st["half"], "p": st["p"],
            "mde": st["mde"], "tost_margin": margin, "tost_p": st["tost_p"], "equivalent": bool(st["tost_p"] < 0.05),
            "identical": bool(st.get("identical", False)), "nb_p": nb if not st.get("identical") else 1.0, "seed": seed_diff(ds, SEED_NAME.get(A.name, "?"), SEED_NAME.get(B.name, "?"))
            if A.name in SEED_NAME and B.name in SEED_NAME else None}


def add_holm(rows, field, out_field):
    for r, q in zip(rows, holm([r[field] for r in rows])):
        r[out_field] = float(q)


def main():
    rep = Report()
    rep("A2 paired tests (experiments/v7_tests.py); differences x1e-3 with 95% half-width in brackets")
    rep("Per-user differences averaged over the test splits of each user. Holm within each family.")
    rep("")
    out = {}
    for ds in ["coat", "yahoo", "kuairand", "kuairec"]:
        key = KEY[ds]
        margin = MARGIN[ds]
        rep(f"=== {NAME[ds]} ({key}; TOST margin +-{margin:.3f}) ===")
        M = load_all(ds, rep)
        ds_out = {"margin": margin, "families": {}}

        # ---- primary family (P1-P4) -----------------------------------------------------------------
        prim = []
        for fam, a, b in primary_pairs(ds, M):
            if fam == "W":
                continue
            if a not in M or b not in M:
                rep(f"  {fam} {a} vs {b}: missing method, skipped")
                continue
            r = compare(ds, M[a], M[b], key, margin)
            r["family"] = fam
            prim.append(r)
        add_holm(prim, "p", "p_holm")
        add_holm(prim, "nb_p", "nb_holm")
        ds_out["families"]["primary"] = prim

        # ---- walk correction (DRUP vs DR) -------------------------------------------------------------
        walk = []
        if "DRUP" in M and "DR" in M:
            r = compare(ds, M["DRUP"], M["DR"], key, margin)
            r["family"] = "W"
            walk.append(r)
        add_holm(walk, "p", "p_holm")
        add_holm(walk, "nb_p", "nb_holm")
        ds_out["families"]["walk"] = walk

        # ---- T6: refinement family vs simple prior in the same learner (+ controls vs backbone) ---------
        ref = []
        base = REFINEMENT_BASE[ds]
        if base in M:
            for name, m in M.items():
                if m.role in ("refinement", "final") and name not in (base, "DR-iALS"):
                    if ds == "kuairec" and name == "DR-BPR-LGN":
                        continue
                    r = compare(ds, m, M[base], key, margin)
                    r["family"] = "T6 refinement vs simple prior"
                    ref.append(r)
            if "DR-iALS" in M and "iALS-p (add prior)" in M:
                r = compare(ds, M["DR-iALS"], M["iALS-p (add prior)"], key, margin)
                r["family"] = "T6 refinement vs simple prior (iALS)"
                ref.append(r)
        else:
            rep(f"  T6: simple prior '{base}' missing on {NAME[ds]}; refinement rows skipped")
        if "LightGCN" in M:
            for name, m in M.items():
                if m.role == "control":
                    r = compare(ds, m, M["LightGCN"], key, margin)
                    r["family"] = "T6 controls vs backbone"
                    ref.append(r)
        for fam in ["T6 refinement vs simple prior", "T6 refinement vs simple prior (iALS)", "T6 controls vs backbone"]:
            sub = [r for r in ref if r["family"] == fam]
            add_holm(sub, "p", "p_holm")
            add_holm(sub, "nb_p", "nb_holm")
        ds_out["families"]["T6"] = ref

        # ---- every method against the backbone (appendix) ------------------------------------------
        allv = []
        if "LightGCN" in M:
            for name, m in M.items():
                if name == "LightGCN" or m.per_user is None:
                    continue
                r = compare(ds, m, M["LightGCN"], key, margin)
                r["family"] = "all vs backbone"
                allv.append(r)
            add_holm(allv, "p", "p_holm")
            add_holm(allv, "nb_p", "nb_holm")
        ds_out["families"]["all_vs_backbone"] = allv
        out[ds] = ds_out

        # ---- report -------------------------------------------------------------------------------------
        def line(r, tag=""):
            s = (f"  {r['family'][:9]:9s} {r['A'][:26]:26s} - {r['B'][:22]:22s} n={r['n_users']:5d} "
                 f"{1000 * r['diff']:+8.2f} [{1000 * r['half']:6.2f}] p={r['p']:.3g} pHolm={r['p_holm']:.3g} "
                 f"NB={r['nb_p']:.3g} NBHolm={r['nb_holm']:.3g} MDE={1000 * r['mde']:6.2f} "
                 f"TOST(+-{r['tost_margin']:.3f}) p={r['tost_p']:.3g}{' equiv' if r['equivalent'] else ''}")
            if r.get("seed"):
                s += f" seeds1-3 {1000 * r['seed']['diff']:+.2f} p={r['seed']['p']:.3g}"
            rep(s + tag)

        rep(" primary family (Holm over P1-P4 of this dataset):")
        for r in prim:
            line(r)
        rep(" walk correction (DRUP - DR):")
        for r in walk:
            line(r)
        rep(" T6 refinement family:")
        for r in ref:
            line(r)
        rep(" every method against LightGCN (appendix):")
        for r in allv:
            line(r)
        rep("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tests.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    rep.save("tests.txt")


if __name__ == "__main__":
    main()
