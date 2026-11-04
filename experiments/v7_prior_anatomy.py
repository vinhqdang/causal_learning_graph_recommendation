"""X1-X4 of DESIGN.md Section 3.4: prior anatomy, estimated propensities, and the wider iALS grid.
Analysis only: reads stored result files (results/v3, v4, v5, v6) and trains nothing.

X1 (T9): Coat (10 splits) and KuaiRec (5 splits). Test nDCG (mean and sd over splits) of iALS-p and LightGCN-p
under the priors add (IPS-fitted), add_noips, pop, shuf_item, and of the prior alone on all four datasets.
Per-user paired differences vs the learner without a prior (iALS, LightGCN) and vs the IPS additive prior,
with Holm over the whole T9 family. The KuaiRec iALS-p additive row of the DRGP matched grid (45 configurations)
is reported as a reference outside the family.
X2 (T10): Coat with estimated propensities (--prop pop) vs the shipped propensities (--prop given).
X3 (summary block in the .txt/.json): offset-only LightGCN-p vs LightGCN on all four datasets (Holm over the 4).
X4 (T11): Coat iALS-wide2 (wider grid) vs the stored iALS and vs the stored iALS-p (Holm over the 2).

Missing inputs and runs that were not executed are skipped with a printed note.
Writes results/v6/prior_anatomy.txt, results/v6/prior_anatomy.json, and
paper/v4/tables/T9_prior_anatomy.tex, T10_estimated_propensity.tex, T11_wider_ials_grid.tex.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import (KEY, MARGIN, NAME, OUT, V3, V4, V5, V6, Report, holm, load_json,  # noqa: E402
                       paired_stats, user_diffs)

TDIR = "paper/v4/tables"
PRIORS = ["add", "add_noips", "pop", "shuf_item", "user_only"]
PRIOR_NAME = {"add": "add (IPS-fitted)", "add_noips": "add, no IPS", "pop": "pop",
              "shuf_item": "shuffled item bias", "user_only": "user only (b_i = 0)"}
PRIOR_TXT = {"add": "add (IPS-fitted)", "add_noips": "add, no IPS", "pop": "pop",
             "shuf_item": "shuffled item bias", "user_only": "user only (b_i=0)"}

# ---------------------------------------------------------------------------------------------
# Sources: (result file, method name). The no-prior learners are the stored runs of the main table.
# ---------------------------------------------------------------------------------------------
SRC = {
    "coat": {
        "ref": {"iALS": (V3 + "learned_coat_given.json", "iALS"), "LightGCN": (V3 + "learned_coat_given.json", "LightGCN")},
        "iALS-p": {"add": (V5 + "learned_coat_add_ials.json", "iALS-p"),
                   "add_noips": (V6 + "learned_coat_add_noips_ials.json", "iALS-p"),
                   "pop": (V6 + "learned_coat_pop_ials.json", "iALS-p"),
                   "shuf_item": (V6 + "learned_coat_shuf_item_ials.json", "iALS-p")},
        "LightGCN-p": {"add": (V4 + "learned_coat_prior_bpr.json", "LightGCN-p"),
                       "add_noips": (V6 + "learned_coat_add_noips_lgn.json", "LightGCN-p"),
                       "pop": (V6 + "learned_coat_pop_lgn.json", "LightGCN-p"),
                       "shuf_item": (V6 + "learned_coat_shuf_item_lgn.json", "LightGCN-p")},
        "alone": {p: (V6 + f"learned_coat_alone_{p}.json", "Prior-alone") for p in PRIORS},
    },
    "kuairec": {
        "ref": {"iALS": (V3 + "learned_kuairec_pop_A.json", "iALS"), "LightGCN": (V3 + "learned_kuairec_pop_B.json", "LightGCN")},
        "iALS-p": {"add": (V6 + "learned_kuairec_add_ials_small.json", "iALS-p-small"),
                   "add_noips": (V6 + "learned_kuairec_add_noips_ials_small.json", "iALS-p-small"),
                   "pop": (V6 + "learned_kuairec_pop_ials_small.json", "iALS-p-small"),
                   "shuf_item": (V6 + "learned_kuairec_shuf_item_ials_small.json", "iALS-p-small")},
        "LightGCN-p": {"add": (V4 + "learned_kuairec_offonly.json", "LightGCN-p")},
        "alone": {p: (V6 + f"learned_kuairec_alone_{p}.json", "Prior-alone") for p in PRIORS},
    },
    "yahoo": {
        "ref": {"LightGCN": (V3 + "learned_yahoo_pop_B.json", "LightGCN")},
        "LightGCN-p": {"add": (V6 + "learned_yahoo_offonly.json", "LightGCN-p")},
        "alone": {p: (V6 + f"learned_yahoo_alone_{p}.json", "Prior-alone") for p in PRIORS},
    },
    "kuairand": {
        "ref": {"LightGCN": (V3 + "learned_kuairand_pop.json", "LightGCN")},
        "LightGCN-p": {"add": (V6 + "learned_kuairand_offonly.json", "LightGCN-p")},
        "alone": {p: (V6 + f"learned_kuairand_alone_{p}.json", "Prior-alone") for p in PRIORS},
    },
}
DS4 = ["coat", "yahoo", "kuairand", "kuairec"]

# X2: Coat, shipped vs estimated propensities (estimated = --prop pop)
X2_SHIP = {"iALS": (V3 + "learned_coat_given.json", "iALS"),
           "LightGCN": (V3 + "learned_coat_given.json", "LightGCN"),
           "iALS-p": (V5 + "learned_coat_add_ials.json", "iALS-p"),
           "LightGCN-p": (V4 + "learned_coat_prior_bpr.json", "LightGCN-p"),
           "iALS-wide": (V5 + "learned_coat_ialswide.json", "iALS-wide"),
           "imputation alone (add)": (V6 + "learned_coat_alone_add.json", "Prior-alone")}
X2_EST = {"iALS-p": (V6 + "learned_coat_pop_add_ials.json", "iALS-p"),
          "LightGCN-p": (V6 + "learned_coat_pop_lgn_p.json", "LightGCN-p"),
          "iALS-wide": (V6 + "learned_coat_pop_ialswide.json", "iALS-wide"),
          "LightGCN": None,   # design X2 lists a 54-configuration run; no prop=pop plain LightGCN file exists
          "imputation alone (add)": (V6 + "learned_coat_pop_alone_add.json", "Prior-alone")}
X2_NOPRIOR = {"iALS-p": "iALS", "iALS-wide": "iALS", "LightGCN-p": "LightGCN"}

# X3: offset-only LightGCN-p vs LightGCN (add prior, shipped propensities for Coat/KuaiRec, pop for Yahoo/KuaiRand)
X3 = {"coat": ((V4 + "learned_coat_prior_bpr.json", "LightGCN-p"), (V3 + "learned_coat_given.json", "LightGCN")),
      "yahoo": ((V6 + "learned_yahoo_offonly.json", "LightGCN-p"), (V3 + "learned_yahoo_pop_B.json", "LightGCN")),
      "kuairand": ((V6 + "learned_kuairand_offonly.json", "LightGCN-p"), (V3 + "learned_kuairand_pop.json", "LightGCN")),
      "kuairec": ((V4 + "learned_kuairec_offonly.json", "LightGCN-p"), (V3 + "learned_kuairec_pop_B.json", "LightGCN"))}

# X4: Coat iALS grids (the grids of run_learned.py; the expected configuration counts are checked)
X4 = [("iALS (stored)", V3 + "learned_coat_given.json", "iALS", "alpha {3,10,30} x reg {1,10,100,1000} x dim {32,64}", 24),
      ("iALS-wide (v5 grid)", V5 + "learned_coat_ialswide.json", "iALS-wide", "alpha {10,30,100} x reg {10,100,300} x dim {32,64}", 18),
      ("iALS-wide2 (wider grid)", V6 + "learned_coat_ialswide2.json", "iALS-wide2", "alpha {30,100,300,1000} x reg {10,100,300,1000} x dim {32,64}", 32),
      ("iALS-p add (stored)", V5 + "learned_coat_add_ials.json", "iALS-p", "alpha {10,30,100} x reg {10,100,300} x pbeta {0.1,0.2,0.3,0.5,1} x dim {32,64}", 90)]


# ---------------------------------------------------------------------------------------------
def load(ds, spec, rep, label):
    """One stored run: per-split test metric, per-user values and selections. None (with a note) if absent."""
    if spec is None:
        return None
    path, method = spec
    if not os.path.exists(path):
        rep.note(f"{label}: missing {path}; row skipped")
        return None
    res = load_json(path)["results"].get(method)
    if res is None or not res.get("per_user"):
        rep.note(f"{label}: {method} with per-user values not found in {path}; row skipped")
        return None
    vals = np.array([s[KEY[ds]] for s in res["per_split"]], dtype=float)
    return {"label": label, "path": path, "method": method, "vals": vals, "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)), "S": len(vals), "per_user": res["per_user"],
            "n_configs": res.get("n_configs"), "chosen": res.get("chosen", [])}


def compare(ds, A, B, rep):
    """Paired per-user difference A - B (user_diffs averages a user over the splits they appear in)."""
    if A is None or B is None:
        return None
    if len(A["per_user"]) != len(B["per_user"]):
        rep.note(f"{A['label']} vs {B['label']}: split counts differ; compared over the shared splits")
    d = user_diffs(A["per_user"], B["per_user"])
    if len(d) < 2:
        rep.note(f"{A['label']} vs {B['label']}: fewer than two common users; not compared")
        return None
    st = paired_stats(d, MARGIN[ds])
    return {"A": A["label"], "B": B["label"], "diff": st["diff"], "half": st["half"], "p": st["p"],
            "n_users": st["n_users"], "identical": bool(st.get("identical", False)), "p_holm": None}


def holm_family(comps):
    """Holm over the non-None comparisons of one family; stores p_holm in each."""
    comps = [c for c in comps if c is not None]
    adj = holm([c["p"] for c in comps]) if comps else []
    for c, a in zip(comps, adj):
        c["p_holm"] = float(a)
    return len(comps)


def sel_counts(chosen, key):
    c = {}
    for ch in chosen:
        if key in ch:
            c[ch[key]] = c.get(ch[key], 0) + 1
    return ", ".join(f"{k:g}:{v}" for k, v in sorted(c.items()))


def fmt_d(c, tex=False):
    if c is None:
        return "--"
    if c["identical"]:
        return "0 (identical)"
    return f"{1000 * c['diff']:+.2f} [{1000 * c['half']:.2f}]"


def fmt_p(c, tex=False):
    if c is None or c["identical"]:
        return "--" if c is None else "1"
    p = c["p_holm"]
    if p < 0.001:
        return "$<$0.001" if tex else "<0.001"
    return f"{p:.3f}"


def fmt_m(r, tex=False):
    if r is None:
        return "--"
    s = f"{r['mean']:.4f} $\\pm$ {r['sd']:.4f}" if tex else f"{r['mean']:.4f} +- {r['sd']:.4f}"
    return s


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def tex_grid(s):
    return esc(s).replace("{", r"\{").replace("}", r"\}")


def tex_table(colspec, header, rows, caption, label, note):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", f"\\par\\smallskip\\raggedright\\footnotesize {note}",
              r"\end{table}", ""]
    return "\n".join(lines)


def write_tex(name, body, rep):
    os.makedirs(TDIR, exist_ok=True)
    with open(os.path.join(TDIR, name), "w") as f:
        f.write(body)
    rep(f"written: {TDIR}/{name}")


# ---------------------------------------------------------------------------------------------
def x1(rep, fam):
    """Rows of T9. Returns (rows, extra) where extra holds the DRGP-grid reference (outside the family)."""
    rows = []
    for ds in ["coat", "kuairec"]:
        s = SRC[ds]
        for learner, ref_name in [("iALS-p", "iALS"), ("LightGCN-p", "LightGCN")]:
            ref = load(ds, s["ref"][ref_name], rep, f"{NAME[ds]} {ref_name} (no prior)")
            rows.append({"ds": ds, "group": f"{ref_name} (no prior)", "prior": "--", "stat": ref,
                         "vs_ref": None, "vs_add": None, "ref": True})
            if learner not in s:
                continue
            notrun = [p for p in PRIORS if p not in s[learner]]
            rep(f"  {NAME[ds]} {learner}: not run for priors {notrun} (no result file)")
            add = None
            for p in PRIORS:
                if p not in s[learner]:
                    continue
                A = load(ds, s[learner][p], rep, f"{NAME[ds]} {learner} {p}")
                if p == "add":
                    add = A
                vs_ref = compare(ds, A, ref, rep)
                vs_add = compare(ds, A, add, rep) if p != "add" else None
                fam.extend([vs_ref, vs_add])
                rows.append({"ds": ds, "group": learner, "prior": p, "stat": A, "vs_ref": vs_ref, "vs_add": vs_add,
                             "ref": False})
    for ds in DS4:
        add_alone = load(ds, SRC[ds]["alone"]["add"], rep, f"{NAME[ds]} prior alone add")
        for p in PRIORS:
            A = load(ds, SRC[ds]["alone"][p], rep, f"{NAME[ds]} prior alone {p}")
            vs_add = compare(ds, A, add_alone, rep) if p != "add" else None
            fam.append(vs_add)
            rows.append({"ds": ds, "group": "prior alone", "prior": p, "stat": A, "vs_ref": None, "vs_add": vs_add,
                         "ref": False})
    # reference outside the family: KuaiRec iALS-p, additive prior, DRGP matched grid (45 configurations)
    extra = {}
    alt = load("kuairec", (V5 + "learned_kuairec_add_ials.json", "iALS-p"), rep, "KuaiRec iALS-p add, DRGP grid")
    small = load("kuairec", SRC["kuairec"]["iALS-p"]["add"], rep, "KuaiRec iALS-p add, X1 grid")
    ref = load("kuairec", SRC["kuairec"]["ref"]["iALS"], rep, "KuaiRec iALS (no prior)")
    extra["kuairec_drgp_grid_add"] = {"stat": alt, "vs_x1_grid_add": compare("kuairec", alt, small, rep),
                                      "vs_iALS": compare("kuairec", alt, ref, rep)}
    return rows, extra


def x2(rep):
    """Rows of T10 (Coat, shipped vs estimated propensities) and the comparison list for the family."""
    ship = {k: load("coat", v, rep, f"Coat {k} (shipped)") for k, v in X2_SHIP.items()}
    est = {k: load("coat", v, rep, f"Coat {k} (estimated)") for k, v in X2_EST.items()}
    if est["LightGCN"] is None:
        rep.note("Coat LightGCN (estimated propensities): no prop=pop plain-LightGCN run in results/v6 "
                 "(design X2 lists 54 configurations); row not computed")
    fam, rows = [], []
    for name in ["iALS-p", "LightGCN-p", "iALS-wide", "imputation alone (add)"]:
        ref_name = X2_NOPRIOR.get(name)
        gain_ship = compare("coat", ship[name], ship[ref_name], rep) if ref_name else None
        gain_est = compare("coat", est[name], ship[ref_name], rep) if (ref_name and est.get(name)) else None
        ev = compare("coat", est[name], ship[name], rep) if (est.get(name) and ship.get(name)) else None
        fam += [gain_ship, gain_est, ev]
        rows.append({"method": name, "prop": "shipped", "stat": ship[name], "vs_noprior": gain_ship,
                     "est_vs_ship": None, "ref_name": ref_name})
        rows.append({"method": name, "prop": "estimated", "stat": est.get(name), "vs_noprior": gain_est,
                     "est_vs_ship": ev, "ref_name": ref_name})
    # plain learners read no propensity (only NAVIP does): shipped run only
    rows.append({"method": "iALS (no prior)", "prop": "shipped", "stat": ship["iALS"], "vs_noprior": None,
                 "est_vs_ship": None, "ref_name": None})
    rows.append({"method": "LightGCN (no prior)", "prop": "shipped", "stat": ship["LightGCN"], "vs_noprior": None,
                 "est_vs_ship": None, "ref_name": None})
    # sanity check: plain iALS-wide does not read the propensities, so its two runs should coincide
    chk = compare("coat", est["iALS-wide"], ship["iALS-wide"], rep)
    if chk is not None:
        rep(f"  check: iALS-wide estimated minus shipped: diff {1000 * chk['diff']:+.4f} x1e-3, identical={chk['identical']}")
    return rows, fam


def x3(rep):
    """Offset-only LightGCN-p vs LightGCN on all four datasets (summary; Holm over the four)."""
    out, fam = {}, []
    for ds in DS4:
        (pa, ma), (pb, mb) = X3[ds]
        A = load(ds, (pa, ma), rep, f"X3 {NAME[ds]} LightGCN-p (add)")
        B = load(ds, (pb, mb), rep, f"X3 {NAME[ds]} LightGCN")
        c = compare(ds, A, B, rep)
        out[ds] = {"A": A, "B": B, "cmp": c}
        fam.append(c)
    holm_family(fam)
    return out


def x4(rep):
    """Rows of T11 and the two comparisons (Holm over the two)."""
    st = {}
    for label, path, mname, grid, n_exp in X4:
        r = load("coat", (path, mname), rep, f"X4 {label}")
        st[label] = r
        if r is not None and r["n_configs"] != n_exp:
            rep.note(f"X4 {label}: {r['n_configs']} configurations stored, expected {n_exp} from the grid")
    wide2, base, pp = st["iALS-wide2 (wider grid)"], st["iALS (stored)"], st["iALS-p add (stored)"]
    cmp_base = compare("coat", wide2, base, rep)
    cmp_pp = compare("coat", wide2, pp, rep)
    holm_family([cmp_base, cmp_pp])
    return st, cmp_base, cmp_pp


def main():
    rep = Report()
    rep("X1-X4 prior anatomy (experiments/v7_prior_anatomy.py; design DESIGN.md Section 3.4)")
    rep("Metric: " + ", ".join(f"{NAME[d]} {KEY[d]}" for d in DS4) + ". Per-user differences averaged over the splits;"
        " Holm within each table family. Differences x1e-3, [half-width of the 95% t interval].")

    # ---------------- X1
    rep("")
    rep("== X1 (T9): prior anatomy")
    fam9 = []
    rows9, extra = x1(rep, fam9)
    n9 = holm_family(fam9)
    rep(f"  family T9: {n9} paired tests (Holm)")
    for r in rows9:
        if r["stat"] is None and not r["ref"]:
            continue
        rep(f"  {NAME[r['ds']]:9s} {r['group']:12s} {PRIOR_TXT.get(r['prior'], r['prior']):22s} "
            f"nDCG {fmt_m(r['stat'])}  vs no prior {fmt_d(r['vs_ref']):18s} pHolm {fmt_p(r['vs_ref']):7s}  "
            f"vs add {fmt_d(r['vs_add']):18s} pHolm {fmt_p(r['vs_add'])}")
    rep("  reference (outside the family): KuaiRec iALS-p, additive prior, DRGP matched grid (45 configurations)")
    alt = extra["kuairec_drgp_grid_add"]
    if alt["stat"] is not None:
        rep(f"    nDCG {fmt_m(alt['stat'])}; vs X1-grid add {fmt_d(alt['vs_x1_grid_add'])} "
            f"(p {alt['vs_x1_grid_add']['p'] if alt['vs_x1_grid_add'] else float('nan'):.3g}); "
            f"vs iALS {fmt_d(alt['vs_iALS'])}")

    # ---------------- X2
    rep("")
    rep("== X2 (T10): Coat, estimated (--prop pop) vs shipped propensities")
    rows10, fam10 = x2(rep)
    n10 = holm_family(fam10)
    rep(f"  family T10: {n10} paired tests (Holm)")
    for r in rows10:
        rep(f"  {r['method']:24s} {r['prop']:10s} nDCG {fmt_m(r['stat'])}  vs no prior {fmt_d(r['vs_noprior']):18s} "
            f"pHolm {fmt_p(r['vs_noprior']):7s}  est-ship {fmt_d(r['est_vs_ship']):18s} pHolm {fmt_p(r['est_vs_ship'])}")

    # ---------------- X3
    rep("")
    rep("== X3 (summary): offset-only LightGCN-p (add prior) vs LightGCN")
    x3res = x3(rep)
    for ds in DS4:
        r = x3res[ds]
        c = r["cmp"]
        if r["A"] is None or r["B"] is None:
            rep(f"  {NAME[ds]:9s} not computed")
            continue
        rep(f"  {NAME[ds]:9s} LightGCN-p {fmt_m(r['A'])}  LightGCN {fmt_m(r['B'])}  diff {fmt_d(c)}  "
            f"pHolm {fmt_p(c)}  (S={r['A']['S']})")

    # ---------------- X4
    rep("")
    rep("== X4 (T11): Coat iALS-wide2 (wider grid) vs stored iALS and iALS-p")
    st4, c_base, c_pp = x4(rep)
    for label, path, mname, grid, n_exp in X4:
        r = st4[label]
        if r is None:
            continue
        rep(f"  {label:22s} cfg {r['n_configs']:3d} nDCG {fmt_m(r)}  alpha {sel_counts(r['chosen'], 'alpha')}  "
            f"reg {sel_counts(r['chosen'], 'reg')}  dim {sel_counts(r['chosen'], 'dim')}")
    rep(f"  iALS-wide2 - iALS (stored): diff {fmt_d(c_base)}  pHolm {fmt_p(c_base)}")
    rep(f"  iALS-wide2 - iALS-p (stored): diff {fmt_d(c_pp)}  pHolm {fmt_p(c_pp)}")

    # ---------------- JSON
    res = {
        "metric": {d: KEY[d] for d in DS4},
        "X1": [{"dataset": r["ds"], "group": r["group"], "prior": r["prior"],
                "nDCG_mean": r["stat"]["mean"] if r["stat"] else None, "nDCG_sd": r["stat"]["sd"] if r["stat"] else None,
                "S": r["stat"]["S"] if r["stat"] else None, "n_configs": r["stat"]["n_configs"] if r["stat"] else None,
                "vs_no_prior": r["vs_ref"], "vs_add": r["vs_add"], "source": r["stat"]["path"] if r["stat"] else None,
                "method": r["stat"]["method"] if r["stat"] else None} for r in rows9],
        "X1_drgp_grid_reference": {"stat_mean": extra["kuairec_drgp_grid_add"]["stat"]["mean"]
                                   if extra["kuairec_drgp_grid_add"]["stat"] else None,
                                   "vs_x1_grid_add": extra["kuairec_drgp_grid_add"]["vs_x1_grid_add"],
                                   "vs_iALS": extra["kuairec_drgp_grid_add"]["vs_iALS"]},
        "X2": [{"method": r["method"], "propensity": r["prop"],
                "nDCG_mean": r["stat"]["mean"] if r["stat"] else None, "nDCG_sd": r["stat"]["sd"] if r["stat"] else None,
                "vs_no_prior": r["vs_noprior"], "est_minus_shipped": r["est_vs_ship"],
                "no_prior_reference": r["ref_name"]} for r in rows10],
        "X3": {ds: {"LightGCN-p_mean": x3res[ds]["A"]["mean"] if x3res[ds]["A"] else None,
                    "LightGCN_mean": x3res[ds]["B"]["mean"] if x3res[ds]["B"] else None,
                    "LightGCN-p_minus_LightGCN": x3res[ds]["cmp"]} for ds in DS4},
        "X4": {"rows": [{"model": label, "n_configs": st4[label]["n_configs"] if st4[label] else None,
                         "nDCG_mean": st4[label]["mean"] if st4[label] else None,
                         "nDCG_sd": st4[label]["sd"] if st4[label] else None, "grid": grid}
                        for label, _, _, grid, _ in X4],
               "wide2_minus_iALS": c_base, "wide2_minus_iALS-p": c_pp},
        "notes": [ln for ln in rep.lines if ln.startswith("NOTE:")],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "prior_anatomy.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rep("")
    rep(f"written: {OUT}/prior_anatomy.json")

    # ---------------- LaTeX: T9
    body9 = []
    for r in rows9:
        if r["stat"] is None and not r["ref"]:
            continue
        if r["ref"]:
            body9.append(f"{NAME[r['ds']]} & {r['group']} & {fmt_m(r['stat'], True)} & -- & -- & -- & --")
        elif r["group"] == "prior alone":
            body9.append(f"{NAME[r['ds']]} & prior alone, {esc(PRIOR_NAME[r['prior']])} & {fmt_m(r['stat'], True)} & "
                         f"-- & -- & {fmt_d(r['vs_add'])} & {fmt_p(r['vs_add'], True)}")
        else:
            body9.append(f"{NAME[r['ds']]} & {r['group']}, {esc(PRIOR_NAME[r['prior']])} & "
                         f"{fmt_m(r['stat'], True)} & {fmt_d(r['vs_ref'])} & {fmt_p(r['vs_ref'], True)} & "
                         f"{fmt_d(r['vs_add'])} & {fmt_p(r['vs_add'], True)}")
    t9 = tex_table(
        "llrrrrr",
        "Dataset & Learner / prior & nDCG (mean $\\pm$ sd) & $\\Delta$ vs no prior $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$ & "
        "$\\Delta$ vs add $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$",
        body9,
        "Prior anatomy. Test nDCG of iALS-p and LightGCN-p under four additive-imputation variants, and of the "
        "prior alone on all four datasets.", "tab:T9",
        "$\\Delta$ = per-user paired difference (mean over splits), $\\times10^{3}$, [half-width of the 95\\% CI]. "
        "No-prior references: stored iALS and LightGCN. The add row is the IPS-fitted additive prior. "
        "$p_\\mathrm{Holm}$ over all paired tests of this table. user only (b\\_i = 0) gives the same score to every item "
        "for a given user, so its ranking is decided by tie-breaking. Coat LightGCN-p and iALS-p use 10 splits, KuaiRec 5. "
        "Rows absent from the table were not run (only the prior-alone rows exist for those priors).")
    write_tex("T9_prior_anatomy.tex", t9, rep)

    # ---------------- LaTeX: T10
    body10 = []
    for r in rows10:
        label = esc(r["method"])
        if r["stat"] is None:
            body10.append(f"{label} & {r['prop']} & not run & -- & -- & -- & --")
            continue
        body10.append(f"{label} & {r['prop']} & {fmt_m(r['stat'], True)} & {fmt_d(r['vs_noprior'])} & "
                      f"{fmt_p(r['vs_noprior'], True)} & {fmt_d(r['est_vs_ship'])} & {fmt_p(r['est_vs_ship'], True)}")
    t10 = tex_table(
        "llrrrrr",
        "Method & Propensities & nDCG (mean $\\pm$ sd) & $\\Delta$ vs no-prior $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$ & "
        "Estimated $-$ shipped $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$",
        body10,
        "Coat with estimated propensities versus the shipped propensities.", "tab:T10",
        "The no-prior reference of each learner is the stored run. Plain iALS and LightGCN do not read the propensities "
        "(only NAVIP does), so the shipped no-prior run is the reference for the estimated rows. "
        "LightGCN with estimated propensities was not run. $p_\\mathrm{Holm}$ over the tests of this table.")
    write_tex("T10_estimated_propensity.tex", t10, rep)

    # ---------------- LaTeX: T11
    body11 = []
    for label, path, mname, grid, n_exp in X4:
        r = st4[label]
        if r is None:
            continue
        if label == "iALS-wide2 (wider grid)":
            d1 = f"{fmt_d(c_base)} & {fmt_p(c_base, True)} & {fmt_d(c_pp)} & {fmt_p(c_pp, True)}"
        elif label == "iALS (stored)":
            d1 = "ref. & -- & -- & --"
        else:
            d1 = "-- & -- & -- & --"
        sel = f"alpha {sel_counts(r['chosen'], 'alpha')}; reg {sel_counts(r['chosen'], 'reg')}"
        body11.append(f"{esc(label)} & {tex_grid(grid)} & {r['n_configs']} & {fmt_m(r, True)} & {d1} & {sel}")
    t11 = tex_table(
        "llrrrrrrl",
        "Model & Grid & Cfg & nDCG (mean $\\pm$ sd) & $\\Delta$ vs iALS $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$ & "
        "$\\Delta$ vs iALS-p $\\times10^{3}$ [CI] & $p_\\mathrm{Holm}$ & Selected values (number of splits)",
        body11,
        "Coat iALS with the wider grid against the stored iALS and iALS-p.", "tab:T11",
        "$\\Delta$ = per-user paired difference (mean over the 10 splits), $\\times10^{3}$, [half-width of the 95\\% CI]. "
        "$p_\\mathrm{Holm}$ over the two comparisons of iALS-wide2. Selected values are counted over the 10 splits.")
    write_tex("T11_wider_ials_grid.tex", t11, rep)

    rep.save("prior_anatomy.txt")
    rep(f"written: {OUT}/prior_anatomy.txt")


if __name__ == "__main__":
    main()
