#!/usr/bin/env python3
"""LaTeX tables of the v8 manuscript, generated from the result JSON files (paper/v8/tables/*.tex).

Every number is computed here from results/v10, results/v11 and results/v12. The statistics are the helper functions
of experiments/final_v12.py (paired per-user differences, the t interval and p, Holm, the Nadeau-Bengio split test,
the method records). They are taken from that file by parsing it and executing only its imports, constants and
function definitions, so the analysis is not re-run and the helpers are not copied. An input that is missing gives a
{TBD:file quantity} cell; nothing is typed by hand.

Usage: python3 experiments/v13_tables.py          (writes paper/v8/tables/*.tex and prints a summary)
"""
import ast
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
TABLES = os.path.join(ROOT, "paper", "v8", "tables")
sys.path.insert(0, EXP)
from significance import nadeau_bengio  # noqa: E402  (importing significance.py runs nothing)

_CTX = None


def load_helpers():
    """Execute the imports, constants and function definitions of experiments/final_v12.py (nothing else)."""
    path = os.path.join(EXP, "final_v12.py")
    tree = ast.parse(open(path).read(), filename=path)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if any((x.isupper() and len(x) > 1 and x != "EXPECTED_N") or x in ("body", "notes", "_cache")
                   for x in names):
                keep.append(node)
    ns = {"__file__": path, "json": json, "os": os, "sys": sys, "np": np, "stats": stats,
          "nadeau_bengio": nadeau_bengio}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])), path, "exec"), ns)
    return ns


# source label of every method, shown inside a {TBD:...} cell when the method is missing
SRC = {
    "LEAF-LGN+DAX": "results/v11/dax_lgn_post.json LEAF-LGN+DAX",
    "LEAF-LGN": "results/v11/dax_lgn_post.json LEAF-LGN",
    "LEAF-LGN(stored)": "results/v10/leaf_lgn.json LEAF-LGN",
    "QP family": "results/v10/learned_kuairand_t_qp.json QP family",
    "QP family+DAX": "results/v11/dax_qp.json QP family+DAX",
    "QP family (retrained)": "results/v11/dax_qp.json QP family",
    "r-AdjNorm": "results/v10/learned_kuairand_t_base.json r-AdjNorm",
    "LightGCN": "results/v10/learned_kuairand_t_base.json LightGCN",
    "LightGCN+DAX": "results/v11/dax_lightgcn.json LightGCN+DAX",
    "EASE": "results/v11/dax_ease.json EASE",
    "EASE+DAX": "results/v11/dax_ease.json EASE+DAX",
    "DIN": "results/v12/din_27k.json DIN",
    "DIN+DAX": "results/v12/din_27k.json DIN+DAX",
    "SIM": "results/v12/sim_27k.json SIM",
    "SIM+DAX": "results/v12/sim_27k.json SIM+DAX",
}
DISPLAY = {"LEAF-LGN(stored)": "LEAF-LGN (stored, v10)", "LEAF-LGN": "LEAF-LGN (post-hoc run)",
           "QP family (retrained)": "QP family (retrained)", "QP family+DAX": "QP family+DAX"}
ABL_NAME = {"full": "LEAF-LGN+DAX", "no_tab": "LEAF-LGN+DAX[no_tab]", "no_life_prior": "LEAF-LGN+DAX[no_life_prior]",
            "no_tag_prior": "LEAF-LGN+DAX[no_tag_prior]", "additive": "LEAF-LGN+DAX[additive]",
            "shuffle_tags": "LEAF-LGN+DAX[shuffle_tags]", "shuffle_items": "LEAF-LGN+DAX[shuffle_items]"}
ABL_LABEL = {"full": "full DAX", "no_tab": "no examination factor kappa_t", "no_life_prior": "no lifelong prior on theta_u",
             "no_tag_prior": "no tag prior on alpha_i", "additive": "additive logit (MRP-style), tab effects",
             "shuffle_tags": "alpha shuffled across items", "shuffle_items": "alpha shuffled within tag"}
ABL_ARMS = ["no_tab", "no_life_prior", "no_tag_prior", "additive", "shuffle_tags", "shuffle_items"]
V10_ARMS = ["shuffle_users", "no_tag", "no_author", "no_lifelong", "no_policy", "lift_only", "familiarity_only",
            "no_music"]
V10_LABEL = {"shuffle_users": "shuffle users", "no_tag": "no tag channels", "no_author": "no author channels",
             "no_lifelong": "no lifelong channels", "no_policy": "no policy channel", "lift_only": "lift only",
             "familiarity_only": "familiarity only", "no_music": "no music channels"}


def context():
    """Build (once) the method records, the pooled user count and the helper namespace."""
    global _CTX
    if _CTX is not None:
        return _CTX
    os.chdir(ROOT)
    H = load_helpers()
    mf = H["method_from"]
    M = {
        "LEAF-LGN+DAX": mf(H["POST"], "LEAF-LGN+DAX"),
        "LEAF-LGN": mf(H["POST"], "LEAF-LGN", "LEAF-LGN (posthoc run)"),
        "LEAF-LGN(stored)": mf(H["STORED_LGN"], "LEAF-LGN", "LEAF-LGN (stored)"),
        "QP family": H["qp_stored"](),
        "QP family+DAX": mf(H["QP_DAX"], "QP family+DAX", "QP family + DAX"),
        "QP family (retrained)": mf(H["QP_DAX"], "QP family", "QP family (retrained)"),
        "r-AdjNorm": mf(H["STORED_BASE"], "r-AdjNorm"),
        "LightGCN": mf(H["STORED_BASE"], "LightGCN"),
        "LightGCN+DAX": mf(H["LGN_GCN"], "LightGCN+DAX", "LightGCN + DAX"),
        "EASE": mf(H["EASE"], "EASE"),
        "EASE+DAX": mf(H["EASE"], "EASE+DAX", "EASE + DAX"),
        "DIN": mf(H["DIN_27K"], "DIN"),
        "DIN+DAX": mf(H["DIN_27K"], "DIN+DAX", "DIN + DAX"),
        "SIM": mf(H["SIM_27K"], "SIM"),
        "SIM+DAX": mf(H["SIM_27K"], "SIM+DAX", "SIM + DAX"),
    }
    N = None
    if M["LEAF-LGN(stored)"] is not None:
        N = len(H["pooled_users"](M["LEAF-LGN(stored)"]))
    # KuaiRand-1K, DIN/SIM 1K, KuaiRand-1K QP family (c1k_methods returns None when a file is missing)
    c1k = H["c1k_methods"]()
    K = {"LEAF-LGN+DAX": None, "QP family": None, "QP family+DAX": None, "DIN": None, "DIN+DAX": None,
         "SIM": None, "SIM+DAX": None}
    if c1k is not None:
        K["LEAF-LGN+DAX"], K["QP family"], K["QP family+DAX"] = c1k
    K["DIN"] = mf(H["DIN_1K"], "DIN", "DIN 1K")
    K["DIN+DAX"] = mf(H["DIN_1K"], "DIN+DAX", "DIN + DAX 1K")
    K["SIM"] = mf(H["SIM_1K"], "SIM", "SIM 1K")
    K["SIM+DAX"] = mf(H["SIM_1K"], "SIM+DAX", "SIM + DAX 1K")
    _CTX = {"H": H, "M": M, "N": N, "K": K}
    return _CTX


def P(a, b):
    """Paired comparison a - b (per-user, the helper of final_v12). None when either side is missing."""
    c = context()
    if a is None or b is None:
        return None
    return c["H"]["pair"](a, b, c["N"])


def P1k(a, b):
    c = context()
    if a is None or b is None:
        return None
    return c["H"]["pair"](a, b, None)


# ---------------------------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------------------------
def esc(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def tbd(src):
    return r"\textnormal{\{TBD:" + esc(src) + r"\}}"


def sci(p):
    """p in the form of the source files: mantissa with two decimals, exponent (x 10^e)."""
    if p == 1.0:
        return "1"
    m, e = f"{p:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def pnb(p):
    """Nadeau-Bengio p with three decimals, as printed in the source files."""
    return f"{p:.3f}"


def dif(r):
    """Paired difference x1e-3 with the 95% half-width, as printed in the source files."""
    if r.get("identical"):
        return r"0.00 [0.00]"
    return f"{r['diff']:+.2f} [{r['half']:.2f}]"


def mean_cell(m):
    c = context()
    mu, sd = c["H"]["mean_sd"](m)
    return f"{mu:.4f}"


def table_env(caption, label, colspec, header, rows, note=""):
    out = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize", r"\caption{" + caption + "}",
           r"\label{" + label + "}", r"\begin{adjustbox}{max width=\linewidth}",
           r"\begin{tabular}{" + colspec + "}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if row == "MIDRULE":
            out.append(r"\midrule")
        elif isinstance(row, str):
            out.append(row)
        else:
            out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        out += [r"\par\smallskip", r"\begin{minipage}{\linewidth}\raggedright\footnotesize " + note +
                r"\end{minipage}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write(name, text):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name), "w") as f:
        f.write(text)


def span(n, text):
    return r"\multicolumn{" + str(n) + "}{l}{" + text + r"} \\"


def summary_line(label, r):
    if r is None:
        return f"  {label}: TBD"
    if r.get("identical"):
        return f"  {label}: identical rankings"
    return (f"  {label}: {r['diff']:+.2f} [{r['half']:.2f}] p={r['p']:.2e} NB p={r['nb']:.3f} "
            f"splits {' '.join(f'{x:+.1f}' for x in r['split_diffs'])}")


# ---------------------------------------------------------------------------------------------
# Table 1: methods (means, spread, search budgets, source)
# ---------------------------------------------------------------------------------------------
def table_methods():
    c = context()
    H, M = c["H"], c["M"]
    rows = []
    for k in ["LEAF-LGN+DAX", "LEAF-LGN", "LEAF-LGN(stored)", "QP family", "QP family+DAX", "QP family (retrained)",
              "r-AdjNorm", "LightGCN", "LightGCN+DAX", "EASE", "EASE+DAX", "DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        m = M.get(k)
        name = DISPLAY.get(k, k)
        if m is None:
            rows.append([esc(name), tbd(SRC[k] + " mean"), tbd(SRC[k] + " sd"), tbd(SRC[k] + " n_configs"),
                         tbd(SRC[k])])
            continue
        mu, sd = H["mean_sd"](m)
        rows.append([esc(name), f"{mu:.4f}", f"{sd:.4f}", f"{m['n_configs']}", esc(m["file"])])
    write("tab_methods.tex", table_env(
        "Methods on KuaiRand-27K (temporal, all users): mean and standard deviation of nDCG@10 over the five splits, "
        "configurations searched per split, and the source file.",
        "tab:methods", "lrrrr", ["Method", "Mean", "SD", "Configurations", "Source"], rows,
        "Configurations: post-hoc runs count backbone trainings times DAX configurations times gammas. "
        "The stored LEAF-LGN counts another unit (not comparable). QP family is the per-split best of the three "
        "stored QP variants on validation."))


# ---------------------------------------------------------------------------------------------
# Table 2: main paired comparisons (Holm family of the main table), with per-user and NB columns
# ---------------------------------------------------------------------------------------------
def main_family(c, keys):
    H, M = c["H"], c["M"]
    ref = M["LEAF-LGN+DAX"]
    res = {k: P(ref, M.get(k)) for k in keys}
    present = [k for k in keys if res[k] is not None]
    ph = dict(zip(present, H["holm"]([res[k]["p"] for k in present])))
    nh = dict(zip(present, H["holm"]([res[k]["nb"] for k in present])))
    return res, ph, nh


def table_main():
    c = context()
    H, M = c["H"], c["M"]
    keys = list(H["FAMILY_V12"])
    res, ph, nh = main_family(c, keys)
    rows = [["LEAF-LGN+DAX (reference)", mean_cell(M["LEAF-LGN+DAX"]) if M["LEAF-LGN+DAX"] else tbd(SRC["LEAF-LGN+DAX"]),
             "n/a", "n/a", "n/a", "n/a", "n/a"], "MIDRULE", span(7, r"Holm family: LEAF-LGN+DAX minus each comparator")]
    missing = []
    for k in keys:
        name = esc(DISPLAY.get(k, k))
        r = res[k]
        if r is None:
            missing.append(k)
            rows.append([name, tbd(SRC[k] + " mean"), tbd(SRC[k] + " paired difference"), tbd(SRC[k]),
                         tbd(SRC[k]), tbd(SRC[k]), tbd(SRC[k])])
            continue
        rows.append([name, mean_cell(M[k]), dif(r), sci(r["p"]), sci(ph[k]), pnb(r["nb"]), sci(nh[k])])
    rows.append("MIDRULE")
    rows.append(span(7, r"Outside the family (raw p values only; not Holm-adjusted)"))
    for k in ["EASE", "EASE+DAX"]:
        r = P(M["LEAF-LGN+DAX"], M.get(k))
        if r is None:
            rows.append([esc(k), tbd(SRC[k] + " mean"), tbd(SRC[k] + " paired difference"), tbd(SRC[k]), "n/a",
                         tbd(SRC[k]), "n/a"])
        else:
            rows.append([esc(k), mean_cell(M[k]), dif(r), sci(r["p"]), "n/a", pnb(r["nb"]), "n/a"])
    note = (f"Per-user columns: paired t test over the pooled test users (n = {c['N']}); Holm within the family, "
            f"over the {len(keys) - len(missing)} comparators present; NB: Nadeau-Bengio corrected resampled t test over "
            f"the five split means (fraction of validation users 0.3); Holm over the same family. "
            f"Missing family members (shown as TBD): {', '.join(missing) if missing else 'none'}. "
            "Differences are per-user means x1e-3 nDCG@10 with the 95\\% half-width.")
    write("tab_main.tex", table_env(
        "Main results on KuaiRand-27K: LEAF-LGN+DAX against every comparator of the main family, with per-user and "
        "split-level (Nadeau-Bengio) tests.",
        "tab:main", "lrccccc",
        ["Comparator", "Mean", r"Diff.\ [95\% hw]", r"$p$", r"$p_{\mathrm{Holm}}$", r"NB $p$",
         r"NB $p_{\mathrm{Holm}}$"], rows, note))
    # sensitivity: the same comparisons with EASE and EASE+DAX inside the family
    keys9 = keys + ["EASE", "EASE+DAX"]
    res9, ph9, nh9 = main_family(c, keys9)
    present9 = [k for k in keys9 if res9[k] is not None]
    rows9 = []
    for k in keys9:
        if res9[k] is None:
            rows9.append([esc(DISPLAY.get(k, k))] + [tbd(SRC[k] + " paired test")] * 4)
            continue
        rows9.append([esc(DISPLAY.get(k, k)), sci(res9[k]["p"]), sci(ph9[k]), pnb(res9[k]["nb"]), sci(nh9[k])])
    write("tab_sens.tex", table_env(
        "Sensitivity of the Holm adjustment: the main family extended by EASE and EASE+DAX (the family size is given "
        "in the column header and counts the comparators whose files exist).",
        "tab:sens", "lrccc", ["Comparator", r"$p$", r"$p_{\mathrm{Holm}}$ (family of $%d$)" % len(present9),
                              r"NB $p$", r"NB $p_{\mathrm{Holm}}$"], rows9,
        "The Holm values differ from Table~\\ref{tab:main} only through the family size."))
    c["fam_missing"] = missing
    return res, ph, nh


# ---------------------------------------------------------------------------------------------
# Table 3: fair-baseline reading and DAX backbone comparisons
# ---------------------------------------------------------------------------------------------
def table_fair():
    c = context()
    M = c["M"]
    rows_def = [
        ("LEAF-LGN+DAX minus QP family+DAX (fair baseline)", "LEAF-LGN+DAX", "QP family+DAX", "main family"),
        ("QP family+DAX minus QP family (DAX on the QP backbone)", "QP family+DAX", "QP family", "raw"),
        ("LEAF-LGN+DAX minus LEAF-LGN (DAX on the LEAF backbone)", "LEAF-LGN+DAX", "LEAF-LGN", "main family"),
        ("LEAF-LGN (stored, v10) minus QP family", "LEAF-LGN(stored)", "QP family", "raw"),
    ]
    rows = []
    for label, a, b, kind in rows_def:
        r = P(M.get(a), M.get(b))
        ca = f"{M[a]['n_configs']}" if M.get(a) else tbd(SRC[a] + " n_configs")
        cb = f"{M[b]['n_configs']}" if M.get(b) else tbd(SRC[b] + " n_configs")
        if r is None:
            rows.append([esc(label), ca + " / " + cb, tbd(f"{SRC[a]} vs {SRC[b]} paired difference"), "", "", ""])
            continue
        rows.append([esc(label), f"{ca} / {cb}", dif(r), sci(r["p"]), pnb(r["nb"]), kind])
    write("tab_fair.tex", table_env(
        "Fair-baseline reading: the DAX increments and the stored LEAF-LGN against the QP family, with their search "
        "budgets (configurations of the first and second method).",
        "tab:fair", "lccccl", ["Comparison", "Configurations", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$", "Test family"],
        rows,
        "Rows marked raw are not Holm-adjusted: the stored v10 comparison is part of a different family (eight "
        "comparisons in the stage 1 run), and the DAX increments are reported beside the main family. The QP family "
        "+ DAX search uses only the QP backbone configurations selected without DAX."))


def table_backbone():
    c = context()
    M = c["M"]
    rows = []
    defs = [("EASE", "EASE", "EASE+DAX"), ("LightGCN", "LightGCN", "LightGCN+DAX"),
            ("QP family (stored)", "QP family", "QP family+DAX"), ("LEAF-LGN (post-hoc run)", "LEAF-LGN", "LEAF-LGN+DAX"),
            ("DIN", "DIN", "DIN+DAX")]
    for label, a, b in defs:
        r = P(M.get(b), M.get(a))
        base = mean_cell(M[a]) if M.get(a) else tbd(SRC[a] + " mean")
        withd = mean_cell(M[b]) if M.get(b) else tbd(SRC[b] + " mean")
        if r is None:
            rows.append([esc(label), base, withd, tbd(f"{SRC[b]} minus {SRC[a]} paired difference"), "", ""])
            continue
        rows.append([esc(label), base, withd, dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append(["SIM", tbd(SRC["SIM"] + " mean"), tbd(SRC["SIM+DAX"] + " mean"),
                 tbd("results/v12/sim_27k.json SIM+DAX minus SIM paired difference"), "", ""])
    write("tab_backbone.tex", table_env(
        "DAX is backbone-agnostic: nDCG@10 without and with the post-hoc DAX module on KuaiRand-27K, paired difference "
        "x1e-3 with the 95\\% half-width, two-sided t test and the Nadeau-Bengio test over the split means.",
        "tab:backbone", "lrcccc", ["Backbone", "Without DAX", "With DAX", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$"],
        rows, "Raw p values (no family). The QP row uses the stored QP family as the reference, as in the fair-baseline "
        "reading."))


# ---------------------------------------------------------------------------------------------
# Table 4: DAX ablations, shuffle controls (gain vs the base of the ablation run)
# ---------------------------------------------------------------------------------------------
def ablation_records():
    c = context()
    H = c["H"]
    mf = H["method_from"]
    base = mf(H["ABLATE"], "LEAF-LGN", "LEAF-LGN (ablation run)")
    arms = {k: mf(H["ABLATE"], ABL_NAME[k], ABL_NAME[k]) for k in ["full"] + ABL_ARMS}
    return base, arms


def table_ablate():
    c = context()
    H = c["H"]
    base, arms = ablation_records()
    N = c["N"]
    if base is None or arms["full"] is None:
        write("tab_ablate.tex", table_env("DAX ablations: not available.", "tab:ablate", "l",
                                          ["Arm"], [[tbd("results/v11/dax_ablate.json full arm and base")]]))
        return
    gain = {k: P(arms[k], base) for k in arms if arms[k] is not None}
    keys = [k for k in gain]
    ph_g = dict(zip(keys, H["holm"]([gain[k]["p"] for k in keys])))
    fm = [k for k in ABL_ARMS if arms.get(k) is not None]
    minus = {k: P(arms["full"], arms[k]) for k in fm}
    ph_m = dict(zip(fm, H["holm"]([minus[k]["p"] for k in fm]))) if fm else {}
    g_full = gain["full"]["diff"]
    rows = []
    for k in ["full"] + ABL_ARMS:
        if arms.get(k) is None:
            rows.append([esc(ABL_LABEL[k]), tbd(f"results/v11/dax_ablate.json {ABL_NAME[k]} mean"), "", "", "", "", "", ""])
            continue
        g = gain[k]
        kept = f"{100 * g['diff'] / g_full:.1f}" if k != "full" else "100.0"
        if k == "full":
            mi = "n/a"
            pm = "n/a"
        else:
            mi = dif(minus[k])
            pm = sci(ph_m[k])
        rows.append([esc(ABL_LABEL[k]), f"{arms[k]['n_configs']}", mean_cell(arms[k]), dif(g), sci(ph_g[k]),
                     kept, mi, pm])
    rows.append("MIDRULE")
    rows.append(span(8, r"Permutation seeds (pooled arm above selects over seeds; separate runs below)"))
    for k in ["shuffle_tags", "shuffle_items"]:
        for s in (0, 1, 2):
            name = f"LEAF-LGN+DAX[{k}|p{s}]"
            r = P(H["method_from"](H["ABLATE"], name, name), base)
            if r is None:
                rows.append([esc(f"{ABL_LABEL[k]}, seed {s}"), "", tbd(f"results/v11/dax_ablate.json {name} gain"), "", "",
                             "", "", ""])
            else:
                kept = f"{100 * r['diff'] / g_full:.1f}"
                rows.append([esc(f"{ABL_LABEL[k]}, seed {s}"), "", "", dif(r), f"NB {pnb(r['nb'])}", kept, "", ""])
    note = (f"Gain: arm minus the LEAF-LGN base of the ablation run (mean {H['mean_sd'](base)[0]:.4f}, 2 backbone trainings, "
            "five gammas, epoch window of 3 around the main selection). Full DAX reproduces the main run's splits. "
            "Holm within the seven gains (column p Holm); the column of full minus arm is Holm-adjusted within the six "
            "ablated arms. Kept: the arm's gain as a percentage of the full gain. Pooled shuffle entries select the permutation "
            "seed on validation.")
    write("tab_ablate.tex", table_env(
        "DAX ablations on KuaiRand-27K (post-hoc): gain over the backbone of the ablation run, the difference to full DAX, "
        "and the shuffle controls.",
        "tab:ablate", "lrcccccc",
        ["Arm", "Configs", "Mean", r"Gain [95\% hw]", r"$p_{\mathrm{Holm}}$", "Kept (\\%)", r"Full minus arm",
         r"$p_{\mathrm{Holm}}$"], rows, note))
    return gain, minus


# ---------------------------------------------------------------------------------------------
# Table 5: KuaiRand-1K replication
# ---------------------------------------------------------------------------------------------
def table_1k():
    c = context()
    H, K = c["H"], c["K"]
    lg, qp, qd = K["LEAF-LGN+DAX"], K["QP family"], K["QP family+DAX"]
    rows = []
    for label, m, src in [("LEAF-LGN+DAX", lg, "results/v11/c_1k_lgn.json LEAF-LGN+DAX"),
                          ("QP family", qp, "results/v11/c_1k_base.json QP methods"),
                          ("QP family+DAX", qd, "results/v11/c_1k_qp.json QP family+DAX")]:
        if m is None:
            rows.append([esc(label), tbd(src + " mean"), "", "", "", "", "", "", ""])
        else:
            mu, sd = H["mean_sd"](m)
            rows.append([esc(label), f"{mu:.4f}", f"{sd:.4f}", f"{m['n_configs']}", "", "", "", "", ""])
    cmp_defs = [("LEAF-LGN+DAX minus QP family", lg, qp), ("LEAF-LGN+DAX minus QP family+DAX", lg, qd),
                ("QP family+DAX minus QP family", qd, qp)]
    res = [P1k(a, b) for _, a, b in cmp_defs]
    present = [r for r in res if r is not None]
    ph = H["holm"]([r["p"] for r in present]) if present else []
    nh = H["holm"]([r["nb"] for r in present]) if present else []
    rows.append("MIDRULE")
    j = 0
    split_notes = []
    for (label, _, _), r in zip(cmp_defs, res):
        if r is None:
            rows.append([esc(label), "", "", "", tbd("results/v11 KuaiRand-1K paired difference"), "", "", "", ""])
            continue
        rows.append([esc(label), "", "", "", dif(r), sci(r["p"]), sci(ph[j]), pnb(r["nb"]), sci(nh[j])])
        split_notes.append(f"{label}: " + " ".join(f"{x:+.1f}" for x in r["split_diffs"]))
        j += 1
    n1k = len(H["pooled_users"](lg)) if lg is not None else None
    n_txt = str(n1k) if n1k is not None else tbd("results/v11/c_1k_lgn.json users")
    cfg_qp = qd["n_configs"] if qd is not None else None
    note = (f"KuaiRand-1K: {n_txt} test users pooled over five splits "
            "(each split 30\\% validation, 70\\% test); 1K users are a subset of the 27K users, a lower-power "
            "replication. The 1K QP family + DAX arm searches all 20 QP configurations ("
            f"{cfg_qp} configurations in total) whereas the 27K arm searches the 10 selected ones (3200); this favours "
            "the baseline and is stated as such. Holm over the three comparisons of this table; NB over the split "
            "means. Split differences (x1e-3): " + "; ".join(split_notes) + ".")
    write("tab_1k.tex", table_env(
        "KuaiRand-1K replication: LEAF-LGN+DAX against the QP family with and without DAX (nDCG@10).",
        "tab:1k", "lrrrccccc",
        ["Method or comparison", "Mean", "SD", "Configs", r"Diff.\ [95\% hw]", r"$p$", r"$p_{\mathrm{Holm}}$", r"NB $p$",
         r"NB $p_{\mathrm{Holm}}$"], rows, note))
    return split_notes


# ---------------------------------------------------------------------------------------------
# Table 6: seeds (training seeds 1-3 for LEAF-LGN+DAX; QP seeds not run)
# ---------------------------------------------------------------------------------------------
def table_seeds():
    c = context()
    H, M, N = c["H"], c["M"], c["N"]
    rows = []
    gains, means = [], []
    entries = [(0, M["LEAF-LGN+DAX"], M["LEAF-LGN"], "results/v11/dax_lgn_post.json")]
    for s in (1, 2, 3):
        path = f"results/v11/c_seeds/lgn_s{s}.json"
        entries.append((s, H["method_from"](path, "LEAF-LGN+DAX", f"seed {s} LEAF-LGN+DAX"),
                        H["method_from"](path, "LEAF-LGN", f"seed {s} LEAF-LGN"), path))
    for s, dax, base, path in entries:
        if dax is None or base is None:
            rows.append([f"{s}", tbd(f"{path} LEAF-LGN+DAX mean"), "", "", "", "", ""])
            continue
        r = H["pair"](dax, base, N)
        mu = H["mean_sd"](dax)[0]
        gains.append(r["diff"])
        means.append(mu)
        rows.append([f"{s}" + (" (main run)" if s == 0 else ""), f"{mu:.4f}", f"{H['mean_sd'](base)[0]:.4f}",
                     f"{dax['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    if len(means) > 1:
        rows.append([r"Spread over seeds", f"{np.mean(means):.4f} $\\pm$ {np.std(means, ddof=1):.4f}", "", "",
                     f"{np.mean(gains):+.2f} $\\pm$ {np.std(gains, ddof=1):.2f}", "", ""])
    rows.append([r"QP family (seeds not run)", tbd("results/v11/c_seeds QP seeds 1-3: qp_s1.json.partial.json incomplete"),
                 "", "", "", "", ""])
    write("tab_seeds.tex", table_env(
        "Training seeds on KuaiRand-27K: LEAF-LGN+DAX and its within-seed gain over LEAF-LGN (same seed), seeds 0 to 3.",
        "tab:seeds", "lrrrccc",
        ["Seed", "LEAF-LGN+DAX", "LEAF-LGN", "Configs", r"Gain [95\% hw]", r"$p$", r"NB $p$"], rows,
        "Seed 0 is the main run with the full backbone grid (12 trainings). Seeds 1 to 3 re-train with the training seed "
        "changed and a reduced backbone grid: lr 3e-3 and wd 1e-4 in every split, w in {0.25, 0.5} for the DAX arm (2 "
        "trainings) and w 0.25 for the base. The epoch, the DAX configuration and gamma are selected on validation per "
        "split. Raw p values, not Holm-adjusted. The QP family was not re-trained under seeds 1 to 3, so no seed "
        "comparison with it is reported."))


# ---------------------------------------------------------------------------------------------
# Table 7: lifelong deep baselines (DIN, SIM) at 1K and 27K
# ---------------------------------------------------------------------------------------------
def table_deep():
    c = context()
    H, M, K = c["H"], c["M"], c["K"]
    rows = [span(6, r"KuaiRand-27K")]
    for k in ["DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        m = M.get(k)
        r = P(M["LEAF-LGN+DAX"], m)
        if m is None:
            rows.append([esc(k), tbd(SRC[k] + " mean"), tbd(SRC[k] + " configs"), tbd(SRC[k] + " difference"),
                         "", ""])
        else:
            rows.append([esc(k), mean_cell(m), f"{m['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    rows.append(span(6, r"KuaiRand-1K"))
    lg = K["LEAF-LGN+DAX"]
    for k in ["DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        m = K.get(k)
        if m is None or lg is None:
            rows.append([esc(k) + " (1K)", tbd(f"results/v12 1K {k} mean"), "", tbd(f"1K {k} difference"), "", ""])
            continue
        r = P1k(lg, m)
        rows.append([esc(k), mean_cell(m), f"{m['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    rows.append(span(6, r"Lifelong sequence models: DIN attends over the last 200 lifelong events; SIM-hard keeps the 50 "
                        r"most recent events sharing the candidate's tag or author."))
    write("tab_deep.tex", table_env(
        "Lifelong deep baselines (DIN and SIM-hard) with and without DAX, and their paired difference to LEAF-LGN+DAX "
        "(LEAF-LGN+DAX minus the comparator, x1e-3 nDCG@10).",
        "tab:deep", "lrccccc",
        ["Comparator", "Mean", "Configs", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$"], rows,
        "Raw p values. DIN and SIM are tuned over four configurations (lr {1e-2, 3e-3} times wd {1e-4, 1e-5}) with "
        "patience 2; their +DAX arms use the post-hoc DAX grid. Entries marked TBD have no result file yet."))


# ---------------------------------------------------------------------------------------------
# Table 8: LEAF controls (v10, LEAF-tied; stored run, the LEAF-LGN ablations are not in a file)
# ---------------------------------------------------------------------------------------------
def table_leafctl():
    c = context()
    H, N = c["H"], c["N"]
    full = H["method_from"]("results/v10/leaf_tied.json", "LEAF-tied", "LEAF-tied")
    arms = {k: H["method_from"]("results/v10/leaf_tied_ablate.json", f"LEAF-tied[{k}]", k) for k in V10_ARMS}
    rows = []
    if full is None:
        write("tab_leafctl.tex", table_env("LEAF controls: not available.", "tab:leafctl", "l", ["Arm"],
                                          [[tbd("results/v10/leaf_tied.json LEAF-tied")]]))
        return
    fm = [k for k in V10_ARMS if arms[k] is not None]
    minus = {k: H["pair"](full, arms[k], N) for k in fm}
    ph = dict(zip(fm, H["holm"]([minus[k]["p"] for k in fm])))
    nh = dict(zip(fm, H["holm"]([minus[k]["nb"] for k in fm])))
    rows.append([r"LEAF-tied (full)", f"{full['n_configs']}", mean_cell(full), "n/a", "n/a", "n/a", "n/a"])
    rows.append("MIDRULE")
    for k in V10_ARMS:
        if arms[k] is None:
            rows.append([esc(V10_LABEL[k]), tbd(f"results/v10/leaf_tied_ablate.json LEAF-tied[{k}]"), "", "", "", "", ""])
            continue
        r = minus[k]
        rows.append([esc(V10_LABEL[k]), f"{arms[k]['n_configs']}", mean_cell(arms[k]), dif(r), sci(ph[k]), pnb(r["nb"]),
                     sci(nh[k])])
    note = ("Full minus arm: positive means the channel helps. Holm within the eight arms; NB over the split means. "
            "The no-music arm gives identical rankings in every split. The LEAF-LGN ablations are not run "
            "(results/v10/leaf\\_lgn\\_ablate.json is missing); these controls use the closed-form LEAF-tied backbone. "
            "The shuffle-users control removes the lifelong gain (design check on the difference between the shuffled "
            "and the no-lifelong arm).")
    write("tab_leafctl.tex", table_env(
        "LEAF controls on KuaiRand-27K (stage 1 stored run, LEAF-tied backbone): arm means and full minus arm.",
        "tab:leafctl", "lrccccc",
        ["Arm", "Configs", "Mean", r"Full minus arm [95\% hw]", r"$p_{\mathrm{Holm}}$", r"NB $p$", r"NB $p_{\mathrm{Holm}}$"],
        rows, note))
    return minus


# ---------------------------------------------------------------------------------------------
# Table 9: permutation seeds and KuaiRand-1K are above; appendix C: chosen configurations per split
# ---------------------------------------------------------------------------------------------
CFG_ORDER = ["method", "lr", "wd", "dim", "layers", "r", "fn", "pbeta", "w", "epoch", "gamma"]
DAX_ORDER = ["prior", "centre", "k_life", "tab", "prior_s"]


def fmt_cfg(ch):
    parts = []
    for k in CFG_ORDER:
        if k in ch:
            v = ch[k]
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    if "dax" in ch and ch["dax"] is not None:
        d = ch["dax"]
        for k in DAX_ORDER:
            if k in d:
                v = d[k]
                parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    for k in ch:
        if k not in CFG_ORDER and k not in ("dax",) and k not in DAX_ORDER:
            parts.append(f"{k}={ch[k]}")
    return ", ".join(parts)


def table_grids():
    c = context()
    H, M, K = c["H"], c["M"], c["K"]
    blocks = []
    specs = [("LEAF-LGN+DAX (main, post-hoc)", M["LEAF-LGN+DAX"]),
             ("LEAF-LGN (post-hoc base)", M["LEAF-LGN"]),
             ("QP family (stored, per-split best variant)", M["QP family"]),
             ("QP family+DAX", M["QP family+DAX"]),
             ("EASE", M["EASE"]), ("EASE+DAX", M["EASE+DAX"]),
             ("LightGCN", M["LightGCN"]), ("r-AdjNorm", M["r-AdjNorm"]),
             ("DIN (27K)", M["DIN"]), ("DIN+DAX (27K)", M["DIN+DAX"]),
             ("SIM (27K)", M["SIM"]), ("SIM+DAX (27K)", M["SIM+DAX"]),
             ("LEAF-LGN+DAX (1K)", K["LEAF-LGN+DAX"]), ("QP family+DAX (1K)", K["QP family+DAX"]),
             ("DIN (1K)", K["DIN"]), ("DIN+DAX (1K)", K["DIN+DAX"]), ("SIM (1K)", K["SIM"]), ("SIM+DAX (1K)", K["SIM+DAX"])]
    index = []
    for i, (label, m) in enumerate(specs):
        name = f"tab_grid_{i:02d}.tex"
        if m is None:
            text = table_env(f"Chosen configurations, {esc(label)}: not available.", f"tab:grid{i:02d}", "l", ["Split"],
                             [[tbd(f"{label} chosen configurations")]])
        else:
            rows = []
            for s, ch in enumerate(m["chosen"]):
                rows.append([f"{s + 1}", r"\texttt{" + esc(fmt_cfg(ch)) + "}"])
            text = table_env(f"Chosen configurations per split: {esc(label)} (nDCG@10 mean {H['mean_sd'](m)[0]:.4f}).",
                             f"tab:grid{i:02d}", "rp{0.82\\linewidth}", ["Split", "Chosen configuration"], rows)
        write(name, text)
        index.append((name, label))
    return index


def main():
    c = context()
    H = c["H"]
    table_methods()
    res, ph, nh = table_main()
    table_fair()
    table_backbone()
    gain_minus = table_ablate()
    table_1k()
    table_seeds()
    table_deep()
    minus = table_leafctl()
    grids = table_grids()
    print("TABLES written to", TABLES)
    print("Holm family (main, present):", [k for k in H["FAMILY_V12"] if c["M"].get(k) is not None])
    print("Missing family members:", c.get("fam_missing"))
    print("Main comparisons (diff [hw], p, p_Holm, NB, NB p_Holm):")
    for k in H["FAMILY_V12"]:
        r = res.get(k)
        print(summary_line(f"LEAF-LGN+DAX - {k}", r), "" if r is None else
              f"p_H={ph[k]:.2e} nb_H={nh[k]:.3e}")
    if gain_minus:
        print("Ablation full minus arm:")
        for k, r in gain_minus[1].items():
            print(summary_line(f"full - {k}", r))
    print("Notes from the helper namespace:")
    for n in H["notes"]:
        print("  -", n)
    print("Grid tables:", len(grids))


if __name__ == "__main__":
    main()
