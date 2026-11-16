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
DISPLAY = {"LEAF-LGN(stored)": "LEAF-LGN (independent tuning)", "LEAF-LGN": "LEAF-LGN",
           "QP family (retrained)": "QP family, re-trained on its selected configurations", "QP family+DAX": "QP family+DAX"}
ABL_NAME = {"full": "LEAF-LGN+DAX", "no_tab": "LEAF-LGN+DAX[no_tab]", "no_life_prior": "LEAF-LGN+DAX[no_life_prior]",
            "no_tag_prior": "LEAF-LGN+DAX[no_tag_prior]", "additive": "LEAF-LGN+DAX[additive]",
            "shuffle_tags": "LEAF-LGN+DAX[shuffle_tags]", "shuffle_items": "LEAF-LGN+DAX[shuffle_items]"}
ABL_LABEL = {"full": "full DAX", "no_tab": r"without scenario factors $\kappa_t$",
             "no_life_prior": r"without lifelong prior on $\theta_u$", "no_tag_prior": r"without tag prior on $\alpha_i$",
             "additive": "additive logit model with user and scenario effects",
             "shuffle_tags": r"$\alpha$ permuted across items", "shuffle_items": r"$\alpha$ permuted within tags"}
ABL_ARMS = ["no_tab", "no_life_prior", "no_tag_prior", "additive", "shuffle_tags", "shuffle_items"]
V10_ARMS = ["shuffle_users", "no_tag", "no_author", "no_lifelong", "no_policy", "lift_only", "familiarity_only",
            "no_music"]
V10_LABEL = {"shuffle_users": "users' lifelong statistics shuffled", "no_tag": "without tag channels",
             "no_author": "without author channels", "no_lifelong": "without lifelong channels",
             "no_policy": "without policy channel", "lift_only": "lift only", "familiarity_only": "familiarity only",
             "no_music": "without music channels"}


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
            rows.append([esc(name), tbd(SRC[k] + " mean"), tbd(SRC[k] + " sd"), tbd(SRC[k] + " configurations")])
            continue
        mu, sd = H["mean_sd"](m)
        rows.append([esc(name), f"{mu:.4f}", f"{sd:.4f}", f"{m['n_configs']:,}"])
    write("tab_methods.tex", table_env(
        "Methods on KuaiRand-27K (temporal protocol, all users): mean and standard deviation of nDCG@10 over the five "
        "splits, and the configurations searched per split.",
        "tab:methods", "lrrr", ["Method", "Mean", "SD", "Configurations"], rows,
        "Configurations searched: backbone trainings times DAX configurations times values of $\\gamma$. "
        "LEAF-LGN (independent tuning) is a separate search of the backbone alone, counted per training and epoch "
        "setting, and is not comparable. The QP family is, in each split, the best of its three variants on validation. "
        "The diagnostic value of Section~\\ref{sec:res-headroom} is a different computation."))


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
    note = (f"Per-user columns: paired t test over the pooled test users (n = {c['N']:,}). Holm adjustment within the "
            f"family, over the {len(keys) - len(missing)} comparators present. NB: Nadeau-Bengio corrected resampled t test "
            "over the five split means, with Holm over the same family. LightGCN is the four-configuration LightGCN "
            "baseline (a LightGCN re-trained with DAX appears in Table~\\ref{tab:backbone}). SIM denotes SIM-hard. "
            "Differences are per-user means of nDCG@10 in units of $10^{-3}$, with the 95\\% half-width.")
    if missing:
        note += " Not available: " + ", ".join(missing) + "."
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
        "in the column header and counts the comparators in the family).",
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
        ("LEAF-LGN (independent tuning) minus QP family", "LEAF-LGN(stored)", "QP family", "raw"),
    ]
    rows = []
    for label, a, b, kind in rows_def:
        r = P(M.get(a), M.get(b))
        ca = f"{M[a]['n_configs']:,}" if M.get(a) else tbd(SRC[a] + " configurations")
        cb = f"{M[b]['n_configs']:,}" if M.get(b) else tbd(SRC[b] + " configurations")
        if r is None:
            rows.append([esc(label), ca + " / " + cb, tbd(f"{SRC[a]} vs {SRC[b]} paired difference"), "", "", ""])
            continue
        rows.append([esc(label), f"{ca} / {cb}", dif(r), sci(r["p"]), pnb(r["nb"]), kind])
    write("tab_fair.tex", table_env(
        "Fair-baseline reading: the DAX increments, and LEAF-LGN without DAX against the QP family, with the "
        "configurations searched by each method (first and second method).",
        "tab:fair", "lccccl", ["Comparison", "Configurations", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$", "Test family"],
        rows,
        "Rows marked raw are not Holm-adjusted. The LEAF-LGN (independent tuning) comparison belongs to the LEAF "
        "development family of eight comparisons, and the DAX increments are reported beside the main family. "
        "The QP family with DAX searches only the QP backbone configurations selected without DAX."))


def table_backbone():
    c = context()
    H, M = c["H"], c["M"]
    # LightGCN: the re-trained base of the run that also fits LightGCN+DAX (3 configurations)
    lgn_ret = H["method_from"](H["LGN_GCN"], "LightGCN", "LightGCN (re-trained)")
    rows = []
    defs = [("EASE", M.get("EASE"), M.get("EASE+DAX")),
            ("LightGCN (re-trained)", lgn_ret, M.get("LightGCN+DAX")),
            ("QP family", M.get("QP family"), M.get("QP family+DAX")),
            ("LEAF-LGN", M.get("LEAF-LGN"), M.get("LEAF-LGN+DAX")),
            ("DIN", M.get("DIN"), M.get("DIN+DAX")),
            ("SIM", M.get("SIM"), M.get("SIM+DAX"))]
    for label, a, b in defs:
        r = P(b, a)
        base = mean_cell(a) if a else tbd(label + " base mean")
        withd = mean_cell(b) if b else tbd(label + " with DAX mean")
        if r is None:
            rows.append([esc(label), base, withd, tbd(label + " paired difference"), "", ""])
            continue
        rows.append([esc(label), base, withd, dif(r), sci(r["p"]), pnb(r["nb"])])
    stored = P(M["LightGCN+DAX"], M["LightGCN"])
    write("tab_backbone.tex", table_env(
        "DAX on six backbones on KuaiRand-27K: nDCG@10 without and with the DAX module, paired difference in units of "
        "$10^{-3}$ with the 95\\% half-width, two-sided t test and Nadeau-Bengio test over the split means. SIM denotes SIM-hard.",
        "tab:backbone", "lrcccc", ["Backbone", "Without DAX", "With DAX", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$"],
        rows, "Raw p values (no family). LightGCN: the base and the DAX variant come from the same experiment "
        "(3 base configurations; 960 with DAX). "
        f"Against the four-configuration LightGCN of Table~\\ref{{tab:main}} the increment is {stored['diff']:+.2f} "
        f"[{stored['half']:.2f}] (NB $p$ {stored['nb']:.3f})."))
    return {"lgn_retrained": lgn_ret, "lgn_stored_increment": stored}


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
                                          ["Variant"], [[tbd("results/v11/dax_ablate.json full DAX and base")]]))
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
            rows.append([ABL_LABEL[k], tbd(f"results/v11/dax_ablate.json {ABL_NAME[k]} mean"), "", "", "", "", "", ""])
            continue
        g = gain[k]
        kept = f"{100 * g['diff'] / g_full:.1f}" if k != "full" else "100.0"
        if k == "full":
            mi = "n/a"
            pm = "n/a"
        else:
            mi = dif(minus[k])
            pm = sci(ph_m[k])
        rows.append([ABL_LABEL[k], f"{arms[k]['n_configs']}", mean_cell(arms[k]), dif(g), sci(ph_g[k]),
                     kept, mi, pm])
    rows.append("MIDRULE")
    rows.append(span(8, r"Separate permutation seeds (the variants above select over the three seeds)"))
    for k in ["shuffle_tags", "shuffle_items"]:
        for s in (0, 1, 2):
            name = f"LEAF-LGN+DAX[{k}|p{s}]"
            r = P(H["method_from"](H["ABLATE"], name, name), base)
            if r is None:
                rows.append([f"{ABL_LABEL[k]}, seed {s}", "", tbd(f"results/v11/dax_ablate.json {name} gain"), "", "",
                             "", "", ""])
            else:
                kept = f"{100 * r['diff'] / g_full:.1f}"
                rows.append([f"{ABL_LABEL[k]}, seed {s}", "", "", dif(r), f"NB {pnb(r['nb'])}", kept, "", ""])
    lift = gain["full"]["diff"] - P(c["M"]["LEAF-LGN+DAX"], c["M"]["LEAF-LGN"])["diff"]
    note = (f"Gain: variant minus the LEAF-LGN base of the ablation experiment (mean {H['mean_sd'](base)[0]:.4f}, two "
            "backbone trainings, five values of $\\gamma$, epoch window of three around the selection of the main "
            "experiment). This base is lower than the LEAF-LGN of Table~\\ref{tab:main} (0.4736), so the full gain here ("
            f"{gain['full']['diff']:+.2f}) exceeds the main increment by {lift:.2f}; the variants are compared through the "
            "full-minus-variant columns, which do not use the base. Holm: the p values of the gains are adjusted within "
            "the seven gains, and those of full minus variant within the six ablated variants. Kept: the gain as a "
            "percentage of the full gain of this experiment, so the percentages inherit the base and are approximate. "
            "Permutation seeds are separate runs, one per seed.")
    write("tab_ablate.tex", table_env(
        "DAX ablations on KuaiRand-27K: gain over the LEAF-LGN base of the ablation experiment, the difference to full "
        "DAX, and the permutation controls.",
        "tab:ablate", "lrcccccc",
        ["Variant", "Configs", "Mean", r"Gain [95\% hw]", r"$p_{\mathrm{Holm}}$", r"Kept (\%)", r"Full minus variant",
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
    cfg_txt = f"{cfg_qp:,}" if cfg_qp else "n/a"
    note = (f"KuaiRand-1K: {n_txt} test users pooled over five splits (each split 30\\% validation, 70\\% test). The users "
            "are a subset of the 27K users, so this is a lower-power replication. The QP family with DAX searches all 20 "
            f"QP configurations on this release ({cfg_txt} configurations in total), whereas at 27K it searches the 10 "
            "selected ones (3,200); this favours the baseline. Holm over the three comparisons of this table; NB over "
            "the split means. Split differences ($\\times10^{-3}$): " + "; ".join(split_notes) + ".")
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
        rows.append([f"{s}" + (" (main experiment)" if s == 0 else ""), f"{mu:.4f}", f"{H['mean_sd'](base)[0]:.4f}",
                     f"{dax['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    if len(means) > 1:
        rows.append([r"Spread over seeds", f"{np.mean(means):.4f} $\\pm$ {np.std(means, ddof=1):.4f}", "", "",
                     f"{np.mean(gains):+.2f} $\\pm$ {np.std(gains, ddof=1):.2f}", "", ""])
    write("tab_seeds.tex", table_env(
        "Training seeds on KuaiRand-27K: LEAF-LGN+DAX and its within-seed gain over LEAF-LGN, for seeds 0 to 3.",
        "tab:seeds", "lrrrccc",
        ["Seed", "LEAF-LGN+DAX", "LEAF-LGN", "Configs", r"Gain [95\% hw]", r"$p$", r"NB $p$"], rows,
        "Seed 0 is the main experiment with the full backbone search (12 trainings). Seeds 1 to 3 re-train with the "
        "training seed changed and a reduced backbone search: $\\mathrm{lr}=3\\times10^{-3}$ and $\\rho=10^{-4}$ in every "
        "split, $w\\in\\{0.25,0.5\\}$ for the DAX variant (two trainings) and $w=0.25$ for the base. The epoch, the DAX "
        "configuration and $\\gamma$ are selected on validation in each split. Raw p values, not Holm-adjusted. The QP "
        "family was not re-trained under seeds 1 to 3, so no seed comparison with it is reported."))


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
            rows.append([esc(k), tbd(SRC[k] + " mean"), tbd(SRC[k] + " configurations"), tbd(SRC[k] + " difference"),
                         "", ""])
        else:
            rows.append([esc(k), mean_cell(m), f"{m['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    rows.append(span(6, r"KuaiRand-1K"))
    lg = K["LEAF-LGN+DAX"]
    for k in ["DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        m = K.get(k)
        if m is None or lg is None:
            rows.append([esc(k), tbd(f"results/v12 1K {k} mean"), "", tbd(f"1K {k} difference"), "", ""])
            continue
        r = P1k(lg, m)
        rows.append([esc(k), mean_cell(m), f"{m['n_configs']}", dif(r), sci(r["p"]), pnb(r["nb"])])
    rows.append("MIDRULE")
    rows.append(span(6, r"Lifelong sequence models: DIN attends over the last 200 lifelong events; SIM-hard keeps the 50 "
                        r"most recent events sharing the candidate's tag or author."))
    write("tab_deep.tex", table_env(
        "Lifelong deep baselines (DIN and SIM-hard) with and without DAX on KuaiRand-27K and KuaiRand-1K, with the "
        "paired difference of LEAF-LGN+DAX minus each comparator (in units of $10^{-3}$ nDCG@10). SIM denotes SIM-hard.",
        "tab:deep", "lrccccc",
        ["Comparator", "Mean", "Configs", r"Diff.\ [95\% hw]", r"$p$", r"NB $p$"], rows,
        "Raw p values. DIN and SIM are each tuned over four configurations (learning rate $\\{10^{-2},3\\times10^{-3}\\}$ "
        "times weight decay $\\{10^{-4},10^{-5}\\}$), and their +DAX variants use the DAX grid. The 1K rows use the 842 "
        "KuaiRand-1K test users."))


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
        write("tab_leafctl.tex", table_env("LEAF controls: not available.", "tab:leafctl", "l", ["Variant"],
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
    note = ("Full minus variant: positive means the channel helps. Holm within the eight variants; NB over the split "
            "means. The variant without music channels gives identical rankings in every split. These controls were "
            "run on the closed-form LEAF-tied backbone.")
    write("tab_leafctl.tex", table_env(
        "LEAF controls on KuaiRand-27K (closed-form LEAF-tied backbone): variant means and full minus variant.",
        "tab:leafctl", "lrccccc",
        ["Variant", "Configs", "Mean", r"Full minus variant [95\% hw]", r"$p_{\mathrm{Holm}}$", r"NB $p$",
         r"NB $p_{\mathrm{Holm}}$"], rows, note))
    return minus


# ---------------------------------------------------------------------------------------------
# Table 9: chosen configurations per split (appendix C)
# ---------------------------------------------------------------------------------------------
def sci_small(v):
    """A power of ten in LaTeX, with the mantissa shown only when it is not one."""
    m, e = f"{v:.0e}".split("e")
    e = int(e)
    return f"$10^{{{e}}}$" if m == "1" else f"${m}\\times10^{{{e}}}$"


def _onoff(v):
    return "on" if v else "off"


# parameter key -> (column header, value formatter); keys of the chosen configurations as stored in the run files
GRID_COLS = {
    "method": ("QP variant", lambda v: esc(str(v))),
    "pbeta": (r"$\beta$ (offset)", lambda v: f"{v:g}"),
    "beta": (r"$\beta$ (popularity)", lambda v: f"{v:g}"),
    "lr": (r"$\mathrm{lr}$", sci_small),
    "wd": (r"$\rho$", sci_small),
    "w": (r"$w$", lambda v: f"{v:g}"),
    "r": ("normalisation exponent", lambda v: f"{v:g}"),
    "epoch": ("epoch", lambda v: f"{int(v)}"),
    "gamma": (r"$\gamma$", lambda v: f"{v:g}"),
    "lam": (r"$\lambda_{\mathrm{EASE}}$", sci_small),
    "prior": (r"$\nu$", lambda v: f"{v:g}"),
    "centre": ("centre", lambda v: esc(str(v))),
    "k_life": (r"$\lambda$", lambda v: f"{v:g}"),
    "tab": ("scenario factors", _onoff),
    "prior_s": ("user groups", lambda v: "none" if v is None else f"{v:g}"),
}
CFG_KEYS = ["method", "pbeta", "beta", "lr", "wd", "w", "r", "epoch", "gamma", "lam"]
DAX_KEYS = ["prior", "centre", "k_life", "tab", "prior_s"]
GRID_NOTE = ("Columns that a method does not use are not shown. The centre is the tag mean or the content regression "
             "of Appendix~\\ref{app:fh}; user groups are none or a strength.")


def grid_columns(chosen):
    keys = []
    for k in CFG_KEYS:
        if any(k in ch for ch in chosen):
            if k == "epoch" and all(ch["epoch"] == 0 for ch in chosen):
                continue
            keys.append(k)
    keys += [k for k in DAX_KEYS if any(k in (ch.get("dax") or {}) for ch in chosen)]
    return keys


def grid_cell(ch, k):
    if k in DAX_KEYS:
        d = ch.get("dax") or {}
        return GRID_COLS[k][1](d[k]) if k in d else "--"
    return GRID_COLS[k][1](ch[k]) if k in ch else "--"


def table_grids():
    c = context()
    H, M, K = c["H"], c["M"], c["K"]
    specs = [("LEAF-LGN+DAX", M["LEAF-LGN+DAX"]),
             ("LEAF-LGN", M["LEAF-LGN"]),
             ("QP family (best of its three variants per split)", M["QP family"]),
             ("QP family+DAX", M["QP family+DAX"]),
             ("EASE", M["EASE"]), ("EASE+DAX", M["EASE+DAX"]),
             ("LightGCN", M["LightGCN"]), ("r-AdjNorm", M["r-AdjNorm"]),
             ("DIN", M["DIN"]), ("DIN+DAX", M["DIN+DAX"]),
             ("SIM (SIM-hard)", M["SIM"]), ("SIM+DAX (SIM-hard)", M["SIM+DAX"]),
             ("LEAF-LGN+DAX, KuaiRand-1K", K["LEAF-LGN+DAX"]), ("QP family+DAX, KuaiRand-1K", K["QP family+DAX"]),
             ("DIN, KuaiRand-1K", K["DIN"]), ("DIN+DAX, KuaiRand-1K", K["DIN+DAX"]),
             ("SIM (SIM-hard), KuaiRand-1K", K["SIM"]), ("SIM+DAX (SIM-hard), KuaiRand-1K", K["SIM+DAX"])]
    index = []
    for i, (label, m) in enumerate(specs):
        name = f"tab_grid_{i:02d}.tex"
        lab = f"tab:grid{i:02d}"
        if m is None:
            text = table_env(f"Chosen configurations, {esc(label)}: not available.", lab, "l", ["Split"],
                             [[tbd(f"{label} chosen configurations")]])
        else:
            keys = grid_columns(m["chosen"])
            header = ["Split"] + [GRID_COLS[k][0] for k in keys]
            rows = [[f"{s + 1}"] + [grid_cell(ch, k) for k in keys] for s, ch in enumerate(m["chosen"])]
            text = table_env(f"Chosen configurations per split: {esc(label)}. Mean nDCG@10 "
                             f"{H['mean_sd'](m)[0]:.4f}.", lab, "l" + "c" * len(keys), header, rows, GRID_NOTE)
        write(name, text)
        index.append((name, label))
    return index


def _line(label, r):
    if r is None:
        return f"{label}: not available"
    if r.get("identical"):
        return f"{label}: identical rankings (0.00 [0.00])"
    return (f"{label}: {r['diff']:+.2f} [{r['half']:.2f}] p={r['p']:.3e} NB p={r['nb']:.4f} "
            f"splits {' '.join(f'{x:+.2f}' for x in r['split_diffs'])}")


def numbers_file(c, res, ph, nh, gain_minus, leaf_minus):
    """Every number of the generated tables that the text quotes, with its family, written to
    results/v13_tables_numbers.txt (computed here from the JSON files; nothing typed by hand)."""
    H, M, K, N = c["H"], c["M"], c["K"], c["N"]
    out = ["Numbers computed by experiments/v13_tables.py from results/v10, results/v11 and results/v12 (JSON).",
           "Differences: per-user paired mean x1e-3 nDCG@10 [95% half-width]; p = two-sided t test over pooled users;",
           "NB p = Nadeau-Bengio test over the five split means (frac_val 0.3); Holm adjustments within the family named.", ""]
    out.append(f"[main family, Holm over {len([k for k in H['FAMILY_V12'] if M.get(k) is not None])} comparators: "
               "table tab:main]")
    for k in H["FAMILY_V12"]:
        r = res.get(k)
        if r is None:
            out.append(f"  LEAF-LGN+DAX - {k}: not available")
            continue
        out.append("  " + _line(f"LEAF-LGN+DAX - {k}", r) +
                   f" p_Holm={ph[k]:.3e} NB_Holm={nh[k]:.4f}")
    keys9 = list(H["FAMILY_V12"]) + ["EASE", "EASE+DAX"]
    res9, ph9, nh9 = main_family(c, keys9)
    present9 = [k for k in keys9 if res9[k] is not None]
    out.append("")
    out.append(f"[sensitivity family, Holm over {len(present9)} comparators: table tab:sens]")
    for k in keys9:
        r = res9[k]
        if r is None:
            out.append(f"  {k}: not available")
            continue
        out.append(f"  {k}: p_Holm={ph9[k]:.3e} NB_Holm={nh9[k]:.4f}")
    out.append("")
    out.append("[backbones, raw p values: table tab:backbone]")
    lgn_ret = H["method_from"](H["LGN_GCN"], "LightGCN", "LightGCN (retrained run)")
    pairs = [("EASE+DAX - EASE", M["EASE+DAX"], M["EASE"]),
             ("LightGCN+DAX - LightGCN (retrained run, dax_lightgcn.json)", M["LightGCN+DAX"], lgn_ret),
             ("LightGCN+DAX - LightGCN (stored, learned_kuairand_t_base.json)", M["LightGCN+DAX"], M["LightGCN"]),
             ("QP family+DAX - QP family (stored)", M["QP family+DAX"], M["QP family"]),
             ("LEAF-LGN+DAX - LEAF-LGN (post-hoc run)", M["LEAF-LGN+DAX"], M["LEAF-LGN"]),
             ("DIN+DAX - DIN", M["DIN+DAX"], M["DIN"]),
             ("SIM+DAX - SIM", M["SIM+DAX"], M["SIM"])]
    for label, a, b in pairs:
        out.append("  " + _line(label, P(a, b)))
    out.append("")
    out.append("[ablations, KuaiRand-27K, base = LEAF-LGN of the ablation run: table tab:ablate]")
    base, arms = ablation_records()
    out.append(f"  base mean {H['mean_sd'](base)[0]:.4f} (n_configs {base['n_configs']}: backbone trainings of the ablation run)")
    gkeys = [k for k in ["full"] + ABL_ARMS if arms.get(k) is not None]
    gains = {k: P(arms[k], base) for k in gkeys}
    gph = dict(zip(gkeys, H["holm"]([gains[k]["p"] for k in gkeys])))
    out.append("  gains (Holm within these %d):" % len(gkeys))
    for k in gkeys:
        out.append("    " + _line(ABL_LABEL[k], gains[k]) + f" p_Holm={gph[k]:.3e}")
    for s_ in (0, 1, 2):
        for k in ["shuffle_tags", "shuffle_items"]:
            name = f"LEAF-LGN+DAX[{k}|p{s_}]"
            r = P(H["method_from"](H["ABLATE"], name, name), base)
            out.append("    " + _line(f"{ABL_LABEL[k]}, permutation seed {s_}", r))
    fm = [k for k in ABL_ARMS if arms.get(k) is not None]
    mh = dict(zip(fm, H["holm"]([gain_minus[k]["p"] for k in fm]))) if gain_minus else {}
    out.append("  full minus arm (Holm within these %d):" % len(fm))
    for k in fm:
        out.append("    " + _line(ABL_LABEL[k], gain_minus[k]) + f" p_Holm={mh[k]:.3e}")
    out.append("")
    out.append("[seeds, KuaiRand-27K: table tab:seeds; raw p values]")
    for s_ in (1, 2, 3):
        path = f"results/v11/c_seeds/lgn_s{s_}.json"
        dax = H["method_from"](path, "LEAF-LGN+DAX", f"seed {s_} LEAF-LGN+DAX")
        bas = H["method_from"](path, "LEAF-LGN", f"seed {s_} LEAF-LGN")
        if dax is None or bas is None:
            out.append(f"  seed {s_}: not available")
            continue
        out.append(f"  seed {s_}: LEAF-LGN+DAX {H['mean_sd'](dax)[0]:.4f}, LEAF-LGN {H['mean_sd'](bas)[0]:.4f}; "
                   + _line("gain", H["pair"](dax, bas, N))[len("gain: "):])
    out.append("")
    out.append("[deep baselines: table tab:deep; raw p values]")
    for k in ["DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        out.append("  " + _line(f"27K LEAF-LGN+DAX - {k}", P(M["LEAF-LGN+DAX"], M[k])) +
                   f" (mean {H['mean_sd'](M[k])[0]:.4f})")
    for k in ["DIN", "DIN+DAX", "SIM", "SIM+DAX"]:
        out.append("  " + _line(f"1K LEAF-LGN+DAX - {k}", P1k(K["LEAF-LGN+DAX"], K[k])) +
                   f" (mean {H['mean_sd'](K[k])[0]:.4f})")
    out.append("")
    out.append("[KuaiRand-1K: table tab:1k; Holm over the three comparisons]")
    cmp = [("LEAF-LGN+DAX - QP family", K["LEAF-LGN+DAX"], K["QP family"]),
           ("LEAF-LGN+DAX - QP family+DAX", K["LEAF-LGN+DAX"], K["QP family+DAX"]),
           ("QP family+DAX - QP family", K["QP family+DAX"], K["QP family"])]
    rr = [P1k(a, b) for _, a, b in cmp]
    pres = [r for r in rr if r is not None]
    hp = H["holm"]([r["p"] for r in pres]); hn = H["holm"]([r["nb"] for r in pres])
    j = 0
    for (label, _, _), r in zip(cmp, rr):
        if r is None:
            out.append(f"  {label}: not available")
            continue
        out.append("  " + _line(label, r) + f" p_Holm={hp[j]:.3e} NB_Holm={hn[j]:.4f}")
        j += 1
    out.append("")
    out.append("[LEAF controls, LEAF-tied, stage 1: table tab:leafctl; Holm over the eight arms]")
    fm = [k for k in V10_ARMS if leaf_minus.get(k) is not None]
    if fm:
        lp = dict(zip(fm, H["holm"]([leaf_minus[k]["p"] for k in fm])))
        for k in fm:
            out.append("  " + _line(f"LEAF-tied - LEAF-tied[{k}]", leaf_minus[k]) + f" p_Holm={lp[k]:.3e}")
    path = os.path.join(ROOT, "results", "v13_tables_numbers.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


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
    bpath = numbers_file(c, res, ph, nh, gain_minus[1] if gain_minus else None, minus or {})
    print("NUMBERS written to", bpath)
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
