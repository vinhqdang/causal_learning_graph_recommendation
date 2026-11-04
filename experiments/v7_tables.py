"""A4: LaTeX tables T4-T8 of paper/v4/DESIGN.md Section 3.2, written from the result files of the v7 scripts
and the stored results. No number is typed by hand.

Inputs: results/v6/tests.json (A2), results/v6/sensitivity.json (A1), the stored results through v7_common.load_all,
and results/v3/matched_config_real.txt (T8 panel C). T8 panels A (MC bias) and B (semi-synthetic) need producer
files that are not part of the v7 pipeline and are not generated here.
Missing inputs are skipped with a printed note.
Writes paper/v4/tables/T4_main.tex ... T8_walk.tex and results/v6/tables.txt.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import KEY, NAME, OUT, Report, load_all, exists  # noqa: E402

TDIR = "paper/v4/tables"
DS = ["coat", "yahoo", "kuairand", "kuairec"]

T4_ROWS = ["Obs", "Pop", "Impute", "IPS", "DR", "DRUP", "EASE", "GF-CF", "EASE-DR", "GF-CF-DR",
           "MF", "IPS-MF", "DR-MF", "BPR-MF", "PDA", "LightGCN", "LightGCN-pt", "r-AdjNorm", "NAVIP", "DR-LightGCN",
           "SimGCL", "MACR", "AdvInfoNCE-LGN", "DR-JL", "MRDR", "iALS", "iALS-wide", "iALS-p (add prior)",
           "LightGCN-p (add prior)", "DR-iALS", "DR-BPR-LGN", "DR-BPR-LGN-1 (Q, lrQ)", "DR-BPR-LGN-1 (Q)",
           "DR-BPR-LGN-p", "DR-BPR-LGN-p2", "QP family (pooled)", "control: shuffled Q", "control: popularity Q"]


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def f4(x, n=4):
    return f"{x:.{n}f}"


def ms(x, n=4):
    return f"${x:+.{n}f}$" if x is not None else "--"


def write(name, body):
    os.makedirs(TDIR, exist_ok=True)
    with open(os.path.join(TDIR, name), "w") as f:
        f.write(body)


def table(colspec, header, rows, caption, label, note=None):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        lines += [f"\\par\\smallskip\\raggedright\\footnotesize {note}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def t4(M):
    cols = ["l"] + ["rr"] * 4
    head = "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAME[d]}}}" for d in DS) + r" \\"
    head += "\n" + " & " + " & ".join("mean (sd) & cfg" for _ in DS)
    rows = []
    for name in T4_ROWS:
        cells = [esc(name)]
        anyv = False
        for d in DS:
            m = M[d].get(name)
            if m is None:
                cells += ["--", "--"]
                continue
            anyv = True
            v = np.array([ps[KEY[d]] for ps in m.per_split])
            cells += [f"{v.mean():.4f} ({v.std(ddof=0):.4f})", str(m.n_configs)]
        if anyv:
            rows.append(" & ".join(cells))
    return table("l" + "rr" * 4, head, rows,
                 "Main benchmark: mean test score over splits (sd over splits, ddof 0) and configuration count per "
                 "method. Metric: nDCG@5 (Coat, Yahoo), nDCG@10 (KuaiRand), nDCG@20 (KuaiRec).",
                 "tab:T4",
                 note="Pooled rows (QP family) pool the configurations of their jobs; a cell marked -- has no run "
                      "for that method and dataset.")


def t5(tests, sens):
    rows = []
    for ds in DS:
        for r in tests.get(ds, {}).get("families", {}).get("primary", []):
            sp = next((p for p in sens.get(ds, {}).get("pairs", [])
                       if p["family"] == r["family"] and p["A"] == r["A"] and p["B"] == r["B"]), None)
            kmin = f"{1000 * sp['expected_diff_kmin']:+.2f}" if sp and sp.get("expected_diff_kmin") is not None else "--"
            seed = r.get("seed")
            seedtxt = f"{1000 * seed['diff']:+.2f} ($p$={seed['p']:.3f})" if seed else "--"
            tost = f"{'yes' if r['equivalent'] else 'no'} ($p$={r['tost_p']:.3f})"
            rows.append(f"{NAME[ds]} & {r['family']} & {esc(r['A'])} -- {esc(r['B'])} & "
                        f"{1000 * r['diff']:+.2f} [{1000 * r['half']:.2f}] & {r['p']:.3f} & {r['p_holm']:.3f} & "
                        f"{r['nb_holm']:.3f} & {1000 * r['mde']:.2f} & {tost} & {kmin} & {seedtxt}")
    return table("llllrrrrlrl", "Dataset & Fam. & Comparison ($A-B$) & Diff.\\ $\\times10^{3}$ [95\\% CI] & $p$ & "
                 "$p_\\mathrm{Holm}$ & $p_\\mathrm{NB,Holm}$ & MDE $\\times10^{3}$ & TOST at margin & Matched-$k$ diff $\\times10^{3}$ & "
                 "Seeds 1--3 diff $\\times10^{3}$",
                 rows, "Primary comparisons P1--P4 (Section~\\ref{sec:protocol}): paired per-user differences, tests, "
                       "MDE, TOST at the pre-specified margins (Section~\\ref{sec:protocol}), and the matched-budget "
                       "difference at $k=k_\\mathrm{min}$.",
                 "tab:T5", note="Margins are the pre-specified margins (Section~\\ref{sec:protocol}) and are provisional "
                                "(Section~\\ref{sec:stats}). Seeds column: only comparisons whose baselines were "
                                "retrained with seeds 1--3 (Section~\\ref{sec:seeds}) are shown.")


def t6(tests, M):
    rows = []
    for ds in DS:
        for r in tests.get(ds, {}).get("families", {}).get("T6", []):
            ma = M[ds].get(r["A"])
            mb = M[ds].get(r["B"])
            mean = f"{ma.stored_mean(KEY[ds]):.4f}" if ma else "--"
            cfg = str(ma.n_configs) if ma else "--"
            tost = f"{'yes' if r['equivalent'] else 'no'}"
            rows.append(f"{NAME[ds]} & {r['family'].replace('T6 ', '')} & {esc(r['A'])} & {mean} & {cfg} & "
                        f"{esc(r['B']) if mb else '--'} & {1000 * r['diff']:+.2f} [{1000 * r['half']:.2f}] & "
                        f"{r['p_holm']:.3f} & {tost}")
    return table("lllrrlrrl", "Dataset & Family & Method & Mean & Cfg & Versus & Diff.\\ $\\times10^{3}$ [95\\% CI] & "
                 "$p_\\mathrm{Holm}$ & TOST equiv.",
                 rows, "Refinement family against the simple debiased prior of the same learner, and the shuffled and "
                       "popularity controls against the backbone (T6).", "tab:T6",
                 note="Holm within each family and dataset. Rows are missing where the simple prior has not been run "
                      "for that dataset.")


def t7(tests, sens):
    rows = []
    for ds in DS:
        for p in sens.get(ds, {}).get("pairs", []):
            t5row = next((r for r in tests.get(ds, {}).get("families", {}).get("primary", [])
                          if r["family"] == p["family"] and r["A"] == p["A"] and r["B"] == p["B"]), None)
            if t5row is None:
                continue
            # sign survival of the split-level columns is judged on the split-mean difference of A1 (same unit as
            # the oracle columns); the Main column and the seed column use the per-user difference of A2
            base = p["valsel_diff"] if p.get("valsel_diff") is not None else t5row["diff"]
            base_user = t5row["diff"]

            def same(v):
                return v is not None and np.sign(v) == np.sign(base)
            main = ("$\\checkmark$" if t5row["p_holm"] < 0.05 else "$\\times$") + f" {1000 * base_user:+.2f}"
            orc = p.get("oracle_diff")
            orc_c = ("$\\checkmark$" if same(orc) else "$\\times$") + (f" {1000 * orc:+.2f}" if orc is not None else "")
            cfo = p.get("config_oracle_diff")
            cfo_c = ("$\\checkmark$" if same(cfo) else "$\\times$") + (f" {1000 * cfo:+.2f}" if cfo is not None else "")
            km = p.get("expected_diff_kmin")
            km_c = ("$\\checkmark$" if same(km) else "$\\times$") + (f" {1000 * km:+.2f}" if km is not None else "")
            nb = "$\\checkmark$" if t5row["nb_holm"] < 0.05 else "$\\times$"
            seed = t5row.get("seed")
            seed_c = ("$\\checkmark$" if (seed and np.sign(seed["diff"]) == np.sign(base_user) and seed["p"] < 0.05)
                      else ("$\\times$" if seed else "n/a"))
            alt = ("$\\checkmark$" if p.get("alt_same_sign") else "$\\times$") if "alt_same_sign" in p else "n/a"
            flip = f"{100 * p['flip_share']:.1f}\\%" if p.get("flip_share") is not None else "n/a"
            rows.append(f"{NAME[ds]} & {p['family']} & {esc(p['A'])} -- {esc(p['B'])} & {main} & {orc_c} & {cfo_c} & "
                        f"{km_c} & {p['kmin']} & {flip} & {seed_c} & {nb} & {alt}")
    return table("lllllllrrlll",
                 "Dataset & Fam. & Comparison & Main (Holm) & Test oracle & Config.\\ oracle & Matched $k$ & $k_\\mathrm{min}$ & "
                 "Flip share & Seeds & NB & Alt.\\ metric",
                 rows, "Sensitivity of the primary conclusions to evaluation choices. A check mark means the sign "
                       "of the difference (and, for Main, significance at $p_\\mathrm{Holm}<0.05$) survives the choice; "
                       "the value is the difference $\\times10^{3}$ under that choice. Main: per-user difference. Oracle, configuration-oracle and matched-budget columns: split-mean difference, compared in sign with the split-mean validation-selected difference.",
                 "tab:T7",
                 note="Flip share: fraction of 1,000 random subgrids of $k_\\mathrm{min}$ configurations per method whose "
                      "sign differs from the full grid. Alt.\\ metric: nDCG@10 (Coat, Yahoo), nDCG@20 (KuaiRand), "
                      "nDCG@50 (KuaiRec), on the stored selections.")


def t8(tests):
    path = "results/v3/matched_config_real.txt"
    rows = []
    kr_note = None
    if exists(path):
        cur = None
        for line in open(path):
            m = re.match(r"^(coat|yahoo) \(", line)
            if m:
                cur = NAME[m.group(1).lower()]
                continue
            m = re.match(r"^\s*([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+(\d+/\d+)", line)
            if m and cur:
                beta = float(m.group(1))
                rows.append(f"{cur} & {beta:g} & {float(m.group(2)):+.2f} & {float(m.group(3)):.2f} & {m.group(4)}")
    for ds in ["kuairec"]:
        w = next((r for r in tests.get(ds, {}).get("families", {}).get("walk", [])), None)
        if w:
            # the KuaiRec row has no configuration grid and no split counts: its paired-test p-value and its
            # equivalence verdict go in the note, not into the share or split-count columns
            rows.append(f"{NAME[ds]} (DRUP $-$ DR, all splits) & -- & {1000 * w['diff']:+.2f} & -- & --")
            kr_note = (f"KuaiRec row: paired test $p$={w['p']:.3f}; "
                       f"{'equivalent' if w['equivalent'] else 'not equivalent'} at the pre-specified margin "
                       f"(Section~\\ref{{sec:protocol}}).")
    return table("llrrl", "Dataset & $\\beta$ & Mean diff.\\ $\\times10^{3}$ & Share of configs $<0$ & Splits with mean $<0$",
                 rows, "Walk correction at matched configurations (Coat, Yahoo), and KuaiRec DRUP minus DR.", "tab:T8",
                 note=kr_note if kr_note else None)


def main():
    rep = Report()
    rep("A4 tables (experiments/v7_tables.py)")
    tests = json.load(open(os.path.join(OUT, "tests.json"))) if exists(os.path.join(OUT, "tests.json"), rep) else {}
    sens = json.load(open(os.path.join(OUT, "sensitivity.json"))) if exists(os.path.join(OUT, "sensitivity.json"), rep) else {}
    M = {}
    for ds in DS:
        M[ds] = load_all(ds, rep)
    write("T4_main.tex", t4(M))
    write("T5_effects.tex", t5(tests, sens))
    write("T6_refinements.tex", t6(tests, M))
    write("T7_sensitivity.tex", t7(tests, sens))
    write("T8_walk.tex", t8(tests))
    rep("written: " + ", ".join(f"{TDIR}/{n}" for n in ["T4_main.tex", "T5_effects.tex", "T6_refinements.tex",
                                                       "T7_sensitivity.tex", "T8_walk.tex"]))
    rep.save("tables.txt")


if __name__ == "__main__":
    main()
