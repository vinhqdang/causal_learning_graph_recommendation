"""A4: LaTeX tables T4-T8, T12 and the refinement configuration-count table (T13), written from the result files of
the v7 scripts and the stored results. No number is typed by hand.

Inputs: results/v6/tests.json (A2), results/v6/sensitivity.json (A1), the stored results through v7_common.load_all,
results/v3/matched_config_real.txt (T8, matched configurations) and the job files of the refinement family.
Missing inputs are skipped with a printed note.
Writes paper/v4/tables/T4_main.tex, T5_effects.tex, T6_refinements.tex, T7_sensitivity.tex, T8_walk.tex,
T12_backbone.tex, T13_refine_cfg.tex and results/v6/tables.txt.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import KEY, NAME, OUT, V4, V5, Report, load_all, exists, _family_jobs  # noqa: E402

TDIR = "paper/v4/tables"
DS = ["coat", "yahoo", "kuairand", "kuairec"]

T4_ROWS = ["Obs", "Pop", "Impute", "IPS", "DR", "DRUP", "EASE", "GF-CF", "EASE-DR", "GF-CF-DR",
           "MF", "IPS-MF", "DR-MF", "BPR-MF", "PDA", "LightGCN", "LightGCN-pt", "r-AdjNorm", "NAVIP", "DR-LightGCN",
           "SimGCL", "MACR", "AdvInfoNCE-LGN", "DR-JL", "MRDR", "iALS", "iALS-wide", "iALS-p (add prior)",
           "LightGCN-p (add prior)", "DR-iALS", "DR-BPR-LGN", "DR-BPR-LGN-1 (Q, lrQ)", "DR-BPR-LGN-1 (Q)",
           "DR-BPR-LGN-p", "DR-BPR-LGN-p2", "QP family (pooled)", "control: shuffled Q", "control: popularity Q"]


DISPLAY = {"iALS-wide": "iALS (search B)", "DR-BPR-LGN-1 (Q, lrQ)": "DR-BPR-LGN-1 (Q-only)",
           "DR-BPR-LGN-1 (Q)": "DR-BPR-LGN-1 (Q-only)"}


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ptx(p):
    """p-value inside running text or a cell note: $p<0.001$ or $p=0.271$."""
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


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
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
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
        cells = [esc(DISPLAY.get(name, name))]
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
            seedtxt = f"{1000 * seed['diff']:+.2f} ({ptx(seed['p'])})" if seed else "--"
            if ds == "coat":
                # Coat: no equivalence claim; the column states which differences the data exclude
                tost = (f"excludes $\\lvert d\\rvert>0.010$ ({ptx(r['tost_p'])})" if r['equivalent']
                        else f"does not exclude $\\lvert d\\rvert>0.010$ ({ptx(r['tost_p'])})")
            else:
                tost = f"{'yes' if r['equivalent'] else 'no'} ({ptx(r['tost_p'])})"
            rows.append(f"{NAME[ds]} & {r['family']} & {esc(r['A'])} -- {esc(r['B'])} & "
                        f"{1000 * r['diff']:+.2f} [{1000 * r['half']:.2f}] & {pfmt(r['p'])} & {pfmt(r['p_holm'])} & "
                        f"{pfmt(r['nb_holm'])} & {1000 * r['mde']:.2f} & {tost} & {kmin} & {seedtxt}")
    return table("llllrrrrlrl", "Dataset & Fam. & Comparison ($A-B$) & Diff.\\ $\\times10^{3}$ [95\\% CI] & $p$ & "
                 "$p_\\mathrm{Holm}$ & $p_\\mathrm{NB,Holm}$ & MDE $\\times10^{3}$ & TOST at margin & Matched-$k$ diff $\\times10^{3}$ & "
                 "Seeds 1--3 diff $\\times10^{3}$",
                 rows, "Primary comparisons P1--P4 (Section~\\ref{sec:protocol}): paired per-user differences, tests, "
                       "MDE, TOST at the sensitivity margins (Section~\\ref{sec:stats}), and the matched-budget "
                       "difference at $k=k_\\mathrm{min}$.",
                 "tab:T5", note="Margins are sensitivity thresholds chosen before the analysis and not derived from the "
                                "literature (Section~\\ref{sec:stats}). On Coat the TOST column states the differences "
                                "the data exclude; equivalence is not claimed there. Seeds column: only comparisons "
                                "whose baselines were retrained with seeds 1--3 (Section~\\ref{sec:seeds}) are shown.")


def t6(tests, M):
    rows = []
    for ds in DS:
        for r in tests.get(ds, {}).get("families", {}).get("T6", []):
            ma = M[ds].get(r["A"])
            mb = M[ds].get(r["B"])
            mean = f"{ma.stored_mean(KEY[ds]):.4f}" if ma else "--"
            cfg = str(ma.n_configs) if ma else "--"
            if ds == "coat":
                tost = "excludes $\\lvert d\\rvert>0.010$" if r['equivalent'] else "does not exclude"
            else:
                tost = f"{'yes' if r['equivalent'] else 'no'}"
            rows.append(f"{NAME[ds]} & {r['family'].replace('T6 ', '')} & {esc(DISPLAY.get(r['A'], r['A']))} & {mean} & {cfg} & "
                        f"{esc(r['B']) if mb else '--'} & {1000 * r['diff']:+.2f} [{1000 * r['half']:.2f}] & "
                        f"{pfmt(r['p_holm'])} & {pfmt(r['nb_holm'])} & {tost}")
    return table("lllrrlrrrl", "Dataset & Family & Method & Mean & Cfg & Versus & Diff.\\ $\\times10^{3}$ [95\\% CI] & "
                 "$p_\\mathrm{Holm}$ & $p_\\mathrm{NB,Holm}$ & TOST at margin (Coat: exclusion, not equivalence)",
                 rows, "Refinement family against the simple debiased prior of the same learner, and the shuffled and "
                       "popularity controls against the backbone.", "tab:T6",
                 note="Holm within each family and dataset; $p_\\mathrm{NB,Holm}$ applies the Nadeau-Bengio correction "
                      "of Section~\\ref{sec:stats} to the same family. TOST at the sensitivity margin: \\emph{yes} means "
                      "equivalence within $\\pm0.010$ (Coat, Yahoo! R3) or $\\pm0.003$ (KuaiRand-Pure, KuaiRec) at level 0.05; "
                      "\\emph{no} means that equivalence is not shown. On Coat the entry states only whether differences "
                      "larger than $\\pm0.010$ are excluded, and equivalence is not claimed. Rows are missing where the "
                      "simple prior has not been run for that dataset.")


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
                 note="Flip share: fraction of 1,000 random subsets of $k_\\mathrm{min}$ configurations per method whose "
                      "sign differs from the full search space. Alt.\\ metric: nDCG@10 (Coat, Yahoo), nDCG@20 (KuaiRand), "
                      "nDCG@50 (KuaiRec), on the stored selections.")


def t8(tests):
    """Walk correction (DRUP minus DR). Panel A: matched configurations, from results/v3/matched_config_real.txt.
    Panel B: selected runs (per-user paired test), from the walk family of tests.json, for every dataset that has it."""
    path = "results/v3/matched_config_real.txt"
    rows = [r"\multicolumn{6}{l}{\emph{Matched configurations (split means, no test)}}"]
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
                rows.append(f"{cur} & {beta:g} & {float(m.group(2)):+.2f} & {float(m.group(3)):.2f} & {m.group(4)} & --")
    rows.append(r"\multicolumn{6}{l}{\emph{Selected runs (per-user paired test)}}")
    notes = []
    for ds in ["coat", "yahoo", "kuairec"]:
        w = next((r for r in tests.get(ds, {}).get("families", {}).get("walk", [])), None)
        if w is None:
            continue
        ci = f" [{1000 * w['half']:.2f}]" if not w.get("identical") else " [0.00]"
        rows.append(f"{NAME[ds]} & -- & {1000 * w['diff']:+.2f}{ci} & -- & -- & {pfmt(w['p_holm'])}")
        if w.get("identical"):
            notes.append("On Yahoo! R3 the selected DRUP and DR runs give identical per-user values, so the walk "
                         "correction changes nothing there and the row is uninformative.")
        elif ds == "coat":
            # Coat: the TOST result excludes differences larger than the margin; equivalence is not claimed
            notes.append(f"Coat: differences larger than $\\pm${w['tost_margin']:.3f} are excluded (TOST {ptx(w['tost_p'])}); "
                         "equivalence is not claimed.")
        else:
            notes.append(f"{NAME[ds]}: TOST {'within' if w['equivalent'] else 'not within'} the sensitivity margin "
                         f"($\\pm${w['tost_margin']:.3f}), {ptx(w['tost_p'])}.")
    note = ("Panel A: difference of the matched-configuration split means, $\\times10^{3}$; the last column is the "
            "number of splits with a negative mean. Panel B: per-user differences with 95\\% half-widths in brackets, "
            "and Holm-adjusted $p$ within the walk family. " + " ".join(notes))
    return table("lllrrr", "Dataset & $\\beta$ & Mean diff.\\ $\\times10^{3}$ [95\\% CI] & Share of configs $<0$ & "
                 "Splits with mean $<0$ & $p_\\mathrm{Holm}$",
                 rows, "Walk correction (DRUP minus DR) at matched configurations and in the selected runs.", "tab:T8",
                 note=note)


def t12(tests):
    """Appendix: every method against LightGCN (the backbone family), one table per dataset."""
    out = []
    for ds in DS:
        rows = []
        for r in tests.get(ds, {}).get("families", {}).get("all_vs_backbone", []):
            if r["A"] == "LightGCN":
                continue
            rows.append(f"{esc(DISPLAY.get(r['A'], r['A']))} & {1000 * r['diff']:+.2f} [{1000 * r['half']:.2f}] & "
                        f"{pfmt(r['p'])} & {pfmt(r['p_holm'])} & {pfmt(r['nb_holm'])} & {1000 * r['mde']:.2f}")
        cap = (f"Appendix: every method against LightGCN on {NAME[ds]} (backbone family, nDCG@{KEY[ds].split('@')[1]}). "
               f"Holm is taken over this family.")
        out.append(table("lrrrrr", "Method & Diff.\\ $\\times10^{3}$ [95\\% CI] & $p$ & $p_\\mathrm{Holm}$ & "
                         "$p_\\mathrm{NB,Holm}$ & MDE $\\times10^{3}$", rows, cap, f"tab:T12-{ds}",
                         note="Per-user paired differences over the test users; the Nadeau-Bengio column is the "
                              "heuristic correction of Section~\\ref{sec:stats}."))
    return "\n".join(out)


def jobs_cfg(jobs):
    """Sum of the stored configuration counts over (result file, method) jobs; None if a job is missing."""
    total = 0
    for path, mname in jobs:
        if not os.path.exists(path):
            return None
        r = json.load(open(path)).get("results", {}).get(mname)
        if r is None:
            return None
        total += int(r.get("n_configs", 0))
    return total


def t13():
    """Configuration counts of the refinement family, the controls and the DRGP runs (appendix)."""
    def fam(ds, name):
        return [j for j in _family_jobs(ds) if j[1] == name]

    def ctl(tag):
        return {"coat": [(V4 + f"learned_coat_abl_DR-BPR-{tag}.json", f"DR-BPR-{tag}")],
                "yahoo": [(V4 + f"learned_yahoo_abl_DR-BPR-{tag}.json", f"DR-BPR-{tag}")]}

    def drgp_prior(ds):
        if ds == "coat":
            return [(V5 + f"learned_coat_drgp_{t}.json", "DRGP")
                    for t in ["f0.05_b1", "f0.05_b2", "f0.1_b1", "f0.1_b2", "f0.2_b1", "f0.2_b2"]]
        return [(V5 + "learned_kuairec_drgp_f0.1.json", "DRGP"), (V5 + "learned_kuairec_drgp_f0.2.json", "DRGP")]

    def drgp_lgn(name):
        return {"coat": [(V5 + "learned_coat_drgp_lgn.json", name)],
                "kuairec": [(V5 + "learned_kuairec_drgp_lgn.json", name)]}

    spec = [
        ("DR-BPR-LGN (Q-weighted LightGCN)", {"coat": [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN")]}),
        ("DR-BPR-LGN-1 (Q-only)", {"coat": [(V4 + "learned_coat_lrQ.json", "DR-BPR-LGN-1")],
                                   **{ds: fam(ds, "DR-BPR-LGN-1") for ds in ["yahoo", "kuairand", "kuairec"]}}),
        ("DR-BPR-LGN-p (offset, $\\beta\\in\\{0.5,1\\}$)", {ds: fam(ds, "DR-BPR-LGN-p") for ds in DS}),
        ("DR-BPR-LGN-p2 (offset, $\\beta\\in\\{1,2\\}$)", {ds: fam(ds, "DR-BPR-LGN-p2") for ds in DS}),
        ("DR-iALS", {"coat": [(V4 + "learned_coat_driALS.json", "DR-iALS")]}),
        ("QP family (pooled over its jobs)", {ds: _family_jobs(ds) for ds in DS}),
        ("Control: shuffled Q", ctl("shuf")),
        ("Control: popularity Q", ctl("pop")),
        ("DRGP prior alone", {"coat": drgp_prior("coat"), "kuairec": drgp_prior("kuairec")}),
        ("DRGP-iALS (iALS-p with the DRGP prior)", {"coat": [(V5 + "learned_coat_drgp_ials.json", "iALS-p")],
                                                     "kuairec": [(V5 + "learned_kuairec_drgp_ials.json", "iALS-p")]}),
        ("DRGP-LGN-bpr", drgp_lgn("DRGP-LGN-bpr")),
        ("DRGP-LGN-dr", drgp_lgn("DRGP-LGN-dr")),
        ("DRGP-LGN-hyb", {"coat": [(V5 + "learned_coat_drgp_lgn.json", "DRGP-LGN-hyb")]}),
    ]
    rows = []
    for name, jobs in spec:
        cells = []
        for ds in DS:
            n = jobs_cfg(jobs[ds]) if jobs.get(ds) else None
            cells.append("--" if n is None else str(n))
        rows.append(" & ".join([esc(name)] + cells))
    return table("lrrrr", "Member & " + " & ".join(NAME[d] for d in DS), rows,
                 "Configurations searched by the refinement family, its controls and the DRGP runs (stored counts).",
                 "tab:T13",
                 note="Counts are the configurations stored in the result files. The QP family pools the jobs listed "
                      "above it for each dataset, and its count is their sum. The DRGP prior alone has one configuration "
                      "per run. Controls are not proposed methods. A dash marks a member that was not run on the dataset.")


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
    write("T12_backbone.tex", t12(tests))
    write("T13_refine_cfg.tex", t13())
    rep("written: " + ", ".join(f"{TDIR}/{n}" for n in ["T4_main.tex", "T5_effects.tex", "T6_refinements.tex",
                                                       "T7_sensitivity.tex", "T8_walk.tex", "T12_backbone.tex",
                                                       "T13_refine_cfg.tex"]))
    rep.save("tables.txt")


if __name__ == "__main__":
    main()
