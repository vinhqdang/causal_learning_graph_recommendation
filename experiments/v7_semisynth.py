"""A5: appendix tables of the semi-synthetic KuaiRec study (results/v3/semisynth_kuairec_dense.json).

The study is a true-label semi-synthetic world. Potential outcomes Y are the labels of the fully observed KuaiRec
matrix, and exposure is drawn from a known mechanism (experiments/semisynth_kuairec.py). The stored values are means
over 30 replications, with standard errors (sd over replications divided by sqrt(30)). The bias of the three-hop term
is relative: mean absolute averaged bias over the true mean absolute three-hop term.

Analysis only: reads the stored file and trains nothing. Missing inputs are skipped with a printed note.
Writes paper/v4/tables/T14_semisynth_ndcg.tex, paper/v4/tables/T15_semisynth_bias.tex and results/v6/semisynth.txt.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import OUT, Report, exists  # noqa: E402

SRC = "results/v3/semisynth_kuairec_dense.json"
TDIR = "paper/v4/tables"
METHODS = ["IPS", "IPS+WC", "DR", "DRUP"]
SRC_NAME = {"indep": "independent", "xfit": "cross-fitted", "split": "20\\% split"}
PROP_NAME = {"true": "true", "est": "estimated"}
CDEG_NAME = {"fixed": "fixed", "yhat": "imputed"}
ORDER_SRC, ORDER_PROP, ORDER_CDEG = ["indep", "xfit", "split"], ["true", "est"], ["fixed", "yhat"]


def table(colspec, header, rows, caption, label, note):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", f"\\par\\smallskip\\raggedright\\footnotesize {note}",
              r"\end{table}", ""]
    return "\n".join(lines)


def cells(R):
    """Cells with nDCG values, in display order: (key, source, propensity, degree weights)."""
    out = []
    for s in ORDER_SRC:
        for p in ORDER_PROP:
            for c in ORDER_CDEG:
                key = f"{s}/{p}/{c}"
                if key in R and "ndcg20" in R[key].get("DRUP", {}):
                    out.append((key, s, p, c))
    return out


def write(name, body, rep):
    os.makedirs(TDIR, exist_ok=True)
    with open(os.path.join(TDIR, name), "w") as f:
        f.write(body)
    rep(f"written: {TDIR}/{name}")


def main():
    rep = Report()
    rep("A5 semi-synthetic KuaiRec appendix tables (experiments/v7_semisynth.py)")
    if not exists(SRC, rep):
        rep.save("semisynth.txt")
        return
    j = json.load(open(SRC))
    R = j["results"]
    info = j.get("info", {})
    cfg = j.get("config", {})
    rep(f"replications {cfg.get('reps')}, density {cfg.get('density')}, beta {cfg.get('beta')}, "
        f"m {info.get('m')} users, n {info.get('n')} items, positive rate {info.get('pos_rate'):.4f}")

    # ---------------- T14: nDCG@20 on unexposed candidates against the true labels
    rows14 = []
    for key, s, p, c in cells(R):
        cells_txt = []
        means = {}
        for m in METHODS:
            v = R[key][m]["ndcg20"]
            means[m] = v["mean"]
            cells_txt.append(f"{v['mean']:.4f} ({v['se']:.4f})")
        d = means["DRUP"] - means["DR"]
        rows14.append(" & ".join([SRC_NAME[s], PROP_NAME[p], CDEG_NAME[c]] + cells_txt + [f"{d:+.4f}"]))
        rep(f"ndcg20 {key}: " + ", ".join(f"{m} {means[m]:.4f}" for m in METHODS) + f"; DRUP-DR {d:+.4f}")
    n_drup_higher = sum(1 for r in rows14 if r.split(" & ")[-1].startswith("+"))
    rep(f"DRUP above DR in {n_drup_higher} of {len(rows14)} cells with nDCG@20")
    t14 = table("lllrrrrr",
                "Nuisance fit & Propensity & Degree weights & IPS & IPS+WC & DR & DRUP & $\\Delta$ DRUP$-$DR",
                rows14,
                "Semi-synthetic KuaiRec study: nDCG@20 on unexposed candidates against the true labels. "
                "Mean over 30 replications with the standard error in parentheses.",
                "tab:T14",
                note="The last column is the difference of the two means, which is not a paired test: the standard "
                     "error of the difference is not stored. Higher values are better. Nuisance fit: "
                     "\\emph{independent} uses a log independent of the one scored, \\emph{cross-fitted} uses folds "
                     "of the same log, and \\emph{20\\% split} uses a 20\\% sample split. The degree weights are "
                     "\\emph{fixed} (the true-degree normalisation) or \\emph{imputed} from the imputation.")
    write("T14_semisynth_ndcg.tex", t14, rep)

    # ---------------- T15: relative bias of the three-hop term
    rows15 = []
    for key, s, p, c in cells(R):
        vals = [f"{R[key][m]['rel_bias_T']:.2f}" for m in METHODS]
        rows15.append(" & ".join([SRC_NAME[s], PROP_NAME[p], CDEG_NAME[c]] + vals))
    if "indep/true/fixed/K5" in R:
        k5 = R["indep/true/fixed/K5"]
        rows15.append(" & ".join(["independent (K=5)", "true", "fixed", "--",
                                  "--", f"{k5['DR']['rel_bias_T5']:.2f}", f"{k5['DRUP']['rel_bias_T5']:.2f}"]))
    t15 = table("lllrrrr",
                "Nuisance fit & Propensity & Degree weights & IPS & IPS+WC & DR & DRUP",
                rows15,
                "Semi-synthetic KuaiRec study: relative bias of the three-hop term, averaged over the draws. "
                "The last row is the five-hop term.",
                "tab:T15",
                note="Relative bias: mean absolute bias of the averaged estimate over the entries of the three-hop "
                     "term, divided by the mean absolute true three-hop term. IPS and IPS+WC (the walk-corrected IPS "
                     "estimator) are shown for reference. The column definitions are those of Table~\\ref{tab:T14}.")
    write("T15_semisynth_bias.tex", t15, rep)
    rep.save("semisynth.txt")


if __name__ == "__main__":
    main()
