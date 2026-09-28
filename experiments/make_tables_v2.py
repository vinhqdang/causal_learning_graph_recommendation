"""LaTeX tables for the manuscript from results/v2/*.json (protocol v2)."""

import json
import os

import numpy as np

R = "results/v2"
T = "paper/tables"
os.makedirs(T, exist_ok=True)
DS = [("coat", "given", "ndcg@5", "Coat"), ("yahoo", "pop", "ndcg@5", "Yahoo!\\,R3"),
      ("kuairec", "pop", "ndcg@20", "KuaiRec")]


def load(path):
    p = os.path.join(R, path)
    return json.load(open(p)) if os.path.exists(p) else None


def results(ds, prop):
    out = {}
    for f in (f"filters_{ds}_{prop}.json", f"learned_{ds}_{prop}.json"):
        j = load(f)
        if j:
            out.update(j["results"])
    return out


def fmt(x, d=4):
    return "--" if x is None else f"{x:.{d}f}"


ROWS = [
    ("Non-personalised", [("Pop", "Popularity"), ("Impute", "Imputation only")]),
    ("Trained, pointwise loss", [("MF", "MF"), ("IPS-MF", "IPS-MF"), ("DR-MF", "DR-MF"),
                                 ("LightGCN-pt", "LightGCN"), ("DR-LightGCN", "DR-LightGCN")]),
    ("Trained, pairwise loss (BPR)", [("BPR-MF", "MF"), ("PDA", "PDA"), ("LightGCN", "LightGCN"),
                                      ("r-AdjNorm", "r-AdjNorm"), ("NAVIP", "NAVIP")]),
    ("Training-free, logged graph", [("Obs", "linear LightGCN / r-AdjNorm"), ("EASE", "EASE"),
                                     ("GF-CF", "GF-CF")]),
    ("Training-free, debiased graph", [("IPS", "IPS adjacency (NAVIP-style)"), ("IPS+WC", "IPS + walk correction"),
                                       ("EASE-DR", "EASE on DR graph"), ("GF-CF-DR", "GF-CF on DR graph"),
                                       ("DR", "DR adjacency"), ("DRUP", "\\textbf{DRUP}"),
                                       ("DR-5hop", "DR adjacency, 5 hops"), ("DRUP-5hop", "\\textbf{DRUP}, 5 hops")]),
]


def accuracy():
    res = {ds: results(ds, prop) for ds, prop, _, _ in DS}
    best = {}
    for ds, prop, key, _ in DS:
        vals = sorted([r["test"][key][0] for r in res[ds].values() if key in r["test"]], reverse=True)
        best[ds] = vals[:2] if vals else []
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             " & Method & Coat nDCG@5 & Yahoo!\\,R3 nDCG@5 & KuaiRec nDCG@20\\\\", "\\midrule"]
    for group, rows in ROWS:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{group}}}}}\\\\")
        for mth, label in rows:
            cells = []
            for ds, prop, key, _ in DS:
                r = res[ds].get(mth)
                if r is None or key not in r["test"]:
                    cells.append("--")
                    continue
                m, s = r["test"][key]
                c = f"{m:.4f}$\\pm${s:.4f}"
                if best[ds] and abs(m - best[ds][0]) < 1e-12:
                    c = "\\textbf{" + c + "}"
                elif len(best[ds]) > 1 and abs(m - best[ds][1]) < 1e-12:
                    c = "\\underline{" + c + "}"
                cells.append(c)
            lines.append(f" & {label} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/accuracy.tex", "w").write("\n".join(lines) + "\n")
    # configuration counts
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Method & Coat & Yahoo!\\,R3 & KuaiRec\\\\", "\\midrule"]
    for group, rows in ROWS:
        for mth, label in rows:
            cells = [str(res[ds].get(mth, {}).get("n_configs", "--")) for ds, _, _, _ in DS]
            lines.append(f"{label} ({group.split(',')[0].lower()}) & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/configs.tex", "w").write("\n".join(lines) + "\n")


SIG = [("DRUP", "DR", "DR adjacency (effect of the correction)"),
       ("DRUP-5hop", "DR-5hop", "DR adjacency (effect of the correction), both 5 hops"),
       ("DRUP", "Obs", "linear LightGCN on the logged graph"),
       ("DRUP", "Impute", "imputation only"),
       ("DRUP", "GF-CF", "GF-CF"), ("DRUP", "EASE", "EASE"),
       ("DRUP", "GF-CF-DR", "GF-CF on DR graph"), ("DRUP", "EASE-DR", "EASE on DR graph"),
       ("DRUP", "LightGCN-pt", "LightGCN (pointwise)"), ("DRUP", "LightGCN", "LightGCN (BPR)"), ("DRUP", "r-AdjNorm", "r-AdjNorm"),
       ("DRUP", "NAVIP", "NAVIP"), ("DRUP", "BPR-MF", "MF (BPR)"), ("DRUP", "DR-MF", "DR-MF"),
       ("DRUP", "PDA", "PDA"), ("DRUP", "DR-LightGCN", "DR-LightGCN")]


def significance():
    lines = ["\\begin{tabular}{l" + "cc" * len(DS) + "}", "\\toprule",
             "DRUP against & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for _, _, _, nm in DS) + "\\\\",
             " & " + " & ".join("$\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$" for _ in DS) + "\\\\", "\\midrule"]
    tabs = {}
    for ds, prop, _, _ in DS:
        j = load(f"significance_{ds}_{prop}.json")
        tabs[ds] = {(r["ref"], r["other"]): r for r in j["rows"]} if j else {}
    for ref, other, label in SIG:
        cells = []
        for ds, _, _, _ in DS:
            r = tabs[ds].get((ref, other))
            if r is None:
                cells += ["--", "--"]
                continue
            cells.append(f"{r['diff']:+.4f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
            p = r["p_holm"]
            cells.append("$<10^{-3}$" if p < 1e-3 else f"{p:.3f}")
        name = label
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/significance.tex", "w").write("\n".join(lines) + "\n")


def mc():
    k3, k5 = load("mc_stress_K3_p1.0.json"), load("mc_stress_K5_p1.0.json")
    el = json.load(open("results/mc_elasticity.json")) if os.path.exists("results/mc_elasticity.json") else None
    bounds = load("mc_bounds.json")
    names = [("IPS", "IPS adjacency"), ("IPS+WC", "IPS + walk correction"), ("DR", "DR adjacency"),
             ("DRUP", "\\textbf{DRUP}")]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{rel.\\ $|$bias$|$, $K=3$} & \\multicolumn{2}{c}{rel.\\ $|$bias$|$, $K=5$} & Kendall $\\tau$ & elasticity\\\\",
             "Estimator & all & unexposed & all & unexposed & ($K=3$) & $\\eta$\\\\", "\\midrule"]
    elmap = {"IPS": "IPS", "IPS+WC": "IPS+WC", "DR": "DR", "DRUP": "DRUP"}
    if el:
        lines.append("Logged graph & -- & -- & -- & -- & -- & " + f"{el['results']['Obs']['elasticity_treated']:+.3f}\\\\")
    for k, lab in names:
        a = k3["results"]["indep"][k] if k3 else None
        b = k5["results"]["indep"][k] if k5 else None
        e = el["results"][elmap[k]]["elasticity_treated"] if el else None
        lines.append(f"{lab} & {fmt(a and a['rel_bias'], 3)} & {fmt(a and a['rel_bias_unexposed'], 3)} & "
                     f"{fmt(b and b['rel_bias'], 3)} & {fmt(b and b['rel_bias_unexposed'], 3)} & "
                     f"{fmt(a and a['kendall_tau'], 3)} & {fmt(e, 3) if e is None else f'{e:+.3f}'}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/mc.tex", "w").write("\n".join(lines) + "\n")
    # stress tests
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Scenario ($K=3$, rel.\\ $|$bias$|$ on unexposed candidates) & IPS & IPS + WC & DR & \\textbf{DRUP}\\\\",
             "\\midrule"]
    sc = [("indep", "independent exposures (Assumption~\\ref{as:unconf})"),
          ("fixedrow", "fixed-size slates per user (Proposition~\\ref{prop:dep})"),
          ("misprop_oracleY", "misspecified propensities, exact imputation"),
          ("misprop", "misspecified propensities, noisy imputation")]
    for key, lab in sc:
        if not k3 or key not in k3["results"]:
            continue
        r = k3["results"][key]
        lines.append(lab + " & " + " & ".join(fmt(r[m]["rel_bias_unexposed"], 3) for m in ("IPS", "IPS+WC", "DR", "DRUP")) + "\\\\")
    if k5:
        for key, lab in sc:
            if key not in k5["results"]:
                continue
            r = k5["results"][key]
            lines.append(lab.replace("(", "($K=5$; ", 1) if "(" in lab else lab + " ($K=5$)")
            lines[-1] = (lab + ", $K=5$") + " & " + " & ".join(fmt(r[m]["rel_bias_unexposed"], 3) for m in ("IPS", "IPS+WC", "DR", "DRUP")) + "\\\\"
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/mc_stress.tex", "w").write("\n".join(lines) + "\n")
    if bounds:
        lines = ["\\begin{tabular}{lcc}", "\\toprule", " & $\\tau=0.3$ & $\\tau=0.1$\\\\", "\\midrule",
                 "variance bound (b) / Monte-Carlo variance, median & " +
                 " & ".join(f"{bounds[t]['var_bound_over_mc_median']:.0f}" for t in ("0.3", "0.1")) + "\\\\",
                 "variance bound (b) / Monte-Carlo variance, minimum & " +
                 " & ".join(f"{bounds[t]['var_bound_over_mc_min']:.0f}" for t in ("0.3", "0.1")) + "\\\\",
                 "McDiarmid half-width (d) / empirical 95\\% quantile, median & " +
                 " & ".join(f"{bounds[t]['t_bound_over_q95_median']:.0f}" for t in ("0.3", "0.1")) + "\\\\",
                 "McDiarmid half-width (d) / empirical 95\\% quantile, minimum & " +
                 " & ".join(f"{bounds[t]['t_bound_over_q95_min']:.0f}" for t in ("0.3", "0.1")) + "\\\\",
                 "\\bottomrule", "\\end{tabular}"]
        open(f"{T}/bounds.tex", "w").write("\n".join(lines) + "\n")


def fat():
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Data & Operator & nDCG & PRU & ECB & Gini@$K$ & user gap [95\\% CI] & coverage\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "fairness" not in j:
            continue
        for mth, lab in (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP")):
            r = j["fairness"]["results"].get(mth)
            if r is None:
                continue
            kk = [k for k in r if k.startswith("ndcg@")][0]
            gk = [k for k in r if k.startswith("gini@")][0]
            ck = [k for k in r if k.startswith("coverage@")][0]
            ci = r.get("group_gap_ci95", [None, None])
            ecb = fmt(r["ECB"], 3) if ds == "kuairec" else "--"
            lines.append(f"{nm if mth == 'Obs' else ''} & {lab} & {r[kk]:.4f} & {r['PRU']:.3f} & {ecb} & "
                         f"{r[gk]:.3f} & {r['group_gap']:.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] & {r[ck]:.3f}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/fat.tex", "w").write("\n".join(lines) + "\n")
    # intervention
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{frozen nuisances} & \\multicolumn{2}{c}{re-fitted, known $p/2$} & \\multicolumn{2}{c}{re-fitted, re-estimated $\\hat p$}\\\\",
             "Data & Operator & rank shift & $\\eta$ & rank shift & $\\eta$ & rank shift & $\\eta$\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "intervene" not in j:
            continue
        iv = j["intervene"]
        for mth, lab in (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP")):
            cells = []
            for v in ("frozen", "refit_known", "refit_estimated"):
                r = iv.get(v, {}).get(mth, {})
                if not r:
                    cells += ["--", "--"]
                    continue
                cells.append(f"{r['shift']['mean']:+.4f}")
                cells.append(f"{r['eta']['mean']:+.3f}")
            lines.append(f"{nm if mth == 'Obs' else ''} & {lab} & " + " & ".join(cells) + "\\\\")
        fr = iv.get("frac_treated_logged_below_clip", {})
        if "DRUP" in fr:
            lines.append(f" & \\multicolumn{{7}}{{l}}{{\\footnotesize treated logged pairs below the DRUP clip after $do(p\\leftarrow p/2)$: {100 * fr['DRUP']:.0f}\\%}}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/intervene.tex", "w").write("\n".join(lines) + "\n")


def tau():
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Data & Operator, $\\tau$ & nDCG & rank shift & $\\eta$ & logged pairs clipped\\\\", "\\midrule"]
    for ds, _, key, nm in DS:
        j = load(f"tau_tradeoff_{ds}_frozen.json")
        if not j:
            continue
        first = True
        for r in j["rows"]:
            kk = [k for k in r if k.startswith("ndcg@")][0]
            lab = "logged graph" if r["method"] == "Obs" else f"DRUP, $\\tau={r['tau']:g}$"
            clip = "--" if r["method"] == "Obs" else f"{100 * r['frac_logged_pairs_clipped']:.0f}\\%"
            lines.append(f"{nm if first else ''} & {lab} & {r[kk]:.4f} & {r['shift']:+.4f} & "
                         f"{r['elasticity']:+.3f}$\\pm${r['elasticity_std']:.3f} & {clip}\\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/tau.tex", "w").write("\n".join(lines) + "\n")


def rerank():
    for ds, prop, key, nm in DS:
        j = load(f"rerank_{ds}_{prop}.json")
        if not j:
            continue
        K = j["K"]
        facs = [r["factor"] for r in j["results"]["DRUP"]]
        head = " & ".join("none" if f == float("inf") or f > 1e9 else f"$c={f:g}$" for f in facs)
        lines = ["\\begin{tabular}{l" + "c" * len(facs) + "}", "\\toprule", f"{nm} & {head}\\\\", "\\midrule"]
        for mth, lab in (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP")):
            rs = j["results"].get(mth, [])
            cells = []
            for r in rs:
                if "error" in r:
                    cells.append("infeas.")
                else:
                    cells.append(f"{r[f'ndcg@{K}']:.3f} ({r['gini']:.2f})")
            lines.append(f"{lab} & " + " & ".join(cells) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(f"{T}/rerank_{ds}.tex", "w").write("\n".join(lines) + "\n")


def frontier():
    lines = ["\\begin{tabular}{llccccc}", "\\toprule", "Data & Operator & Gini $\\le0.3$ & $\\le0.4$ & $\\le0.5$ & $\\le0.6$ & no cap\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"frontier_{ds}_{prop}.json")
        if not j:
            continue
        for mth, lab in (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP")):
            s = j["summary"].get(mth, {})
            cells = [fmt(s.get(c), 3) for c in ("0.3", "0.4", "0.5", "0.6", "1.0")]
            lines.append(f"{nm if mth == 'Obs' else ''} & {lab} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/frontier.tex", "w").write("\n".join(lines) + "\n")


def explain_attack_privacy():
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Data & max.\\ rel.\\ error & top-1 rank after 5 removals (attr.\\ / random / similarity) & no CF explanation & median CF size\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "explain" not in j:
            continue
        e = j["explain"]
        d5 = e["deletion_rank_of_top1"]
        lines.append(f"{nm} & {e['max_rel_error_completeness_additivity']:.1e} & "
                     f"{d5['DRUP-attribution']['5']:.2f} / {d5['random']['5']:.2f} / {d5['item-similarity']['5']:.2f} & "
                     f"{100 * e['frac_no_cf']:.0f}\\% & {fmt(e['minimal_cf_size_median'], 1)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/explain.tex", "w").write("\n".join(lines) + "\n")
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "attack" not in j:
            continue
        at = j["attack"]["results"]
        budgets = list(at["DRUP"]["hit@K"].keys())
        lines = ["\\begin{tabular}{l" + "c" * len(budgets) + "|" + "c" * len(budgets) + "}", "\\toprule",
                 f"{nm} & \\multicolumn{{{len(budgets)}}}{{c|}}{{realised hit rate, $F=$}} & \\multicolumn{{{len(budgets)}}}{{c}}{{certified fraction, $F=$}}\\\\",
                 " & " + " & ".join(budgets) + " & " + " & ".join(budgets) + "\\\\", "\\midrule"]
        for mth, lab in (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP")):
            r = at.get(mth)
            if r is None:
                continue
            lines.append(f"{lab} & " + " & ".join(f"{r['hit@K'][b]:.3f}" for b in budgets) + " & " +
                         " & ".join(f"{r['certified_frac'][b]:.2f}" for b in budgets) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(f"{T}/attack_{ds}.tex", "w").write("\n".join(lines) + "\n")
    # re-fitted nuisances
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Data & Operator & fake users $F$ & hit rate (frozen / re-fitted) & certified pairs & violated\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "attack_refit" not in j:
            continue
        for mth, r in j["attack_refit"].items():
            bs = r["budgets"]
            hits = ", ".join(f"{r['hit_frozen'][str(b)]:.3f}/{r['hit_refit'][str(b)]:.3f}" for b in bs)
            lines.append(f"{nm} & {mth} & {', '.join(map(str, bs))} & {hits} & {r['certified_pairs']} & {r['violations_under_refit']}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/attack_refit.tex", "w").write("\n".join(lines) + "\n")
    # privacy
    lines = ["\\begin{tabular}{llccccccc}", "\\toprule",
             "Data & rank & $\\epsilon=0.5$ & 1 & 2 & 4 & 8 & 16 & $\\infty$\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "privacy" not in j:
            continue
        pr = j["privacy"]["results"]
        for rk in ("8", "32", "128", "None"):
            cells = [f"{pr[e]['ndcg'][rk]:.3f}" for e in ("0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "inf") if e in pr]
            lines.append(f"{nm if rk == '8' else ''} & {'full' if rk == 'None' else rk} & " + " & ".join(cells) + "\\\\")
        lines.append(f" & \\multicolumn{{8}}{{l}}{{\\footnotesize non-private DRUP (all nuisances) {j['privacy']['non_private_full_nuisance_ndcg']:.3f}; popularity {j['privacy']['pop_ndcg']:.3f}; {j['privacy']['n_public']} public users}}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/privacy.tex", "w").write("\n".join(lines) + "\n")


def sparse():
    j = load("sparse_regime_kuairec.json")
    if not j:
        return
    lines = ["\\begin{tabular}{cccccccc}", "\\toprule",
             "thinning $q$ & density & DR & DRUP & DR 5 hops & DRUP 5 hops & top-20 overlap & Spearman\\\\", "\\midrule"]
    for r in j["rows"]:
        ov = min(p["overlap@20"] for p in r["agreement"])
        sp = min(p["spearman"] for p in r["agreement"])
        lines.append(f"{r['q']:g} & {100 * r['density']:.2f}\\% & " +
                     " & ".join(f"{r[m]['mean']:.4f}" for m in ("DR", "DRUP", "DR-5hop", "DRUP-5hop")) +
                     f" & {ov:.3f} & {sp:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/sparse.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    for f in (accuracy, significance, mc, fat, tau, rerank, frontier, explain_attack_privacy, sparse):
        try:
            f()
        except Exception as e:           # a missing result should not block the others
            print("skip", f.__name__, repr(e))
    print("tables written to", T)
