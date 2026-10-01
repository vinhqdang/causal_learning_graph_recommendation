"""LaTeX tables for the manuscript from results/v2/*.json (protocol v2)."""

import glob
import json
import os

import numpy as np

R = "results/v2"
R3 = "results/v3"          # round-3 results take precedence where they exist
T = "paper/tables"
os.makedirs(T, exist_ok=True)
DS = [("coat", "given", "ndcg@5", "Coat"), ("yahoo", "pop", "ndcg@5", "Yahoo!\\,R3"),
      ("kuairand", "pop", "ndcg@10", "KuaiRand"), ("kuairec", "pop", "ndcg@20", "KuaiRec")]


def load(path):
    for root in (R3, R):
        p = os.path.join(root, path)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def results(ds, prop):
    """Training-free: round-2 results updated by round-3 ones (same protocol, AUC
    added); trained: round-3 results only where they exist (round-2 models
    stopped on all splits jointly). Sample-split rows: mean and standard
    deviation over the draws of the split-off pairs."""
    out = {}
    for root in (R, R3):
        p = os.path.join(root, f"filters_{ds}_{prop}.json")
        if os.path.exists(p):
            out.update(json.load(open(p))["results"])
    j = load(f"learned_{ds}_{prop}.json")
    if j:
        out.update(j["results"])
    draws = [json.load(open(p))["results"] for p in
             sorted(glob.glob(os.path.join(R3, f"filters_{ds}_{prop}_split[1-9].json")))]
    for m in ("DR-split", "DRUP-split"):
        if m not in out or not all(m in d for d in draws) or not draws:
            continue
        r = dict(out[m])
        r["draw0"] = out[m]["test"]
        r["test"] = {}
        for k, (v0, _) in out[m]["test"].items():
            v = [v0] + [d[m]["test"][k][0] for d in draws if k in d[m]["test"]]
            r["test"][k] = [float(np.mean(v)), float(np.std(v, ddof=1))]
        r["n_draws"] = 1 + len(draws)
        out[m] = r
    return out


def loadfat(ds, prop, base=True):
    """Audit results of a dataset: round-2 file (if base), overlaid by the round-3 sections
    (_v3, and _v3x for sections that were run separately)."""
    out = {}
    names = ([f"fat_{ds}_{prop}.json"] if base else []) + [f"fat_{ds}_{prop}_v3.json", f"fat_{ds}_{prop}_v3x.json"]
    for nm in names:
        j = load(nm)
        if j:
            out.update(j)
    return out or None


def fmt(x, d=4):
    return "--" if x is None else f"{x:.{d}f}"


ROWS = [
    ("Non-personalised", [("Pop", "Popularity"), ("Impute", "Imputation only")]),
    ("Trained, pointwise loss", [("MF", "MF"), ("IPS-MF", "IPS-MF"), ("DR-MF", "DR-MF"), ("DR-JL", "DR-JL"),
                                 ("MRDR", "MRDR"), ("iALS", "iALS"), ("LightGCN-pt", "LightGCN"),
                                 ("DR-LightGCN", "DR-LightGCN")]),
    ("Trained, pairwise or sampled loss", [("BPR-MF", "MF (BPR)"), ("PDA", "PDA"), ("MACR", "MACR"),
                                           ("LightGCN", "LightGCN (BPR)"), ("r-AdjNorm", "r-AdjNorm"),
                                           ("NAVIP", "NAVIP"), ("SimGCL", "SimGCL")]),
    ("Training-free, logged graph", [("Obs", "linear LightGCN / r-AdjNorm"), ("EASE", "EASE"),
                                     ("GF-CF", "GF-CF"), ("BSPM", "BSPM")]),
    ("Training-free, debiased graph", [("IPS", "IPS adjacency (NAVIP-style)"), ("IPS+WC", "IPS + walk correction"),
                                       ("EASE-DR", "EASE on DR graph"), ("GF-CF-DR", "GF-CF on DR graph"),
                                       ("BSPM-DR", "BSPM on DR graph"),
                                       ("DR", "DR adjacency"), ("DRUP", "DRUP"),
                                       ("DR-5hop", "DR adjacency, 5 hops"), ("DRUP-5hop", "DRUP, 5 hops")]),
    ("Training-free, sample split (Assumption 2 holds)", [("DR-split", "DR adjacency, sample split"),
                                                          ("DRUP-split", "DRUP-split")]),
]


def accuracy():
    res = {ds: results(ds, prop) for ds, prop, _, _ in DS}
    nd = len(DS)

    def table(metric_of, fname, header, digits=4, std=True):
        best = {}
        for ds, prop, key, _ in DS:
            k = metric_of(key)
            vals = sorted([r["test"][k][0] for r in res[ds].values() if k in r["test"]], reverse=True)
            best[ds] = vals[:2]
        lines = ["\\begin{tabular}{ll" + "c" * nd + "}", "\\toprule",
                 " & Method & " + " & ".join(header(nm, key) for _, _, key, nm in DS) + "\\\\", "\\midrule"]
        for group, rows in ROWS:
            lines.append(f"\\multicolumn{{{nd + 2}}}{{l}}{{\\emph{{{group}}}}}\\\\")
            for mth, label in rows:
                cells = []
                for ds, prop, key, _ in DS:
                    r = res[ds].get(mth)
                    k = metric_of(key)
                    if r is None or k not in r["test"]:
                        cells.append("--")
                        continue
                    m, sd = r["test"][k]
                    c = f"{m:.{digits}f}$\\pm${sd:.{digits}f}" if std else f"{m:.{digits}f}"
                    if best[ds] and abs(m - best[ds][0]) < 1e-12:
                        c = "\\textbf{" + c + "}"
                    elif len(best[ds]) > 1 and abs(m - best[ds][1]) < 1e-12:
                        c = "\\underline{" + c + "}"
                    cells.append(c)
                lines.append(f" & {label} & " + " & ".join(cells) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(f"{T}/{fname}", "w").write("\n".join(lines) + "\n")
    table(lambda key: key, "accuracy.tex", lambda nm, key: f"{nm} {key.replace('ndcg', 'nDCG')}")
    table(lambda key: "auc", "accuracy_auc.tex", lambda nm, key: f"{nm} AUC", digits=3, std=False)
    table(lambda key: key.replace("ndcg", "recall"), "accuracy_recall.tex",
          lambda nm, key: f"{nm} {key.replace('ndcg', 'Recall')}", digits=3, std=False)
    # configuration counts
    lines = ["\\begin{tabular}{l" + "c" * nd + "}", "\\toprule",
             "Method & " + " & ".join(nm for _, _, _, nm in DS) + "\\\\", "\\midrule"]
    short = {"Trained, pointwise loss": "trained, pointwise", "Trained, pairwise or sampled loss": "trained, pairwise",
             "Training-free, sample split (Assumption 2 holds)": "training-free"}
    for group, rows in ROWS:
        for mth, label in rows:
            cells = [str(res[ds].get(mth, {}).get("n_configs", "--")) for ds, _, _, _ in DS]
            lines.append(f"{label} ({short.get(group, group.split(',')[0].lower())}) & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/configs.tex", "w").write("\n".join(lines) + "\n")


SIG = [("DRUP", "DR", "DRUP vs DR adjacency (the correction)"),
       ("DRUP-5hop", "DR-5hop", "DRUP vs DR adjacency, both 5 hops"),
       ("DRUP-split", "DR-split", "DRUP-split vs DR adjacency, sample split"),
       ("DRUP-split", "DRUP", "DRUP-split vs DRUP (cost of the split)"),
       ("DRUP", "Obs", "DRUP vs linear LightGCN, logged graph"),
       ("DRUP", "Impute", "DRUP vs imputation only"),
       ("DRUP", "GF-CF", "DRUP vs GF-CF"),
       ("DRUP", "BSPM", "DRUP vs BSPM"),
       ("DRUP", "GF-CF-DR", "DRUP vs GF-CF on DR graph"),
       ("DRUP", "iALS", "DRUP vs iALS"),
       ("DRUP", "DR-JL", "DRUP vs DR-JL"),
       ("DRUP", "LightGCN-pt", "DRUP vs LightGCN (pointwise)"), ("DRUP", "LightGCN", "DRUP vs LightGCN (BPR)"),
       ("DRUP", "r-AdjNorm", "DRUP vs r-AdjNorm"), ("DRUP", "NAVIP", "DRUP vs NAVIP"),
       ("DRUP", "SimGCL", "DRUP vs SimGCL"),
       ("DRUP-split", "iALS", "DRUP-split vs iALS"),
       ("DRUP-split", "DR-JL", "DRUP-split vs DR-JL"),
       ("DRUP-split", "LightGCN-pt", "DRUP-split vs LightGCN (pointwise)"),
       ("DRUP-split", "r-AdjNorm", "DRUP-split vs r-AdjNorm")]


def significance():
    lines = ["\\begin{tabular}{l" + "cc" * len(DS) + "}", "\\toprule",
             "Comparison & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for _, _, _, nm in DS) + "\\\\",
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
             ("DRUP", "DRUP")]
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
             "Scenario ($K=3$, rel.\\ $|$bias$|$ on unexposed candidates) & IPS & IPS + WC & DR & DRUP\\\\",
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
        j = loadfat(ds, prop)
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
                         f"{r[gk]:.3f} & {r['group0'] - r['group1']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] & {r[ck]:.3f}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/fat.tex", "w").write("\n".join(lines) + "\n")
    # intervention
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{frozen nuisances} & \\multicolumn{2}{c}{re-fitted, known $p/2$} & \\multicolumn{2}{c}{re-fitted, re-estimated $\\hat p$}\\\\",
             "Data & Operator & rank shift & $\\eta$ & rank shift & $\\eta$ & rank shift & $\\eta$\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = loadfat(ds, prop)
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
        j = load(f"fat_{ds}_{prop}_v3.json")        # round 3: fixed configuration, public constants
        if not j or "privacy" not in j:
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

def protocol():
    """Monte-Carlo gap between Assumption 2 and the cross-fitted protocol."""
    settings = [("", "_split_m40n60_p1", "40$\\times$60, 11"),
                ("_m60n120_p1", "_split_m60n120_p1", "60$\\times$120, 23"),
                ("_m60n120_p3", "_split_m60n120_p3", "60$\\times$120, 60")]
    rows = [("indep", "Yhat", "independent log"), ("xfit", "Yhat", "cross-fitted (protocol)"),
            ("xfit", "Wx", "cross-fitted, $W$-degrees"), ("split50", "Yhat", "sample split, 50\\%"),
            ("split20", "Yhat", "sample split, 20\\%")]
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{IPS edges} & \\multicolumn{4}{c}{DR edges}\\\\",
             "Model & Nuisances & uncorr. & DRUP & uncorr. & DRUP & $s$, data $c$ & $s$, $\\hat Y$ $c$\\\\", "\\midrule"]
    for base, split, lab in settings:
        jb, js = load(f"mc_protocol{base}.json"), load(f"mc_protocol{split}.json")
        if not jb:
            continue
        r = dict(jb["results"])
        if js:
            r.update(js["results"])
        first = True
        for src, deg, slab in rows:
            k = f"{src}/true/{deg}"
            if f"{k}/IPS" not in r:
                continue
            a, b = r[f"{k}/IPS"], r[f"{k}/DR"]
            cells = [a["rel_bias_T_uncorrected"], a["rel_bias_T_corrected"], b["rel_bias_T_uncorrected"],
                     b["rel_bias_T_corrected"], b["rel_bias_s_data_consts"], b["rel_bias_s_yhat_consts"]]
            lines.append(f"{lab if first else ''} & {slab} & " + " & ".join(f"{c:.2f}" for c in cells) + "\\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/protocol.tex", "w").write("\n".join(lines) + "\n")
    # full table for the smallest model: estimated propensities and in-sample nuisances
    j = load("mc_protocol.json")
    if not j:
        return
    r = j["results"]
    lines = ["\\begin{tabular}{lllcccccc}", "\\toprule",
             " & & & \\multicolumn{3}{c}{IPS edges} & \\multicolumn{3}{c}{DR edges}\\\\",
             "Propensities & Nuisances & degrees & uncorr. & DRUP & $s$, data $c$ & uncorr. & DRUP & $s$, data $c$\\\\", "\\midrule"]
    for prop, plab in (("true", "true"), ("pop", "estimated")):
        first = True
        for src, slab in (("indep", "independent log"), ("xfit", "cross-fitted"), ("insample", "same log")):
            for deg, dlab in (("Yhat", "$\\hat Y$"), ("Wx", "$W$")):
                cells = []
                for edge in ("IPS", "DR"):
                    x = r[f"{src}/{prop}/{deg}/{edge}"]
                    cells += [x["rel_bias_T_uncorrected"], x["rel_bias_T_corrected"], x["rel_bias_s_data_consts"]]
                lines.append(f"{plab if first else ''} & {slab} & {dlab} & " + " & ".join(f"{c:.2f}" for c in cells) + "\\\\")
                first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/protocol_full.tex", "w").write("\n".join(lines) + "\n")

def semisynth_table(fname, outname, props=("true",)):
    """Semi-synthetic KuaiRec: score errors and score-based decisions."""
    path = f"results/v3/{fname}"
    if not os.path.exists(path):
        return
    j = json.load(open(path))
    r = j["results"]
    src_lab = {"indep": "independent log", "xfit": "cross-fitted", "split": "sample split"}
    c_lab = {"fixed": "fixed", "yhat": "$\\hat Y$"}
    lines = ["\\begin{tabular}{llllccccc}", "\\toprule",
             " & & & & & \\multicolumn{2}{c}{utility error (\\%)} & & \\\\",
             "$p$ & Nuisances & $C$ & Estimator & RMSE $T$ & random & popular & regret & nDCG@20\\\\",
             "\\midrule"]
    for prop in props:
        for src in ("indep", "xfit", "split"):
            for cdeg in ("fixed", "yhat"):
                key = f"{src}/{prop}/{cdeg}"
                if key not in r:
                    continue
                first = True
                for nm in ("IPS", "IPS+WC", "DR", "DRUP"):
                    x = r[key][nm]

                    def u(k):
                        if k not in x:
                            return "--"
                        return f"{100 * x[k]['mean']:+.1f}$\\pm${100 * x[k]['se']:.1f}"

                    def v(k, d=3):
                        return f"{x[k]['mean']:.{d}f}" if k in x else "--"
                    lab = "true" if prop == "true" else "est."
                    rm = f"{x['rel_rmse_T']:.1f}" if cdeg == "fixed" else "--"
                    lines.append(f"{lab if first else ''} & {src_lab[src] if first else ''} & "
                                 f"{c_lab[cdeg] if first else ''} & {nm} & {rm} & "
                                 f"{u('util_rand')} & {u('util_pop')} & {v('cap_regret')} & {v('ndcg20')}\\\\")
                    first = False
                lines.append("\\midrule")
    k5 = [k for k in r if k.endswith("/K5")]
    for k in k5:
        lines.append(f"\\multicolumn{{9}}{{l}}{{Five hops (independent log, true $p$, fixed $C$): relative $|$bias$|$ "
                     f"of $T_5$, DR {r[k]['DR']['rel_bias_T5']:.2f}, DRUP {r[k]['DRUP']['rel_bias_T5']:.2f}.}}\\\\")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/{outname}", "w").write("\n".join(lines) + "\n")


def semisynth():
    semisynth_table("semisynth_kuairec_dense.json", "semisynth.tex")
    semisynth_table("semisynth_kuairec_dense.json", "semisynth_est.tex", props=("est",))
    semisynth_table("semisynth_kuairec_sparse.json", "semisynth_sparse.tex", props=("true", "est"))

def intervene_extra():
    """Audit controls: popularity, imputation only, DRUP-split, and the negative
    control in which the auditor's propensities are not updated."""
    rows = [("Pop/refit", "popularity (re-fitted)"), ("Impute/refit_known", "imputation only (re-fitted)"),
            ("IPS/misspecified", "IPS adjacency, propensities not updated"),
            ("DR/misspecified", "DR adjacency, propensities not updated"),
            ("DRUP/misspecified", "DRUP, propensities not updated"),
            ("DRUP-split/frozen", "DRUP-split, $\\hat Y$ and $C$ frozen"),
            ("DRUP-split/refit_known", "DRUP-split, re-fitted (known change)"),
            ("DRUP-split/refit_estimated", "DRUP-split, re-fitted (estimated)")]
    dss = [(ds, prop, nm) for ds, prop, _, nm in DS]
    js = {ds: loadfat(ds, prop, base=False) for ds, prop, _ in dss}
    js = {k: v for k, v in js.items() if v and "intervene_extra" in v}
    if not js:
        return
    cols = [(ds, nm) for ds, _, nm in dss if ds in js]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "$\\eta$ & " + " & ".join(nm for _, nm in cols) + "\\\\", "\\midrule"]
    for key, lab in rows:
        cells = []
        for ds, _ in cols:
            x = js[ds]["intervene_extra"]["results"].get(key)
            cells.append(f"{x['eta']['mean']:+.3f}" if x and "eta" in x else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    for nm, lab in (("Obs", "logged graph"), ("DR", "DR adjacency"), ("DRUP", "DRUP"), ("DRUP-split", "DRUP-split")):
        cells = [f"{js[ds]['intervene_extra']['spearman_with_imputation'].get(nm, float('nan')):.2f}" for ds, _ in cols]
        lines.append(f"rank corr.\\ with $\\hat Y$: {lab} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/intervene_extra.tex", "w").write("\n".join(lines) + "\n")


BUDGET_TRAINED = ["LightGCN-pt", "LightGCN", "r-AdjNorm", "NAVIP", "DR-LightGCN", "PDA", "BPR-MF", "DR-MF", "MF", "IPS-MF",
                  "DR-JL", "MRDR", "iALS", "MACR", "SimGCL"]


def budget():
    """Expected test score of DRUP under the tuning budget of the best trained
    model, counted in configurations and in validation evaluations."""
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{configurations} & \\multicolumn{2}{c}{evaluations} & & \\\\",
             "Data & best trained model & $k$ & DRUP & $k$ & DRUP & DRUP, all & trained\\\\",
             "\\midrule"]

    def at(curve, k):
        ks = sorted(int(x) for x in curve)
        kk = max([x for x in ks if x <= k] or [ks[0]])
        return curve[str(kk)], kk

    for ds, prop, key, nm in DS:
        j, je = load(f"budget_{ds}.json"), load(f"budget_eval_{ds}.json")
        if not j or not je:
            continue
        res = results(ds, prop)
        tr = [(res[m]["test"][key][0], m) for m in BUDGET_TRAINED if m in res and m in j]
        if not tr:
            continue
        bt, m = max(tr)
        kc = j[m]["n_configs"]
        ke = je[m]["n_configs"]
        dc, kc2 = at(j["DRUP"]["curve"], kc)
        de, ke2 = at(je["DRUP"]["curve"], ke)
        lines.append(f"{nm} & {NAMES.get(m, m)} & {kc} & {dc:.4f} & {ke} & {de:.4f} & "
                     f"{res['DRUP']['test'][key][0]:.4f} & {bt:.4f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/budget.tex", "w").write("\n".join(lines) + "\n")


NAMES = {"LightGCN-pt": "LightGCN, pointwise", "LightGCN": "LightGCN, BPR", "r-AdjNorm": "r-AdjNorm"}


if __name__ == "__main__":
    for f in (accuracy, significance, mc, fat, tau, rerank, frontier, explain_attack_privacy, sparse, protocol, budget, semisynth, intervene_extra):
        try:
            f()
        except Exception as e:           # a missing result should not block the others
            print("skip", f.__name__, repr(e))
    print("tables written to", T)


def compact():
    """Compact main-text versions of the re-ranking, attack and privacy tables."""
    ops = (("Obs", "logged graph"), ("IPS", "IPS adjacency"), ("DR", "DR adjacency"), ("DRUP", "DRUP"))
    # re-ranking: no cap, c = 5, c = 2
    lines = ["\\begin{tabular}{llccc}", "\\toprule", "Data & Operator & no cap & $c=5$ & $c=2$\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"rerank_{ds}_{prop}.json")
        if not j:
            continue
        K = j["K"]
        for mth, lab in ops:
            rs = {r["factor"]: r for r in j["results"].get(mth, []) if "error" not in r}
            cells = []
            for f in (float("inf"), 5.0, 2.0):
                r = next((v for k, v in rs.items() if (k == f) or (f == float("inf") and k > 1e9)), None)
                cells.append("--" if r is None else f"{r[f'ndcg@{K}']:.3f} ({r['gini']:.2f})")
            lines.append(f"{nm if mth == 'Obs' else ''} & {lab} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/rerank_compact.tex", "w").write("\n".join(lines) + "\n")
    # attack: hit rate and certified fraction at three budgets per dataset
    pick = {"coat": ["1", "5", "20"], "yahoo": ["1", "5", "100"], "kuairec": ["1", "5", "200"]}
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Data & Operator & \\multicolumn{3}{c}{realised hit rate} & \\multicolumn{3}{c}{certified fraction}\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}.json")
        if not j or "attack" not in j:
            continue
        bs = pick[ds]
        lines.append(f"{nm} & $F=$ & " + " & ".join(bs) + " & " + " & ".join(bs) + "\\\\")
        for mth, lab in ops:
            r = j["attack"]["results"].get(mth)
            if r is None:
                continue
            lines.append(f" & {lab} & " + " & ".join(f"{r['hit@K'][b]:.3f}" for b in bs) + " & " +
                         " & ".join(f"{r['certified_frac'][b]:.2f}" for b in bs) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(f"{T}/attack_compact.tex", "w").write("\n".join(lines) + "\n")
    # privacy: fixed rank 32
    lines = ["\\begin{tabular}{lccccccccc}", "\\toprule",
             "Data & $\\epsilon=0.5$ & 1 & 2 & 4 & 8 & 16 & $\\infty$ & all nuisances & popularity\\\\", "\\midrule"]
    for ds, prop, key, nm in DS:
        j = load(f"fat_{ds}_{prop}_v3.json")        # round 3: fixed configuration, public constants
        if not j or "privacy" not in j:
            j = load(f"fat_{ds}_{prop}.json")
        if not j or "privacy" not in j:
            continue
        pr = j["privacy"]["results"]
        cells = [f"{pr[e]['ndcg']['32']:.3f}" for e in ("0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "inf") if e in pr]
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {j['privacy']['non_private_full_nuisance_ndcg']:.3f} & {j['privacy']['pop_ndcg']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/privacy_compact.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    compact()
