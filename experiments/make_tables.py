"""Generate the LaTeX tables of the paper from results/*.json."""

import json
import os

R = "results"
# propensity model selected on validation for DRUP on Yahoo!R3 (see
# experiments/select_yahoo.py); FAT / re-ranking runs use that model
YPROP = open(os.path.join("results", "yahoo_prop.txt")).read().strip() \
    if os.path.exists(os.path.join("results", "yahoo_prop.txt")) else "pop"
OUT = os.path.join("paper", "tables")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(mv, best=False, second=False):
    s = f"{mv[0]:.4f}$\\pm${mv[1]:.4f}"
    if best:
        return f"\\textbf{{{s}}}"
    if second:
        return f"\\underline{{{s}}}"
    return s


def accuracy_table():
    specs = [("coat", "given", ["ndcg@5", "recall@5", "ndcg@10"], "Coat"),
             ("kuairec", "pop", ["ndcg@20", "recall@20", "ndcg@50"], "KuaiRec"),
             ("yahoo", "sel", ["ndcg@5", "recall@5", "ndcg@10"], "Yahoo!\\,R3")]
    order = [("Pop", "filters"), ("Impute", "filters"), ("MF", "learned"), ("IPS-MF", "learned"),
             ("DR-MF", "learned"), ("LightGCN", "learned"), ("NAVIP", "learned"),
             ("DR-LightGCN", "learned"), ("Obs", "filters"), ("IPS", "filters"),
             ("IPS+WC", "filters"), ("DR", "filters"), ("DRUP", "filters"),
             ("DR-5hop", "filters5"), ("DRUP-5hop", "filters5"),
             ("DR-LR", "lr"), ("DRUP-LR", "lr")]
    names = {"Obs": "Linear LightGCN (logged graph)", "IPS": "IPS adjacency (NAVIP-style)",
             "IPS+WC": "IPS adjacency + WC", "DR": "DR adjacency", "DRUP": "\\textbf{DRUP (ours)}",
             "Impute": "Imputation only", "Pop": "Popularity", "NAVIP": "NAVIP-LightGCN",
             "DR-5hop": "DR adjacency, 5 hops", "DRUP-5hop": "\\textbf{DRUP, 5 hops (ours)}",
             "DR-LR": "DR adjacency, low-rank $\\hat Y$",
             "DRUP-LR": "\\textbf{DRUP, low-rank $\\hat Y$ (ours)}"}
    lines = []
    for ds, prop, mets, title in specs:
        fl = load(f"filters_{ds}_{prop}.json")
        le = load(f"learned_{ds}_{prop}_tuned.json") or load(f"learned_{ds}_{prop}.json")
        f5 = load(f"filters5_{ds}_{prop}.json") or load(f"filters5_{ds}_{prop}_lr.json")
        lr = load(f"filters_{ds}_{prop}_lr.json")
        if fl is None:
            continue
        res = {}
        for mth, src in order:
            if src == "lr":
                j, key = lr, mth[:-3]
            else:
                j, key = {"filters": fl, "learned": le, "filters5": f5}[src], mth
            if j and key in j["results"]:
                res[mth] = j["results"][key]["test"]
        lines.append(f"\\multicolumn{{{1 + len(mets)}}}{{l}}{{\\textit{{{title}}}}}\\\\")
        ranks = {}
        for met in mets:
            vals = sorted({v[met][0] for v in res.values()}, reverse=True)
            ranks[met] = vals
        for mth, _ in order:
            if mth not in res:
                continue
            cells = []
            for met in mets:
                v = res[mth][met]
                cells.append(fmt(v, v[0] == ranks[met][0], len(ranks[met]) > 1 and v[0] == ranks[met][1]))
            lines.append(f"{names.get(mth, mth)} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    hdr = ("\\begin{tabular}{lccc}\n\\toprule\nMethod & \\multicolumn{3}{c}{metrics (Coat, Yahoo!\\,R3: nDCG@5, Recall@5, nDCG@10;"
           " KuaiRec: nDCG@20, Recall@20, nDCG@50)}\\\\\n\\midrule\n")
    body = "\n".join(lines[:-1])
    return hdr + body + "\n\\bottomrule\n\\end{tabular}\n"


def mc_table():
    u = load("mc_unbiasedness.json")
    u5 = load("mc_unbiasedness_K5.json")
    e = load("mc_elasticity.json")
    names = {"naive-obs": "Obs", "IPS": "IPS", "IPS+WC": "IPS+WC", "DR": "DR", "DR+WC": "DRUP"}
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{$K=3$} & $K=5$ & \\\\",
             "Estimator & rel.\\,|bias| & frac.\\ $|z|>3$ & rel.\\,RMSE & rel.\\,|bias| & elasticity $\\eta$\\\\", "\\midrule"]
    for k, nm in names.items():
        r = u["results"][k]
        r5 = u5["results"][k]["rel_abs_bias"] if u5 else float("nan")
        ek = "Obs" if k == "naive-obs" else ("DRUP" if k == "DR+WC" else k)
        el = e["results"][ek]["elasticity_treated"] if e else float("nan")
        lines.append(f"{nm} & {r['rel_abs_bias']:.3f} & {r['frac_|z|>3']:.4f} & {r['rel_rmse']:.2f} & {r5:.3f} & {el:+.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def rerank_table():
    """nDCG@K and Gini@K of each operator under exposure caps (exact solver)."""
    out = []
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec"), ("yahoo", YPROP, "Yahoo!\\,R3")]:
        f = load(f"rerank_{ds}_{prop}.json")
        if not f:
            continue
        res = dict(f["results"])
        flr = load(f"rerank_{ds}_{prop}_lr.json")
        if flr:
            res["DRUP-LR"] = flr["results"]["DRUP"]
        facs = [r["factor"] for r in next(iter(res.values())) if "error" not in r]
        fl = ["none" if f_ == float("inf") else f"{f_:g}" for f_ in facs]
        block = ["\\begin{tabular}{l" + "c" * len(facs) + "}", "\\toprule",
                 f"\\textit{{{title}}}, cap factor $c$ & " + " & ".join(fl) + "\\\\", "\\midrule"]
        for mth, rs in res.items():
            rr = {r["factor"]: r for r in rs if "error" not in r}
            cells = [f"{rr[x][f'ndcg@{f[chr(75)]}']:.3f} ({rr[x]['gini']:.2f})" if x in rr else "--" for x in facs]
            block.append(mth + " & " + " & ".join(cells) + "\\\\")
        block += ["\\bottomrule", "\\end{tabular}"]
        out.append("\n".join(block) + "\n")
    return out


def fat_table():
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Data & Method & nDCG & ECB$\\downarrow$ & PRU$\\downarrow$ & Gini$\\downarrow$ & user gap$\\downarrow$ & rank shift under $do(p/2)$\\\\",
             "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec"), ("yahoo", YPROP, "Yahoo!\\,R3")]:
        f = load(f"fat_{ds}_{prop}.json")
        if not f or "fairness" not in f:
            continue
        rows_ = [(m_, r_, f.get("intervene", {}).get(m_, {})) for m_, r_ in f["fairness"]["results"].items()]
        flr = load(f"fat_{ds}_{prop}_lr.json")
        if flr and "fairness" in flr:
            rows_.append(("DRUP-LR", flr["fairness"]["results"]["DRUP"], flr.get("intervene", {}).get("DRUP", {})))
        for mth, r, iv in rows_:
            nd = [v for k, v in r.items() if k.startswith("ndcg")][0]
            gi = [v for k, v in r.items() if k.startswith("gini")][0]
            shift = f"{iv['mean']:+.4f}$\\pm${iv['std']:.4f}" if iv else "--"
            lines.append(f"{title} & {mth} & {nd:.4f} & {r['ECB']:.3f} & {r['PRU']:.3f} & {gi:.3f} & {r['group_gap']:.4f} & {shift}\\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def attack_table():
    """One sub-table per dataset (their attack budgets differ)."""
    blocks = []
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec"), ("yahoo", YPROP, "Yahoo!\\,R3")]:
        f = load(f"fat_{ds}_{prop}.json")
        if not f or "attack" not in f:
            continue
        res = f["attack"]["results"]
        budgets = list(next(iter(res.values()))["hit@K"].keys())
        nb = len(budgets)
        out = ["\\begin{tabular}{l" + "c" * (2 * nb) + "}", "\\toprule",
               f"\\multicolumn{{{1 + 2 * nb}}}{{l}}{{\\textit{{{title}}}: $K$={f['attack']['K']}, "
               f"fake profile size {f['attack']['L'] + 1}}}\\\\",
               f" & \\multicolumn{{{nb}}}{{c}}{{hit rate of target in top-$K$ $\\downarrow$}} & "
               f"\\multicolumn{{{nb}}}{{c}}{{certified fraction $\\uparrow$}}\\\\",
               "$F$ & " + " & ".join(budgets) + " & " + " & ".join(budgets) + "\\\\", "\\midrule"]
        for mth, r in res.items():
            out.append(mth + " & " + " & ".join(f"{r['hit@K'][b]:.3f}" for b in budgets) + " & "
                       + " & ".join(f"{r['certified_frac'][b]:.3f}" for b in budgets) + "\\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        blocks.append("\n".join(out))
    return blocks


def explain_privacy_table():
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Data & max rel.\\ error & mean min.\\ CF size & median & users without CF\\\\", "\\midrule"]
    priv = ["\\begin{tabular}{l" + "c" * 8 + "}", "\\toprule",
            "Data & $\\epsilon$=0.5 & 1 & 2 & 4 & 8 & 16 & $\\infty$ (clipped) & non-private / Pop\\\\", "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec"), ("yahoo", YPROP, "Yahoo!\\,R3")]:
        f = load(f"fat_{ds}_{prop}.json")
        if not f:
            continue
        if "explain" in f:
            e = f["explain"]
            lines.append(f"{title} & {e['max_rel_error_completeness_additivity']:.1e} & {e['minimal_cf_size_mean']:.2f} & "
                         f"{e['minimal_cf_size_median']:.0f} & {e['frac_no_cf']:.2f}\\\\")
        if "privacy" in f:
            p = f["privacy"]["results"]
            cells = [f"{p[k]['ndcg']:.4f}" for k in ["0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "inf"]]
            priv.append(f"{title} & " + " & ".join(cells) +
                        f" & {f['privacy']['non_private_ndcg']:.4f} / {f['privacy']['pop_ndcg']:.4f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    priv += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", "\n".join(priv) + "\n"


def frontier_table():
    """Best nDCG under Gini caps, recomputed from the stored grid points with
    dataset-specific caps (KuaiRec lists are far more concentrated)."""
    caps = {"coat": [0.4, 0.5, 0.6, 1.0], "kuairec": [0.97, 0.98, 0.99, 1.0]}
    lines = ["\\begin{tabular}{llcccc}", "\\toprule", "Data & Method & \\multicolumn{4}{c}{Gini cap}\\\\", "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec"), ("yahoo", YPROP, "Yahoo!\\,R3")]:
        f = load(f"frontier_{ds}_{prop}.json")
        if not f:
            continue
        lines.append(f" & & " + " & ".join(f"$\\le${c}" if c < 1 else "none" for c in caps[ds]) + "\\\\")
        for mth, pts in f["points"].items():
            cells = []
            for c in caps[ds]:
                v = max([p["ndcg"] for p in pts if p["gini"] <= c], default=None)
                cells.append("--" if v is None else f"{v:.4f}")
            lines.append(f"{title} & {mth} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    tabs = {"accuracy": accuracy_table(), "mc": mc_table(), "fat": fat_table(),
            "frontier": frontier_table()}
    for name, blk in zip(["rerank_coat", "rerank_kuairec"], rerank_table()):
        tabs[name] = blk
    for name, blk in zip(["attack_coat", "attack_kuairec"], attack_table()):
        tabs[name] = blk + "\n"
    tabs["explain"], tabs["privacy"] = explain_privacy_table()
    for k, v in tabs.items():
        with open(os.path.join(OUT, f"{k}.tex"), "w") as f:
            f.write(v)
        print("wrote", k)


if __name__ == "__main__":
    main()


def significance():
    """Paired t-tests of DRUP against every baseline on the splits both share."""
    from scipy.stats import ttest_rel
    rows = []
    for ds, prop, met, title in [("coat", "given", "ndcg@5", "Coat"), ("kuairec", "pop", "ndcg@20", "KuaiRec"),
                                 ("yahoo", "sel", "ndcg@5", "Yahoo!\\,R3")]:
        fl = load(f"filters_{ds}_{prop}.json")
        le = load(f"learned_{ds}_{prop}_tuned.json") or load(f"learned_{ds}_{prop}.json")
        lr = load(f"filters_{ds}_{prop}_lr.json")
        if fl is None:
            continue
        f5 = load(f"filters5_{ds}_{prop}_lr.json")
        variants = [("DRUP-LR", lr["results"]["DRUP"])] if lr else [("DRUP", fl["results"]["DRUP"])]
        if f5:
            variants.append(("DRUP-5hop-LR", f5["results"]["DRUP-5hop"]))
        allres = dict(fl["results"])
        if lr:
            allres["DRUP (additive $\\hat Y$)"] = allres.pop("DRUP")
            allres["Impute-LR"] = lr["results"]["Impute"]
        if le:
            allres.update(le["results"])
        learned = set(le["results"]) if le else set()
        pairs = [(o, r_o, mth, r) for o, r_o in variants for mth, r in allres.items()
                 if mth != "DRUP" and (o == variants[0][0] or mth in learned)]
        if f5 and lr:
            pairs.append(("DRUP-5hop-LR", f5["results"]["DRUP-5hop"], "DRUP-LR (3 hops)", lr["results"]["DRUP"]))
        for oname, r_o, mth, r in pairs:
            drup = [x[met] for x in r_o["per_split"]]
            ours = (oname,)
            other = [x[met] for x in r["per_split"]]
            k = min(len(drup), len(other))
            diff = sum(d - o for d, o in zip(drup[:k], other[:k])) / k
            p = ttest_rel(drup[:k], other[:k]).pvalue if k > 1 else float("nan")
            ptxt = "identical" if (p != p and abs(diff) < 1e-12) else f"{p:.3g}"
            rows.append(f"{title} & {ours[0]} vs {mth} & {k} & {diff:+.4f} & {ptxt}\\\\")
        rows.append("\\midrule")
    return "\n".join(["\\begin{tabular}{llccc}", "\\toprule",
                      "Data & comparison & splits & mean diff. & $p$ (paired $t$)\\\\",
                      "\\midrule"] + rows[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n"


if __name__ == "__main__":
    with open(os.path.join(OUT, "significance.tex"), "w") as f:
        f.write(significance())
    print(open(os.path.join(OUT, "significance.tex")).read())
