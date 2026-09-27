"""Generate the LaTeX tables of the paper from results/*.json."""

import json
import os

R = "results"
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
             ("kuairec", "pop", ["ndcg@20", "recall@20", "ndcg@50"], "KuaiRec")]
    order = [("Pop", "filters"), ("Impute", "filters"), ("MF", "learned"), ("IPS-MF", "learned"),
             ("DR-MF", "learned"), ("LightGCN", "learned"), ("NAVIP", "learned"),
             ("DR-LightGCN", "learned"), ("Obs", "filters"), ("IPS", "filters"),
             ("IPS+WC", "filters"), ("DR", "filters"), ("DRUP", "filters")]
    names = {"Obs": "Linear LightGCN (logged graph)", "IPS": "IPS adjacency (NAVIP-style)",
             "IPS+WC": "IPS adjacency + WC", "DR": "DR adjacency", "DRUP": "\\textbf{DRUP (ours)}",
             "Impute": "Imputation only", "Pop": "Popularity", "NAVIP": "NAVIP-LightGCN"}
    lines = []
    for ds, prop, mets, title in specs:
        fl = load(f"filters_{ds}_{prop}.json")
        le = load(f"learned_{ds}_{prop}.json")
        if fl is None:
            continue
        res = {}
        for mth, src in order:
            j = fl if src == "filters" else le
            if j and mth in j["results"]:
                res[mth] = j["results"][mth]["test"]
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
    hdr = ("\\begin{tabular}{lccc}\n\\toprule\nMethod & \\multicolumn{3}{c}{metrics (Coat: nDCG@5, Recall@5, nDCG@10;"
           " KuaiRec: nDCG@20, Recall@20, nDCG@50)}\\\\\n\\midrule\n")
    body = "\n".join(lines[:-1])
    return hdr + body + "\n\\bottomrule\n\\end{tabular}\n"


def mc_table():
    u = load("mc_unbiasedness.json")
    e = load("mc_elasticity.json")
    names = {"naive-obs": "Obs", "IPS": "IPS", "IPS+WC": "IPS+WC", "DR": "DR", "DR+WC": "DRUP"}
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Estimator & rel.\\,|bias| & frac.\\ $|z|>3$ & rel.\\,RMSE & exposure elasticity $\\eta$\\\\", "\\midrule"]
    for k, nm in names.items():
        r = u["results"][k]
        ek = "Obs" if k == "naive-obs" else ("DRUP" if k == "DR+WC" else k)
        el = e["results"][ek]["elasticity_treated"] if e else float("nan")
        lines.append(f"{nm} & {r['rel_abs_bias']:.3f} & {r['frac_|z|>3']:.4f} & {r['rel_rmse']:.2f} & {el:+.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def fat_table():
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Data & Method & nDCG & ECB$\\downarrow$ & PRU$\\downarrow$ & Gini$\\downarrow$ & user gap$\\downarrow$ & rank shift under $do(p/2)$\\\\",
             "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec")]:
        f = load(f"fat_{ds}_{prop}.json")
        if not f or "fairness" not in f:
            continue
        for mth, r in f["fairness"]["results"].items():
            nd = [v for k, v in r.items() if k.startswith("ndcg")][0]
            gi = [v for k, v in r.items() if k.startswith("gini")][0]
            iv = f.get("intervene", {}).get(mth, {})
            shift = f"{iv['mean']:+.4f}$\\pm${iv['std']:.4f}" if iv else "--"
            lines.append(f"{title} & {mth} & {nd:.4f} & {r['ECB']:.3f} & {r['PRU']:.3f} & {gi:.3f} & {r['group_gap']:.4f} & {shift}\\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def attack_table():
    out = []
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec")]:
        f = load(f"fat_{ds}_{prop}.json")
        if not f or "attack" not in f:
            continue
        res = f["attack"]["results"]
        budgets = list(next(iter(res.values()))["hit@K"].keys())
        out.append(f"\\multicolumn{{{1 + 2 * len(budgets)}}}{{l}}{{\\textit{{{title}}} (K={f['attack']['K']}, profile size {f['attack']['L'] + 1})}}\\\\")
        out.append("Method & " + " & ".join(f"$F$={b}" for b in budgets) + " & " + " & ".join(f"$F$={b}" for b in budgets) + "\\\\")
        for mth, r in res.items():
            out.append(mth + " & " + " & ".join(f"{r['hit@K'][b]:.3f}" for b in budgets) + " & "
                       + " & ".join(f"{r['certified_frac'][b]:.3f}" for b in budgets) + "\\\\")
        out.append("\\midrule")
    if not out:
        return ""
    ncol = out[1].count("&") + 1
    hdr = ["\\begin{tabular}{l" + "c" * (ncol - 1) + "}", "\\toprule",
           f" & \\multicolumn{{{(ncol - 1) // 2}}}{{c}}{{hit rate of target in top-$K$ $\\downarrow$}} & "
           f"\\multicolumn{{{(ncol - 1) // 2}}}{{c}}{{certified fraction $\\uparrow$}}\\\\", "\\midrule"]
    return "\n".join(hdr + out[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def explain_privacy_table():
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Data & max rel.\\ error & mean min.\\ CF size & median & users without CF\\\\", "\\midrule"]
    priv = ["\\begin{tabular}{l" + "c" * 8 + "}", "\\toprule",
            "Data & $\\epsilon$=0.5 & 1 & 2 & 4 & 8 & 16 & $\\infty$ (clipped) & non-private / Pop\\\\", "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec")]:
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
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Data & Method & Gini$\\le$0.3 & $\\le$0.4 & $\\le$0.5 & $\\le$0.6 & any\\\\", "\\midrule"]
    for ds, prop, title in [("coat", "given", "Coat"), ("kuairec", "pop", "KuaiRec")]:
        f = load(f"frontier_{ds}_{prop}.json")
        if not f:
            continue
        for mth, s in f["summary"].items():
            cells = [("--" if s[c] is None else f"{s[c]:.4f}") for c in ["0.3", "0.4", "0.5", "0.6", "1.0"]]
            lines.append(f"{title} & {mth} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    tabs = {"accuracy": accuracy_table(), "mc": mc_table(), "fat": fat_table(),
            "attack": attack_table(), "frontier": frontier_table()}
    tabs["explain"], tabs["privacy"] = explain_privacy_table()
    for k, v in tabs.items():
        with open(os.path.join(OUT, f"{k}.tex"), "w") as f:
            f.write(v)
        print("wrote", k)


if __name__ == "__main__":
    main()
