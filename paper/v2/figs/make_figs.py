"""Figures for manuscript v2, Section 6. Reads only result files; writes PDFs next to this script.

Figure 2 (fig_forest.pdf): per-split test difference between the final method and the strongest
baseline of each dataset; paired means and 95% CI half-widths are parsed from results/v4/final_summary.txt.
Figure 3 (fig_ablation.pdf): ablation means and standard deviations from the stored JSON files.
Figure 4 (fig_walk.pdf): left, matched-configuration DRUP minus DR from results/v3/matched_config_real.txt;
right, semi-synthetic DRUP minus DR nDCG@20 from results/v3/rank_dose.json (same statistic as
experiments/rank_dose_table.py).

Run from any directory:  python3 paper/v2/figs/make_figs.py
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments"))
import final_config as fc  # noqa: E402

ld = lambda f: json.load(open(f))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def combine_per_split(ds):
    """Per-split test values of the validated combination (as final_analysis.combine)."""
    key = fc.KEY[ds]
    rs = [ld(f)["results"][n] for f, n in fc.FINAL_JOBS[ds] if os.path.exists(f)]
    out = []
    for s in range(len(rs[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in rs]))
        out.append(rs[j]["per_split"][s][key])
    return np.array(out)


def baseline_block(ds_name):
    """Per-split values and test means of all baselines of one dataset, from the SETS part files."""
    metric, files, names = fc.SETS[ds_name]
    base = {}
    for f in files:
        if os.path.exists(f):
            for n, v in ld(f)["results"].items():
                base.setdefault(n, v)  # first occurrence wins, as the test means of final_analysis
    return base, metric


def parse_final_summary():
    """Paired mean diff (x1e-3) and CI half-width per dataset and baseline, from final_summary.txt."""
    txt = open(ROOT / "results/v4/final_summary.txt").read()
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^(Coat|Yahoo|KuaiRand|KuaiRec):", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"vs (\S+)\s+([\d.]+) \(\s*\d+ cfgs\)\s+diff\s+([+-][\d.]+)e-3 \[\s*([\d.]+)\]", line)
        if m and cur:
            out[(cur, m.group(1))] = (float(m.group(3)), float(m.group(4)))
    return out


# ---------- Figure 2: per-split forest ----------
def fig_forest():
    summary = parse_final_summary()
    rows = [("Coat", "iALS"), ("Yahoo", "r-AdjNorm"), ("KuaiRand", "r-AdjNorm"), ("KuaiRec", "DR-JL")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (ds, base_name) in enumerate(rows):
        y = len(rows) - 1 - i
        base, _ = baseline_block(ds)
        mine = combine_per_split(ds.lower())
        b = np.array([v[fc.KEY[ds.lower()]] for v in base[base_name]["per_split"]])
        assert len(b) == len(mine), (ds, len(b), len(mine))
        assert abs(b.mean() - base[base_name]["test"][fc.KEY[ds.lower()]][0]) < 1e-3, ds
        d = 1000 * (mine - b)
        ax.scatter(d, np.full(len(d), y) + np.linspace(-0.12, 0.12, len(d)), s=10, color="0.55", zorder=2)
        mean, half = summary[(ds, base_name)]
        ax.errorbar([mean], [y], xerr=[[half], [half]], fmt="D", color="C0", ms=5, capsize=3, zorder=3)
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{ds} vs {bn}" for ds, bn in rows][::-1])
    ax.set_xlabel(r"Final minus baseline ($\times 10^{-3}$ nDCG); dots: splits, diamonds: paired mean [95% CI]")
    fig.tight_layout()
    fig.savefig(HERE / "fig_forest.pdf")
    plt.close(fig)


# ---------- Figure 3: ablation bars ----------
def fig_ablation():
    def mean_sd(path, name):
        m, s = ld(path)["results"][name]["test"]["ndcg@5"]
        return m, s

    lgn_coat = mean_sd(ROOT / "results/v3/learned_coat_given.json", "LightGCN")
    lgn_yahoo = None
    for f in sorted(glob.glob(str(ROOT / "results/v3/learned_yahoo_pop*.json"))):
        if "LightGCN" in ld(f)["results"]:
            lgn_yahoo = mean_sd(f, "LightGCN")
            break
    coat = [("LightGCN", lgn_coat),
            ("Q", mean_sd("results/v4/learned_coat_drbpr.json", "DR-BPR-LGN")),
            ("Q(Yhat)", mean_sd("results/v4/learned_coat_abl_DR-BPR-yhat.json", "DR-BPR-yhat")),
            ("Q(pop)", mean_sd("results/v4/learned_coat_abl_DR-BPR-pop.json", "DR-BPR-pop")),
            ("Q(shuffled)", mean_sd("results/v4/learned_coat_abl_DR-BPR-shuf.json", "DR-BPR-shuf"))]
    yahoo = [("LightGCN", lgn_yahoo),
             ("Q", mean_sd("results/v4/learned_yahoo_drbpr_lr3e-3.json", "DR-BPR-LGN-1")),
             ("Q(pop)", mean_sd("results/v4/learned_yahoo_abl_DR-BPR-pop.json", "DR-BPR-pop")),
             ("Q(shuffled)", mean_sd("results/v4/learned_yahoo_abl_DR-BPR-shuf.json", "DR-BPR-shuf"))]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [5, 4]})
    for ax, data, title, lo, hi in [(axes[0], coat, "Coat (nDCG@5)", 0.55, 0.6),
                                    (axes[1], yahoo, "Yahoo! R3 (nDCG@5)", 0.64, 0.69)]:
        names = [d[0] for d in data]
        means = [d[1][0] for d in data]
        sds = [d[1][1] for d in data]
        colors = ["0.6"] + ["C0"] * (len(data) - 1)
        ax.bar(range(len(data)), means, yerr=sds, color=colors, capsize=3, width=0.65, ecolor="k")
        ax.axhline(means[0], color="k", lw=0.7, ls="--")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("test nDCG (mean, sd over splits)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_ablation.pdf")
    plt.close(fig)


# ---------- Figure 4: walk correction ----------
def parse_matched():
    txt = open(ROOT / "results/v3/matched_config_real.txt").read().splitlines()
    out, cur = {}, None
    for line in txt:
        m = re.match(r"^(coat|yahoo) \(", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = re.match(r"^\s*([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)", line)
        if m and cur:
            out[cur].append((float(m.group(1)), float(m.group(2))))
    return out


def fig_walk():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    mc = parse_matched()
    for name, color in [("coat", "C0"), ("yahoo", "C1")]:
        b = np.array([x[0] for x in mc[name]])
        d = np.array([x[1] for x in mc[name]])
        ax1.plot(b, d, marker="o", ms=3, color=color, label=name.capitalize())
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\beta$ (log scale)")
    ax1.set_ylabel(r"DRUP $-$ DR, mean test nDCG@5 ($\times 10^{-3}$)")
    ax1.set_title("Matched configurations (real data)", fontsize=8)
    ax1.legend(frameon=False, fontsize=7)

    r = ld(ROOT / "results/v3/rank_dose.json")
    betas = ["0.3", "1.0", "3.0", "10.0", "1000.0"]
    xs = np.array([float(b) for b in betas])
    for c, v in r.items():
        means, halves = [], []
        for bb in betas:
            dd = np.array([x[f"DRUP|{bb}|ndcg"] - x[f"DR|{bb}|ndcg"] for x in v["rows"]])
            means.append(1000 * dd.mean())
            halves.append(1000 * stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)))
        style = "-" if "_g1.0" not in c else "--"
        ax2.errorbar(xs * 1.04 ** (list(r).index(c) - 2.5), means, yerr=halves, fmt=style + "o", ms=2.5,
                     lw=0.8, capsize=1.5, elinewidth=0.6, color=f"C{list(r).index(c) % 10}",
                     label=c.replace("_d0.13", "").replace("_", " "))
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\beta$ (log scale)")
    ax2.set_ylabel(r"DRUP $-$ DR, nDCG@20 ($\times 10^{-3}$)")
    ax2.set_title("Semi-synthetic KuaiRec, fixed degree weights", fontsize=8)
    ax2.legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_walk.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_ablation()
    fig_walk()
    print("wrote", sorted(p.name for p in HERE.glob("fig_*.pdf")))
