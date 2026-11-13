#!/usr/bin/env python3
"""Figures of the v8 manuscript (paper/v8/figs/*.pdf), built from the result JSON files through experiments/v13_tables.py.

fig_forest.pdf   per-split paired differences of LEAF-LGN+DAX against each comparator of the main family (dots: the
                 five splits; diamond: pooled per-user mean with the 95% half-width).
fig_ablate.pdf   DAX ablations: gain over the ablation-run base with the 95% half-width; dots: the five splits.
fig_seeds.pdf    training seeds 0 to 3: LEAF-LGN+DAX mean nDCG@10 (left) and within-seed gain over LEAF-LGN (right).

Missing inputs are skipped and listed in the caption-less panel as "not run" labels; nothing is typed by hand.
Usage: python3 experiments/v13_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v13_tables as T  # noqa: E402

FIGS = os.path.join(T.ROOT, "paper", "v8", "figs")
INK = "#222222"       # primary ink
MUTED = "#7a7a7a"     # secondary ink (zero line, grid)
ACCENT = "#1f5a8c"    # single hue for the highlighted series

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42,
})


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)


def fig_forest(c):
    H, M = c["H"], c["M"]
    keys = list(H["FAMILY_V12"])
    rows = []
    for k in keys:
        r = T.P(M["LEAF-LGN+DAX"], M.get(k))
        rows.append((k, r))
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    for yi, (k, r) in zip(y, rows):
        if r is None:
            ax.text(1.0, yi, "not run yet (TBD)", color=MUTED, va="center", ha="left", fontsize=7.5)
        else:
            for d in r["split_diffs"]:
                ax.plot(d, yi, "o", ms=3.2, color=MUTED, mec="none", zorder=2)
            ax.errorbar(r["diff"], yi, xerr=r["half"], fmt="D", ms=4.5, color=ACCENT, ecolor=ACCENT,
                        elinewidth=1.2, capsize=2.5, zorder=3)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([T.DISPLAY.get(k, k) for k, _ in rows])
    ax.set_xlabel(r"LEAF-LGN+DAX minus comparator, nDCG@10 $\times 10^{-3}$ (per-user mean, 95\% half-width)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.5, zorder=0)
    _save(fig, "fig_forest.pdf")


def fig_ablate(c):
    base, arms = T.ablation_records()
    if base is None or arms.get("full") is None:
        print("fig_ablate: skipped (dax_ablate.json incomplete)")
        return
    order = ["full", "no_tab", "no_life_prior", "no_tag_prior", "additive", "shuffle_tags", "shuffle_items"]
    gains = []
    for k in order:
        if arms.get(k) is None:
            gains.append((k, None))
        else:
            gains.append((k, T.P(arms[k], base)))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    for xi, (k, r) in zip(x, gains):
        if r is None:
            ax.text(xi, 0, "TBD", ha="center", va="bottom", color=MUTED, fontsize=7.5)
            continue
        col = ACCENT if k == "full" else MUTED
        ax.bar(xi, r["diff"], width=0.62, color=col, alpha=0.85, zorder=2)
        ax.errorbar(xi, r["diff"], yerr=r["half"], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
        ax.plot([xi] * len(r["split_diffs"]), r["split_diffs"], "o", ms=2.6, color=INK, mec="none", zorder=4)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x)
    short = {"full": "full", "no_tab": "no kappa", "no_life_prior": "no life\nprior", "no_tag_prior": "no tag\nprior",
             "additive": "additive", "shuffle_tags": "shuffle\nacross", "shuffle_items": "shuffle\nwithin tag"}
    ax.set_xticklabels([short[k] for k in order], fontsize=7)
    ax.set_ylabel(r"gain over LEAF-LGN base, nDCG@10 $\times 10^{-3}$")
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    _save(fig, "fig_ablate.pdf")


def fig_seeds(c):
    H, M, N = c["H"], c["M"], c["N"]
    entries = [(0, M["LEAF-LGN+DAX"], M["LEAF-LGN"])]
    for s in (1, 2, 3):
        path = f"results/v11/c_seeds/lgn_s{s}.json"
        entries.append((s, H["method_from"](path, "LEAF-LGN+DAX"), H["method_from"](path, "LEAF-LGN")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6))
    for s, dax, base in entries:
        if dax is None or base is None:
            a1.text(s, 0.476, "TBD", ha="center", color=MUTED, fontsize=7.5)
            continue
        mu, sd = H["mean_sd"](dax)
        a1.errorbar(s, mu, yerr=sd, fmt="o", ms=4, color=ACCENT, ecolor=INK, elinewidth=0.9, capsize=2.5)
        r = H["pair"](dax, base, N)
        a2.errorbar(s, r["diff"], yerr=r["half"], fmt="D", ms=4, color=ACCENT, ecolor=INK, elinewidth=0.9, capsize=2.5)
    a1.set_xticks([0, 1, 2, 3])
    a1.set_xlabel("training seed")
    a1.set_ylabel("LEAF-LGN+DAX nDCG@10 (mean, SD over splits)")
    a1.grid(axis="y", color="#e5e5e5", lw=0.5)
    a2.axhline(0, color=INK, lw=0.7)
    a2.set_xticks([0, 1, 2, 3])
    a2.set_xlabel("training seed")
    a2.set_ylabel(r"gain over LEAF-LGN, $\times 10^{-3}$ (95\% hw)")
    a2.grid(axis="y", color="#e5e5e5", lw=0.5)
    fig.tight_layout()
    _save(fig, "fig_seeds.pdf")


def main():
    c = T.context()
    fig_forest(c)
    fig_ablate(c)
    fig_seeds(c)
    print("FIGS written to", FIGS)


if __name__ == "__main__":
    main()
