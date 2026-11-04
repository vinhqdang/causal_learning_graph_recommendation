"""A3: user bootstrap of the rank distribution of the leading methods (DESIGN.md F4).

For each dataset the methods are the eight with the highest mean validation score among the non-control
methods (selection on validation, not on test). Their per-user, per-split test values come from the stored
selected runs (pooled families use the job chosen per split by validation). The bootstrap draws users with
replacement, and for Coat (ten splits) also splits with replacement; 2,000 resamples, the same draw for all
methods. For each resample the methods are ranked by mean test score (1 = best). Output: P(rank = r) per method.
The leader is the method with the highest mean validation score; its P(rank 1) is reported.
Writes results/v6/bootstrap_rank.txt, results/v6/bootstrap_rank.json and paper/v4/figs/rank_distribution.pdf.
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import KEY, NAME, OUT, Report, load_all, val_mean  # noqa: E402

B = 2000
TOP = 8
FIG = "paper/v4/figs/rank_distribution.pdf"


def matrix(m, users, S):
    """Users x splits matrix of per-user test values (nan where the user is not a test user of the split)."""
    idx = {u: i for i, u in enumerate(users)}
    M = np.full((len(users), S), np.nan)
    for s, pu in enumerate(m.per_user):
        for u, v in pu.items():
            if u in idx:
                M[idx[u], s] = v
    return M


def bootstrap(mats, resample_splits, rng):
    """Returns P matrix [methods x ranks] and the rank of each method per resample."""
    names = list(mats)
    U, S = next(iter(mats.values())).shape
    K = len(names)
    ranks = np.zeros((B, K), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in range(B):
            ui = rng.integers(0, U, U)
            si = rng.integers(0, S, S) if resample_splits else np.arange(S)
            stats = np.array([np.nanmean(mats[n][np.ix_(ui, si)]) for n in names])
            order = np.argsort(-stats, kind="stable")
            ranks[b, order] = np.arange(1, K + 1)
    P = np.stack([(ranks == r).mean(axis=0) for r in range(1, K + 1)], axis=1)
    return names, P


def main():
    rep = Report()
    rep(f"A3 user bootstrap of rank distributions (experiments/v7_bootstrap_rank.py), B={B}")
    rep("Methods: top 8 by mean validation score among non-control methods. Ranks by mean test (1 = best).")
    rep("")
    out = {}
    panels = []
    for ds in ["coat", "yahoo", "kuairand", "kuairec"]:
        key = KEY[ds]
        M = load_all(ds, rep)
        pool = [m for m in M.values() if m.role != "control" and m.per_user is not None]
        pool.sort(key=lambda m: -val_mean(m))
        sel = pool[:TOP]
        rep(f"=== {NAME[ds]} ({key}) ===")
        if len(sel) < 2:
            rep("  fewer than two methods available; skipped")
            rep("")
            continue
        users = sorted(set.intersection(*[set(u for pu in m.per_user for u in pu) for m in sel]))
        S = len(sel[0].per_split)
        # keep only users that are present in every split of every selected method
        mats = {m.name: matrix(m, users, S) for m in sel}
        keep = np.all(np.stack([~np.isnan(mats[m.name]).all(axis=1) for m in sel]), axis=0)
        mats = {n: M_[keep] for n, M_ in mats.items()}
        rng = np.random.default_rng(0)
        names, P = bootstrap(mats, resample_splits=(ds == "coat"), rng=rng)
        leader = max(sel, key=val_mean).name
        rep(f"  users {int(keep.sum())}, splits {S}, resample splits: {ds == 'coat'}")
        rep(f"  {'method':24s} {'val':>7s} {'mean':>7s}  " + " ".join(f"{'r' + str(r):>6s}" for r in range(1, len(names) + 1)))
        rows = []
        for i, n in enumerate(names):
            mean_test = float(np.nanmean(mats[n]))
            val = val_mean(M[n])
            rows.append({"method": n, "val_mean": val, "test_mean": mean_test, "P_rank": P[i].tolist()})
            rep(f"  {n[:24]:24s} {val:7.4f} {mean_test:7.4f}  " + " ".join(f"{p:6.3f}" for p in P[i]))
        li = names.index(leader)
        rep(f"  leader by validation: {leader}; P(rank 1) = {P[li, 0]:.3f}")
        rep("")
        out[ds] = {"users": int(keep.sum()), "splits": S, "resample_splits": ds == "coat",
                   "leader_by_validation": leader, "P_rank1_leader": float(P[li, 0]), "rows": rows}
        panels.append((ds, names, P))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bootstrap_rank.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    rep.save("bootstrap_rank.txt")

    # ---- figure: one heatmap per dataset, sequential single hue, cells labelled ------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not panels:
        return
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol + 2.2, 0.42 * TOP + 1.4), squeeze=False)
    ink = "#1f2328"
    for ax, (ds, names, P) in zip(axes[0], panels):
        im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(NAME[ds], fontsize=9, color=ink)
        ax.set_xticks(range(P.shape[1]))
        ax.set_xticklabels([str(r) for r in range(1, P.shape[1] + 1)], fontsize=7, color=ink)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ds == panels[0][0] else [""] * len(names), fontsize=7, color=ink)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else ink)
        ax.set_xlabel("rank", fontsize=7, color=ink)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("probability", fontsize=7, color=ink)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Bootstrap rank distribution of the leading methods (2,000 user resamples)", fontsize=9, color=ink)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    rep(f"figure written: {FIG}")
    rep.save("bootstrap_rank.txt")


if __name__ == "__main__":
    main()
