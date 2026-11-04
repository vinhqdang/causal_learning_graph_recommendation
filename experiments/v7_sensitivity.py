"""A1: sensitivity of the conclusions to evaluation choices, from the stored banks (no training).

Per dataset and method (trained methods with a bank, and training-free operators with a bank):
  (a) validation-selected test score, replayed with the patience rule of the bank (= R0 reconciled value);
  (b) per-split test oracle: best test over every evaluated (configuration, epoch) row of each split;
  (c) configuration-level oracle: the configuration whose mean test (each configuration at its
      validation-selected epoch) is highest;
  (d) expected maximum test score vs number of configurations k (budget_curve.expected, Dodge et al. 2019);
  (e) random subgrid: 1,000 draws of k_min configurations per method, the share of draws in which the sign of a
      primary comparison flips (k_min = the smaller grid of the pair);
  (f) alternative metric: the sign of the same comparison on another metric of the stored selections.
Pairs are the comparisons of DESIGN.md Section 2.3 that exist for each dataset (v7_common.primary_pairs).
Inputs that do not exist yet (X-runs) are skipped with a printed note.
Output: results/v6/sensitivity.txt, results/v6/sensitivity.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7_common import (KEY, NAME, OUT, Report, load_all, primary_pairs, select, val_mean)  # noqa: E402
from budget_curve import expected  # noqa: E402

DRAWS = 1000
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ALT = {"coat": "ndcg@10", "yahoo": "ndcg@10", "kuairand": "ndcg@20", "kuairec": "ndcg@50"}


def oracle_split(m, matched=True):
    """Per split: the best test over the evaluated rows of all jobs.
    matched=True: only the rows that the stop rule of the bank could select from (the candidate set of the
    validation selection); matched=False: every row the bank holds, including epochs after the stop."""
    best = np.full(m.S, -np.inf)
    for groups, _, _, _, P in m.tables:
        for cfg, rows in groups:
            cur = [-1.0] * m.S
            bad = [0] * m.S
            done = [False] * m.S
            for r in rows:
                for s in range(m.S):
                    if done[s] and matched:
                        continue
                    best[s] = max(best[s], r["test"][s])
                    if not matched:
                        continue
                    v = r["val"][s]
                    if v > cur[s]:
                        cur[s], bad[s] = v, 0
                    else:
                        bad[s] += 1
                        if P is not None and bad[s] >= P:
                            done[s] = True
    return best


def curve_at(m, ks):
    """Expected maximum test vs k on the pooled configuration set."""
    _, V, T = m.table()
    cfgs = [[(V[n, s], T[n, s]) for s in range(m.S)] for n in range(V.shape[0])]
    N = V.shape[0]
    return {k: expected(cfgs, k) for k in sorted(set(k for k in ks if k <= N))}


def budgets_for(N):
    return sorted(set([b for b in BUDGETS if b <= N] + [N]))


def sign_changes(ks, diffs):
    out = []
    for i in range(1, len(ks)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]):
            out.append(ks[i])
    return out


def main():
    rep = Report()
    rep("A1 sensitivity of conclusions to evaluation choices (experiments/v7_sensitivity.py)")
    rep("Reconciled selection = replay of the bank with its recorded patience (R0). Test values are split means.")
    rep("")
    out = {}
    for ds in ["coat", "yahoo", "kuairand", "kuairec"]:
        key = KEY[ds]
        rep(f"=== {NAME[ds]} ({key}) ===")
        M = load_all(ds, rep)
        rows = []
        rep(f"{'method':24s} {'role':12s} {'cfgs':>5s} {'stored':>8s} {'valsel':>8s} {'oracle':>8s} {'infl':>7s} "
            f"{'cfgOr':>8s} {'inflC':>7s} {'E[k=8]':>8s} {'E[N]':>8s}")
        ds_out = {"methods": {}, "pairs": [], "alt_metric": ALT[ds]}
        for name, m in M.items():
            if m.tables is None:
                continue
            _, V, T = m.table()
            N = V.shape[0]
            _, _, tsel = select(V, T)
            valsel = float(tsel.mean())
            stored = m.stored_mean(key)
            orc = float(oracle_split(m).mean())
            orc_all = float(oracle_split(m, matched=False).mean())
            cfg_orc = float(T.mean(axis=1).max())
            cur = curve_at(m, budgets_for(N))
            e8 = cur.get(min(8, N))
            eN = cur.get(N)
            rows.append((name, m.role, N, stored, valsel, orc, cfg_orc))
            rep(f"{name:24s} {m.role:12s} {N:5d} {stored:8.4f} {valsel:8.4f} {orc:8.4f} {orc - valsel:+7.4f} "
                f"{cfg_orc:8.4f} {cfg_orc - valsel:+7.4f} {e8:8.4f} {eN:8.4f}")
            ds_out["methods"][name] = {"role": m.role, "n_configs": N, "stored_mean": stored,
                                       "val_selected_replay": valsel, "test_oracle_split": orc,
                                       "inflation_oracle": orc - valsel, "config_oracle": cfg_orc,
                                       "inflation_config_oracle": cfg_orc - valsel,
                                       "expected_max": {str(k): v for k, v in cur.items()}}
        trained = [r for r in rows if r[1] not in ("training-free",)]
        if trained:
            infl = np.array([r[5] - r[4] for r in trained])
            rep(f"  inflation (oracle - validation-selected), trained methods: mean {infl.mean():+.4f}, "
                f"range {infl.min():+.4f} to {infl.max():+.4f}")
            ds_out["inflation_trained_mean"] = float(infl.mean())
            ds_out["inflation_trained_range"] = [float(infl.min()), float(infl.max())]
        rep("")
        # ---- pairwise conclusions -----------------------------------------------------------------
        rep("  primary comparisons (A minus B; valsel = validation-selected test; oracle = per-split test oracle)")
        rep(f"  {'fam':4s} {'A':24s} {'B':20s} {'valsel':>9s} {'oracle':>9s} {'cfgOr':>9s} {'E@kmin':>9s} "
            f"{'kmin':>5s} {'flip%':>6s} {'alt':>9s} {'altSign':>8s} {'signchg k':>14s}")
        for fam, a, b in primary_pairs(ds, M):
            if a not in M or b not in M:
                rep(f"  {fam:4s} {a} vs {b}: missing method, skipped")
                continue
            A, B = M[a], M[b]
            pair = {"family": fam, "A": a, "B": b}
            if A.tables is not None and B.tables is not None:
                _, VA, TA = A.table()
                _, VB, TB = B.table()
                _, _, tA = select(VA, TA)
                _, _, tB = select(VB, TB)
                full = float((tA - tB).mean())
                pair["valsel_diff"] = full
                oA, oB = oracle_split(A), oracle_split(B)
                pair["oracle_diff"] = float((oA - oB).mean())
                pair["oracle_all_diff"] = float((oracle_split(A, False) - oracle_split(B, False)).mean())
                pair["config_oracle_diff"] = float(TA.mean(axis=1).max() - TB.mean(axis=1).max())
                kmin = min(VA.shape[0], VB.shape[0])
                ks = budgets_for(max(VA.shape[0], VB.shape[0]))
                ks = sorted(set(ks + [kmin]))
                cA, cB = curve_at(A, ks), curve_at(B, ks)
                kk = [k for k in ks if k in cA and k in cB]
                dd = [cA[k] - cB[k] for k in kk]
                pair["expected_diff_by_k"] = {str(k): d for k, d in zip(kk, dd)}
                pair["expected_diff_kmin"] = cA[kmin] - cB[kmin]
                pair["kmin"] = kmin
                pair["sign_changes_k"] = sign_changes(kk, dd)
                # (e) random subgrid of k_min configurations per method
                rng = np.random.default_rng(0)
                fs = np.sign(full) if full != 0 else 1.0
                flips = 0
                for _ in range(DRAWS if abs(full) > 1e-12 else 0):
                    ia = rng.choice(VA.shape[0], kmin, replace=False)
                    ib = rng.choice(VB.shape[0], kmin, replace=False)
                    _, _, ta = select(VA[ia], TA[ia])
                    _, _, tb = select(VB[ib], TB[ib])
                    if np.sign((ta - tb).mean()) != fs:
                        flips += 1
                pair["flip_share"] = flips / DRAWS if abs(full) > 1e-12 else None
            else:
                pair["valsel_diff"] = None
            # (f) alternative metric of the stored selections (same sign as the main metric?)
            alt = ALT[ds]
            pair["stored_diff"] = float(np.mean([x[key] - y[key] for x, y in zip(A.per_split, B.per_split)]))
            if all(alt in r for r in A.per_split) and all(alt in r for r in B.per_split):
                pair["alt_diff"] = float(np.mean([x[alt] - y[alt] for x, y in zip(A.per_split, B.per_split)]))
                pair["alt_same_sign"] = bool(np.sign(pair["alt_diff"]) == np.sign(pair["stored_diff"]))
            ds_out["pairs"].append(pair)
            f4 = lambda v: f"{v:+.4f}" if v is not None else "-"
            if pair.get("flip_share") is not None or pair.get("oracle_diff") is not None:
                sg = ("same" if pair.get("alt_same_sign") else "flip") if "alt_same_sign" in pair else "-"
                rep(f"  {fam:4s} {a[:24]:24s} {b[:20]:20s} {f4(pair['valsel_diff']):>9s} {f4(pair['oracle_diff']):>9s} "
                    f"{f4(pair['config_oracle_diff']):>9s} {f4(pair['expected_diff_kmin']):>9s} {pair['kmin']:5d} "
                    f"{(100 * pair['flip_share']) if pair['flip_share'] is not None else float('nan'):6.1f} "
                    f"{f4(pair.get('alt_diff')):>9s} {sg:>8s} "
                    f"{str(pair['sign_changes_k']):>14s}")
            else:
                rep(f"  {fam:4s} {a[:24]:24s} {b[:20]:20s} stored selections only (no bank): stored diff "
                    f"{pair['stored_diff']:+.4f}")
        rep("")
        out[ds] = ds_out
        rep("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    rep.save("sensitivity.txt")


if __name__ == "__main__":
    main()
