"""R0: reconcile the stored results with a replay of their validation/test banks.

For every (result file, method) that has a bank, the training-time selection of run_learned.py is replayed
(replay_bank in v7_common.py) with the patience recorded in the bank. The replayed test mean is compared with
the stored mean (results[...]["test"][key][0]). The replay is also tested against the other patience values,
to see which rule reproduces the stored chosen (configuration, epoch) of every split. Training-free operators
(results/v2/bank_filters_*) are replayed by plain validation argmax.

Gaps above 0.002 are reported with the cause that the checks support. No model is trained.
Output: results/v6/reconcile.txt, results/v6/reconcile.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_common import (OUT, Report, bank_path, bank_rows, cfg_key, load_json, replay_bank, same_cfg, select)  # noqa: E402

GAP = 0.002
CANDIDATE_P = [None, 1, 2, 3, 5, 10]   # None = no stop (the bank's full sequence)


def stored_chosen_match(stored_chosen, idx, groups, E, trained):
    """True if, for every split, the stored chosen configuration (with epoch) equals the replayed one."""
    for s, ch in enumerate(stored_chosen):
        n = idx[s]
        cfg = dict(groups[n][0])
        if trained:
            cfg["epoch"] = int(E[n, s])
            if not same_cfg({k: v for k, v in ch.items()}, cfg):
                return False
        else:
            if not same_cfg({k: v for k, v in ch.items() if k != "epoch"}, cfg):
                return False
    return True


def classify(gap, rec, trained):
    if abs(gap) <= GAP:
        return "match"
    if rec["match_patience"] and rec["recorded_patience"] in rec["match_patience"]:
        return "unexplained (patience reproduces selections but mean differs)"
    if rec["match_patience"]:
        return f"stored selections reproduced only with patience {rec['match_patience']} (bank records {rec['recorded_patience']})"
    if not rec["chosen_in_bank"]:
        return "stored selections are not configurations of the bank (bank from another run or merged jobs)"
    if rec["n_configs_stored"] is not None and rec["n_configs_stored"] != rec["n_configs_bank"]:
        return (f"configuration count differs (stored {rec['n_configs_stored']}, bank {rec['n_configs_bank']}): "
                "bank and result file from different runs")
    if rec["val_max_abs_diff"] > 1e-9:
        return "validation values differ at the stored selection: bank and result file from different runs"
    if rec["test_max_abs_diff"] > 1e-9:
        return "test values differ at identical selections: evaluation or seed mismatch"
    return "unexplained"


def main():
    rep = Report()
    rep("R0 reconciliation of stored results with bank replay (experiments/v7_reconcile.py)")
    rep(f"gap threshold {GAP}; replay candidates for patience: {CANDIDATE_P}")
    rep("")
    records = []

    # ---- learned models (training with patience) -----------------------------------------------------
    res_files = sorted(p for p in glob.glob("results/v[345]/learned_*.json") if ".partial" not in p)
    for rp in res_files:
        js = load_json(rp)
        if "results" not in js:
            continue
        for method, r in js["results"].items():
            bk = bank_rows(rp, method)
            if bk is None:
                records.append({"file": rp, "method": method, "status": "no bank next to this result file"})
                continue
            bargs, rows = bk
            key = bank_key = load_json(bank_path(rp)).get("key")
            S = len(r["per_split"])
            P_rec = bargs.get("patience")
            groups, V, T, E = replay_bank(rows, S, P_rec)
            idx, vsel, tsel = select(V, T)
            stored_mean = r["test"][key][0]
            replay_mean = float(tsel.mean())
            # which rules reproduce the stored selection?
            matches = []
            for P in CANDIDATE_P:
                g2, V2, T2, E2 = replay_bank(rows, S, P)
                i2, _, _ = select(V2, T2)
                if stored_chosen_match(r["chosen"], i2, g2, E2, trained=True):
                    matches.append(P)
            chosen_in_bank = stored_chosen_match(r["chosen"], idx, groups, E, trained=True)
            vmax = float(np.max(np.abs(vsel - np.array(r["val_best"]))))
            tmax = float(np.max(np.abs(tsel - np.array([ps[key] for ps in r["per_split"]]))))
            rec = {"file": rp, "method": method, "status": "replayed", "metric": key,
                   "stored_mean": stored_mean, "replay_mean": replay_mean, "gap": replay_mean - stored_mean,
                   "recorded_patience": P_rec, "match_patience": matches, "chosen_in_bank": chosen_in_bank,
                   "n_configs_stored": r.get("n_configs"), "n_configs_bank": len(groups),
                   "val_max_abs_diff": vmax, "test_max_abs_diff": tmax,
                   "seed_check_mean": (r.get("seed_check") or {}).get("mean")}
            rec["cause"] = classify(rec["gap"], rec, trained=True)
            records.append(rec)

    # ---- training-free operators (no patience; argmax of validation) ----------------------------------
    for bp in sorted(glob.glob("results/v2/bank_filters_*.json")):
        bj = load_json(bp)
        ds = bj["args"]["dataset"]
        prop = bj["args"]["prop"]
        rp = f"results/v2/filters_{ds}_{prop}.json"
        key = bj["key"]
        if not os.path.exists(rp):
            rep(f"NOTE: {rp} missing for {bp}")
            continue
        res = load_json(rp)["results"]
        for method, rows in bj["banks"].items():
            if method not in res:
                continue
            r = res[method]
            S = len(r["per_split"])
            groups, V, T, E = replay_bank(rows, S, None)
            idx, vsel, tsel = select(V, T)
            stored_mean = r["test"][key][0]
            replay_mean = float(tsel.mean())
            chosen_ok = stored_chosen_match(r["chosen"], idx, groups, E, trained=False)
            vmax = float(np.max(np.abs(vsel - np.array(r["val_best"]))))
            tmax = float(np.max(np.abs(tsel - np.array([ps[key] for ps in r["per_split"]]))))
            rec = {"file": rp, "method": method, "status": "replayed (training-free)", "metric": key,
                   "stored_mean": stored_mean, "replay_mean": replay_mean, "gap": replay_mean - stored_mean,
                   "recorded_patience": None, "match_patience": [None] if chosen_ok else [],
                   "chosen_in_bank": chosen_ok, "n_configs_stored": r.get("n_configs"),
                   "n_configs_bank": len(groups), "val_max_abs_diff": vmax, "test_max_abs_diff": tmax,
                   "seed_check_mean": None}
            rec["cause"] = classify(rec["gap"], rec, trained=False)
            records.append(rec)

    # ---- v3 training-free results have no bank: listed, not reconciled --------------------------------
    v3f = sorted(glob.glob("results/v3/filters_*.json"))
    nobank = [p for p in v3f if "_split" not in p]

    # ---- report ---------------------------------------------------------------------------------------
    rep("method-level table (replayed mean vs stored mean, metric of the bank)")
    rep(f"{'file':42s} {'method':14s} {'stored':>8s} {'replay':>8s} {'gap':>9s}  patience-match  cause")
    flagged = []
    for rec in records:
        if rec["status"] == "no bank next to this result file":
            rep(f"{rec['file']:42s} {rec['method']:14s}  -- no bank: not reconciled")
            continue
        pm = ",".join("none" if p is None else str(p) for p in rec["match_patience"]) or "-"
        rep(f"{rec['file']:42s} {rec['method']:14s} {rec['stored_mean']:8.4f} {rec['replay_mean']:8.4f} "
            f"{rec['gap']:+9.5f}  {pm:14s}  {rec['cause'] if abs(rec['gap']) > GAP else ''}")
        if abs(rec["gap"]) > GAP:
            flagged.append(rec)
    rep("")
    n_ok = sum(1 for r in records if r.get("gap") is not None and abs(r["gap"]) <= GAP)
    n_rep = sum(1 for r in records if r.get("gap") is not None)
    rep(f"replayed methods: {n_rep}; within {GAP}: {n_ok}; above {GAP}: {len(flagged)}")
    rep(f"result files of v3 with training-free results but no bank (not reconciled): {len(nobank)} files")
    rep("")
    rep("GAPS above threshold")
    for rec in flagged:
        rep(f"  {rec['file']} :: {rec['method']}: stored {rec['stored_mean']:.4f}, replay {rec['replay_mean']:.4f} "
            f"(gap {rec['gap']:+.4f}); cause: {rec['cause']}; seed_check mean {rec['seed_check_mean']}")
    if not flagged:
        rep("  none")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reconcile.json"), "w") as f:
        json.dump({"threshold": GAP, "records": records}, f, indent=1, default=float)
    rep.save("reconcile.txt")


if __name__ == "__main__":
    main()
