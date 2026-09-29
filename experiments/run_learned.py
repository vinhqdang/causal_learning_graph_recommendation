"""Trained baselines on the same splits / protocol as run_filters.py.

Every configuration is trained once on the MNAR log (the training data are
the same for every split); after each epoch it is evaluated on the validation
and test part of every split, and each split selects its own
(configuration, epoch) by validation nDCG. This is per-split tuning with
early stopping at the cost of a single training run per configuration.
Per-user test metrics of the selected model are stored for user-level tests.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.data import load_coat, load_yahoo, split_test  # noqa: E402
from drup.estimation import baseline_imputation, clip_propensity, get_propensity  # noqa: E402
from drup.learned import MF, LightGCN, train  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

# name: (model kind, loss, extra grid)
METHODS = {
    "MF": ("mf", "naive", {}),
    "IPS-MF": ("mf", "ips", {}),
    "DR-MF": ("mf", "dr", {}),
    "BPR-MF": ("mf", "bpr", {}),
    "PDA": ("mf", "pda", {"gamma": [0.1, 0.2]}),
    "LightGCN": ("lgn", "bpr", {}),
    "LightGCN-pt": ("lgn", "naive", {}),
    "r-AdjNorm": ("lgn", "bpr", {"r": [0.3, 0.7]}),
    "NAVIP": ("navip", "bpr", {}),
    "DR-LightGCN": ("lgn", "dr", {}),
}


def build(kind, m, n, d, O, Y, P, layers, r):
    if kind == "mf":
        return MF(m, n, d)
    eu, ei = torch.nonzero(O * Y > 0, as_tuple=True)
    w = torch.ones(len(eu)) if kind == "lgn" else (1.0 / P[eu, ei]).float()
    return LightGCN(m, n, d, eu, ei, w, layers=layers, r=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coat")
    ap.add_argument("--prop", default="given")
    ap.add_argument("--seeds", type=int, default=10, help="number of val/test splits")
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--dims", type=int, nargs="+", default=[64])
    ap.add_argument("--layers", type=int, nargs="+", default=[2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--pairs_per_epoch", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed_check", type=int, default=3,
                    help="re-train the split-0 selection with this many seeds (0: skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump_bank", default=None,
                    help="write every (configuration, epoch)'s validation and test score per split "
                         "instead of the result file (for tuning-budget curves)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.dataset == "coat":
        d = load_coat()
        ks, key, by = (5, 10), "ndcg@5", "entry"
    elif a.dataset == "yahoo":
        d = load_yahoo()
        ks, key, by = (5, 10), "ndcg@5", "user"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        ks, key, by = (10, 20, 50), "ndcg@20", "user"
    O, Y = d["O"].float(), d["Y"].float()
    m, n = O.shape
    P = get_propensity(d, O.double(), Y.double(), a.prop).float()
    P = clip_propensity(P, a.floor)
    Yhat = baseline_imputation(O.double(), Y.double(), P.double(), lam=5.0).float()
    pop = (O * Y).sum(0)
    item_pop = (pop / pop.max()).clamp_min(1.0 / pop.max())
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(d["test"], a.frac_val, s, by=by) for s in range(a.seeds)]
    out = a.out or f"results/v2/learned_{a.dataset}_{a.prop}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results = json.load(open(out))["results"] if os.path.exists(out) else {}

    def run(kind, loss, cfg, seed, record):
        model = build(kind, m, n, cfg["dim"], O, Y, P, cfg["layers"], cfg.get("r", 0.5))

        def on_epoch(mod, ep):
            S = mod.full_scores(rows)
            vals = [evaluate(S, v, ks, row_of)[key] for v, _ in splits]
            if record is not None:
                record(vals, S, ep)
            return float(np.mean(vals))
        return train(model, O, Y, P, Yhat, loss, cfg["lr"], cfg["wd"], a.epochs, a.batch, on_epoch,
                     patience=a.patience, pairs_per_epoch=a.pairs_per_epoch, seed=seed,
                     item_pop=item_pop, gamma=cfg.get("gamma", 0.1))

    dumps = {}
    for name in a.methods:
        kind, loss, extra = METHODS[name]
        t0 = time.time()
        grid_keys = ["lr", "wd", "dim", "layers"] + list(extra)
        grid_vals = [a.lrs, a.wds, a.dims, a.layers if kind != "mf" else [0]] + [extra[k] for k in extra]
        best = [(-1.0, None, None, None) for _ in splits]       # (val, cfg, test agg, test per-user)
        bank = []
        n_cfg = 0
        for vals in itertools.product(*grid_vals):
            cfg = dict(zip(grid_keys, vals))
            n_cfg += 1

            def record(vs, S, ep, cfg=cfg):
                if a.dump_bank:
                    bank.append({"cfg": dict(cfg, epoch=ep), "val": vs,
                                 "test": [evaluate(S, t, ks, row_of)[key] for _, t in splits]})
                    return
                for s, v in enumerate(vs):
                    if v > best[s][0]:
                        agg, pu = evaluate(S, splits[s][1], ks, row_of, per_user=key)
                        best[s] = (v, dict(cfg, epoch=ep), agg, pu)
            tc = time.time()
            run(kind, loss, cfg, 0, record)
            print(f"  {name} {cfg} done [{time.time() - tc:.0f}s] best split0 val {best[0][0]:.4f}",
                  flush=True)
        if a.dump_bank:
            dumps[name] = bank
            os.makedirs(os.path.dirname(a.dump_bank), exist_ok=True)
            with open(a.dump_bank, "w") as f:
                json.dump({"args": vars(a), "key": key, "banks": dumps}, f)
            print(f"{name:12s} dumped {len(bank)} (cfg, epoch) rows [{time.time() - t0:.0f}s]", flush=True)
            continue
        per_split = [b[2] for b in best]
        agg = {k: (float(np.mean([r[k] for r in per_split])), float(np.std([r[k] for r in per_split])))
               for k in per_split[0]}
        res = {"test": agg, "chosen": [b[1] for b in best], "per_split": per_split,
               "val_best": [b[0] for b in best],
               "per_user": [{str(u): v for u, v in b[3].items()} for b in best], "n_configs": n_cfg}
        if a.seed_check > 1:
            cfg0 = {k: v for k, v in best[0][1].items() if k != "epoch"}
            tv = []
            for sd in range(a.seed_check):
                rec = {"v": -1, "t": None}

                def record0(vs, S, ep, rec=rec):
                    if vs[0] > rec["v"]:
                        rec["v"] = vs[0]
                        rec["t"] = evaluate(S, splits[0][1], ks, row_of)[key]
                run(kind, loss, cfg0, 100 + sd, record0)
                tv.append(rec["t"])
            res["seed_check"] = {"cfg": cfg0, "test": tv, "mean": float(np.mean(tv)),
                                 "std": float(np.std(tv))}
        results[name] = res
        txt = "  ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in agg.items())
        sc = res.get("seed_check", {})
        print(f"{name:12s} {txt}   [{time.time() - t0:.0f}s, {n_cfg} cfgs] seeds {sc.get('mean', 0):.4f}"
              f"±{sc.get('std', 0):.4f}", flush=True)
        with open(out, "w") as f:
            json.dump({"args": vars(a), "results": results}, f)


if __name__ == "__main__":
    main()
