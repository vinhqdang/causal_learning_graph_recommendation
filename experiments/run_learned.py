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
from drup.learned import IALS, MACR, MF, LightGCN, SimGCL, train, train_ials  # noqa: E402
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
    "DR-JL": ("mf", "drjl", {}),
    "MRDR": ("mf", "mrdr", {}),
    "MACR": ("macr", "macr", {"c": [0.1, 0.3]}),
    "SimGCL": ("simgcl", "simgcl", {"cl": [0.1, 0.5]}),
    "iALS": ("ials", None, {"alpha": [3.0, 10.0, 30.0], "reg": [1.0, 10.0, 100.0, 1000.0]}),
}
GRAPH = ("lgn", "navip", "simgcl")


def build(kind, m, n, d, O, Y, P, cfg):
    if kind == "mf":
        return MF(m, n, d)
    if kind == "macr":
        return MACR(m, n, d, c=cfg["c"])
    eu, ei = torch.nonzero(O * Y > 0, as_tuple=True)
    if kind == "ials":
        return IALS(m, n, d, eu, ei, alpha=cfg["alpha"], reg=cfg["reg"])
    w = (1.0 / P[eu, ei]).float() if kind == "navip" else torch.ones(len(eu))
    if kind == "simgcl":
        return SimGCL(m, n, d, eu, ei, w, layers=cfg["layers"], r=0.5)
    return LightGCN(m, n, d, eu, ei, w, layers=cfg["layers"], r=cfg.get("r", 0.5))


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
    ap.add_argument("--wide", nargs="*", default=[],
                    help="methods that search --wide_layers and --wide_dims instead of --layers/--dims")
    ap.add_argument("--wide_layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--wide_dims", type=int, nargs="+", default=[64, 128])
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
    elif a.dataset == "kuairand":
        d = torch.load("data/raw/kuairand.pt", weights_only=False)
        ks, key, by = (5, 10, 20), "ndcg@10", "user"
    else:
        d = torch.load("data/raw/kuairec.pt", weights_only=False)
        ks, key, by = (10, 20, 50), "ndcg@20", "user"
    O, Y = d["O"].float(), d["Y"].float()
    m, n = O.shape
    # nuisances in float64 for the small datasets, float32 for the large ones (memory)
    ndt = torch.float64 if m * n < 5e7 else torch.float32
    P = get_propensity(d, O.to(ndt), Y.to(ndt), a.prop).float()
    P = clip_propensity(P, a.floor)
    Yhat = baseline_imputation(O.to(ndt), Y.to(ndt), P.to(ndt), lam=5.0).float()
    del d["O"], d["Y"]
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
        """Train one configuration. record(vals, S, ep) returns True when every
        split has stopped: each split stops on its own validation score after
        ``patience`` epochs without improvement and then keeps its selection,
        so no split's selection depends on another split's data."""
        torch.manual_seed(seed)
        model = build(kind, m, n, cfg["dim"], O, Y, P, cfg)

        def on_epoch(mod, ep):
            with torch.no_grad():
                S = mod.full_scores(rows)
            vals = [evaluate(S, v, ks, row_of)[key] for v, _ in splits]
            return bool(record(vals, S, ep))
        if kind == "ials":
            return train_ials(model, a.epochs, on_epoch)
        return train(model, O, Y, P, Yhat, loss, cfg["lr"], cfg["wd"], a.epochs, a.batch, on_epoch,
                     patience=a.patience, pairs_per_epoch=a.pairs_per_epoch, seed=seed,
                     item_pop=item_pop, gamma=cfg.get("gamma", 0.1), cl_weight=cfg.get("cl", 0.1))

    bank_path = a.dump_bank or os.path.join(os.path.dirname(out), "bank_" + os.path.basename(out))
    dumps = json.load(open(bank_path))["banks"] if os.path.exists(bank_path) else {}
    for name in a.methods:
        if name in results:
            print(f"{name}: already in {out}, skipped", flush=True)
            continue
        kind, loss, extra = METHODS[name]
        t0 = time.time()
        wide = name in a.wide
        dims = a.wide_dims if wide else a.dims
        layers = (a.wide_layers if wide else a.layers) if kind in GRAPH else [0]
        if kind == "ials":
            grid_keys, grid_vals = ["dim"] + list(extra), [dims] + [extra[k] for k in extra]
        else:
            grid_keys = ["lr", "wd", "dim", "layers"] + list(extra)
            grid_vals = [a.lrs, a.wds, dims, layers] + [extra[k] for k in extra]
        best = [(-1.0, None, None, None) for _ in splits]       # (val, cfg, test agg, test per-user)
        bank = []
        n_cfg = 0
        # checkpoint after every configuration, so that a restarted run resumes
        ckpt = out + f".{name}.partial.json"
        done_cfgs = []
        if os.path.exists(ckpt):
            cp = json.load(open(ckpt))
            done_cfgs, bank = cp["done"], cp["bank"]
            best = [(b[0], b[1], b[2], {int(k): v for k, v in b[3].items()} if b[3] else None) for b in cp["best"]]
            print(f"  {name}: resuming after {len(done_cfgs)} configurations", flush=True)
        for vals in itertools.product(*grid_vals):
            cfg = dict(zip(grid_keys, vals))
            n_cfg += 1
            if cfg in done_cfgs:
                continue
            st = {"best": [-1.0] * len(splits), "bad": [0] * len(splits), "done": [False] * len(splits)}

            def record(vs, S, ep, cfg=cfg, st=st):
                bank.append({"cfg": dict(cfg, epoch=ep), "val": vs,
                             "test": [evaluate(S, t, ks, row_of)[key] for _, t in splits]})
                for s, v in enumerate(vs):
                    if st["done"][s]:
                        continue
                    if v > st["best"][s]:
                        st["best"][s], st["bad"][s] = v, 0
                        if v > best[s][0]:
                            agg, pu = evaluate(S, splits[s][1], ks, row_of, per_user=key)
                            best[s] = (v, dict(cfg, epoch=ep), agg, pu)
                    else:
                        st["bad"][s] += 1
                        if st["bad"][s] >= a.patience:
                            st["done"][s] = True
                return all(st["done"])
            tc = time.time()
            run(kind, loss, cfg, 0, record)
            print(f"  {name} {cfg} done [{time.time() - tc:.0f}s] best split0 val {best[0][0]:.4f}",
                  flush=True)
            done_cfgs.append(cfg)
            with open(ckpt + ".tmp", "w") as f:
                json.dump({"done": done_cfgs, "bank": bank,
                           "best": [(b[0], b[1], b[2], {str(k): v for k, v in b[3].items()} if b[3] else None)
                                    for b in best]}, f)
            os.replace(ckpt + ".tmp", ckpt)
        dumps[name] = bank
        with open(bank_path, "w") as f:
            json.dump({"args": vars(a), "key": key, "banks": dumps}, f)
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
                rec = {"v": -1, "t": None, "bad": 0}

                def record0(vs, S, ep, rec=rec):
                    if vs[0] > rec["v"]:
                        rec["v"], rec["bad"] = vs[0], 0
                        rec["t"] = evaluate(S, splits[0][1], ks, row_of)[key]
                    else:
                        rec["bad"] += 1
                    return rec["bad"] >= a.patience
                run(kind, loss, cfg0, 100 + sd, record0)
                tv.append(rec["t"])
            res["seed_check"] = {"cfg": cfg0, "test": tv, "mean": float(np.mean(tv)),
                                 "std": float(np.std(tv))}
        results[name] = res
        if os.path.exists(ckpt):
            os.remove(ckpt)
        txt = "  ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in agg.items())
        sc = res.get("seed_check", {})
        print(f"{name:12s} {txt}   [{time.time() - t0:.0f}s, {n_cfg} cfgs] seeds {sc.get('mean', 0):.4f}"
              f"±{sc.get('std', 0):.4f}", flush=True)
        with open(out, "w") as f:
            json.dump({"args": vars(a), "results": results}, f)


if __name__ == "__main__":
    main()
