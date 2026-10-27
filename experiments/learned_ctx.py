"""Shared helpers of experiments/run_ensemble.py and experiments/run_seeds.py.

prepare() repeats the data and nuisance preparation of run_learned.main, and
train_cfg() repeats run_learned.run with the caller's record closure in place
of run_learned's own. run_learned itself is not modified.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import run_learned as RL  # noqa: E402
from drup.data import load_coat, load_yahoo, split_test  # noqa: E402
from drup.estimation import baseline_imputation, clip_propensity, get_propensity, lowrank_imputation  # noqa: E402
from drup.learned import train, train_ials  # noqa: E402
from final_config import KEY  # noqa: E402,F401

IALS_KINDS = ("ials", "drials", "drials_lr")


def prepare(dataset, prop, floor, imp, rank, nsplits, frac_val, threads):
    """Data, propensity, imputation, item popularity and splits as in run_learned.main.

    imp None is treated as "add". Writes RL.PRIOR[0] and RL.QMAT[0] in place.
    """
    torch.set_num_threads(threads)
    if dataset == "coat":
        d, ks, key, by = load_coat(), (5, 10), "ndcg@5", "entry"
    elif dataset == "yahoo":
        d, ks, key, by = load_yahoo(), (5, 10), "ndcg@5", "user"
    elif dataset == "kuairand":
        d, ks, key, by = torch.load("data/raw/kuairand.pt", weights_only=False), (5, 10, 20), "ndcg@10", "user"
    else:
        d, ks, key, by = torch.load("data/raw/kuairec.pt", weights_only=False), (10, 20, 50), "ndcg@20", "user"
    O, Y = d["O"].float(), d["Y"].float()
    m, n = O.shape
    # nuisances in float64 for the small datasets, float32 for the large ones (memory)
    ndt = torch.float64 if m * n < 5e7 else torch.float32
    P = get_propensity(d, O.to(ndt), Y.to(ndt), prop).float()
    P = clip_propensity(P, floor)
    if imp == "lr":
        Yhat = lowrank_imputation(O.to(ndt), Y.to(ndt), P.to(ndt), rank=rank, ridge=5.0).float().clamp(0.0, 1.0)
    elif imp in (None, "add"):
        Yhat = baseline_imputation(O.to(ndt), Y.to(ndt), P.to(ndt), lam=5.0).float()
    else:
        raise ValueError(f"unknown imp {imp!r}")
    _lg = torch.logit(Yhat.clamp(1e-3, 1 - 1e-3))
    RL.PRIOR[0] = (_lg - _lg.mean()) / _lg.std()
    RL.QMAT[0] = (Yhat * (1.0 - P) / (1.0 - P * Yhat).clamp_min(1e-6)).clamp(0.0, 1.0)
    del d["O"], d["Y"]
    pop = (O * Y).sum(0)
    item_pop = (pop / pop.max()).clamp_min(1.0 / pop.max())
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(d["test"], frac_val, s, by=by) for s in range(nsplits)]
    return {"O": O, "Y": Y, "P": P, "Yhat": Yhat, "item_pop": item_pop, "rows": rows, "row_of": row_of,
            "splits": splits, "ks": ks, "key": key, "by": by, "m": m, "n": n}


def job_args(path):
    return json.load(open(path))["args"]


def check_shared(paths):
    """Jobs used together must share prop, floor, imp (missing or "add"), seeds and frac_val.
    Returns the args of the first job."""
    ref, sig0, p0 = None, None, None
    for p in paths:
        a = job_args(p)
        if a.get("imp") not in (None, "add"):
            raise ValueError(f"{p}: imp {a.get('imp')!r} is not add")
        sig = (a["prop"], a["floor"], a["seeds"], a["frac_val"])
        if ref is None:
            ref, sig0, p0 = a, sig, p
        elif sig != sig0:
            raise ValueError(f"{p}: {sig} differs from {p0}: {sig0}")
    return ref


def train_cfg(ctx, name, cfg, seed, ja, record):
    """One training run of method ``name`` with configuration ``cfg`` and ``seed``, as run_learned.run.

    record(S, ep) receives the rows-only score matrix after every epoch (epoch index from 0)
    and returns True to stop training.
    """
    kind, loss, _ = RL.METHODS[name]
    cfg = {k: v for k, v in cfg.items() if k != "epoch"}
    torch.set_num_threads(ja.get("threads", 1))
    torch.manual_seed(seed)
    m, n, O, Y, P, rows = ctx["m"], ctx["n"], ctx["O"], ctx["Y"], ctx["P"], ctx["rows"]
    model = RL.build(kind, m, n, cfg["dim"], O, Y, P, cfg, seed=seed)

    def on_epoch(mod, ep):
        with torch.no_grad():
            S = mod.full_scores(rows)
        return bool(record(S, ep))

    if kind in IALS_KINDS:
        return train_ials(model, ja["epochs"], on_epoch)
    return train(model, O, Y, P, ctx["Yhat"], loss, cfg["lr"], cfg["wd"], ja["epochs"], ja["batch"], on_epoch,
                 patience=ja["patience"], pairs_per_epoch=ja.get("pairs_per_epoch"), seed=seed,
                 item_pop=ctx["item_pop"], gamma=cfg.get("gamma", 0.1), cl_weight=cfg.get("cl", 0.1),
                 tau=cfg.get("tau", 0.2), fn=cfg.get("fn", 0.0), ipsw=cfg.get("ipsw", 0),
                 qmode=cfg.get("qmode", "dr"),
                 adv_lr=ja.get("adv_lr", 1e-4), adv_interval=ja.get("adv_interval", 5), adv_eta=cfg.get("eta", 0),
                 n_neg=(ja.get("n_neg", 64) if loss in ("advinfonce", "dradvinfonce") else 32),
                 adv_min_steps=ja.get("adv_min_steps", 100))
