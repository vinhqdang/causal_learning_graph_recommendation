"""DIN or SIM-hard lifelong baselines on KuaiRand (drup/din.py), with DAX added post hoc (the fair comparison of
LEAF-LGN + DAX, paper/v12/).

Training: BPR on the pre-cut in-pool clicks (the training signal of LightGCN and LEAF-LGN; the random-exposure log is
used for validation and test only). Every epoch the model scores every validation and test candidate; the DAX table
(run_dax.dax_arms, the default 32-configuration grid, or --dax_grid small) and gamma are then selected post hoc on
validation per split, exactly as for LEAF-LGN + DAX (run_dax.Posthoc, selection over epoch x DAX x gamma).

--arch din   DIN: target attention over the whole lifelong sequence (last L events, drup/din.py)
--arch sim   SIM-hard: the K most recent sequence events sharing the candidate's tag or author, then attention

Grid (defaults, 4 CPU cores): lr {1e-2, 3e-3} x wd {1e-4, 1e-5}, dim 16, hidden 16, 12 epochs at most, patience 2 per
split (run_dax's patience rule, on the best value over the epoch's configurations). Output: the run_dax format,
results {"DIN", "DIN+DAX"} (or "SIM", "SIM+DAX") with per_user keyed by dataset row ids, n_configs = trainings x DAX
configurations x gammas for the DAX arm (trainings for the base), n_evals = validation evaluations. Resumable through
OUT.partial.json (written after every training).

--device cpu (default) or cuda: the DIN model, the sequence tables and the batches are computed on the device; the
data, the splits, the sampling (CPU generator), the model selection and the metrics stay on the CPU. --device is not
part of the stored arguments (the output and the partial file are the same on both devices).

Inputs: data/raw/seq_<27k|1k>.npz from experiments/v12_build_seq.py (--dataset selects it).

python3 experiments/run_din.py --arch din --dataset kuairand27k --out results/v12/din_27k.json
"""

import argparse
import itertools
import json
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import run_dax as RD  # noqa: E402
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.device import get_device  # noqa: E402
from drup.din import DIN, fit_bpr, load_sequences, score_rows  # noqa: E402
from drup.dax import item_content_codes, lifelong_rate, tag_codes, user_strata  # noqa: E402
from drup.leaf import CUT_MS, Layout, check_temporal_protocol  # noqa: E402

DATA = os.path.join(ROOT, "data", "raw")
SEQ = {"kuairand27k": os.path.join(DATA, "seq_27k.npz"), "kuairand1k": os.path.join(DATA, "seq_1k.npz")}
KEY = "ndcg@10"
INF = float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["din", "sim"])
    ap.add_argument("--dataset", default="kuairand27k", choices=sorted(SEQ))
    ap.add_argument("--seq", default=None, help="sequence file (default data/raw/seq_<dataset>.npz)")
    ap.add_argument("--L", type=int, default=200, help="events per user used (the last L of the stored window)")
    ap.add_argument("--sim_k", type=int, default=50, help="SIM-hard: most recent matching events kept")
    ap.add_argument("--dax_grid", default="full", choices=["full", "small"])
    ap.add_argument("--gammas", type=float, nargs="+", default=list(RD.GAMMAS))
    ap.add_argument("--no_zcache", action="store_true")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-5])
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--mlp", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="number of validation / test splits")
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--n_users", type=int, default=0, help="0 = all users (smoke runs use a subset)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device of the model computations")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = get_device(a.device)
    torch.set_num_threads(a.threads)
    path = a.out + ".partial.json"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    ds = RD.DATASETS[a.dataset]
    seq_path = a.seq or SEQ[a.dataset]

    d = load_kuairand_rich(version=ds["version"], n_users=(a.n_users or None), train_until=CUT_MS,
                           dense_dtype=torch.float32, with_signals=False)
    print("temporal protocol:", check_temporal_protocol(d), flush=True)
    test = d["test"]
    m, n = len(d["users_raw"]), len(d["item_feat"])
    row_ids = np.array([u for u, _, _ in test], dtype=np.int64)
    lengths = np.array([len(it) for _, it, _ in test], dtype=np.int64)
    layout = Layout(test)
    splits = RD.make_splits(test, a.seeds, a.frac_val)
    seq = load_sequences(seq_path, d["users_raw"], d["item_feat"], L=a.L)
    assert seq["cut_ms"] == CUT_MS, "the sequence file was built with another cut-off"
    tok, valid = seq["tok"].to(dev), seq["valid"].to(dev)
    X = (d["O"] * d["Y"]) > 0
    del d["O"], d["Y"]                       # the DIN fit reads only the clicked pairs X
    pos_u, pos_i = torch.nonzero(X, as_tuple=True)
    print(f"{ds['label']}: {m} users, {n} items, {len(test)} test users in {len(layout.buckets)} candidate buckets, "
          f"{len(pos_u)} training clicks, sequences: {int((seq['length'] > 0).sum())} users with history, "
          f"mean length {float(seq['length'].mean()):.1f} [{time.time() - t0:.0f}s]", flush=True)

    life = lifelong_rate(d["users_raw"], ds["lifelong"])
    strata = user_strata(d["users_raw"], ds["lifelong"])
    tags = tag_codes(d["item_feat"])
    content = item_content_codes(d["item_feat"], d["items_raw"], video=ds["video"], path=ds["lifelong"])
    base_name = a.arch.upper()
    dax_name = base_name + "+DAX"
    grid = RD.post_grid(a.dax_grid == "small")
    arms = RD.dax_arms(d, dax_name, grid, [], life, strata, content, tags, [0])
    post = RD.Posthoc(layout, splits, row_ids, lengths, base_name, arms, base_std=True, gammas=a.gammas,
                      cache_z=not a.no_zcache)
    sig = json.loads(json.dumps({k: v for k, v in vars(a).items() if k not in ("out", "threads", "device")}))
    state = {"done": []}
    if os.path.exists(path):
        st = json.load(open(path))
        if st["sig"] != sig:
            raise SystemExit(f"{path} was written with other arguments; remove it or change --out")
        state, sel = st["state"], st["sel"]
        post.load(sel)
        print(f"resuming: {len(state['done'])} trainings done", flush=True)

    def save():
        with open(path + ".tmp", "w") as f:
            json.dump({"sig": sig, "state": state, "sel": post.states()}, f)
        os.replace(path + ".tmp", path)

    S = len(splits)
    for lr, wd in itertools.product(a.lrs, a.wds):
        key = f"{a.arch}|lr{lr}|wd{wd}"
        if key in state["done"]:
            continue
        t1 = time.time()
        torch.manual_seed(a.train_seed)
        # built on the CPU (the initialisation is the same on both devices), then moved
        model = DIN(m, n, seq["n_tag"], seq["n_author"], seq["n_music"], seq["cand"], d=a.dim, hidden=a.hidden,
                    mlp=a.mlp, sim_k=(a.sim_k if a.arch == "sim" else 0)).to(dev)
        best, bad, done = [-1.0] * S, [0] * S, [False] * S

        def on_epoch(mod, ep, lr=lr, wd=wd, best=best, bad=bad, done=done):
            base_b = [score_rows(mod, bk["rows"], bk["cand"], tok, valid, device=dev) for bk in layout.buckets]
            vals = post.add_epoch(base_b, dict(lr=lr, wd=wd), ep, active=[not x for x in done])
            RD.patience_update(best, bad, done, vals, a.patience)
            print(f"    {key} epoch {ep} [{time.time() - t1:.0f}s] "
                  f"val {' '.join('-' if v is None else f'{v:.4f}' for v in vals)}", flush=True)
            return all(done)
        ran = fit_bpr(model, X, tok, valid, lr, wd, a.epochs, a.batch, on_epoch, seed=a.train_seed, device=dev)
        print(f"  {key}: {ran} epochs [{time.time() - t1:.0f}s]", flush=True)
        del model
        state["done"].append(key)
        save()

    res = {}
    for nm in (base_name, dax_name):
        res[nm] = RD.result_of(post.sels[nm].result())
    for nm, r in res.items():
        print(f"{nm}: test {r['test']['ndcg@10'][0]:.4f} +- {r['test']['ndcg@10'][1]:.4f} "
              f"(val {np.mean(r['val_best']):.4f}, {r['n_configs']} configurations, {r['n_evals']} evaluations) "
              f"chosen {r['chosen'][0]}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"args": sig, "results": res, "plan": None, "dataset": a.dataset}, f)
    if os.path.exists(path):
        os.remove(path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"wrote {a.out} [{time.time() - t0:.0f}s, peak RSS {peak:.0f} MB]", flush=True)


if __name__ == "__main__":
    main()
