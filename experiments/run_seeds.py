"""Training-seed robustness: retrain the selected configurations with fresh seeds.

For each split the configuration selected in the existing files is kept (epoch dropped) and
retrained with each seed; each split re-selects its epoch by early stopping on its own
validation part, exactly as run_learned.run / record. Splits that share a configuration share
one training run per seed. DRUP is deterministic and not retrained.

Checkpoint: after every (method, group, seed) the finished entries are rewritten atomically to
<out>.partial.json, keyed "method|group hash|seed", and skipped on restart.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import learned_ctx as LC  # noqa: E402
import run_ensemble as RE  # noqa: E402  (FINAL_JOBS and final_choice)
from drup.metrics import evaluate  # noqa: E402

# baseline (method name -> part file with its "chosen" per split), as named in the task
SRC = {
    "coat": {"iALS": "results/v3/learned_coat_given.json", "DR-LightGCN": "results/v3/learned_coat_given.json"},
    "yahoo": {"r-AdjNorm": "results/v3/learned_yahoo_pop_A.json", "LightGCN": "results/v3/learned_yahoo_pop_B.json"},
    "kuairand": {"r-AdjNorm": "results/v3/learned_kuairand_pop.json", "LightGCN": "results/v3/learned_kuairand_pop.json"},
    "kuairec": {"DR-JL": "results/v3/learned_kuairec_pop_A.json", "MRDR": "results/v3/learned_kuairec_pop_A.json"},
}


def md5h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def run_group(ctx, name, cfg, seed, ja, sp):
    """Train once with ``seed`` for the splits sp (all sharing cfg); per-split early stopping."""
    ks, key, row_of, splits = ctx["ks"], ctx["key"], ctx["row_of"], ctx["splits"]
    st = {"best": [-1.0] * len(sp), "bad": [0] * len(sp), "done": [False] * len(sp)}
    best = [None] * len(sp)

    def record(S, ep):
        for j, s in enumerate(sp):
            v = evaluate(S, splits[s][0], ks, row_of)[key]
            if st["done"][j]:
                continue
            if v > st["best"][j]:
                st["best"][j], st["bad"][j] = v, 0
                agg, pu = evaluate(S, splits[s][1], ks, row_of, per_user=key)
                best[j] = (v, dict(cfg, epoch=ep), agg, pu)
            else:
                st["bad"][j] += 1
                if st["bad"][j] >= ja["patience"]:
                    st["done"][j] = True
        return all(st["done"])

    LC.train_cfg(ctx, name, cfg, seed, ja, record)
    if any(b is None for b in best):
        raise RuntimeError(f"{name} seed {seed}: a split never evaluated")
    return {"splits": list(sp), "chosen": [b[1] for b in best], "val_best": [b[0] for b in best],
            "per_split": [b[2] for b in best], "per_user": [{str(u): v for u, v in b[3].items()} for b in best]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["coat", "yahoo", "kuairand", "kuairec"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = a.dataset
    out = a.out or f"results/v4/seeds_{ds}.json"
    partial_path = out + ".partial.json"
    key = LC.KEY[ds]

    # per method: per split (file, name, chosen-with-epoch)
    choices = RE.final_choice(ds, key)
    methods = {"final": [(f, n, ch) for f, n, ch, _ in choices]}
    for name, path in SRC[ds].items():
        r = json.load(open(path))["results"][name]
        methods[name] = [(path, name, c) for c in r["chosen"]]
    files = sorted({f for src in methods.values() for f, _, _ in src})
    ref = LC.check_shared(files)
    ctx = LC.prepare(ds, ref["prop"], ref["floor"], ref.get("imp"), ref.get("rank", 16), ref["seeds"],
                     ref["frac_val"], ref.get("threads", 1))
    nsp = len(ctx["splits"])
    assert all(len(src) == nsp for src in methods.values())

    partial = json.load(open(partial_path)) if os.path.exists(partial_path) else {}
    results = {}
    for name, src in methods.items():
        groups = {}
        for s, (f, n, ch) in enumerate(src):
            cfg = {k: v for k, v in ch.items() if k != "epoch"}
            groups.setdefault(json.dumps([f, n, cfg], sort_keys=True), []).append(s)
        seeds_out = {str(sd): {"per_user": [None] * nsp, "per_split": [None] * nsp,
                               "chosen": [None] * nsp, "val_best": [None] * nsp} for sd in a.seeds}
        for gk, sp in groups.items():
            f, n, cfg = json.loads(gk)
            h = md5h([f, n, cfg])
            ja = LC.job_args(f)
            for sd in a.seeds:
                pk = f"{name}|{h}|{sd}"
                if pk not in partial:
                    print(f"{name} {n} seed {sd} splits {sp}: training", flush=True)
                    partial[pk] = run_group(ctx, n, cfg, sd, ja, sp)
                    save_json(partial_path, partial)
                ent = partial[pk]
                for j, s in enumerate(ent["splits"]):
                    so = seeds_out[str(sd)]
                    so["per_user"][s] = ent["per_user"][j]
                    so["per_split"][s] = ent["per_split"][j]
                    so["chosen"][s] = ent["chosen"][j]
                    so["val_best"][s] = ent["val_best"][j]
        results[name] = {"seeds": seeds_out}
        print(f"{name}: done", flush=True)
    save_json(out, {"args": vars(a), "results": results})
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
