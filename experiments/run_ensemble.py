"""Ensemble of component A (the final trained DR-BPR-LGN + offset method) and component B (DRUP).

Three stages, each a separate process. Every score matrix (test-user rows, float32) is
cached in --cache, and work whose cache file already exists is skipped (restart-safe):
  --stage drup     DRUP scores of each split's selected configuration
  --stage trained  retrain each split's selected component-A configuration to its epoch
  --stage blend    z-normalise both components over unlogged items, choose w per split on validation
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import learned_ctx as LC  # noqa: E402
import run_filters as RF  # noqa: E402
import run_learned as RL  # noqa: E402,F401
from drup.data import split_test  # noqa: E402
from drup.metrics import evaluate  # noqa: E402
from drup.pipeline import Nuisance  # noqa: E402

PROP = {"coat": "given", "yahoo": "pop", "kuairand": "pop", "kuairec": "pop"}
# (file, method name) of the final method, copied from the SETS[...][1] lists of experiments/final_analysis.py
V4 = "results/v4/"
FINAL_JOBS = {
    "coat": [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN"), (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p"),
             (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")],
    "yahoo": [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] +
             [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")],
    "kuairand": [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                 (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")],
    "kuairec": [(V4 + f"learned_kuairec_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3")] +
               [(V4 + f"learned_kuairec_p2_lr{lr}.json", "DR-BPR-LGN-p2") for lr in ("1e-2", "3e-3", "1e-3")] +
               [(V4 + "learned_kuairec_p_lr3e-3.json", "DR-BPR-LGN-p")],
}
WGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def filt_path(ds):
    return f"results/v3/filters_{ds}_{PROP[ds]}.json"


def md5h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def load_index(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def final_choice(ds, key=None):
    """Per split, the best (file, name, chosen, val) over the final-method jobs (as final_analysis.combine).
    ``key`` is accepted for signature parity; val_best is already stored under each job's metric."""
    rs = [(f, n, json.load(open(f))["results"][n]) for f, n in FINAL_JOBS[ds] if os.path.exists(f)]
    out = []
    for s in range(len(rs[0][2]["val_best"])):
        j = int(np.argmax([r["val_best"][s] for _, _, r in rs]))
        f, n, r = rs[j]
        out.append((f, n, r["chosen"][s], r["val_best"][s]))
    return out


def rowz(S, mask):
    """Row-wise z-normalisation over the unlogged items (mask = 1)."""
    cnt = mask.sum(1)
    mu = (S * mask).sum(1) / cnt
    sd = (((S - mu[:, None]) ** 2 * mask).sum(1) / cnt).sqrt().clamp_min(1e-12)
    return (S - mu[:, None]) / sd[:, None]


def select_w(zA, zB, val, ks, row_of, key):
    """First argmax over WGRID of the validation metric. Receives the validation list only."""
    curve = [float(evaluate((1 - w) * zA + w * zB, val, ks, row_of)[key]) for w in WGRID]
    return WGRID[int(np.argmax(curve))], curve


def stage_drup(ds, cache):
    fpath = filt_path(ds)
    fa = json.load(open(fpath))["args"]
    fres = json.load(open(fpath))["results"]["DRUP"]
    chosen, val_best = fres["chosen"], fres["val_best"]
    nsp = fa["seeds"]
    idx_path = os.path.join(cache, f"drup_{ds}_index.json")
    index = load_index(idx_path)
    groups = {}
    for s in range(nsp):
        base = {k: v for k, v in chosen[s].items() if k != "beta"}
        groups.setdefault(json.dumps(base, sort_keys=True), []).append(s)

    def order(g):
        b = json.loads(g)
        return (b["floor"], b["lam"], b.get("imp", "add"))

    todo = [g for g in sorted(groups, key=order)
            if not all(str(s) in index and os.path.exists(index[str(s)]["cache"]) and index[str(s)]["cfg"] == chosen[s]
                       for s in groups[g])]
    if not todo:
        print("drup: all splits cached", flush=True)
        return
    d, ks, key, by = RF.load_dataset(ds, getattr(torch, fa["dtype"]))
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(d["test"], fa["frac_val"], s, by=by) for s in range(nsp)]
    nz = Nuisance(d, d["O"], d["Y"], fa["prop"], K=fa["xfit"], seed=0)
    for g in todo:
        base = json.loads(g)
        bank = RF.score_bank(d, "DRUP", base, nz, rows)
        for s in groups[g]:
            beta = chosen[s]["beta"]
            hits = [S for c, S in bank if c["beta"] == beta]
            assert len(hits) == 1, f"split {s}: {len(hits)} bank entries with beta {beta}"
            S = hits[0]
            val = evaluate(S, splits[s][0], ks, row_of)[key]
            if abs(val - val_best[s]) > 1e-4:
                raise RuntimeError(f"split {s}: DRUP validation {val:.6f} != stored {val_best[s]:.6f}")
            h = md5h({"base": base, "beta": beta})
            path = os.path.join(cache, f"drup_{ds}_{h}.pt")
            if not os.path.exists(path):
                torch.save(S.detach().float().cpu(), path)
            index[str(s)] = {"file": fpath, "name": "DRUP", "cfg": chosen[s], "hash": h,
                             "stored_val": val_best[s], "repro_val": val, "cache": path}
            print(f"drup split {s}: val {val:.6f} stored {val_best[s]:.6f}", flush=True)
        save_json(idx_path, index)
        del bank


def stage_trained(ds, cache, threads):
    jobs = [(f, n) for f, n in FINAL_JOBS[ds] if os.path.exists(f)]
    ref = LC.check_shared([f for f, _ in jobs])
    key = LC.KEY[ds]
    choices = final_choice(ds, key)
    ctx = LC.prepare(ds, ref["prop"], ref["floor"], ref.get("imp"), ref.get("rank", 16), ref["seeds"],
                     ref["frac_val"], threads)
    assert len(choices) == len(ctx["splits"])
    idx_path = os.path.join(cache, f"trained_{ds}_index.json")
    index = load_index(idx_path)
    groups = {}
    for s, (f, n, ch, _) in enumerate(choices):
        cfg = {k: v for k, v in ch.items() if k != "epoch"}
        groups.setdefault(json.dumps([f, n, cfg], sort_keys=True), []).append(s)
    for gk, sp in groups.items():
        if all(str(s) in index and os.path.exists(index[str(s)]["cache"]) and index[str(s)]["cfg"] == choices[s][2]
               for s in sp):
            continue
        f, n, cfg = json.loads(gk)
        ja = LC.job_args(f)
        need = {}
        for s in sp:
            need.setdefault(choices[s][2]["epoch"], []).append(s)
        last = max(need)
        hashes = {ep: md5h({"f": f, "n": n, "cfg": cfg, "ep": ep}) for ep in need}
        repro, paths = {}, {}

        def record(S, ep):
            if ep in need:
                path = os.path.join(cache, f"trained_{ds}_{hashes[ep]}.pt")
                paths[ep] = path
                if not os.path.exists(path):
                    torch.save(S.float().cpu(), path)
                for s in need[ep]:
                    rv = evaluate(S, ctx["splits"][s][0], ctx["ks"], ctx["row_of"])[ctx["key"]]
                    repro[s] = float(rv)
                    if abs(rv - choices[s][3]) > 2e-3:
                        print(f"WARNING split {s} {n} {cfg} ep {ep}: repro val {rv:.6f} vs stored {choices[s][3]:.6f}",
                              flush=True)
            return ep >= last

        print(f"training {n} {cfg} for epochs {sorted(need)} on splits {sp}", flush=True)
        LC.train_cfg(ctx, n, cfg, 0, ja, record)
        missing = [s for s in sp if s not in repro]
        if missing:
            raise RuntimeError(f"training stopped before the selected epochs for splits {missing}")
        for s in sp:
            ep = choices[s][2]["epoch"]
            index[str(s)] = {"file": f, "name": n, "cfg": choices[s][2], "hash": hashes[ep],
                             "stored_val": choices[s][3], "repro_val": repro[s], "cache": paths[ep]}
        save_json(idx_path, index)


def stage_blend(ds, cache, out, args):
    fpath = filt_path(ds)
    fa = json.load(open(fpath))["args"]
    fres = json.load(open(fpath))["results"]["DRUP"]
    d, ks, key, by = RF.load_dataset(ds, torch.float32)
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    splits = [split_test(d["test"], fa["frac_val"], s, by=by) for s in range(fa["seeds"])]
    mask = (d["O"][rows] == 0).float()          # unlogged items of the training log only
    del d["O"], d["Y"]
    dix = load_index(os.path.join(cache, f"drup_{ds}_index.json"))
    tix = load_index(os.path.join(cache, f"trained_{ds}_index.json"))
    choices = final_choice(ds, key)
    assert len(choices) == len(splits), "component A and B use different numbers of splits"

    per = {k: {"per_split": [], "per_user": [], "chosen": [], "val_best": []}
           for k in ("Ens-A+DRUP", "A-repro", "DRUP-repro", "Ens-fixed-0.5")}
    repro = {"A": [], "B": []}
    for s in range(len(splits)):
        val, test = splits[s]
        A, B = tix[str(s)], dix[str(s)]
        assert A["cfg"] == choices[s][2], f"split {s}: trained index does not carry the selected A configuration"
        assert B["cfg"] == fres["chosen"][s], f"split {s}: DRUP index does not carry the selected B configuration"
        zA = rowz(torch.load(A["cache"]), mask)
        zB = rowz(torch.load(B["cache"]), mask)
        assert torch.isfinite(zA).all() and torch.isfinite(zB).all()
        # leakage checks: validation and test are disjoint
        if by == "user":
            assert not ({u for u, _, _ in val} & {u for u, _, _ in test}), "user overlap"
        else:
            vm = {u: set(np.asarray(it).tolist()) for u, it, _ in val}
            for u, it, _ in test:
                if u in vm:
                    assert not (vm[u] & set(np.asarray(it).tolist())), f"item overlap for user {u}"
        w_s, curve = select_w(zA, zB, val, ks, row_of, key)
        # select_w never receives the test lists: re-selecting after permuting every test relevance array must agree
        rng = np.random.default_rng(12345 + s)
        test_perm = [(u, it, rel[rng.permutation(len(rel))]) for u, it, rel in test]
        w_chk, _ = select_w(zA, zB, val, ks, row_of, key)
        assert w_chk == w_s and len(test_perm) == len(test)

        def score(w, test=test, zA=zA, zB=zB):
            return evaluate((1 - w) * zA + w * zB, test, ks, row_of, per_user=key)

        cfgA, cfgB = choices[s][2], fres["chosen"][s]
        plan = [("Ens-A+DRUP", w_s, {"w": w_s, "A": cfgA, "B": cfgB, "val_curve": curve}, curve[WGRID.index(w_s)]),
                ("A-repro", 0.0, {"w": 0.0, "A": cfgA, "B": cfgB}, curve[0]),
                ("DRUP-repro", 1.0, {"w": 1.0, "A": cfgA, "B": cfgB}, curve[-1]),
                ("Ens-fixed-0.5", 0.5, {"w": 0.5, "A": cfgA, "B": cfgB}, curve[WGRID.index(0.5)])]
        for name, w, ch, vb in plan:
            agg, pu = score(w)
            per[name]["per_split"].append(agg)
            per[name]["per_user"].append({str(u): v for u, v in pu.items()})
            per[name]["chosen"].append(ch)
            per[name]["val_best"].append(float(vb))
        repro["A"].append({"stored": float(choices[s][3]), "repro": A["repro_val"]})
        repro["B"].append({"stored": float(fres["val_best"][s]), "repro": B["repro_val"]})
        print(f"split {s}: w {w_s:.1f}  test {per['Ens-A+DRUP']['per_split'][-1][key]:.4f}", flush=True)

    results = {}
    for name, r in per.items():
        agg = {k: (float(np.mean([x[k] for x in r["per_split"]])), float(np.std([x[k] for x in r["per_split"]])))
               for k in r["per_split"][0]}
        results[name] = {"test": agg, "chosen": r["chosen"], "per_split": r["per_split"],
                         "val_best": r["val_best"], "per_user": r["per_user"],
                         "n_configs": len(WGRID) if name == "Ens-A+DRUP" else 1}
    save_json(out, {"args": vars(args), "results": results, "repro": repro})
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["coat", "yahoo", "kuairand", "kuairec"])
    ap.add_argument("--stage", required=True, choices=["drup", "trained", "blend"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cache", default="results/v4/cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.cache, exist_ok=True)
    out = args.out or f"results/v4/ensemble_{args.dataset}.json"
    if args.stage == "drup":
        stage_drup(args.dataset, args.cache)
    elif args.stage == "trained":
        stage_trained(args.dataset, args.cache, args.threads)
    else:
        stage_blend(args.dataset, args.cache, out, args)


if __name__ == "__main__":
    main()
