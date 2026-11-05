"""LEAF on KuaiRand (paper/v6/DESIGN.md, Sections 5 and 6, item 5).

Variants (--variant):
  tied  closed-form collective autoencoder with the tied lifelong channels. --gamma grid: channel weights chosen on
        validation as in the probes (tag lift, author familiarity, author lift, music familiarity); --gamma fit:
        weights of all six channels fitted by least squares on the closed-form residual (ridge grid --ridge).
  full  closed-form collective autoencoder with free lifelong-key weights W (--families, --lam_phi).
  lgn   LEAF-LGN: LightGCN trained jointly with the six channel weights, the popularity offset and the policy channel.
--ablate lists arms, each run as its own method: full, no_lifelong, no_policy, lift_only, familiarity_only, no_tag,
no_author, no_music, shuffle_users (the channel tables of other users).
Output: the run_learned result format, results[name] = {test, per_split, val_best, chosen, per_user, n_configs};
per_split holds the ndcg@10 of each split only. Resumable: OUT.partial.json is written after every block.

python3 experiments/run_leaf.py --dataset kuairand_t --variant tied --seeds 5 --out results/v10/leaf_tied.json
(the lifelong window follows the dataset: kuairand_t -> "pre" (before the cut-off), kuairand_all -> "all".)
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.leaf import (CHANNELS, CUT_MS, LEAFLGN, BucketSelector, LifelongTables, Layout, PhiFull,  # noqa: E402
                       check_temporal_protocol, collective_gram, collective_solve, compact_base, fit_bpr_policy,
                       fit_tied, make_lightgcn, protocol_split)

LIFELONG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "lifelong_27k.npz")
CH_GRID = {0: (0.0, 0.5, 1.0), 3: (0.0, 0.5, 1.0, 2.0), 2: (0.0, 0.5), 5: (0.0, 0.5)}   # channel index -> grid
KEY = "ndcg@10"


def arm_spec(name):
    """Channel mask (1 = on, order of CHANNELS), whether the policy weight is fixed to 0, and the table shuffle."""
    mask, w0, shuffle = np.ones(6), False, False
    if name == "full":
        pass
    elif name == "no_lifelong":
        mask[:] = 0.0
    elif name == "no_policy":
        w0 = True
    elif name == "lift_only":
        mask[[1, 3, 5]] = 0.0
    elif name == "familiarity_only":
        mask[[0, 2, 4]] = 0.0
    elif name == "no_tag":
        mask[[0, 1]] = 0.0
    elif name == "no_author":
        mask[[2, 3]] = 0.0
    elif name == "no_music":
        mask[[4, 5]] = 0.0
    elif name == "shuffle_users":
        shuffle = True
    else:
        raise ValueError(f"unknown arm {name!r}")
    return {"name": name, "mask": mask, "w0": w0, "shuffle": shuffle}


def arm_ws(arm, ws):
    return (0.0,) if arm["w0"] else tuple(ws)


class Ctx:
    """Lifelong tables of the chosen window; a new LifelongTables per (s, shuffle), built on demand."""

    def __init__(self, path, window, users_raw, item_feat):
        L = np.load(path, allow_pickle=True)
        self.npz = {k: L[k] for k in L.files if f"_{window}" in k or k.endswith("_values")}
        self.window, self.users_raw, self.item_feat = window, users_raw, item_feat

    def build(self, s, shuffle):
        return LifelongTables(self.npz, self.users_raw, self.item_feat, window=self.window, s=s, shuffle=shuffle)


def load_state(path, sig):
    if not os.path.exists(path):
        return None
    st = json.load(open(path))
    if st["sig"] != sig:
        raise SystemExit(f"{path} was written with other arguments; remove it or change --out")
    return st


def save_state(path, sig, state, sels):
    obj = {"sig": sig, "done": state["done"], "sel": {k: v.state() for k, v in sels.items()}}
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def tied_bases(O, Y, layout, ws, lams, n):
    """Row-standardised closed-form scores per (w, lam), one matrix per candidate bucket (the B matrices are not kept)."""
    bases = {}
    for w in ws:
        G, C = collective_gram(O, Y, w)
        for lam in lams:
            B, _ = collective_solve(G, C, n, lam)
            bases[(w, lam)] = [compact_base(O, Y, w, B, bk["rows"], bk["cand"]) for bk in layout.buckets]
            del B
        del G, C
    return bases


def run_tied(ctx, a, arms, sels, state, path, sig, O, Y, layout, pop, n):
    ws = sorted({w for arm in arms for w in arm_ws(arm, a.w)})
    t0 = time.time()
    bases = tied_bases(O, Y, layout, ws, a.lam, n)
    print(f"closed-form bases for {len(bases)} (w, lam) pairs [{time.time() - t0:.0f}s]", flush=True)
    pop_b = [pop[bk["cand"]] for bk in layout.buckets]
    blocks = sorted({(arm["shuffle"], s) for arm in arms for s in a.s})
    for shuffle, s in blocks:
        key = f"tied|shuffle{int(shuffle)}|s{s}"
        if key in state["done"]:
            continue
        t0 = time.time()
        tabs = ctx.build(s, shuffle)
        chan = [tabs.compact(bk["rows"], bk["cand"]) for bk in layout.buckets]     # per bucket (U_b, L_b, 6)
        for arm in arms:
            if arm["shuffle"] != shuffle:
                continue
            sel = sels[arm["name"]]
            for w in arm_ws(arm, a.w):
                for lam in a.lam:
                    base = bases[(w, lam)]
                    if a.gamma == "fit":
                        G, C = collective_gram(O, Y, w)
                        B, _ = collective_solve(G, C, n, lam)
                        del G, C
                        for ridge in a.ridge:
                            gamma = fit_tied(O, Y, B, w, tabs, ridge=ridge, mask=arm["mask"])
                            for beta in a.beta:
                                Sc = [bs + ch @ gamma + beta * pb for bs, ch, pb in zip(base, chan, pop_b)]
                                sel.add(Sc, dict(lam=lam, w=w, s=s, beta=beta, ridge=ridge,
                                                 gamma=[round(float(x), 4) for x in gamma]))
                        del B
                    else:
                        gc = list(CH_GRID)
                        vals_per = [CH_GRID[c] if arm["mask"][c] else (0.0,) for c in gc]
                        for beta in a.beta:
                            for combo in itertools.product(*vals_per):
                                gw = torch.zeros(6)
                                for c, g in zip(gc, combo):
                                    gw[c] = g
                                Sc = [bs + ch @ gw + beta * pb for bs, ch, pb in zip(base, chan, pop_b)]
                                sel.add(Sc, dict(lam=lam, w=w, s=s, beta=beta,
                                                 **{CHANNELS[c]: g for c, g in zip(gc, combo)}))
            print(f"  {arm['name']} s={s} shuffle={shuffle}: {sel.n} configurations [{time.time() - t0:.0f}s]",
                  flush=True)
        del tabs, chan
        state["done"].append(key)
        save_state(path, sig, state, sels)


def run_full(ctx, a, sel, state, path, sig, O, Y, layout, pop, n):
    pop_b = [pop[bk["cand"]] for bk in layout.buckets]
    for fam, s, w in itertools.product(a.families, a.s, a.w):
        key = f"full|{'+'.join(fam)}|s{s}|w{w}"
        if key in state["done"]:
            continue
        t0 = time.time()
        tabs = ctx.build(s, False)
        phi = PhiFull(tabs, fam, min_imp=a.min_imp)
        G, C = collective_gram(O, Y, w, phi)
        print(f"  Gram {key}: dimension {G.shape[0]} [{time.time() - t0:.0f}s]", flush=True)
        for lam, lp in itertools.product(a.lam, a.lam_phi):
            B, W = collective_solve(G, C, n, lam, lp)
            base = [compact_base(O, Y, w, B, bk["rows"], bk["cand"], phi, W) for bk in layout.buckets]
            del B, W
            for beta in a.beta:
                Sc = [bs + beta * pb for bs, pb in zip(base, pop_b)]
                sel.add(Sc, dict(lam=lam, lam_phi=lp, w=w, s=s, beta=beta, family="+".join(fam)))
        del G, C, tabs, phi
        print(f"  {key}: {sel.n} configurations [{time.time() - t0:.0f}s]", flush=True)
        state["done"].append(key)
        save_state(path, sig, state, {"LEAF-full": sel})


def run_lgn(ctx, a, arms, sels, state, path, sig, O, Y, layout, pop, n, m):
    S = a.seeds
    for arm in arms:
        sel = sels[arm["name"]]
        tabs = ctx.build(a.s[0], arm["shuffle"])
        for lr, wd, w in itertools.product(a.lrs, a.wds, arm_ws(arm, a.w)):
            key = f"lgn|{arm['name']}|lr{lr}|wd{wd}|w{w}"
            if key in state["done"]:
                continue
            t0 = time.time()
            base = make_lightgcn(m, n, Y, a.dim, a.layers)
            model = LEAFLGN(base, tabs, pop, arm["mask"])
            best, bad, done = [-1.0] * S, [0] * S, [False] * S

            def on_epoch(mod, ep, lr=lr, wd=wd, w=w, best=best, bad=bad, done=done):
                with torch.no_grad():
                    emb = mod.embeddings()
                    Sc = [mod(bk["rows"][:, None], bk["cand"], emb) for bk in layout.buckets]
                vals = sel.add(Sc, dict(lr=lr, wd=wd, w=w, epoch=ep), active=[not x for x in done])
                for s_, v in enumerate(vals):
                    if v is None:
                        continue
                    if v > best[s_]:
                        best[s_], bad[s_] = v, 0
                    else:
                        bad[s_] += 1
                        if bad[s_] >= a.patience:
                            done[s_] = True
                print(f"    {arm['name']} lr {lr} wd {wd} w {w} ep {ep} val {np.mean([x for x in vals if x is not None]):.4f}",
                      flush=True)
                return all(done)

            fit_bpr_policy(model, O, Y, w, lr, wd, a.epochs, a.batch, on_epoch, seed=0)
            print(f"  {key}: {sel.n} epochs [{time.time() - t0:.0f}s]", flush=True)
            del model, base
            state["done"].append(key)
            save_state(path, sig, state, sels)
        del tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kuairand_t", choices=["kuairand_t", "kuairand_all"])
    ap.add_argument("--variant", required=True, choices=["tied", "full", "lgn"])
    ap.add_argument("--gamma", default="grid", choices=["grid", "fit"])
    ap.add_argument("--ablate", nargs="+", default=["full"])
    ap.add_argument("--seeds", type=int, default=5, help="number of validation / test splits")
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--n_users", type=int, default=0, help="0 = all users (smoke runs use a subset)")
    ap.add_argument("--lam", type=float, nargs="+", default=[1000.0, 3000.0, 10000.0])
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--s", type=float, nargs="+", default=[5.0, 20.0])
    ap.add_argument("--ridge", type=float, nargs="+", default=[0.0, 10.0])
    ap.add_argument("--lam_phi", type=float, nargs="+", default=[1e2, 1e3, 1e4])
    ap.add_argument("--families", nargs="+", default=["tag", "tag+author_id", "tag+author_id+music_id"])
    ap.add_argument("--min_imp", type=int, default=20)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-5])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.families = [tuple(f.split("+")) for f in a.families]
    torch.set_num_threads(a.threads)
    out = a.out or f"results/v10/leaf_{a.variant}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    path = out + ".partial.json"
    t0 = time.time()

    cut = a.dataset == "kuairand_t"
    d = load_kuairand_rich(version="pure", n_users=(a.n_users or None), train_until=(CUT_MS if cut else None),
                           dense_dtype=torch.float32, with_signals=False)
    if cut:
        print("temporal protocol:", check_temporal_protocol(d), flush=True)
    O, Y = d["O"], d["Y"]
    m, n = O.shape
    test = d["test"]
    U = len(test)
    rows_ids = np.array([u for u, _, _ in test], dtype=np.int64)
    layout = Layout(test)
    ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
    splits = [protocol_split(ctest, s, a.frac_val) for s in range(a.seeds)]
    X = O * Y
    pop = torch.log1p(X.sum(0))
    pop = (pop - pop.mean()) / pop.std().clamp_min(1e-12)
    del X
    window = "pre" if cut else "all"
    ctx = Ctx(LIFELONG, window, d["users_raw"], d["item_feat"])
    lifelong_users = np.isin(d["users_raw"], ctx.npz[f"tot_{window}_user"])
    cells = sum(bk["cand"].numel() for bk in layout.buckets)
    print(f"{m} users, {n} items, {U} test users in {len(layout.buckets)} candidate buckets ({cells} padded cells), "
          f"lifelong history for "
          f"{lifelong_users.mean():.3f} of users [{time.time() - t0:.0f}s]", flush=True)

    if a.variant == "full":
        arms, sel_names = [], ["LEAF-full"]
    else:
        base_name = {"tied": "LEAF-tied", "lgn": "LEAF-LGN"}[a.variant]
        arms = [arm_spec(x) for x in a.ablate]
        sel_names = [base_name + ("" if x == "full" else f"[{x}]") for x in a.ablate]
        for arm, nm in zip(arms, sel_names):
            arm["name"] = nm
    sels = {nm: BucketSelector(layout, splits, (10,), KEY, rows_ids) for nm in sel_names}
    # the signature is normalised to JSON types, so that a resumed run compares like with like
    sig = json.loads(json.dumps({k: ["+".join(f) for f in v] if k == "families" else v for k, v in vars(a).items()
                                 if k not in ("out", "threads")}))
    state = load_state(path, sig)
    if state is None:
        state = {"done": []}
    else:
        for k, v in state["sel"].items():
            sels[k].load_state(v)
        print(f"resuming: {len(state['done'])} blocks done", flush=True)
    if a.variant == "tied":
        run_tied(ctx, a, arms, sels, state, path, sig, O, Y, layout, pop, n)
    elif a.variant == "full":
        run_full(ctx, a, sels["LEAF-full"], state, path, sig, O, Y, layout, pop, n)
    else:
        run_lgn(ctx, a, arms, sels, state, path, sig, O, Y, layout, pop, n, m)

    res = {}
    for nm, sel in sels.items():
        r = sel.result()
        res[nm] = {"test": {KEY: [r["test_mean"], r["test_sd"]]}, "chosen": r["chosen"], "per_split": r["per_split"],
                   "val_best": r["val_best"], "per_user": r["per_user"], "n_configs": r["n_configs"]}
        print(f"{nm}: test {r['test_mean']:.4f} +- {r['test_sd']:.4f} (val {r['val_mean']:.4f}, "
              f"{r['n_configs']} configurations) chosen {r['chosen'][0]}", flush=True)
    with open(out, "w") as f:
        json.dump({"args": sig, "results": res}, f)
    if os.path.exists(path):
        os.remove(path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"wrote {out} [{time.time() - t0:.0f}s, peak RSS {peak:.0f} MB]", flush=True)


if __name__ == "__main__":
    main()
