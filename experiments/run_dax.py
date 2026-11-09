"""DAX on KuaiRand (paper/v7/DESIGN.md, Sections 5 and 6, item 2): detection-adjusted item appeal added to a backbone.

Backbones (--backbone):
  leaf_lgn      LEAF-LGN (the lr, wd, w grid of experiments/run_leaf.py --variant lgn)
  ease          EASE(click + w shown) + beta * popularity (E0 of the probes); DAX post hoc on top (E1)
  lightgcn      LightGCN, retrained at the stored configurations of results/v10/learned_kuairand_t_base.json
  qp_posthoc    the QP family (DR-BPR-LGN-1 / -p2 / -p), retrained at the stored configurations of
                results/v10/learned_kuairand_t_qp.json; the fair baseline of the gates is QP family + DAX
Modes (--mode, leaf_lgn and lightgcn; qp_posthoc and ease are posthoc):
  posthoc       the backbone is trained alone; at every epoch every DAX configuration and gamma is evaluated on
                validation (selection over epoch x DAX x gamma, per split)
  joint         LEAF-LGN + DAX with the DAX weight trained by the BPR objective (drup.dax.LEAFLGNDAX), one
                training per (lr, wd, w, DAX configuration)
--dax_grid: posthoc full = the 32 configurations of the probe P3* (prior {80, 160} x centre {tag, content} x k_life
{0, 100} x tab {off, on} x stratum cells {off, 80}); posthoc small = prior 80, tag centre, k_life 100, tab on.
joint full = prior {40, 80, 160} x k_life {0, 100}, tab on; joint small = prior 80, k_life 100, tab on.
--ablate (leaf_lgn posthoc only; the full arm is always run): no_tab, no_life_prior, no_tag_prior, additive
(MRP-style logistic with tab effects), shuffle_items (alpha permuted within tag), shuffle_tags (alpha permuted
across all items). Every arm is its own method, selected like the full arm.

Output: the run_learned / run_leaf format, results[name] = {test, chosen, per_split, val_best, per_user, n_configs,
n_evals}. n_configs counts configurations (backbone training x DAX configuration x gamma), n_evals the validation
evaluations (configuration x epoch). per_user keys are dataset row ids. Resumable: OUT.partial.json is written after
every training block (one LEAF-LGN training, one joint training, one learned-backbone training, or one EASE w).

The DAX table is fitted on the pre-cut standard log only (drup.dax.fit_dax_dataset). The learned backbones use the
unchanged training code of experiments/learned_ctx.py on the same protocol.

python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --seeds 5 --out results/v11/dax_lgn_post.json
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
import learned_ctx as LC  # noqa: E402
import run_learned as RL  # noqa: E402
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.dax import (DAXChannel, LEAFLGNDAX, all_perm, fit_additive, fit_dax_dataset, item_content_codes,  # noqa: E402
                      lifelong_rate, masked_rowstd, pre_cut, tag_codes, user_strata, within_code_perm)
from drup.estimation import baseline_imputation, clip_propensity, get_propensity  # noqa: E402
from drup.leaf import (CUT_MS, LEAFLGN, BucketSelector, Layout, LifelongTables, check_temporal_protocol,  # noqa: E402
                       collective_gram, collective_solve, compact_base, fit_bpr_policy, make_lightgcn,
                       protocol_split)

LIFELONG = os.path.join(ROOT, "data", "raw", "lifelong_27k.npz")
CACHE = os.path.join(ROOT, "data", "raw", "kuairand_t.pt")
V10 = os.path.join(ROOT, "results", "v10")
SIG_EXCLUDE = ("out", "threads")
KEY = "ndcg@10"
GAMMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
ABLATIONS = ("no_tab", "no_life_prior", "no_tag_prior", "additive", "shuffle_items", "shuffle_tags")
LAM_E = (5.0, 20.0, 80.0)


def post_grid(small=False):
    if small:
        return [dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)]
    return [dict(prior=p, centre=c, k_life=k, tab=t, prior_s=ps)
            for p, c, k, t, ps in itertools.product((80.0, 160.0), ("tag", "content"), (0.0, 100.0), (False, True),
                                                    (None, 80.0))]


def joint_grid(small=False):
    if small:
        return [dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)]
    return [dict(prior=p, centre="tag", k_life=k, tab=True, prior_s=None)
            for p, k in itertools.product((40.0, 80.0, 160.0), (0.0, 100.0))]


def check_cache(path, d):
    """The cached kuairand_t.pt (the input of the stored learned runs) must hold the test lists and users of this
    loader; a stale cache would make the numbers incomparable (as in experiments/run_leaf.py)."""
    C = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    same = len(C["test"]) == len(d["test"]) and all(
        x[0] == y[0] and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])
        for x, y in zip(C["test"], d["test"]))
    assert same, f"{path}: test lists differ from the loader (stale cache?)"
    assert np.array_equal(np.asarray(C["users_raw"]), np.asarray(d["users_raw"])), f"{path}: users_raw differ"
    print(f"cache {path}: test lists and users match the loader", flush=True)


def lifelong_npz():
    L = np.load(LIFELONG, allow_pickle=True)
    return {k: L[k] for k in L.files if "_pre" in k or k.endswith("_values")}


def learned_context(d, prop="pop", floor=0.05, threads=2):
    """The data and nuisances of experiments/learned_ctx.prepare (imputation "add"), taken from the loaded dict d.
    Writes RL.PRIOR[0] and RL.QMAT[0] as prepare does."""
    torch.set_num_threads(threads)
    O, Y = d["O"].float(), d["Y"].float()
    m, n = O.shape
    P = clip_propensity(get_propensity(d, O, Y, prop).float(), floor)
    Yhat = baseline_imputation(O, Y, P, lam=5.0).float()
    lg = torch.logit(Yhat.clamp(1e-3, 1 - 1e-3))
    RL.PRIOR[0] = (lg - lg.mean()) / lg.std()
    RL.QMAT[0] = (Yhat * (1.0 - P) / (1.0 - P * Yhat).clamp_min(1e-6)).clamp(0.0, 1.0)
    pop = (O * Y).sum(0)
    item_pop = (pop / pop.max()).clamp_min(1.0 / pop.max())
    rows = torch.tensor(sorted({u for u, _, _ in d["test"]}))
    return {"O": O, "Y": Y, "P": P, "Yhat": Yhat, "item_pop": item_pop, "rows": rows, "m": m, "n": n}


class Posthoc:
    """Validation selection of the backbone alone (base) and of the backbone plus DAX. Every arm is a list of
    (DAX configuration, DAXChannel) and every configuration is combined with every gamma of GAMMAS. add_epoch
    handles one epoch of one training and returns the best validation value per split over all of its evaluations,
    which drives the per-split patience of the training."""

    def __init__(self, layout, splits, row_ids, lengths, base_name, arms, base_std=True):
        self.layout, self.base_std, self.n_splits = layout, base_std, len(splits)
        self.rows = [bk["rows"] for bk in layout.buckets]
        self.cand = [bk["cand"] for bk in layout.buckets]
        self.mask = [torch.arange(bk["cand"].shape[1])[None, :] < torch.as_tensor(lengths[bk["k"]])[:, None]
                     for bk in layout.buckets]
        self.base_name, self.arms = base_name, arms
        self.sels = {base_name: BucketSelector(layout, splits, (10,), KEY, row_ids)}
        for name, _ in arms:
            self.sels[name] = BucketSelector(layout, splits, (10,), KEY, row_ids)

    def add_epoch(self, base_b, cfg, ep, active):
        best = [None] * self.n_splits

        def take(vals):
            for s, v in enumerate(vals):
                if v is not None:
                    best[s] = v if best[s] is None else max(best[s], v)
        base = [masked_rowstd(b, mk) for b, mk in zip(base_b, self.mask)] if self.base_std else base_b
        take(self.sels[self.base_name].add(base, dict(cfg, epoch=ep), active=active, new_config=(ep == 0)))
        for name, cfgs in self.arms:
            for dcfg, ch in cfgs:
                Z = [masked_rowstd(ch.compact(r, c), mk) for r, c, mk in zip(self.rows, self.cand, self.mask)]
                for g in GAMMAS:
                    S = [b + g * z for b, z in zip(base, Z)]
                    take(self.sels[name].add(S, dict(cfg, epoch=ep, dax=dcfg, gamma=g), active=active,
                                             new_config=(ep == 0)))
        return best

    def states(self):
        return {k: v.state() for k, v in self.sels.items()}

    def load(self, st):
        for k, v in st.items():
            self.sels[k].load_state(v)


def dax_arms(d, dax_name, grid, ablate, life, strata, content, tags):
    """[(name, [(DAX configuration, DAXChannel), ...])] for the full arm and every requested ablation. The fits
    are cached: the ablations reuse the full-grid fits (shuffles permute them)."""
    cache = {}
    log = pre_cut(d["log"])
    n = len(tags)

    def fit(kw):
        key = json.dumps(kw, sort_keys=True)
        if key not in cache:
            t0 = time.time()
            la, _, _, la_s = fit_dax_dataset(d, content=content, life=life, strata=strata, **kw)
            print(f"  DAX fit {key} [{time.time() - t0:.1f}s]", flush=True)
            cache[key] = (la.numpy(), la_s.numpy() if la_s is not None else None)
        return cache[key]

    def channel(kw, perm=None):
        la, la_s = fit(kw)
        if perm is not None:
            la = la[perm]
            la_s = la_s[perm] if la_s is not None else None
        return DAXChannel(la, strata=strata if la_s is not None else None, logit_alpha_strata=la_s)

    arms = [(dax_name, [(c, channel(c)) for c in grid])]
    for ab in ablate:
        if ab == "no_tab":
            cfgs = [(c, channel(c)) for c in grid if not c["tab"]]
        elif ab == "no_life_prior":
            cfgs = [(c, channel(c)) for c in grid if c["k_life"] == 0.0]
        elif ab == "no_tag_prior":
            cfgs = [(dict(c, centre="global"), channel(dict(c, centre="global")))
                    for c in grid if c["centre"] == "tag"]
        elif ab == "additive":
            cfgs = []
            for lam in LAM_E:
                la = fit_additive(log, len(d["users_raw"]), n, tags, lam_e=lam, tab=True).numpy()
                cfgs.append((dict(model="additive", lam_e=lam), DAXChannel(la)))
        elif ab == "shuffle_items":
            perm = within_code_perm(tags, seed=0)
            cfgs = [(dict(c, shuffle="within_tag"), channel(c, perm)) for c in grid]
        elif ab == "shuffle_tags":
            perm = all_perm(n, seed=0)
            cfgs = [(dict(c, shuffle="all_items"), channel(c, perm)) for c in grid]
        else:
            raise ValueError(f"unknown ablation {ab!r}")
        arms.append((f"{dax_name}[{ab}]", cfgs))
    return arms


def patience_update(best, bad, done, vals, patience):
    for s, v in enumerate(vals):
        if v is None or done[s]:
            continue
        if v > best[s]:
            best[s], bad[s] = v, 0
        else:
            bad[s] += 1
            if bad[s] >= patience:
                done[s] = True


def run_leaf_posthoc(a, d, post, state, save, layout, pop, m, n):
    """LEAF-LGN trained alone (the lr, wd, w grid of run_leaf); DAX and gamma selected post hoc at every epoch."""
    tabs = LifelongTables(lifelong_npz(), d["users_raw"], d["item_feat"], window="pre", s=a.s)
    O, Y = d["O"], d["Y"]
    S = post.n_splits
    for lr, wd, w in itertools.product(a.lrs, a.wds, a.w):
        key = f"lgn|lr{lr}|wd{wd}|w{w}"
        if key in state["done"]:
            continue
        t0 = time.time()
        base = make_lightgcn(m, n, Y, a.dim, a.layers)
        model = LEAFLGN(base, tabs, pop, None)
        best, bad, done = [-1.0] * S, [0] * S, [False] * S

        def on_epoch(mod, ep, lr=lr, wd=wd, w=w, best=best, bad=bad, done=done):
            with torch.no_grad():
                emb = mod.embeddings()
                base_b = [mod(bk["rows"][:, None], bk["cand"], emb) for bk in layout.buckets]
            vals = post.add_epoch(base_b, dict(lr=lr, wd=wd, w=w), ep, active=[not x for x in done])
            patience_update(best, bad, done, vals, a.patience)
            return all(done)
        fit_bpr_policy(model, O, Y, w, lr, wd, a.epochs, a.batch, on_epoch, seed=0)
        print(f"  {key} [{time.time() - t0:.0f}s]", flush=True)
        del model, base
        state["done"].append(key)
        save()


def run_leaf_joint(a, d, sel, state, save, layout, pop, m, n, grid, life, strata):
    """LEAF-LGN + DAX with the DAX weight trained by the BPR objective (drup.dax.LEAFLGNDAX)."""
    tabs = LifelongTables(lifelong_npz(), d["users_raw"], d["item_feat"], window="pre", s=a.s)
    O, Y = d["O"], d["Y"]
    S = len(sel.best)
    for dcfg in grid:
        la, _, _, _ = fit_dax_dataset(d, life=life, strata=strata, **dcfg)
        chan = DAXChannel(la.numpy())
        for lr, wd, w in itertools.product(a.lrs, a.wds, a.w):
            key = f"joint|{json.dumps(dcfg, sort_keys=True)}|lr{lr}|wd{wd}|w{w}"
            if key in state["done"]:
                continue
            t0 = time.time()
            base = make_lightgcn(m, n, Y, a.dim, a.layers)
            model = LEAFLGNDAX(base, tabs, pop, chan)
            best, bad, done = [-1.0] * S, [0] * S, [False] * S

            def on_epoch(mod, ep, lr=lr, wd=wd, w=w, dcfg=dcfg, best=best, bad=bad, done=done):
                with torch.no_grad():
                    emb = mod.embeddings()
                    Sc = [mod(bk["rows"][:, None], bk["cand"], emb) for bk in layout.buckets]
                vals = sel.add(Sc, dict(lr=lr, wd=wd, w=w, dax=dcfg, epoch=ep), active=[not x for x in done],
                               new_config=(ep == 0))
                patience_update(best, bad, done, vals, a.patience)
                return all(done)
            fit_bpr_policy(model, O, Y, w, lr, wd, a.epochs, a.batch, on_epoch, seed=0)
            print(f"  {key}: g_dax {float(model.g_dax):.3f} [{time.time() - t0:.0f}s]", flush=True)
            del model, base
            state["done"].append(key)
            save()


def run_ease(a, d, post, state, save, layout, pop, n):
    """EASE(click + w shown) + beta * popularity (E0), with the DAX arm post hoc (E1). The base is the closed-form
    score of drup.leaf.compact_base (row-standardised over all items), as in the probes."""
    O, Y = d["O"], d["Y"]
    pop_b = [masked_rowstd(pop[bk["cand"]], mk) for bk, mk in zip(layout.buckets, post.mask)]
    for w in a.w:
        key = f"ease|w{w}"
        if key in state["done"]:
            continue
        t0 = time.time()
        G, Cc = collective_gram(O, Y, w)
        for lam in a.lam:
            B, _ = collective_solve(G, Cc, n, lam)
            base = [compact_base(O, Y, w, B, bk["rows"], bk["cand"]) for bk in layout.buckets]
            del B
            for beta in a.beta:
                post.add_epoch([b + beta * p for b, p in zip(base, pop_b)], dict(w=w, lam=lam, beta=beta), 0,
                               active=[True] * post.n_splits)
        del G, Cc
        print(f"  {key} [{time.time() - t0:.0f}s]", flush=True)
        state["done"].append(key)
        save()


def run_learned_posthoc(a, d, post, state, save, layout):
    """LightGCN or the QP family, each stored configuration of any split retrained with the unchanged training code
    (experiments/learned_ctx.train_cfg); selection over epoch x DAX x gamma per split."""
    ctx = learned_context(d, threads=a.threads)
    ja = {"epochs": a.epochs, "batch": a.batch, "patience": a.patience, "threads": a.threads, "n_neg": 64,
          "adv_lr": 1e-4, "adv_interval": 5, "adv_min_steps": 100, "pairs_per_epoch": None}
    if a.backbone == "lightgcn":
        src = json.load(open(os.path.join(V10, "learned_kuairand_t_base.json")))["results"]
        methods = ["LightGCN"]
    else:
        src = json.load(open(os.path.join(V10, "learned_kuairand_t_qp.json")))["results"]
        methods = ["DR-BPR-LGN-1", "DR-BPR-LGN-p2", "DR-BPR-LGN-p"]
    jobs = []
    for meth in methods:
        for c in src[meth]["chosen"]:
            cfg = {k: v for k, v in c.items() if k != "epoch"}
            if (meth, cfg) not in jobs:
                jobs.append((meth, cfg))
    print(f"{len(jobs)} (method, configuration) trainings", flush=True)
    S = post.n_splits
    k_idx = [torch.as_tensor(bk["k"]) for bk in layout.buckets]
    for meth, cfg in jobs:
        key = f"{meth}|{json.dumps(cfg, sort_keys=True)}"
        if key in state["done"]:
            continue
        t0 = time.time()
        best, bad, done = [-1.0] * S, [0] * S, [False] * S

        def record(Sd, ep, cfg=cfg, meth=meth, best=best, bad=bad, done=done):
            base_b = [Sd[kk[:, None], bk["cand"]] for kk, bk in zip(k_idx, layout.buckets)]
            vals = post.add_epoch(base_b, dict(method=meth, **cfg), ep, active=[not x for x in done])
            patience_update(best, bad, done, vals, a.patience)
            return all(done)
        LC.train_cfg(ctx, meth, cfg, 0, ja, record)
        print(f"  {key} [{time.time() - t0:.0f}s]", flush=True)
        state["done"].append(key)
        save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["leaf_lgn", "ease", "lightgcn", "qp_posthoc"])
    ap.add_argument("--mode", default="posthoc", choices=["posthoc", "joint"])
    ap.add_argument("--dax_grid", default="full", choices=["full", "small"])
    ap.add_argument("--ablate", nargs="*", default=[], choices=list(ABLATIONS))
    ap.add_argument("--seeds", type=int, default=5, help="number of validation / test splits")
    ap.add_argument("--frac_val", type=float, default=0.3)
    ap.add_argument("--n_users", type=int, default=0, help="0 = all users (smoke runs use a subset)")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-5])
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--lam", type=float, nargs="+", default=[1000.0, 3000.0, 10000.0])
    ap.add_argument("--s", type=float, default=5.0, help="Beta-binomial shrinkage of the lifelong lift (LEAF-LGN)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.backbone in ("qp_posthoc", "ease"):
        a.mode = "posthoc"
    if a.backbone in ("ease", "qp_posthoc", "lightgcn") and a.mode == "joint":
        ap.error("--mode joint is for leaf_lgn")
    if a.backbone != "leaf_lgn" and a.ablate:
        ap.error("--ablate is for leaf_lgn")
    torch.set_num_threads(a.threads)
    path = a.out + ".partial.json"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()

    d = load_kuairand_rich(version="pure", n_users=(a.n_users or None), train_until=CUT_MS,
                           dense_dtype=torch.float32, with_signals=False)
    print("temporal protocol:", check_temporal_protocol(d), flush=True)
    if a.n_users == 0:
        check_cache(CACHE, d)
    test = d["test"]
    m, n = len(d["users_raw"]), len(d["item_feat"])
    row_ids = np.array([u for u, _, _ in test], dtype=np.int64)
    lengths = np.array([len(it) for _, it, _ in test], dtype=np.int64)
    layout = Layout(test)
    ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
    splits = [protocol_split(ctest, s, a.frac_val) for s in range(a.seeds)]
    X = d["O"] * d["Y"]
    pop = torch.log1p(X.sum(0))
    pop = (pop - pop.mean()) / pop.std().clamp_min(1e-12)
    del X
    print(f"{m} users, {n} items, {len(test)} test users in {len(layout.buckets)} candidate buckets "
          f"[{time.time() - t0:.0f}s]", flush=True)

    life = lifelong_rate(d["users_raw"], LIFELONG)
    strata = user_strata(d["users_raw"], LIFELONG)
    tags = tag_codes(d["item_feat"])
    dax_name = {"leaf_lgn": "LEAF-LGN", "ease": "EASE", "lightgcn": "LightGCN", "qp_posthoc": "QP family"}[a.backbone]
    dax_name += "+DAX" if a.mode == "posthoc" else "+DAX (joint)"
    base_name = dax_name.split("+DAX")[0]

    sig = json.loads(json.dumps({k: v for k, v in vars(a).items() if k not in SIG_EXCLUDE}))
    state = {"done": []}
    if a.mode == "posthoc":
        grid = post_grid(a.dax_grid == "small")
        content = item_content_codes(d["item_feat"], d["items_raw"])
        arms = dax_arms(d, dax_name, grid, a.ablate, life, strata, content, tags)
        if a.backbone == "ease":
            arms = arms[:1]
        post = Posthoc(layout, splits, row_ids, lengths, base_name, arms, base_std=(a.backbone != "ease"))
        names = list(post.sels)
        sels = post.sels
    else:
        grid = joint_grid(a.dax_grid == "small")
        post = BucketSelector(layout, splits, (10,), KEY, row_ids)
        names = [dax_name]
        sels = {dax_name: post}

    if os.path.exists(path):
        st = json.load(open(path))
        if st["sig"] != sig:
            raise SystemExit(f"{path} was written with other arguments; remove it or change --out")
        state = st["state"]
        if a.mode == "posthoc":
            post.load(st["sel"])
        else:
            post.load_state(st["sel"][dax_name])
        print(f"resuming: {len(state['done'])} blocks done", flush=True)

    def save():
        sel = post.states() if a.mode == "posthoc" else {dax_name: post.state()}
        with open(path + ".tmp", "w") as f:
            json.dump({"sig": sig, "state": state, "sel": sel}, f)
        os.replace(path + ".tmp", path)

    if a.backbone == "leaf_lgn" and a.mode == "posthoc":
        run_leaf_posthoc(a, d, post, state, save, layout, pop, m, n)
    elif a.backbone == "leaf_lgn":
        run_leaf_joint(a, d, post, state, save, layout, pop, m, n, grid, life, strata)
    elif a.backbone == "ease":
        run_ease(a, d, post, state, save, layout, pop, n)
    else:
        run_learned_posthoc(a, d, post, state, save, layout)

    res = {}
    for nm in names:
        r = sels[nm].result()
        res[nm] = {"test": {KEY: [r["test_mean"], r["test_sd"]]}, "chosen": r["chosen"], "per_split": r["per_split"],
                   "val_best": r["val_best"], "per_user": r["per_user"], "n_configs": r["n_configs"],
                   "n_evals": r["n_evals"]}
        print(f"{nm}: test {r['test_mean']:.4f} +- {r['test_sd']:.4f} (val {r['val_mean']:.4f}, "
              f"{r['n_configs']} configurations, {r['n_evals']} evaluations) chosen {r['chosen'][0]}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"args": sig, "results": res}, f)
    if os.path.exists(path):
        os.remove(path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"wrote {a.out} [{time.time() - t0:.0f}s, peak RSS {peak:.0f} MB]", flush=True)


if __name__ == "__main__":
    main()
