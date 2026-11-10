"""DAX on KuaiRand (paper/v7/DESIGN.md, Sections 5 and 6, item 2): detection-adjusted item appeal added to a backbone.

Backbones (--backbone):
  leaf_lgn      LEAF-LGN (the lr, wd, w grid of experiments/run_leaf.py --variant lgn)
  ease          EASE(click + w shown) + beta * popularity (E0 of the probes); DAX post hoc on top (E1)
  lightgcn      LightGCN, retrained at the stored configurations of results/v10/learned_kuairand_t_base.json
  qp_posthoc    the QP family (DR-BPR-LGN-1 / -p2 / -p), retrained at the stored configurations of
                results/v10/learned_kuairand_t_qp.json (--qp_source stored, the default) or at every configuration of
                the run_learned grid (--qp_source grid, for a dataset without stored selections, KuaiRand-1K); the
                fair baseline of the gates is QP family + DAX
Modes (--mode, leaf_lgn and lightgcn; qp_posthoc and ease are posthoc):
  posthoc       the backbone is trained alone; at every epoch every DAX configuration and gamma is evaluated on
                validation (selection over epoch x DAX x gamma, per split)
  joint         LEAF-LGN + DAX with the DAX weight trained by the BPR objective (drup.dax.LEAFLGNDAX), one
                training per (lr, wd, w, DAX configuration)
--dax_grid: posthoc full = the 32 configurations of the probe P3* (prior {80, 160} x centre {tag, content} x k_life
{0, 100} x tab {off, on} x stratum cells {off, 80}); posthoc small = prior 80, tag centre, k_life 100, tab on.
joint full = prior {40, 80, 160} x k_life {0, 100}, tab on; joint small = prior 80, k_life 100, tab on.
--ablate (leaf_lgn posthoc only; the full arm is always run): each arm is its own method, selected like the full arm.
  no_tab, no_life_prior, no_tag_prior   the DAX grid with that switch at its off value (drup.dax.dax_variant), duplicates
                                        removed (16 configurations each on the full grid)
  additive      MRP-style logistic with tab effects (drup.dax.fit_additive), 32 configurations matched to the DAX
                grid (additive_grid): item ridge lam_e, tag-effect ridge lam_g, tab on / off
  shuffle_items alpha permuted within tag (the tag mean kept); shuffle_tags alpha permuted across all items
                (both run once per permutation seed, --perm_seeds). Each seed is its own arm ("...[shuffle_items|p0]"),
                and "...[shuffle_items]" is the mean of the seeds' per-split test values.
--select_from FILE (leaf_lgn posthoc only): the LEAF-LGN+DAX configurations chosen per split in FILE (a results file of
this script). Only the (lr, wd, w) trainings chosen by some split are run. With --epoch_window W >= 0 each split reads
only the epochs e_s - W .. e_s + W around its chosen epoch e_s (the ablation setting), with no patience; with W < 0
(default) every split reads every epoch of those trainings, with patience, as in the main run.
--train_seed S: the seed of the model initialisation and of the BPR sampling (the splits do not depend on it, see
make_splits). The main run and the Stage C replicates use S = 0 (main) and 1, 2, 3.
--dataset kuairand27k (default) or kuairand1k (KuaiRand-1K, all 1000 users, lifelong tables data/raw/lifelong_1k.npz).
--gammas, --no_zcache: gamma grid (default the 10 gammas of the main run); no_zcache recomputes every arm's DAX channel at
every epoch instead of caching it (memory: the cache holds one 2.3M-cell matrix per arm and configuration).

Output: the run_learned / run_leaf format, results[name] = {test, chosen, per_split, val_best, per_user, n_configs,
n_evals}, plus "plan" (the trainings and the epoch windows read by each split) and "dataset". n_configs counts
configurations (backbone training x DAX configuration x gamma), n_evals the validation evaluations (configuration x
epoch). per_user keys are dataset row ids. Resumable: OUT.partial.json is written after every training block (one
LEAF-LGN training, one joint training, one learned-backbone training, or one EASE w).

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
from drup.dax import (DAXChannel, LEAFLGNDAX, all_perm, dax_variant, fit_additive, fit_dax_dataset,  # noqa: E402
                      item_content_codes, lifelong_rate, masked_rowstd, pre_cut, tag_codes, user_strata,
                      within_code_perm)
from drup.estimation import baseline_imputation, clip_propensity, get_propensity  # noqa: E402
from drup.leaf import (CUT_MS, LEAFLGN, BucketSelector, Layout, LifelongTables, check_temporal_protocol,  # noqa: E402
                       collective_gram, collective_solve, compact_base, fit_bpr_policy, make_lightgcn,
                       protocol_split)

DATA = os.path.join(ROOT, "data", "raw")
V10 = os.path.join(ROOT, "results", "v10")
SIG_EXCLUDE = ("out", "threads")
KEY = "ndcg@10"
GAMMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
GAMMAS_SMALL = (0.1, 0.25, 0.5, 0.7, 1.0)          # the gamma grid of the ablation setting (--gammas)
ABLATIONS = ("no_tab", "no_life_prior", "no_tag_prior", "additive", "shuffle_items", "shuffle_tags")
ADD_LAM_E = (5.0, 20.0, 80.0, 320.0)
ADD_LAM_G = (0.1, 1.0, 10.0, 100.0)
QP_METHODS = ("DR-BPR-LGN-1", "DR-BPR-LGN-p2", "DR-BPR-LGN-p")
INF = float("inf")
DATASETS = {
    "kuairand27k": {"version": "pure", "label": "KuaiRand-27K", "lifelong": os.path.join(DATA, "lifelong_27k.npz"),
                    "video": os.path.join(DATA, "KuaiRand-Pure", "data", "video_features_basic_pure.csv"),
                    "cache": os.path.join(DATA, "kuairand_t.pt")},
    "kuairand1k": {"version": "1k", "label": "KuaiRand-1K", "lifelong": os.path.join(DATA, "lifelong_1k.npz"),
                   "video": os.path.join(DATA, "KuaiRand-1K", "data", "video_features_basic_1k.csv"),
                   "cache": os.path.join(DATA, "kuairand_1k_t.pt")},
}


PRIOR_DEFAULT = (80.0, 160.0)      # prior strength grid of the main run (--prior_grid extends it)
KLIFE_DEFAULT = (0.0, 100.0)       # k_life grid of the main run (--klife_grid extends it)


def post_grid(small=False, priors=PRIOR_DEFAULT, klifes=KLIFE_DEFAULT):
    """Post-hoc DAX grid. The defaults are the 32 configurations of the main run (prior {80, 160} x centre {tag,
    content} x k_life {0, 100} x tab x stratum cells {off, 80}); priors and klifes extend the two grids (v12)."""
    if small:
        return [dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)]
    return [dict(prior=float(p), centre=c, k_life=float(k), tab=t, prior_s=ps)
            for p, c, k, t, ps in itertools.product(priors, ("tag", "content"), klifes, (False, True),
                                                    (None, 80.0))]


def joint_grid(small=False):
    if small:
        return [dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)]
    return [dict(prior=p, centre="tag", k_life=k, tab=True, prior_s=None)
            for p, k in itertools.product((40.0, 80.0, 160.0), (0.0, 100.0))]


def additive_grid():
    """32 configurations of the MRP-style arm, the size of the full DAX grid: item ridge lam_e x tag-effect ridge lam_g
    x tab effects on / off (the fit is drup.dax.fit_additive)."""
    return [dict(model="additive", lam_e=e, lam_g=g, tab=t)
            for e, g, t in itertools.product(ADD_LAM_E, ADD_LAM_G, (False, True))]


def make_splits(test, n_splits, frac_val):
    """Validation / test splits of the test users (drup.leaf.protocol_split, seeds 0 .. n_splits-1). They depend on the
    split index and frac_val only: a replicate with another --train_seed reads the same splits."""
    ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
    return [protocol_split(ctest, s, frac_val) for s in range(n_splits)]


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


def lifelong_npz(path):
    L = np.load(path, allow_pickle=True)
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
    (DAX configuration, DAXChannel) and every configuration is combined with every gamma of self.gammas. add_epoch
    handles one epoch of one training and returns the best validation value per split over the evaluations it
    scored (None for the splits not active), which drives the per-split patience of the training. A split that is not
    active at an epoch is not scored there (windowed selection)."""

    def __init__(self, layout, splits, row_ids, lengths, base_name, arms, base_std=True, gammas=GAMMAS, cache_z=True):
        self.layout, self.base_std, self.n_splits = layout, base_std, len(splits)
        self.rows = [bk["rows"] for bk in layout.buckets]
        self.cand = [bk["cand"] for bk in layout.buckets]
        self.mask = [torch.arange(bk["cand"].shape[1])[None, :] < torch.as_tensor(lengths[bk["k"]])[:, None]
                     for bk in layout.buckets]
        self.base_name, self.arms = base_name, arms
        self.gammas, self.cache_z = tuple(gammas), cache_z
        self.zcache = {}      # (arm, configuration index) -> row-standardised DAX channel per bucket (epoch-free)
        self.sels = {base_name: BucketSelector(layout, splits, (10,), KEY, row_ids)}
        for name, _ in arms:
            self.sels[name] = BucketSelector(layout, splits, (10,), KEY, row_ids)

    def channel_z(self, name, j, ch):
        if not self.cache_z:
            return [masked_rowstd(ch.compact(r, c), mk) for r, c, mk in zip(self.rows, self.cand, self.mask)]
        if (name, j) not in self.zcache:
            self.zcache[(name, j)] = [masked_rowstd(ch.compact(r, c), mk)
                                      for r, c, mk in zip(self.rows, self.cand, self.mask)]
        return self.zcache[(name, j)]

    def add_epoch(self, base_b, cfg, ep, active):
        best = [None] * self.n_splits

        def take(vals):
            for s, v in enumerate(vals):
                if v is not None:
                    best[s] = v if best[s] is None else max(best[s], v)
        any_on = any(active)
        if not any_on and ep > 0:          # nothing to score at this epoch
            return best
        base = None
        if any_on:
            base = [masked_rowstd(b, mk) for b, mk in zip(base_b, self.mask)] if self.base_std else base_b
        take(self.sels[self.base_name].add(base, dict(cfg, epoch=ep), active=active, new_config=(ep == 0)))
        for name, cfgs in self.arms:
            for j, (dcfg, ch) in enumerate(cfgs):
                Z = self.channel_z(name, j, ch) if any_on else None
                for g in self.gammas:
                    S = [b + g * z for b, z in zip(base, Z)] if any_on else None
                    take(self.sels[name].add(S, dict(cfg, epoch=ep, dax=dcfg, gamma=g), active=active,
                                             new_config=(ep == 0)))
        return best

    def states(self):
        return {k: v.state() for k, v in self.sels.items()}

    def load(self, st):
        for k, v in st.items():
            self.sels[k].load_state(v)


def dax_arms(d, dax_name, grid, ablate, life, strata, content, tags, perm_seeds):
    """[(name, [(DAX configuration, DAXChannel), ...])] for the full arm and every requested ablation. The fits are
    cached: the shuffles and the switched configurations reuse the fits of the full grid."""
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
        if ab in ("no_tab", "no_life_prior", "no_tag_prior"):
            kws = []
            for c in grid:
                kw = dax_variant(c, ab)
                if kw not in kws:
                    kws.append(kw)
            arms.append((f"{dax_name}[{ab}]", [(kw, channel(kw)) for kw in kws]))
        elif ab == "additive":
            cfgs = []
            for c in additive_grid():
                la = fit_additive(log, len(d["users_raw"]), n, tags, lam_e=c["lam_e"], lam_g=c["lam_g"],
                                  tab=c["tab"]).numpy()
                cfgs.append((c, DAXChannel(la)))
            arms.append((f"{dax_name}[additive]", cfgs))
        elif ab in ("shuffle_items", "shuffle_tags"):
            for p in perm_seeds:
                perm = within_code_perm(tags, seed=p) if ab == "shuffle_items" else all_perm(n, seed=p)
                cfgs = [(dict(c, shuffle=ab, perm_seed=p), channel(c, perm)) for c in grid]
                arms.append((f"{dax_name}[{ab}|p{p}]", cfgs))
        else:
            raise ValueError(f"unknown ablation {ab!r}")
    return arms


def leaf_plan(a, S, select=None):
    """The LEAF-LGN trainings and the epochs each split reads. A list of dicts (lr, wd, w, splits, lo, hi).
    Without select: the lr x wd x w grid, every split reads every epoch (lo 0, hi inf: patience decides).
    With select (the results file of a run of this script): one training per (lr, wd, w) that some split chose in
    LEAF-LGN+DAX. With a.epoch_window < 0 every split reads every epoch of those trainings; with W >= 0 split s reads
    the epochs e_s - W .. e_s + W (capped to the run length) of the training it chose, e_s its chosen epoch."""
    if select is None:
        return [dict(lr=lr, wd=wd, w=w, splits=list(range(S)), lo={s: 0 for s in range(S)},
                     hi={s: INF for s in range(S)})
                for lr, wd, w in itertools.product(a.lrs, a.wds, a.w)]
    chosen = json.load(open(select))["results"]["LEAF-LGN+DAX"]["chosen"]
    assert len(chosen) == S, f"{select} has {len(chosen)} splits, the run has {S}"
    groups = {}
    for s, c in enumerate(chosen):
        groups.setdefault((c["lr"], c["wd"], c["w"]), []).append((s, int(c["epoch"])))
    plans = []
    for (lr, wd, w), items in groups.items():
        if a.epoch_window < 0:
            splits = list(range(S))
            lo, hi = {s: 0 for s in splits}, {s: INF for s in splits}
        else:
            splits = [s for s, _ in items]
            lo = {s: max(0, e - a.epoch_window) for s, e in items}
            hi = {s: min(a.epochs - 1, e + a.epoch_window) for s, e in items}
        plans.append(dict(lr=lr, wd=wd, w=w, splits=splits, lo=lo, hi=hi))
    return plans


def qp_jobs(a):
    """(method, configuration) trainings of the QP family (or of LightGCN). Stored: the configurations chosen by any
    split of the stored v10 run (the main run's search space). Grid: every configuration of run_learned's grid (lr x wd
    x the method's extra grid, dim 64, 2 layers)."""
    if a.backbone == "lightgcn":
        src = json.load(open(os.path.join(V10, "learned_kuairand_t_base.json")))["results"]
        methods = ["LightGCN"]
    else:
        src = json.load(open(os.path.join(V10, "learned_kuairand_t_qp.json")))["results"]
        methods = list(QP_METHODS)
    jobs = []
    if a.qp_source == "stored":
        for meth in methods:
            for c in src[meth]["chosen"]:
                cfg = {k: v for k, v in c.items() if k != "epoch"}
                if (meth, cfg) not in jobs:
                    jobs.append((meth, cfg))
        return jobs
    for meth in methods:
        extra = RL.METHODS[meth][2]
        for vals in itertools.product(a.lrs, a.wds, *extra.values()):
            cfg = dict(lr=vals[0], wd=vals[1], dim=64, layers=2, **dict(zip(extra.keys(), vals[2:])))
            jobs.append((meth, cfg))
    return jobs


def qp_select_jobs(a):
    """QP-family trainings restricted to the configurations chosen per split in --select_from (a results file of this
    script, method "QP family+DAX"): one job per distinct (method, configuration), with the splits that chose it and
    the window e_s - W .. e_s + W of each of them (W = --epoch_window >= 0). Same trainings as qp_jobs, fewer epochs
    read and no DAX search outside the window."""
    chosen = json.load(open(a.select_from))["results"]["QP family+DAX"]["chosen"]
    S = len(chosen)
    groups = {}
    for s, c in enumerate(chosen):
        cfg = {k: v for k, v in c.items() if k not in ("epoch", "dax", "gamma")}
        meth = cfg.pop("method")
        groups.setdefault(json.dumps([meth, cfg], sort_keys=True), (meth, cfg, []))[2].append((s, int(c["epoch"])))
    jobs = []
    for meth, cfg, items in groups.values():
        W = a.epoch_window
        win = dict(splits=[s for s, _ in items], lo={s: max(0, e - W) for s, e in items},
                   hi={s: min(a.epochs - 1, e + W) for s, e in items})
        jobs.append((meth, cfg, win))
    assert len({s for _, _, w in jobs for s in w["splits"]}) == S, "a split chose no configuration"
    return jobs


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


def run_leaf_posthoc(a, d, post, state, save, layout, plans, npz, pop, m, n):
    """LEAF-LGN trained alone, one training per plan entry (the lr, wd, w grid of run_leaf unless --select_from); DAX and
    gamma selected post hoc on validation, per split, over the epochs that split reads."""
    tabs = LifelongTables(npz, d["users_raw"], d["item_feat"], window="pre", s=a.s)
    O, Y = d["O"], d["Y"]
    S = post.n_splits
    patience = 10 ** 9 if a.epoch_window >= 0 else a.patience    # windowed splits read their window, no patience
    for pl in plans:
        lr, wd, w = pl["lr"], pl["wd"], pl["w"]
        key = f"lgn|lr{lr}|wd{wd}|w{w}"
        if key in state["done"]:
            continue
        t0 = time.time()
        base = make_lightgcn(m, n, Y, a.dim, a.layers, seed=a.train_seed)
        model = LEAFLGN(base, tabs, pop, None)
        best, bad = [-1.0] * S, [0] * S
        done = [s not in pl["splits"] for s in range(S)]

        def on_epoch(mod, ep, pl=pl, lr=lr, wd=wd, w=w, best=best, bad=bad, done=done):
            active = [(not done[s]) and pl["lo"].get(s, 0) <= ep <= pl["hi"].get(s, INF) for s in range(S)]
            base_b = None
            if any(active):
                with torch.no_grad():
                    emb = mod.embeddings()
                    base_b = [mod(bk["rows"][:, None], bk["cand"], emb) for bk in layout.buckets]
            vals = post.add_epoch(base_b, dict(lr=lr, wd=wd, w=w), ep, active=active)
            patience_update(best, bad, done, vals, patience)
            return all(done[s] or ep >= pl["hi"][s] for s in pl["splits"])
        fit_bpr_policy(model, O, Y, w, lr, wd, a.epochs, a.batch, on_epoch, seed=a.train_seed)
        print(f"  {key}: splits {pl['splits']} [{time.time() - t0:.0f}s]", flush=True)
        del model, base
        state["done"].append(key)
        save()


def run_leaf_joint(a, d, sel, state, save, layout, pop, m, n, grid, npz, life, strata):
    """LEAF-LGN + DAX with the DAX weight trained by the BPR objective (drup.dax.LEAFLGNDAX)."""
    tabs = LifelongTables(npz, d["users_raw"], d["item_feat"], window="pre", s=a.s)
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
            base = make_lightgcn(m, n, Y, a.dim, a.layers, seed=a.train_seed)
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
            fit_bpr_policy(model, O, Y, w, lr, wd, a.epochs, a.batch, on_epoch, seed=a.train_seed)
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


def run_learned_posthoc(a, d, post, state, save, layout, jobs):
    """LightGCN or the QP family, each (method, configuration) of jobs retrained with the unchanged training code
    (experiments/learned_ctx.train_cfg, seed --train_seed); selection over epoch x DAX x gamma per split. A job is
    (method, configuration) or (method, configuration, window) with window = {splits, lo, hi} (qp_select_jobs): only
    the listed splits read the epochs lo..hi of that training, and the training has no patience then."""
    ctx = learned_context(d, threads=a.threads)
    ja = {"epochs": a.epochs, "batch": a.batch, "patience": a.patience, "threads": a.threads, "n_neg": 64,
          "adv_lr": 1e-4, "adv_interval": 5, "adv_min_steps": 100, "pairs_per_epoch": None}
    print(f"{len(jobs)} (method, configuration) trainings", flush=True)
    S = post.n_splits
    k_idx = [torch.as_tensor(bk["k"]) for bk in layout.buckets]
    for job in jobs:
        meth, cfg = job[0], job[1]
        win = job[2] if len(job) > 2 else None
        key = f"{meth}|{json.dumps(cfg, sort_keys=True)}"
        if key in state["done"]:
            continue
        t0 = time.time()
        pat = a.patience if win is None else 10 ** 9
        ja_job = dict(ja, patience=pat)
        best, bad = [-1.0] * S, [0] * S
        done = [False] * S if win is None else [s not in win["splits"] for s in range(S)]

        def active_at(ep, done=done, win=win):
            if win is None:
                return [not x for x in done]
            return [(not done[s]) and win["lo"].get(s, 0) <= ep <= win["hi"].get(s, INF) for s in range(S)]

        def record(Sd, ep, cfg=cfg, meth=meth, best=best, bad=bad, done=done, win=win, pat=pat):
            base_b = [Sd[kk[:, None], bk["cand"]] for kk, bk in zip(k_idx, layout.buckets)]
            vals = post.add_epoch(base_b, dict(method=meth, **cfg), ep, active=active_at(ep))
            patience_update(best, bad, done, vals, pat)
            if win is None:
                return all(done)
            return all(done[s] or ep >= win["hi"][s] for s in win["splits"])
        LC.train_cfg(ctx, meth, cfg, a.train_seed, ja_job, record)
        print(f"  {key} [{time.time() - t0:.0f}s]", flush=True)
        state["done"].append(key)
        save()


def average_arms(rs):
    """Mean over the permutation seeds of one shuffle arm. Per split the per-user test values are averaged over the
    seeds (every seed selects its own configuration on validation); the arm's test value is the mean over users, as in
    BucketSelector. n_configs and n_evals are summed over the seeds."""
    S = len(rs[0]["per_split"])
    per_user, per_split = [], []
    for s in range(S):
        acc = {}
        for r in rs:
            for u, v in r["per_user"][s].items():
                acc.setdefault(u, []).append(v)
        pu = {u: float(np.mean(v)) for u, v in acc.items()}
        per_user.append(pu)
        per_split.append({KEY: float(np.mean(list(pu.values())))})
    tests = [p[KEY] for p in per_split]
    return {"test": {KEY: [float(np.mean(tests)), float(np.std(tests))]}, "chosen": rs[0]["chosen"],
            "chosen_by_perm_seed": [r["chosen"] for r in rs], "per_split": per_split,
            "val_best": [float(np.mean([r["val_best"][s] for r in rs])) for s in range(S)],
            "per_user": per_user, "n_configs": sum(r["n_configs"] for r in rs),
            "n_evals": sum(r["n_evals"] for r in rs)}


def result_of(r):
    return {"test": {KEY: [r["test_mean"], r["test_sd"]]}, "chosen": r["chosen"], "per_split": r["per_split"],
            "val_best": r["val_best"], "per_user": r["per_user"], "n_configs": r["n_configs"],
            "n_evals": r["n_evals"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["leaf_lgn", "ease", "lightgcn", "qp_posthoc"])
    ap.add_argument("--mode", default="posthoc", choices=["posthoc", "joint"])
    ap.add_argument("--dataset", default="kuairand27k", choices=sorted(DATASETS))
    ap.add_argument("--dax_grid", default="full", choices=["full", "small"])
    ap.add_argument("--ablate", nargs="*", default=[], choices=list(ABLATIONS))
    ap.add_argument("--perm_seeds", type=int, nargs="+", default=[0], help="permutation seeds of the shuffle arms")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    ap.add_argument("--no_zcache", action="store_true",
                    help="recompute every arm's DAX channel at every epoch instead of caching it")
    ap.add_argument("--prior_grid", type=float, nargs="+", default=list(PRIOR_DEFAULT),
                    help="posthoc full grid: prior strengths (default 80 160; v12 extension e.g. 10 20 40 80 160)")
    ap.add_argument("--klife_grid", type=float, nargs="+", default=list(KLIFE_DEFAULT),
                    help="posthoc full grid: k_life values (default 0 100; v12 extension e.g. 0 100 200 400 1000)")
    ap.add_argument("--select_from", default=None,
                    help="chosen configurations to restrict to: LEAF-LGN+DAX of a results file (leaf_lgn), or "
                         "QP family+DAX of a results file (qp_posthoc)")
    ap.add_argument("--epoch_window", type=int, default=-1,
                    help="with --select_from: epochs read by each split around its chosen epoch (-1: all epochs)")
    ap.add_argument("--qp_source", default="stored", choices=["stored", "grid"])
    ap.add_argument("--train_seed", type=int, default=0, help="model initialisation and BPR sampling seed")
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
    if a.backbone not in ("leaf_lgn", "qp_posthoc") and (a.ablate or a.select_from):
        ap.error("--ablate is for leaf_lgn; --select_from is for leaf_lgn and qp_posthoc")
    if a.backbone == "qp_posthoc" and a.ablate:
        ap.error("--ablate is for leaf_lgn")
    if a.backbone == "qp_posthoc" and a.select_from and a.epoch_window < 0:
        ap.error("qp_posthoc with --select_from needs --epoch_window >= 0")
    if a.mode == "joint" and (a.ablate or a.select_from):
        ap.error("--ablate and --select_from are for --mode posthoc (the ablation arms are post hoc)")
    if a.epoch_window >= 0 and a.select_from is None:
        ap.error("--epoch_window needs --select_from")
    if a.qp_source == "grid" and a.backbone != "qp_posthoc":
        ap.error("--qp_source grid is for qp_posthoc")
    torch.set_num_threads(a.threads)
    path = a.out + ".partial.json"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    ds = DATASETS[a.dataset]

    d = load_kuairand_rich(version=ds["version"], n_users=(a.n_users or None), train_until=CUT_MS,
                           dense_dtype=torch.float32, with_signals=False)
    print("temporal protocol:", check_temporal_protocol(d), flush=True)
    if a.n_users == 0 and os.path.exists(ds["cache"]):
        check_cache(ds["cache"], d)
    test = d["test"]
    m, n = len(d["users_raw"]), len(d["item_feat"])
    row_ids = np.array([u for u, _, _ in test], dtype=np.int64)
    lengths = np.array([len(it) for _, it, _ in test], dtype=np.int64)
    layout = Layout(test)
    splits = make_splits(test, a.seeds, a.frac_val)
    X = d["O"] * d["Y"]
    pop = torch.log1p(X.sum(0))
    pop = (pop - pop.mean()) / pop.std().clamp_min(1e-12)
    del X
    print(f"{ds['label']}: {m} users, {n} items, {len(test)} test users in {len(layout.buckets)} candidate buckets "
          f"[{time.time() - t0:.0f}s]", flush=True)

    npz = lifelong_npz(ds["lifelong"])
    life = lifelong_rate(d["users_raw"], ds["lifelong"])
    strata = user_strata(d["users_raw"], ds["lifelong"])
    tags = tag_codes(d["item_feat"])
    dax_name = {"leaf_lgn": "LEAF-LGN", "ease": "EASE", "lightgcn": "LightGCN", "qp_posthoc": "QP family"}[a.backbone]
    dax_name += "+DAX" if a.mode == "posthoc" else "+DAX (joint)"
    base_name = dax_name.split("+DAX")[0]

    sig = json.loads(json.dumps({k: v for k, v in vars(a).items() if k not in SIG_EXCLUDE}))
    state = {"done": []}
    if a.mode == "posthoc":
        grid = post_grid(a.dax_grid == "small", tuple(a.prior_grid), tuple(a.klife_grid))
        content = item_content_codes(d["item_feat"], d["items_raw"], video=ds["video"], path=ds["lifelong"])
        arms = dax_arms(d, dax_name, grid, a.ablate, life, strata, content, tags, a.perm_seeds)
        if a.backbone == "ease":
            arms = arms[:1]
        post = Posthoc(layout, splits, row_ids, lengths, base_name, arms, base_std=(a.backbone != "ease"),
                       gammas=a.gammas, cache_z=not a.no_zcache)
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

    plans = None
    if a.backbone == "leaf_lgn" and a.mode == "posthoc":
        plans = leaf_plan(a, len(splits), a.select_from)
        print("trainings: " + "; ".join(f"lr {p['lr']} wd {p['wd']} w {p['w']} splits {p['splits']}" for p in plans),
              flush=True)

    def save():
        sel = post.states() if a.mode == "posthoc" else {dax_name: post.state()}
        with open(path + ".tmp", "w") as f:
            json.dump({"sig": sig, "state": state, "sel": sel}, f)
        os.replace(path + ".tmp", path)

    if plans is not None:
        run_leaf_posthoc(a, d, post, state, save, layout, plans, npz, pop, m, n)
    elif a.backbone == "leaf_lgn":
        run_leaf_joint(a, d, post, state, save, layout, pop, m, n, grid, npz, life, strata)
    elif a.backbone == "ease":
        run_ease(a, d, post, state, save, layout, pop, n)
    else:
        run_learned_posthoc(a, d, post, state, save, layout, qp_select_jobs(a) if a.select_from else qp_jobs(a))

    res = {}
    for nm in names:
        res[nm] = result_of(sels[nm].result())
    for ab in a.ablate:
        if ab in ("shuffle_items", "shuffle_tags"):
            parts = [f"{dax_name}[{ab}|p{p}]" for p in a.perm_seeds]
            res[f"{dax_name}[{ab}]"] = average_arms([res[pn] for pn in parts])
    for nm, r in res.items():
        print(f"{nm}: test {r['test']['ndcg@10'][0]:.4f} +- {r['test']['ndcg@10'][1]:.4f} "
              f"(val {np.mean(r['val_best']):.4f}, {r['n_configs']} configurations, {r['n_evals']} evaluations) "
              f"chosen {r['chosen'][0]}", flush=True)
    plan_out = None if plans is None else [dict(lr=p["lr"], wd=p["wd"], w=p["w"], splits=p["splits"],
                                                 lo={str(k): v for k, v in p["lo"].items()},
                                                 hi={str(k): (None if v == INF else v) for k, v in p["hi"].items()})
                                           for p in plans]
    with open(a.out, "w") as f:
        json.dump({"args": sig, "results": res, "plan": plan_out, "dataset": a.dataset}, f)
    if os.path.exists(path):
        os.remove(path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"wrote {a.out} [{time.time() - t0:.0f}s, peak RSS {peak:.0f} MB]", flush=True)


if __name__ == "__main__":
    main()
