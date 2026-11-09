"""Stage B / C code paths of DAX (paper/v7/DESIGN.md, Section 7): the training seed changes the model and not the
splits; the ablation arms reduce to full DAX when their switch is off; the restricted epoch windows of the ablation
label; the cache-free channel path selects the same configurations; the KuaiRand-1K loader and its lifelong tables
pass their leak checks (the last two need data/raw and are skipped without it)."""

import json
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from drup.dax import DAXChannel, dax_variant, fit_dax_dataset  # noqa: E402
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.leaf import CUT_MS, Layout, check_temporal_protocol, fit_bpr_policy, make_lightgcn  # noqa: E402
import run_dax as RD  # noqa: E402

T0 = CUT_MS


def _toy(seed=0, m=120, n=60, per_user=30, n_tags=6):
    rng = np.random.default_rng(seed)
    tag = np.arange(n) % n_tags
    rows = []
    for u in range(m):
        it = rng.choice(n, per_user, replace=False)
        tb = rng.integers(0, 3, per_user)
        y = (rng.random(per_user) < 0.3 * np.array([0.2, 1.0, 0.5])[tb]).astype(float)
        rows.append(pd.DataFrame({"u": u, "i": it, "tab": tb, "is_click": y, "time_ms": T0 - 10 ** 6}))
    d = {"log": pd.concat(rows, ignore_index=True), "item_feat": pd.DataFrame({"tag": [str(t) for t in tag]}),
         "users_raw": np.arange(m)}
    return d, rng.uniform(0.01, 0.5, m)


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a[:3], b[:3]))


def test_train_seed_changes_the_model_and_not_the_splits():
    """Seeds 1 and 0 give different LightGCN initialisations and different trained weights; the same seed reproduces the
    weights exactly. The splits take no training seed: make_splits depends on the test list, the count and frac_val."""
    import inspect
    assert "train_seed" not in inspect.signature(RD.make_splits).parameters
    rng = np.random.default_rng(0)
    m, n = 40, 30
    O = torch.zeros(m, n)
    Y = torch.zeros(m, n)
    for u in range(m):
        it = rng.choice(n, 12, replace=False)
        O[u, it] = 1.0
        Y[u, it[:3]] = 1.0
    emb = {}
    for seed in (0, 0, 1):
        base = make_lightgcn(m, n, Y, 8, 2, seed=seed)
        model = base
        for _ in range(2):
            fit_bpr_policy(model, O, Y, 0.5, 1e-2, 1e-4, 2, 64, lambda mod, ep: False, seed=seed)
        emb.setdefault(seed, []).append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.equal(emb[0][0], emb[0][1])
    assert not torch.allclose(emb[0][0], emb[1][0])
    test = [(u, np.arange(n)[:10], (rng.random(10) < 0.3).astype(float)) for u in range(m)]
    s_a = RD.make_splits(test, 3, 0.3)
    s_b = RD.make_splits(test, 3, 0.3)
    for (va, ta), (vb, tb) in zip(s_a, s_b):
        assert [u for u, _, _ in va] == [u for u, _, _ in vb] and [u for u, _, _ in ta] == [u for u, _, _ in tb]


def test_switch_off_is_the_full_dax_arm():
    """dax_variant(cfg) with no switch is cfg itself, and the fit is bitwise the full fit; each switched variant is the
    fit with that switch at its off value (no_tab: tab off; no_life_prior: k_life 0; no_tag_prior: global centre)."""
    d, life = _toy(1)
    for cfg in RD.post_grid():
        if cfg["centre"] != "tag" or cfg["prior_s"] is not None:
            continue            # the content centre needs item covariates, the stratum cells need user strata
        full = fit_dax_dataset(d, life=life, **cfg)
        assert dax_variant(cfg) == cfg
        assert _same(fit_dax_dataset(d, life=life, **dax_variant(cfg, None)), full)
        assert _same(fit_dax_dataset(d, life=life, **dax_variant(cfg, "no_tab")),
                     fit_dax_dataset(d, life=life, **dict(cfg, tab=False)))
        assert _same(fit_dax_dataset(d, life=life, **dax_variant(cfg, "no_life_prior")),
                     fit_dax_dataset(d, life=None, **dict(cfg, k_life=0.0)))
        assert _same(fit_dax_dataset(d, life=life, **dax_variant(cfg, "no_tag_prior")),
                     fit_dax_dataset(d, life=life, **dict(cfg, centre="global")))
    switched = {ab: {repr(dax_variant(c, ab)) for c in RD.post_grid()} for ab in ("no_tab", "no_life_prior",
                                                                                 "no_tag_prior")}
    assert all(len(v) == 16 for v in switched.values()), {k: len(v) for k, v in switched.items()}
    assert len(RD.post_grid()) == 32 and len(RD.additive_grid()) == 32


def test_epoch_windows_and_restricted_trainings(tmp_path):
    chosen = [dict(lr=0.003, wd=1e-4, w=0.25, epoch=23), dict(lr=0.003, wd=1e-4, w=0.5, epoch=25),
              dict(lr=0.003, wd=1e-4, w=0.5, epoch=27), dict(lr=0.003, wd=1e-4, w=0.25, epoch=14),
              dict(lr=0.003, wd=1e-4, w=0.5, epoch=27)]
    path = tmp_path / "post.json"
    path.write_text(json.dumps({"results": {"LEAF-LGN+DAX": {"chosen": chosen}}}))
    a = types.SimpleNamespace(lrs=[1e-2], wds=[1e-4], w=[0.0], epoch_window=3, epochs=40)
    plans = RD.leaf_plan(a, 5, str(path))
    by_w = {p["w"]: p for p in plans}
    assert sorted(by_w) == [0.25, 0.5]
    assert by_w[0.25]["splits"] == [0, 3] and by_w[0.25]["lo"] == {0: 20, 3: 11} and by_w[0.25]["hi"] == {0: 26, 3: 17}
    assert by_w[0.5]["splits"] == [1, 2, 4] and by_w[0.5]["hi"] == {1: 28, 2: 30, 4: 30}
    a.epoch_window = -1
    plans = RD.leaf_plan(a, 5, str(path))
    assert all(p["splits"] == [0, 1, 2, 3, 4] and p["hi"][0] == float("inf") for p in plans)
    grid = RD.leaf_plan(types.SimpleNamespace(lrs=[1e-2, 3e-3], wds=[1e-4], w=[0.0, 0.5]), 5)
    assert len(grid) == 4 and all(p["splits"] == [0, 1, 2, 3, 4] for p in grid)


def test_average_of_shuffle_seeds_and_qp_job_grid():
    def res(vals):
        return {"per_split": [{"ndcg@10": float(np.mean(list(v.values())))} for v in vals],
                "per_user": [dict(v) for v in vals], "val_best": [0.5] * len(vals), "chosen": [{"c": 1}] * len(vals),
                "n_configs": 320, "n_evals": 100}
    r0 = res([{"1": 0.2, "2": 0.4}])
    r1 = res([{"1": 0.4, "2": 0.6}])
    avg = RD.average_arms([r0, r1])
    assert abs(avg["per_split"][0]["ndcg@10"] - 0.40) < 1e-12   # per user: (0.3 + 0.5) / 2 for users 1, 2 -> mean 0.4
    assert avg["n_configs"] == 640 and avg["test"]["ndcg@10"][0] == avg["per_split"][0]["ndcg@10"]
    args = types.SimpleNamespace(backbone="qp_posthoc", qp_source="grid", lrs=[1e-2, 3e-3], wds=[1e-5, 1e-4])
    assert len(RD.qp_jobs(args)) == 20          # DR-BPR-LGN-1: 4, -p2: 8, -p: 8
    args.qp_source = "stored"
    assert len(RD.qp_jobs(args)) == 10          # the 10 distinct configurations chosen by the 5 splits of the v10 run


def test_posthoc_cache_free_channel_selects_the_same():
    """Posthoc with and without the epoch-free channel cache (--no_zcache) gives the same selections and test values."""
    rng = np.random.default_rng(3)
    test = []
    for u in range(60):
        it = np.arange(rng.integers(12, 40))
        test.append((u, it, (rng.random(len(it)) < 0.2).astype(float)))
    layout = Layout(test)
    splits = RD.make_splits(test, 2, 0.3)
    row_ids = np.array([u for u, _, _ in test])
    lengths = np.array([len(it) for _, it, _ in test])
    base_b = [torch.as_tensor(rng.normal(size=(len(bk["k"]), bk["cand"].shape[1])), dtype=torch.float32)
              for bk in layout.buckets]
    arms = [("X", [({"a": 1}, DAXChannel(rng.normal(size=60))), ({"a": 2}, DAXChannel(rng.normal(size=60)))])]
    out = {}
    for cache in (True, False):
        post = RD.Posthoc(layout, splits, row_ids, lengths, "B", arms, base_std=True, gammas=(0.0, 0.5, 1.0),
                          cache_z=cache)
        for ep in range(3):
            post.add_epoch(base_b, {"lr": 1.0}, ep, active=[True, True])
        out[cache] = post.sels["X"].result()
    assert out[True]["chosen"] == out[False]["chosen"]
    assert out[True]["test_mean"] == out[False]["test_mean"]


def _have(path):
    return os.path.exists(os.path.join(ROOT, path))


def test_kuairand1k_loader_leak_checks():
    if not _have("data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv"):
        import pytest
        pytest.skip("KuaiRand-1K files are not in data/raw")
    d = load_kuairand_rich(version="1k", n_users=200, train_until=T0, dense_dtype=torch.float32, with_signals=False)
    leak = d["meta"]["leak"]
    assert leak["ranked_pairs_present_in_training_log"] == 0
    assert leak["test_events_present_in_training_log"] == 0
    assert leak["training_uses_random_log"] is False
    check_temporal_protocol(d)
    assert not d["outpool"].video_id.isin(set(d["items_raw"].tolist())).any()    # outside-pool history only
    assert int(d["test_log"].time_ms.min()) >= T0 > int(d["log"].time_ms.max())


def test_kuairand1k_lifelong_tables_match_the_raw_log_before_the_cut():
    """The 1K lifelong counts of a few users equal the counts recomputed from the raw standard logs: impressions before
    the cut-off, of videos outside the 1K random pool (the leak and window rules of experiments/v9_build_lifelong.py)."""
    base = os.path.join(ROOT, "data", "raw", "KuaiRand-1K", "data")
    if not (os.path.exists(os.path.join(base, "log_standard_4_08_to_4_21_1k.csv")) and
            os.path.exists(os.path.join(ROOT, "data", "raw", "lifelong_1k.npz"))):
        import pytest
        pytest.skip("KuaiRand-1K files or data/raw/lifelong_1k.npz are missing")
    L = np.load(os.path.join(ROOT, "data", "raw", "lifelong_1k.npz"), allow_pickle=True)
    users = np.asarray(L["tot_pre_user"])[:6]
    pool = set(pd.read_csv(os.path.join(base, "log_random_4_22_to_5_08_1k.csv"), usecols=["video_id"])
               .video_id.unique().tolist())
    parts = []
    for name in ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"):
        df = pd.read_csv(os.path.join(base, name), usecols=["user_id", "video_id", "time_ms", "is_click"])
        df = df[df.user_id.isin(users) & (df.time_ms < T0) & ~df.video_id.isin(pool)]
        parts.append(df.groupby("user_id").agg(n=("is_click", "size"), c=("is_click", "sum")))
    raw = pd.concat(parts).groupby(level=0).sum().reindex(users).fillna(0)
    tot = pd.DataFrame(np.asarray(L["tot_pre"]), index=np.asarray(L["tot_pre_user"]), columns=["n", "c", "l"])
    got = tot.reindex(users)[["n", "c"]].fillna(0).values
    assert np.array_equal(got, raw[["n", "c"]].values.astype(float)), (got, raw)
