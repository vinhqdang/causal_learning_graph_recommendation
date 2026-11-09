"""DAX: EM of the detection x attractiveness model, leakage of the pre-cut fit, negative controls, and the probe
reproduction (paper/v7/DESIGN.md, Sections 5, 6 and 7 gate B2).

The probe reproduction (last test) reads the split-0 LEAF-LGN scores stored in results/v11_probes/lgn_s0.pt (written by
experiments/v11_leaf_scores.py) and needs the KuaiRand temporal dataset (data/raw).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drup.dax import (DAXChannel, all_perm, fit_additive, fit_dax, fit_dax_dataset, masked_rowstd,  # noqa: E402
                      within_code_perm)
from drup.leaf import CUT_MS, BucketSelector, Layout, protocol_split  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

T0 = CUT_MS


def click_world(seed=0, m=1500, n=300, n_tags=8, per_user=60, tabs=3, item_sd=1.0, theta_range=None):
    """Clicks of a click model: P(click) = theta_u * kappa_t * alpha_i with tab factors kappa = (0.2, 1, 0.5).
    theta ~ Beta(2, 3), or Uniform(theta_range) when given."""
    rng = np.random.default_rng(seed)
    tag = np.arange(n) % n_tags
    alpha = 1 / (1 + np.exp(-(rng.normal(0, 0.5, n_tags)[tag] + rng.normal(0, item_sd, n) - 0.5)))
    theta = rng.beta(2, 3, m) if theta_range is None else rng.uniform(*theta_range, m)
    kappa = np.array([0.2, 1.0, 0.5])[:tabs]
    u = np.repeat(np.arange(m), per_user)
    i = np.concatenate([rng.choice(n, per_user, replace=False) for _ in range(m)])
    tb = rng.integers(0, tabs, m * per_user)
    y = (rng.random(m * per_user) < theta[u] * kappa[tb] * alpha[i]).astype(float)
    log = pd.DataFrame({"u": u, "i": i, "tab": tb, "is_click": y, "time_ms": T0 - 10_000_000})
    return log, alpha, theta, tag, m, n


def loglik(log, th, ka, al):
    t = np.clip(th[log.u.values] * ka[log.tab.values.astype(int)] * al[log.i.values], 1e-9, 1 - 1e-9)
    y = log.is_click.values
    return float((y * np.log(t) + (1 - y) * np.log(1 - t)).sum())


def test_em_likelihood_is_monotone_without_priors():
    """Closed-form EM with negligible priors climbs the observed-data likelihood at every iteration (tab factors on)."""
    for seed in range(3):
        log, _, _, tag, m, n = click_world(seed, m=800, n=200, per_user=60)
        vals = []
        for it in range(1, 26):
            la, th, ka, _ = fit_dax(log, m, n, tag, iters=it, prior=1e-9, k_life=0.0, tab=True)
            al = torch.sigmoid(la).double().numpy()
            vals.append(loglik(log, th.double().numpy(), ka.double().numpy(), al))
        d = np.diff(vals)
        assert d.min() >= -1e-6, (seed, d.min())
        assert vals[-1] - vals[0] > 1000.0


def test_em_recovers_attractiveness_and_examination_ranking():
    log, alpha, theta, tag, m, n = click_world(1, m=1500, n=300)
    la, th, ka, _ = fit_dax(log, m, n, tag, prior=5.0, k_life=0.0, tab=True, iters=60)
    rho_a = stats.spearmanr(la.numpy(), np.log(alpha / (1 - alpha)))[0]
    rho_t = stats.spearmanr(th.numpy(), theta)[0]
    assert rho_a > 0.8, rho_a
    # examination is estimated from 60 impressions per user: the ranking is noise-limited (about 0.79 here)
    assert rho_t > 0.75, rho_t
    # the tab factors are recovered up to the reference (tab 1 = 1)
    assert np.allclose(ka.numpy(), [0.2, 1.0, 0.5], atol=0.08), ka


def test_em_recovers_examination_up_to_the_global_scale():
    """theta is recovered within 0.05 once the one global scale is removed. The clicks depend on theta and alpha only
    through their product, so (c theta, alpha / c) has the same likelihood for every c: the absolute level of theta
    is not identified (the rankings are). The scale is removed here with the true mean of theta (a test-only
    oracle). With 8000 impressions per user and theta in (0.1, 0.4) the largest error is about 0.03 (mean absolute
    error about 0.01). Higher theta makes the raw fit exceed the bound 1 on the scale ridge and clip, so the range is
    kept below it."""
    log, _, theta, tag, m, n = click_world(7, m=40, n=10000, per_user=8000, theta_range=(0.1, 0.4))
    _, th, ka, _ = fit_dax(log, m, n, tag, prior=5.0, k_life=0.0, tab=True, iters=60)
    th = th.numpy().astype(float)
    th = th * (theta.mean() / th.mean())
    err = np.abs(th - theta)
    assert err.max() < 0.05, (err.max(), err.mean())
    assert np.allclose(ka.numpy(), [0.2, 1.0, 0.5], atol=0.08), ka


def _dataset(log, test, test_log, m, n, tags):
    item_feat = pd.DataFrame({"tag": [str(t) for t in tags]})
    return {"log": log, "test": test, "test_log": test_log, "users_raw": np.arange(1000, 1000 + m),
            "item_feat": item_feat}


def test_dax_fit_reads_only_pre_cut_rows_and_no_labels():
    """Post-cut rows of the log, the random log and the test labels never change the DAX parameters."""
    log, _, _, tag, m, n = click_world(2, m=300, n=120, per_user=40)
    rng = np.random.default_rng(3)
    test = [(u, np.arange(n)[:20], (rng.random(20) < 0.3).astype(float)) for u in range(m)]
    test_log = pd.DataFrame({"u": rng.integers(0, m, 500), "i": rng.integers(0, n, 500), "tab": 1,
                             "is_click": rng.integers(0, 2, 500).astype(float), "time_ms": T0 + 1000})
    life = np.full(m, 0.2)
    kw = dict(life=life, prior=80.0, k_life=100.0, tab=True)
    clean = _dataset(log, test, test_log, m, n, tag)
    ref = fit_dax_dataset(clean, **kw)
    # (a) post-cut rows added to the log, with out-of-range codes: they must not be read at all
    post_rows = pd.DataFrame({"u": rng.integers(0, m, 4000), "i": rng.integers(0, n, 4000), "tab": 9,
                              "is_click": 1.0, "time_ms": T0 + rng.integers(0, 10**6, 4000)})
    dirty = _dataset(pd.concat([log, post_rows], ignore_index=True), test, test_log, m, n, tag)
    # (b) every label of the random period flipped
    flipped_test = [(u, it, 1.0 - rel) for u, it, rel in test]
    flipped_log = test_log.assign(is_click=1.0 - test_log.is_click)
    flipped = _dataset(log, flipped_test, flipped_log, m, n, tag)
    for other in (dirty, flipped):
        out = fit_dax_dataset(other, **kw)
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])
    # (c) removing every post-cut row of the dict leaves the same parameters
    cleaned = _dataset(pd.concat([log, post_rows], ignore_index=True), test, test_log, m, n, tag)
    cleaned["log"] = cleaned["log"][cleaned["log"].time_ms < T0].reset_index(drop=True)
    out = fit_dax_dataset(cleaned, **kw)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


def test_additive_arm_tracks_item_appeal():
    log, alpha, _, tag, m, n = click_world(4, m=1200, n=200, per_user=60)
    la = fit_additive(log, m, n, tag, lam_e=5.0, tab=True).numpy()
    assert np.isfinite(la).all()
    assert stats.spearmanr(la, alpha)[0] > 0.5


def test_within_tag_control_removes_the_item_gain():
    """Synthetic exposure confound: high-appeal items are shown mostly in the low-examination tab, so raw clicks
    under-rank them. DAX (item appeal) adds the lost part on a uniform random test; the same appeal permuted within
    tag (tag mean kept, item part destroyed) must lose most of that gain (the test asserts a drop of at least 70%,
    stricter than the 30% of gate B2)."""
    rng = np.random.default_rng(5)
    m, n, n_tags, per_user = 1200, 400, 8, 60
    tag = np.arange(n) % n_tags
    alpha = 1 / (1 + np.exp(-(rng.normal(0, 0.5, n_tags)[tag] + rng.normal(0, 1.2, n) - 0.5)))
    theta = rng.beta(2, 3, m)
    kappa = np.array([0.1, 1.0])
    p_tab0 = np.clip(0.15 + 0.7 * (alpha - alpha.min()) / (alpha.max() - alpha.min()), 0, 1)
    rows = []
    for u in range(m):
        it = rng.choice(n, per_user, replace=False)
        t0 = rng.random(per_user) < p_tab0[it]
        tb = np.where(t0, 0, 1)
        y = (rng.random(per_user) < theta[u] * kappa[tb] * alpha[it]).astype(float)
        rows.append(pd.DataFrame({"u": u, "i": it, "tab": tb, "is_click": y}))
    log = pd.concat(rows, ignore_index=True)
    test = []
    for u in range(m):
        it = rng.choice(n, 40, replace=False)
        rel = (rng.random(40) < alpha[it]).astype(float)
        test.append((u, it, rel))
    pop = np.log1p(np.bincount(log.i.values, weights=log.is_click.values, minlength=n))
    pop = (pop - pop.mean()) / pop.std()
    la, _, _, _ = fit_dax(log, m, n, tag, prior=80.0, k_life=0.0, tab=True, iters=60)
    la = la.numpy()
    z = (la - la.mean()) / la.std()
    perm = within_code_perm(tag, seed=0)
    zs = z[perm]
    S0 = np.tile(pop, (m, 1))
    base = evaluate(S0, test, ks=(10,))["ndcg@10"]
    full = evaluate(S0 + z[None, :], test, ks=(10,))["ndcg@10"]
    shuf = evaluate(S0 + zs[None, :], test, ks=(10,))["ndcg@10"]
    gain, gain_shuf = full - base, shuf - base
    assert gain > 0.01, (base, full)
    assert gain_shuf < 0.3 * gain, (gain, gain_shuf)


def test_permutations_keep_the_tag_or_not():
    tag = np.array([0, 0, 0, 1, 1, 2])
    p = within_code_perm(tag, seed=0)
    assert sorted(p.tolist()) == list(range(6))
    assert all(tag[p[j]] == tag[j] for j in range(6))
    q = all_perm(50, seed=0)
    assert sorted(q.tolist()) == list(range(50))


def test_channel_interface_and_joint_module_shapes():
    rng = np.random.default_rng(6)
    n, m, S = 40, 7, 3
    ch = DAXChannel(rng.normal(size=n), strata=rng.integers(0, S, m), logit_alpha_strata=rng.normal(size=(n, S)))
    rows = torch.arange(m)
    cand = torch.as_tensor(rng.integers(0, n, (m, 5)))
    c = ch.compact(rows, cand)
    assert c.shape == (m, 5)
    assert torch.allclose(c, ch.pairs(rows[:, None], cand))
    assert ch.dense_rows(rows).shape == (m, n)
    mask = torch.ones(m, 5, dtype=torch.bool)
    z = masked_rowstd(c, mask)
    assert torch.allclose(z.mean(1), torch.zeros(m), atol=1e-5)


SPLIT0_SCORES = os.path.join(ROOT, "results", "v11_probes", "lgn_s0.pt")


def test_post_hoc_dax_reproduces_the_probe_split0():
    """Post-hoc DAX added to the stored split-0 LEAF-LGN scores gives the probe's number for split 0 (tolerance 1e-4):
    LEAF-LGN + DAX full grid, chosen configuration prior 80, tag centre, k_life 100, tab on, stratum cells 80, gamma 0.5
    (results/v11_probes/probe_pbm.json)."""
    from drup.data2 import load_kuairand_rich
    from drup.dax import item_content_codes, lifelong_rate, user_strata
    probe = json.load(open(os.path.join(ROOT, "results", "v11_probes", "probe_pbm.json")))["LEAF-LGN + DAX full grid"]
    cfg = probe["chosen"][0]
    want = probe["per_split"][0]["ndcg@10"]
    d = load_kuairand_rich(version="pure", n_users=None, train_until=T0, dense_dtype=torch.float32,
                           with_signals=False)
    test = d["test"]
    layout = Layout(test)
    lengths = np.array([len(it) for _, it, _ in test])
    ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
    splits = [protocol_split(ctest, s, 0.3) for s in range(5)]
    row_ids = np.array([u for u, _, _ in test])
    base = torch.load(SPLIT0_SCORES, weights_only=False)
    mask = [torch.arange(bk["cand"].shape[1])[None, :] < torch.as_tensor(lengths[bk["k"]])[:, None]
            for bk in layout.buckets]
    kw = {k: cfg[k] for k in ("prior", "centre", "k_life", "tab", "prior_s")}
    life = lifelong_rate(d["users_raw"])
    strata = user_strata(d["users_raw"]) if kw["prior_s"] is not None else None
    cov = item_content_codes(d["item_feat"], d["items_raw"]) if kw["centre"] == "content" else None
    la, _, _, la_s = fit_dax_dataset(d, life=life, strata=strata, content=cov, **kw)
    ch = DAXChannel(la.numpy(), strata=strata, logit_alpha_strata=None if la_s is None else la_s.numpy())
    rows = [bk["rows"] for bk in layout.buckets]
    cands = [bk["cand"] for bk in layout.buckets]
    S = [masked_rowstd(b.float(), mk) + cfg["gamma"] * masked_rowstd(ch.compact(r, c), mk)
         for b, r, c, mk in zip(base, rows, cands, mask)]
    sel = BucketSelector(layout, splits, (10,), "ndcg@10", row_ids)
    sel.add(S, cfg, active=[True, False, False, False, False])
    got = sel.best[0]["test"]
    assert abs(got - want) < 1e-4, (got, want)
