"""Control arms and the exact Bernoulli M-step of drup.dax (v16).

(a) control_item_term: popularity = log(1 + pre-cut clicks), raw_ctr = (clicks + 1) / (impressions + 2) on a tiny log;
(b) dax_variant no_groups (strata off, 16 distinct configurations on the full grid) and exact_em (exact_mstep True,
    32 distinct configurations);
(c) exact_theta_kappa maximises the Bernoulli expected complete log-likelihood with its prior terms (checked against a
    dense grid, per user and per tab);
(d) fit_dax with exact_mstep False (default) is the ratio form, and exact_mstep True runs and improves the observed-data
    Bernoulli log-likelihood of the fit relative to the ratio form's own E-step values on this tiny log.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from drup.dax import control_item_term, dax_variant, exact_theta_kappa, fit_dax  # noqa: E402
import run_dax as RD  # noqa: E402


def _tiny_log():
    """6 items in 3 tags (item 5 has no impression), 4 users, tabs 0 and 1 (the same log as test_dax_ablation)."""
    rows = [
        (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (0, 3, 1, 1),
        (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 0, 0), (1, 3, 0, 0),
        (2, 2, 1, 1), (2, 2, 0, 0), (2, 4, 1, 0), (2, 1, 1, 0),
        (3, 4, 0, 1), (3, 0, 1, 0), (3, 3, 0, 0), (3, 4, 1, 1),
    ]
    log = pd.DataFrame(rows, columns=["u", "i", "tab", "is_click"])
    log["is_click"] = log.is_click.astype(float)
    tag = np.array([0, 0, 1, 1, 2, 2])
    return log, 4, 6, tag


def test_popularity_and_raw_ctr_terms():
    log, m, n, tag = _tiny_log()
    clicks = np.bincount(log.i.values, weights=log.is_click.values, minlength=n)
    imps = np.bincount(log.i.values, minlength=n).astype(float)
    pop = control_item_term(log, n, "popularity")
    ctr = control_item_term(log, n, "raw_ctr")
    assert np.allclose(pop, np.log1p(clicks))
    assert pop[5] == 0.0 and np.log1p(clicks[0]) > np.log1p(clicks[1])
    assert np.allclose(ctr, (clicks + 1.0) / (imps + 2.0))
    assert ctr[5] == 0.5                       # no impression: the (1, 2) pseudo-count prior mean
    with pytest.raises(ValueError):
        control_item_term(log, n, "tag")


def test_variants_no_groups_and_exact_em():
    grid = RD.post_grid()
    assert len(grid) == 32
    ng = {json.dumps(dax_variant(c, "no_groups"), sort_keys=True) for c in grid}
    assert len(ng) == 16
    assert all(json.loads(k)["prior_s"] is None for k in ng)
    assert all("exact_mstep" not in json.loads(k) for k in ng)
    ex = [dax_variant(c, "exact_em") for c in grid]
    assert all(c["exact_mstep"] is True for c in ex)
    assert len({json.dumps(c, sort_keys=True) for c in ex}) == 32
    assert dax_variant(grid[0], None) == grid[0]
    for name in ("popularity", "raw_ctr", "no_groups", "exact_em"):
        assert name in RD.ABLATIONS


def _obj_theta(t, pE, kj, pos, neg):
    p = t * kj
    return np.sum(pE * np.log(p) + (1 - pE) * np.log(1 - p)) + pos * np.log(t) + neg * np.log(1 - t)


def _obj_kappa(k, pE, th_j, pos):
    p = k * th_j
    return np.sum(pE * np.log(p) + (1 - pE) * np.log(1 - p)) + np.log(k) - k


@pytest.mark.parametrize("pos, neg", [(1.0, 1.0), (30.0, 10.0), (0.2, 5.0)])
def test_exact_theta_maximises_bernoulli_objective(pos, neg):
    rng = np.random.default_rng(0)
    m, T, N = 5, 2, 60
    u = rng.integers(0, m, N)
    u[:3] = [0, 1, 2]                          # users 3 and 4 have no impression unless drawn
    tb = rng.integers(0, T, N)
    pE = rng.uniform(0.05, 0.95, N)
    ka = np.array([0.7, 1.3])
    th0 = np.full(m, 0.5)
    th, _ = exact_theta_kappa(torch.as_tensor(pE), torch.as_tensor(u), torch.as_tensor(tb), torch.as_tensor(th0),
                              torch.as_tensor(ka), m, T, pos, neg, tab=True, normalise=False)
    th = th.numpy()
    for user in range(m):
        sel = u == user
        kj = ka[tb[sel]]
        hi = min(1.0, (1 - 1e-9) / kj.max()) if sel.any() else 1.0
        grid = np.linspace(1e-3, hi, 20001)
        vals = np.array([_obj_theta(g, pE[sel], kj, pos, neg) for g in grid]) if sel.any() else \
            pos * np.log(grid) + neg * np.log(1 - grid)
        best = vals.max()
        assert _obj_theta(th[user], pE[sel], kj, pos, neg) >= best - 1e-7
    for user in range(m):
        if not (u == user).any():              # no impressions: the prior mean pos / (pos + neg)
            assert np.isclose(th[user], pos / (pos + neg))


def test_exact_kappa_maximises_bernoulli_objective():
    rng = np.random.default_rng(1)
    m, T, N = 4, 3, 80
    u = rng.integers(0, m, N)
    tb = rng.integers(0, T, N)
    pE = rng.uniform(0.05, 0.95, N)
    th = rng.uniform(0.2, 0.9, m)
    theta, kappa = exact_theta_kappa(torch.as_tensor(pE), torch.as_tensor(u), torch.as_tensor(tb),
                                     torch.as_tensor(th), torch.ones(T, dtype=torch.float64), m, T, 1.0, 1.0,
                                     tab=True, normalise=False)
    kappa, theta = kappa.numpy(), theta.numpy()
    tj = theta[u]                              # the kappa step uses the theta returned by the same call
    for t in range(T):
        sel = tb == t
        hi = (1 - 1e-9) / tj[sel].max()
        grid = np.linspace(1e-6, hi, 20001)
        vals = np.array([_obj_kappa(g, pE[sel], tj[sel], 1.0) for g in grid])
        assert _obj_kappa(kappa[t], pE[sel], tj[sel], 1.0) >= vals.max() - 1e-7


def test_exact_mstep_off_is_the_ratio_form_and_on_runs():
    log, m, n, tag = _tiny_log()
    life = np.array([0.1, 0.5, 0.9, 0.3])
    base = fit_dax(log, m, n, tag, prior=10.0, k_life=100.0, life_rate=life, tab=True, iters=8)
    off = fit_dax(log, m, n, tag, prior=10.0, k_life=100.0, life_rate=life, tab=True, iters=8, exact_mstep=False)
    assert torch.equal(base[0], off[0]) and torch.equal(base[1], off[1]) and torch.equal(base[2], off[2])
    on = fit_dax(log, m, n, tag, prior=10.0, k_life=100.0, life_rate=life, tab=True, iters=8, exact_mstep=True)
    assert torch.isfinite(on[0]).all() and torch.isfinite(on[1]).all()
    assert float(on[1].min()) >= 1e-3 and float(on[1].max()) <= 1.0
    assert on[2][1].item() == pytest.approx(1.0)
    assert not torch.equal(on[1], base[1])


def test_exact_theta_prior_mean_for_a_user_without_impressions():
    u = torch.tensor([0, 0, 1])
    tb = torch.tensor([0, 1, 0])
    pE = torch.tensor([0.4, 0.9, 0.2], dtype=torch.float64)
    th, ka = exact_theta_kappa(pE, u, tb, torch.full((3,), 0.5, dtype=torch.float64),
                               torch.ones(2, dtype=torch.float64), 3, 2, 30.0, 10.0, tab=True)
    assert np.isclose(float(th[2]), 0.75)                  # pos / (pos + neg)
    assert float(ka[1]) == pytest.approx(1.0)
