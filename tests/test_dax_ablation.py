"""Examination-off ablation (no_exam) and the stand-alone rankers (backbone none) of experiments/run_dax.py.

(a) fit_dax(exam=False) equals the closed-form tag-shrunk click rate alpha_i = (clicks_i + prior c_i) / (n_i + prior),
    c_i = (clicks of the tag + 1) / (impressions of the tag + 2), theta = kappa = 1 (the tab factor has no effect);
(b) dax_variant(cfg, "no_exam") switches exam, tab and k_life off, and only that arm carries an "exam" key;
(c) the no_exam arm of the post-hoc grid has 8 distinct configurations (prior x centre x stratum cells);
(d) --backbone none runs end to end on a small subset (with the popularity, raw_ctr, no_groups arms and Random):
    the methods, their configuration counts and value ranges.
(d) needs the KuaiRand-Pure inputs (skipped without them).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from drup.dax import dax_variant, fit_dax  # noqa: E402
import run_dax as RD  # noqa: E402


def _tiny_log():
    """6 items in 3 tags (item 5 has no impression), 4 users, tabs 0 and 1: a click log with known counts."""
    rows = [  # (u, i, tab, is_click)
        (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (0, 3, 1, 1),
        (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 0, 0), (1, 3, 0, 0),
        (2, 2, 1, 1), (2, 2, 0, 0), (2, 4, 1, 0), (2, 1, 1, 0),
        (3, 4, 0, 1), (3, 0, 1, 0), (3, 3, 0, 0), (3, 4, 1, 1),
    ]
    log = pd.DataFrame(rows, columns=["u", "i", "tab", "is_click"])
    log["is_click"] = log.is_click.astype(float)
    tag = np.array([0, 0, 1, 1, 2, 2])              # item 5 is in tag 2 without impressions
    return log, 4, 6, tag


def _closed_form_tag_rate(log, n, tag, prior):
    clicks = np.bincount(log.i.values, weights=log.is_click.values, minlength=n)
    imps = np.bincount(log.i.values, minlength=n).astype(float)
    G = tag.max() + 1
    tag_clicks = np.bincount(tag, weights=clicks, minlength=G)
    tag_imps = np.bincount(tag, weights=imps, minlength=G)
    c = ((tag_clicks + 1.0) / (tag_imps + 2.0))[tag]
    return (clicks + prior * c) / (imps + prior)


@pytest.mark.parametrize("prior", [0.5, 3.0, 40.0])
def test_no_exam_is_the_tag_shrunk_click_rate(prior):
    log, m, n, tag = _tiny_log()
    want = _closed_form_tag_rate(log, n, tag, prior)
    la, th, ka, _ = fit_dax(log, m, n, tag, prior=prior, k_life=0.0, tab=False, exam=False, iters=3)
    assert np.allclose(torch.sigmoid(la).double().numpy(), want, atol=1e-6)
    assert torch.equal(th, torch.ones(m))
    assert torch.equal(ka, torch.ones(2, dtype=torch.float64))


def test_no_exam_ignores_the_tab_factor_and_the_lifelong_prior():
    log, m, n, tag = _tiny_log()
    life = np.array([0.1, 0.5, 0.9, 0.3])
    a = fit_dax(log, m, n, tag, prior=10.0, k_life=100.0, life_rate=life, tab=True, exam=False, iters=5)
    b = fit_dax(log, m, n, tag, prior=10.0, k_life=0.0, tab=False, exam=False, iters=5)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[2], torch.ones(2, dtype=torch.float64))


def test_exam_default_is_the_full_model():
    log, m, n, tag = _tiny_log()
    a = fit_dax(log, m, n, tag, prior=10.0, k_life=0.0, tab=True, iters=5)
    b = fit_dax(log, m, n, tag, prior=10.0, k_life=0.0, tab=True, exam=True, iters=5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert not torch.equal(a[2], torch.ones(2, dtype=torch.float64))


def test_dax_variant_no_exam_and_keys():
    cfg = dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)
    assert dax_variant(cfg, "no_exam") == dict(cfg, exam=False, tab=False, k_life=0.0)
    assert dax_variant(cfg, None) == cfg and "exam" not in dax_variant(cfg, None)
    for ab in ("no_tab", "no_life_prior", "no_tag_prior"):
        assert "exam" not in dax_variant(cfg, ab)
    grid = RD.post_grid()
    assert all("exam" not in c for c in grid)
    distinct = {json.dumps(dax_variant(c, "no_exam"), sort_keys=True) for c in grid}
    assert len(distinct) == 8
    assert "no_exam" in RD.ABLATIONS


KUAIRAND = os.path.join(ROOT, "data", "raw", "KuaiRand-Pure", "data", "video_features_basic_pure.csv")


@pytest.mark.skipif(not os.path.exists(KUAIRAND), reason="KuaiRand-Pure inputs are not in data/raw")
def test_backbone_none_runs_end_to_end(tmp_path):
    out = tmp_path / "none.json"
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "run_dax.py"), "--backbone", "none",
           "--ablate", "no_exam", "popularity", "raw_ctr", "no_groups", "--n_users", "60", "--seeds", "2",
           "--dax_grid", "small", "--gammas", "0.5", "1.0", "--threads", "2", "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out))["results"]
    assert set(res) == {"DAX alone", "DAX alone[no_exam]", "DAX alone[no_groups]", "DAX alone+pop", "DAX alone+ctr",
                        "Popularity", "Random"}
    assert res["Popularity"]["n_configs"] == 1 and res["Random"]["n_configs"] == 1
    assert res["DAX alone"]["n_configs"] == 2 and res["DAX alone[no_exam]"]["n_configs"] == 2   # 1 config x 2 gammas
    assert res["DAX alone+pop"]["n_configs"] == 2 and res["DAX alone+ctr"]["n_configs"] == 2
    assert res["Random"]["chosen"][0] == {"method": "Random"}
    assert res["DAX alone+pop"]["chosen"][0]["dax"] == {"model": "popularity"}
    for nm, r_ in res.items():
        assert len(r_["per_split"]) == 2 and len(r_["chosen"]) == 2 and len(r_["val_best"]) == 2
        assert all(0.0 <= v <= 1.0 for p in r_["per_user"] for v in p.values())
        assert np.isfinite(r_["test"]["ndcg@10"][0])
    assert res["Popularity"]["chosen"][0] == {"method": "Popularity"}
    assert set(res["DAX alone"]["chosen"][0]) >= {"backbone", "epoch", "dax", "gamma"}
