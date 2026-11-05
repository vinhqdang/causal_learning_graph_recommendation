"""LEAF closed form, LEAF-LGN interface and the temporal protocol (DESIGN.md Sections 5, 6, 9)."""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drup.data2 import load_kuairand_rich  # noqa: E402
from drup.leaf import (CUT_MS, LEAFLGN, LifelongTables, check_temporal_protocol,  # noqa: E402
                       collective_ease, collective_gram, collective_solve, fit_bpr_policy, make_lightgcn,
                       protocol_split)
from leaf_synth import world  # noqa: E402


class DensePhi:
    """Phi block given as a dense matrix (the interface PhiFull provides)."""

    def __init__(self, P):
        self.P, self.p = P, P.shape[1]

    def chunk(self, rows):
        return self.P[rows]


def _ease(X, lam):
    """Reference EASE (Steck, WWW 2019), as in experiments/v9_probes/common.ease."""
    I = torch.linalg.inv(X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype))
    B = -I / torch.diag(I)[None, :]
    B.fill_diagonal_(0.0)
    return B


def test_collective_ease_without_channels_is_ease():
    torch.manual_seed(0)
    m, n = 40, 12
    O = (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    Y = O * (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    B, W = collective_ease(O, Y, w=0.0, lam=3.0)
    assert W is None
    assert torch.allclose(B, _ease(O * Y, 3.0), atol=1e-8)
    assert torch.diagonal(B).abs().max() < 1e-12


def test_constrained_solution_solves_each_column_with_zero_diagonal():
    torch.manual_seed(1)
    m, n, p = 50, 8, 3
    O = (torch.rand(m, n, dtype=torch.float64) < 0.6).double()
    Y = O * (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    P = torch.rand(m, p, dtype=torch.float64)
    w, lam, lam_phi = 0.5, 2.0, 0.7
    G, C = collective_gram(O, Y, w, DensePhi(P))
    G0 = G.clone()
    B, W = collective_solve(G, C, n, lam, lam_phi)
    assert torch.equal(G, G0), "collective_solve must restore the Gram matrix"
    assert torch.diagonal(B).abs().max() < 1e-12
    X = O * Y
    Z = X + w * (O - X)
    for j in range(n):
        keep = [k for k in range(n) if k != j]
        A = torch.cat([Z[:, keep], P], 1)
        reg = torch.diag(torch.tensor([lam] * (n - 1) + [lam_phi] * p, dtype=torch.float64))
        sol = torch.linalg.solve(A.T @ A + reg, A.T @ X[:, j])
        assert torch.allclose(B[keep, j], sol[: n - 1], atol=1e-8), j
        assert torch.allclose(W[:, j], sol[n - 1:], atol=1e-8), j


def test_leaf_lgn_scores_agree_between_pairs_and_full_rows():
    d = world(seed=2, m=60)
    O, Y = d["O"], d["Y"]
    m, n = O.shape
    tabs = LifelongTables(d["npz"], d["users_raw"], d["item_feat"], window="pre", s=5.0)
    pop = torch.log1p((O * Y).sum(0))
    pop = (pop - pop.mean()) / pop.std()
    torch.manual_seed(0)
    base = make_lightgcn(m, n, Y, dim=8, layers=2)
    model = LEAFLGN(base, tabs, pop, mask=np.array([1, 1, 0, 0, 1, 1.0]))
    with torch.no_grad():
        model.gain.copy_(torch.randn(6))
        model.beta.fill_(0.3)
    model.eval()
    rows = torch.arange(m)
    with torch.no_grad():
        S = model.full_scores(rows)
        u = torch.randint(0, m, (200,))
        i = torch.randint(0, n, (200,))
        f = model(u, i)
    assert torch.allclose(S[u, i], f, atol=1e-4)
    # masked channels do not enter the score
    assert torch.equal(model.gain[2:4] * model.mask[2:4], torch.zeros(2))


def test_policy_bpr_trains_channel_weights_and_stops_on_request():
    d = world(seed=6, m=60)
    O, Y = d["O"], d["Y"]
    m, n = O.shape
    tabs = LifelongTables(d["npz"], d["users_raw"], d["item_feat"], window="pre", s=5.0)
    pop = torch.zeros(n)
    base = make_lightgcn(m, n, Y, dim=8, layers=2)
    model = LEAFLGN(base, tabs, pop)
    seen = []

    def on_epoch(mod, ep):
        seen.append(ep)
        return ep == 1

    fit_bpr_policy(model, O, Y, 0.5, 1e-2, 1e-4, epochs=10, batch=256, on_epoch=on_epoch, seed=0)
    assert seen == [0, 1]
    assert float(model.gain.detach().abs().sum()) > 0.0


def test_protocol_split_is_reproducible_and_disjoint():
    d = world(seed=7, m=120)
    test = d["test"]
    v1, t1 = protocol_split(test, seed=0)
    v2, t2 = protocol_split(test, seed=0)
    assert [u for u, _, _ in v1] == [u for u, _, _ in v2]
    assert [u for u, _, _ in t1] == [u for u, _, _ in t2]
    v3, _ = protocol_split(test, seed=1)
    assert [u for u, _, _ in v1] != [u for u, _, _ in v3]
    uv = {u for u, _, _ in v1}
    ut = {u for u, _, _ in t1}
    assert not uv & ut and len(uv) + len(ut) == len(test)
    assert len(uv) == round(0.3 * len(test))


PURE = os.path.join(ROOT, "data", "raw", "KuaiRand-Pure", "data", "log_standard_4_08_to_4_21_pure.csv")


def test_temporal_protocol_on_kuairand_is_reproducible_and_disjoint():
    if not os.path.exists(PURE):
        import pytest
        pytest.skip("KuaiRand-Pure logs not present")
    a = load_kuairand_rich(version="pure", n_users=200, seed=0, train_until=CUT_MS, with_signals=False)
    b = load_kuairand_rich(version="pure", n_users=200, seed=0, train_until=CUT_MS, with_signals=False)
    info = check_temporal_protocol(a, CUT_MS)
    assert info["max_train_time_ms"] < CUT_MS <= info["min_test_time_ms"]
    assert torch.equal(a["O"], b["O"]) and len(a["test"]) == len(b["test"])
    assert all(x[0] == y[0] and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])
               for x, y in zip(a["test"], b["test"]))
    val, tst = protocol_split(a["test"], seed=0)
    assert not {u for u, _, _ in val} & {u for u, _, _ in tst}
    # the training matrices contain no random-exposure impression: every training time is before the cut-off
    assert int(a["log"].time_ms.max()) < CUT_MS
