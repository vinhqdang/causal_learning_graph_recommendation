"""Checks of the DRGP prior and of the iALS-with-offset learner.

(a) loo_three_hop equals a brute-force leave-edge-out recomputation of the corrected 3-hop.
(b) --prior add reproduces the scores of the existing LightGCN-p and iALS runs stored in
    tests/data/drgp_add_golden.json (produced by the code before the DRGP change).
(c) drgp_prior reads only the training log: no uniform, validation or test entry is used.
(d) The iALS-with-offset user solve equals the dense weighted normal equations with target y - beta z.
(e) Leave-edge-out features do not move at (u, i) when (O_ui, Y_ui) changes, with Yhat and P fixed;
    the plain features do.
(f) The iALS-with-offset item solve equals the dense weighted normal equations.
"""
import inspect
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drup.learned import IALS, IALSP  # noqa: E402
from drup.propagation import degree_weights, drgp_prior, edge_estimate, gf_features, loo_three_hop, three_hop  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "data", "drgp_add_golden.json")


def test_loo_three_hop_matches_brute_force():
    torch.manual_seed(0)
    m, n = 6, 8
    O = (torch.rand(m, n, dtype=torch.float64) < 0.4).double()
    Y = (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    P = 0.1 + 0.8 * torch.rand(m, n, dtype=torch.float64)
    Yhat = torch.rand(m, n, dtype=torch.float64)
    W = edge_estimate(O, Y, P, Yhat)
    C = degree_weights(W, 0.5, D=Yhat)
    L = loo_three_hop(W, C, Yhat)
    worst = 0.0
    for u in range(m):
        for i in range(n):
            W2 = W.clone()
            W2[u, i] = Yhat[u, i]
            worst = max(worst, abs(float(three_hop(W2, C)[u, i] - L[u, i])))
    assert worst < 1e-10


def test_leave_edge_out_features_ignore_own_exposure_and_outcome():
    torch.manual_seed(3)
    m, n = 7, 9
    O = (torch.rand(m, n, dtype=torch.float64) < 0.4).double()
    Y = (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    P = 0.2 + 0.6 * torch.rand(m, n, dtype=torch.float64)
    Yhat = torch.rand(m, n, dtype=torch.float64)
    u, i = 3, 4
    O[u, i], Y[u, i] = 1.0, 1.0          # a logged positive, so that each flip below moves W_ui
    base_feats, names = gf_features(O, Y, P, Yhat, alphas=(0.5, 0.7), obs=True, loo=True, standardise=False)
    plain_feats, plain_names = gf_features(O, Y, P, Yhat, alphas=(0.5, 0.7), obs=True, loo=False,
                                           standardise=False)
    assert names == ["logit_yhat", "dr3_0.5", "dr3_0.7", "obs3"] and plain_names == ["W", "dr3_0.5", "dr3_0.7", "obs3"]
    for o_new, y_new in [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        O2, Y2 = O.clone(), Y.clone()
        O2[u, i], Y2[u, i] = o_new, y_new
        new_feats, _ = gf_features(O2, Y2, P, Yhat, alphas=(0.5, 0.7), obs=True, loo=True, standardise=False)
        for k, name in enumerate(names):
            assert abs(float(new_feats[k][u, i] - base_feats[k][u, i])) < 1e-10, (name, o_new, y_new)
        if (o_new, y_new) != (1.0, 1.0):
            # the plain features depend on the pair: the flip changes W_ui, hence the plain terms
            plain_new, _ = gf_features(O2, Y2, P, Yhat, alphas=(0.5, 0.7), obs=True, loo=False, standardise=False)
            for k, name in enumerate(plain_names):
                assert abs(float(plain_new[k][u, i] - plain_feats[k][u, i])) > 1e-8, (name, o_new, y_new)


def test_ialsp_item_solve_matches_dense_normal_equations():
    torch.manual_seed(4)
    m, n, d, alpha, reg, beta = 9, 7, 3, 10.0, 2.0, 0.7
    O = (torch.rand(m, n) < 0.4).float()
    Y = (torch.rand(m, n) < 0.5).float()
    Z = torch.randn(m, n)
    eu, ei = torch.nonzero(O * Y > 0, as_tuple=True)
    model = IALSP(m, n, d, eu, ei, Z, beta=beta, alpha=alpha, reg=reg, seed=1)
    U = torch.randn(m, d)
    model.U = U.clone()
    model._solve(model.V, model.U, ei, eu, n, model.Z.T)
    pos = (O * Y > 0).float()
    c = 1.0 + alpha * pos
    t = pos - beta * Z
    for i in range(n):
        A = (c[:, i][:, None, None] * U[:, :, None] * U[:, None, :]).sum(0) + reg * torch.eye(d)
        b = (c[:, i] * t[:, i]) @ U
        x = torch.linalg.solve(A, b)
        assert torch.allclose(model.V[i], x, atol=1e-4)


def test_prior_add_reproduces_previous_scores(tmp_path):
    golden = json.load(open(GOLDEN))
    out, bank = tmp_path / "learned.json", tmp_path / "bank.json"
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "run_learned.py"), "--dataset", "coat", "--prop", "given",
           "--seeds", "1", "--threads", "1", "--dims", "16", "--epochs", "3", "--lrs", "3e-3", "--wds", "1e-5",
           "--floor", "0.1", "--seed_check", "0", "--prior", "add", "--methods", "LightGCN-p", "iALS",
           "--out", str(out), "--dump_bank", str(bank)]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    banks = json.load(open(bank))["banks"]
    assert set(banks) == set(golden["banks"])
    for name, entries in golden["banks"].items():
        got = banks[name]
        assert len(got) == len(entries), name
        for g, e in zip(got, entries):
            assert g["cfg"] == e["cfg"], name
            assert max(abs(a - b) for a, b in zip(g["val"], e["val"])) < 1e-9, name
            assert max(abs(a - b) for a, b in zip(g["test"], e["test"])) < 1e-9, name


class _Spy(dict):
    """Records which keys of the dataset dict are read."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = set()

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)


def test_drgp_prior_uses_only_training_log():
    forbidden = {"test", "val", "valid", "validation", "uniform", "U", "split", "rows", "S_test", "S_val"}
    assert not forbidden & set(inspect.signature(drgp_prior).parameters)
    torch.manual_seed(1)
    m, n = 30, 40
    O = (torch.rand(m, n, dtype=torch.float64) < 0.3).double()
    Y = (torch.rand(m, n, dtype=torch.float64) < 0.5).double()
    P = 0.2 + 0.6 * torch.rand(m, n, dtype=torch.float64)
    d = _Spy({"P_given": P, "test": "UNIFORM-SENTINEL", "val": "VALIDATION-SENTINEL"})
    pi, z, theta = drgp_prior(d, O, Y, "given", 0.1, 5.0, alphas=(0.5,), n_buckets=2, ridge=1.0, K=3)
    assert d.seen <= {"P_given"}
    pi2, z2, theta2 = drgp_prior({"P_given": P.clone()}, O, Y, "given", 0.1, 5.0, alphas=(0.5,),
                                 n_buckets=2, ridge=1.0, K=3)
    assert torch.equal(pi, pi2) and torch.equal(z, z2) and torch.equal(theta, theta2)
    assert torch.isfinite(pi).all() and torch.isfinite(z).all()


def test_ialsp_user_solve_matches_dense_normal_equations():
    torch.manual_seed(2)
    m, n, d, alpha, reg, beta = 9, 7, 3, 10.0, 2.0, 0.7
    O = (torch.rand(m, n) < 0.4).float()
    Y = (torch.rand(m, n) < 0.5).float()
    Z = torch.randn(m, n)
    eu, ei = torch.nonzero(O * Y > 0, as_tuple=True)
    model = IALSP(m, n, d, eu, ei, Z, beta=beta, alpha=alpha, reg=reg, seed=0)
    V = model.V.clone()
    model._solve(model.U, model.V, eu, ei, m, Z)
    pos = (O * Y > 0).float()
    c = 1.0 + alpha * pos
    t = pos - beta * Z
    for u in range(m):
        A = (c[u][:, None, None] * V[:, :, None] * V[:, None, :]).sum(0) + reg * torch.eye(d)
        b = (c[u] * t[u]) @ V
        x = torch.linalg.solve(A, b)
        assert torch.allclose(model.U[u], x, atol=1e-4)
    # beta = 0 reduces to plain iALS
    a_, b_ = IALS(m, n, d, eu, ei, alpha=alpha, reg=reg, seed=0), IALSP(m, n, d, eu, ei, Z, beta=0.0, alpha=alpha,
                                                                     reg=reg, seed=0)
    for _ in range(3):
        a_.step()
        b_.step()
    assert torch.allclose(a_.U, b_.U, atol=1e-5) and torch.allclose(a_.V, b_.V, atol=1e-5)
