"""Exact walk-corrected propagation for any odd number of hops K.

A K-hop user->item walk u = U_0 - J_1 - U_1 - J_2 - ... - U_r - J_{r+1} = i
(r = (K-1)/2) uses the edges (U_t, J_{t+1}) and (U_{t+1}, J_{t+1}). Whenever
free indices coincide, edges can repeat and the plug-in product picks up
W_e^k, whose mean is not Y_e^k = Y_e. The unbiased replacement of the
repeated factor is C_e^k W_e (idempotent correction, docs/THEORY.md).

Which edges repeat depends only on the *equality pattern* pi of the index
variables: a pair of set partitions, one of the user variables
{U_0, ..., U_r} and one of the item variables {J_1, ..., J_{r+1}}. For every
pattern with at least one repeated edge we need

    S_pi = sum over assignments whose equality pattern is exactly pi of F_pi,

where F_pi is the product over pi's distinct edges of C^k W (corrected) or
Wt^k (plug-in). "Exactly pi" sums are obtained from unconstrained
("at least sigma") sums by Moebius inversion on the partition lattice:

    S_pi = sum_{sigma >= pi} mu(pi, sigma) g_pi(sigma),
    mu(pi, sigma) = prod over blocks of sigma of (-1)^(b-1) (b-1)!,

with b the number of pi-blocks merged into that sigma-block. Each g_pi(sigma)
is a tensor contraction over sigma's blocks (evaluated with einsum).
The estimator is the plug-in K-hop product plus
sum_pi (S_pi[corrected] - S_pi[plug-in]); it is exactly unbiased for the
K-hop propagation of the full-exposure graph (Theorem 1 extends verbatim).
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import torch


def set_partitions(n):
    """All set partitions of range(n) as tuples of block labels (restricted
    growth strings)."""
    out = []

    def rec(i, labels, nb):
        if i == n:
            out.append(tuple(labels))
            return
        for b in range(nb + 1):
            rec(i + 1, labels + [b], max(nb, b + 1))
    rec(0, [], 0)
    return out


def _coarser(p, q):
    """True if partition q is coarser than or equal to p (p refines q)."""
    m = {}
    for bp, bq in zip(p, q):
        if m.setdefault(bp, bq) != bq:
            return False
    return True


def _mu(p, q):
    """Moebius function of the partition lattice between p <= q."""
    counts = {}
    for bq in set(q):
        counts[bq] = len({bp for bp, b2 in zip(p, q) if b2 == bq})
    val = 1
    for b in counts.values():
        val *= (-1) ** (b - 1) * math.factorial(b - 1)
    return val


def walk_edges(K):
    """Edges of a K-hop walk as (user_var, item_var) pairs."""
    assert K % 2 == 1 and K >= 1
    r = (K - 1) // 2
    edges = []
    for t in range(r + 1):
        edges.append((t, t))            # (U_t, J_{t+1})  item var index t
        if t < r:
            edges.append((t + 1, t))    # (U_{t+1}, J_{t+1})
    return edges, r + 1, r + 1         # edges, #user vars, #item vars


@lru_cache(maxsize=None)
def correction_plan(K):
    """List of (mu, sigma_user, sigma_item, [(ublock, iblock, k), ...]) terms.

    Each term contributes mu * (einsum_corrected - einsum_plugin) where the
    factor list gives, for each distinct edge of pattern pi, its endpoints in
    sigma's blocks and its multiplicity k under pi.
    """
    edges, nu, ni = walk_edges(K)
    P_u, P_i = set_partitions(nu), set_partitions(ni)
    plan = []
    for pu, pi in product(P_u, P_i):
        mult = {}
        for (uv, iv) in edges:
            key = (pu[uv], pi[iv])
            mult[key] = mult.get(key, 0) + 1
        if all(k == 1 for k in mult.values()):
            continue                      # no repeated edge: plug-in is exact
        for su in P_u:
            if not _coarser(pu, su):
                continue
            for si in P_i:
                if not _coarser(pi, si):
                    continue
                mu = _mu(pu, su) * _mu(pi, si)
                # map pi-blocks to sigma-blocks
                mu_map = {pu[v]: su[v] for v in range(nu)}
                mi_map = {pi[v]: si[v] for v in range(ni)}
                factors = tuple((mu_map[ub], mi_map[ib], k) for (ub, ib), k in mult.items())
                plan.append((mu, su, si, factors))
    return tuple(plan)


def _contract(factors, su, si, mats, rows):
    """einsum over sigma blocks. The user block containing U_0 is the output
    row index 'a' (restricted to ``rows``); the item block containing the last
    item variable is the output column index 'b'."""
    u_letters = {}
    i_letters = {}
    pool = iter("cdefghjklmnopqrstuvwxyz")
    u_out = su[0]
    i_out = si[-1]
    subs, ops = [], []
    for ub, ib, k in factors:
        ul = "a" if ub == u_out else u_letters.setdefault(ub, next(pool))
        il = "b" if ib == i_out else i_letters.setdefault(ib, next(pool))
        M = mats[k]
        ops.append(M[rows] if ul == "a" else M)
        subs.append(ul + il)
    # Factors on the same index pair are multiplied elementwise first:
    # einsum path search handles such repeated operands badly (it can build
    # a (rows, m, n) intermediate for 'ac,dc,dc,db->ab').
    merged = {}
    for sub, op in zip(subs, ops):
        merged[sub] = op if sub not in merged else merged[sub] * op
    expr = ",".join(merged) + "->ab"
    return torch.einsum(expr, *merged.values())


def plugin_khop(Wt, K, rows=None):
    """Plug-in K-hop product Wt (Wt^T Wt)^r restricted to ``rows``."""
    r = (K - 1) // 2
    X = Wt if rows is None else Wt[rows]
    for _ in range(r):
        X = (X @ Wt.T) @ Wt
    return X


def khop(W, C, K, rows=None, correct=True):
    """Walk-corrected (exactly unbiased) K-hop propagation over C * W."""
    Wt = C * W
    naive = plugin_khop(Wt, K, rows)
    if not correct or K == 1:
        return naive
    rows_ = torch.arange(W.shape[0]) if rows is None else rows
    kmax = K
    corr_m = {k: (C ** k) * W for k in range(1, kmax + 1)}
    plug_m = {k: Wt ** k for k in range(1, kmax + 1)}
    total = torch.zeros_like(naive)
    for mu, su, si, factors in correction_plan(K):
        total += mu * (_contract(factors, su, si, corr_m, rows_) - _contract(factors, su, si, plug_m, rows_))
    return naive + total
