"""Exposure-constrained re-ranking with a certified optimality gap.

Given scores s_ui on each user's candidate set, choose K items per user so
that every item is recommended to at most cap_i users:

    max_x  sum_ui s_ui x_ui   s.t.  sum_i x_ui = K,  sum_u x_ui <= cap_i,
                                     x_ui in {0, 1},  x_ui = 0 off-candidates.

This is a bipartite b-matching: the constraint matrix is totally unimodular,
so the LP relaxation has an integral optimum and strong duality holds. The
Lagrangian dual

    D(lam) = sum_u topK_i (s_ui - lam_i) + sum_i lam_i cap_i,   lam >= 0,

upper-bounds the optimum for every lam. We minimise it by projected
subgradient steps, take each user's top-K under the best prices, repair the
(few) capacity violations greedily, and return the primal value P together
with the certificate D(lam*) - P >= OPT - P (docs/THEORY.md, Theorem 9).
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _topk_masked(S, mask, K):
    X = torch.where(mask, S, torch.full_like(S, -float("inf")))
    val, idx = torch.topk(X, K, dim=1)
    return val, idx


def dual_value(S, mask, lam, cap, K):
    val, _ = _topk_masked(S - lam[None, :], mask, K)
    return float(val.sum() + (lam * cap).sum())


def rerank(S, mask, K, cap, iters=300, step0=None, seed=0):
    """S, mask: (R, N) scores and candidate mask; cap: (N,) capacities.

    Returns (alloc (R, K) item indices, info dict with primal, dual, gap,
    max load / cap).
    """
    cols = torch.nonzero(mask.any(0)).flatten()      # work on candidate items only
    if len(cols) < mask.shape[1]:
        alloc_c, info = rerank(S[:, cols], mask[:, cols], K, cap[cols], iters, step0, seed)
        return cols[alloc_c], info
    S = S.double()
    cap = cap.double()
    R, N = S.shape
    kmin = int(mask.sum(1).min())
    assert kmin >= K, "every user needs at least K candidates"
    finite = S[mask]
    scale = float(finite.abs().mean().clamp_min(1e-12))
    step0 = step0 or scale
    lam = torch.zeros(N, dtype=torch.float64)
    best_lam, best_dual = lam.clone(), dual_value(S, mask, lam, cap, K)
    for t in range(1, iters + 1):
        _, idx = _topk_masked(S - lam[None, :], mask, K)
        load = torch.zeros(N, dtype=torch.float64).index_add_(0, idx.flatten(), torch.ones(idx.numel(), dtype=torch.float64))
        g = cap - load                                   # subgradient of D
        gn = float(g.norm())
        if gn == 0:
            break
        lam = (lam - step0 / math.sqrt(t) * g / gn * math.sqrt(N)).clamp_min(0.0)
        dv = dual_value(S, mask, lam, cap, K)
        if dv < best_dual:
            best_dual, best_lam = dv, lam.clone()
    alloc = _repair(S, mask, best_lam, cap, K)
    primal = float(S.gather(1, alloc).sum())
    load = torch.zeros(N, dtype=torch.float64).index_add_(0, alloc.flatten(), torch.ones(alloc.numel(), dtype=torch.float64))
    return alloc, {
        "primal": primal, "dual": best_dual,
        "rel_gap": (best_dual - primal) / max(abs(primal), 1e-12),
        "max_load_over_cap": float((load / cap.clamp_min(1)).max()),
        "feasible": bool((load <= cap + 1e-9).all()),
    }


def _repair(S, mask, lam, cap, K):
    """Top-K under prices, then move users off over-capacity items to their
    best alternative with spare capacity, smallest reduced-score loss first
    (vectorised over the users of each over-capacity item)."""
    R, N = S.shape
    Sr = torch.where(mask, S - lam[None, :], torch.full_like(S, -float("inf")))
    _, idx = torch.topk(Sr, K, dim=1)
    alloc = idx.clone()
    assigned = torch.zeros(R, N, dtype=torch.bool)
    assigned.scatter_(1, alloc, True)
    load = assigned.sum(0).double()
    for _ in range(10 * N):
        over = torch.nonzero(load > cap + 1e-9).flatten()
        if len(over) == 0:
            break
        i = int(over[torch.argmax(load[over] - cap[over])])
        users = torch.nonzero(assigned[:, i]).flatten()
        spare = load < cap - 1e-9
        ok = mask[users] & ~assigned[users] & spare[None, :]
        alt = torch.where(ok, Sr[users], torch.full_like(Sr[users], -float("inf")))
        best_val, best_j = alt.max(1)
        valid = torch.isfinite(best_val)
        if not bool(valid.any()):
            raise RuntimeError("capacity constraints infeasible for this candidate structure")
        loss = torch.where(valid, Sr[users, i] - best_val, torch.full_like(best_val, float("inf")))
        excess = int(round(float(load[i] - cap[i])))
        moved = 0
        for t in torch.argsort(loss).tolist():
            if moved >= excess or not bool(valid[t]):
                break
            u, j = int(users[t]), int(best_j[t])
            if load[j] >= cap[j] - 1e-9:
                continue                      # filled meanwhile; next sweep
            k = int((alloc[u] == i).nonzero()[0])
            alloc[u, k] = j
            assigned[u, i], assigned[u, j] = False, True
            load[i] -= 1
            load[j] += 1
            moved += 1
    # order each user's allocation by original score
    order = torch.argsort(S.gather(1, alloc), dim=1, descending=True)
    return alloc.gather(1, order)


def uniform_caps(mask, K, factor):
    """cap_i = ceil(factor * (R K / #candidate items)) for candidate items.
    factor = inf means no constraint. Coverage >= R K / cap (deterministic)."""
    R = mask.shape[0]
    items = mask.any(0)
    fair = R * K / int(items.sum())
    if math.isinf(factor):
        c = float(R)
    else:
        c = float(max(1, math.ceil(factor * fair)))
    return torch.where(items, torch.full((mask.shape[1],), c, dtype=torch.float64),
                       torch.zeros(mask.shape[1], dtype=torch.float64))


def rerank_exact(S, mask, K, cap, resolution=1e6):
    """Exact b-matching by min-cost flow (OR-Tools).

    Scores are mapped affinely to integers in [0, resolution]; the optimum of
    the integer problem is optimal for the real scores up to
    R K * range / resolution, which is reported as the certified gap.
    """
    from ortools.graph.python import min_cost_flow

    cols = torch.nonzero(mask.any(0)).flatten()
    Sc, Mc, capc = S[:, cols].double(), mask[:, cols], cap[cols].double()
    R, N = Sc.shape
    lo, hi = float(Sc[Mc].min()), float(Sc[Mc].max())
    rng = max(hi - lo, 1e-12)
    uu, ii = torch.nonzero(Mc, as_tuple=True)
    cost = -np.rint((Sc[uu, ii].numpy() - lo) / rng * resolution).astype(np.int64)
    src, sink = 0, R + N + 1
    tails = np.concatenate([np.zeros(R, np.int64), 1 + uu.numpy(), 1 + R + np.arange(N)])
    heads = np.concatenate([1 + np.arange(R), 1 + R + ii.numpy(), np.full(N, sink)])
    caps = np.concatenate([np.full(R, K), np.ones(len(uu), np.int64), capc.numpy().astype(np.int64)])
    costs = np.concatenate([np.zeros(R, np.int64), cost, np.zeros(N, np.int64)])
    f = min_cost_flow.SimpleMinCostFlow()
    arcs = f.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)
    supplies = np.zeros(R + N + 2, np.int64)
    supplies[src], supplies[sink] = R * K, -R * K
    f.set_nodes_supplies(np.arange(R + N + 2), supplies)
    status = f.solve()
    if status != f.OPTIMAL:
        raise RuntimeError(f"min-cost flow status {status} (infeasible caps?)")
    flow = f.flows(arcs[R:R + len(uu)])
    sel = flow > 0
    alloc = torch.zeros(R, K, dtype=torch.long)
    fill = np.zeros(R, np.int64)
    for u, i in zip(uu.numpy()[sel], ii.numpy()[sel]):
        alloc[u, fill[u]] = int(i)
        fill[u] += 1
    order = torch.argsort(Sc.gather(1, alloc), dim=1, descending=True)
    alloc = alloc.gather(1, order)
    primal = float(Sc.gather(1, alloc).sum())
    load = torch.zeros(N, dtype=torch.float64).index_add_(0, alloc.flatten(), torch.ones(alloc.numel(), dtype=torch.float64))
    gap = R * K * rng / resolution    # each chosen score is rounded by <= rng / (2 resolution)
    return cols[alloc], {"primal": primal, "dual": primal + gap, "rel_gap": gap / max(abs(primal), 1e-12),
                         "max_load_over_cap": float((load / capc.clamp_min(1)).max()),
                         "feasible": bool((load <= capc + 1e-9).all()), "solver": "min-cost-flow"}
