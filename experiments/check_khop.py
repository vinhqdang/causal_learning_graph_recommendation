"""Exactness of the K-hop engine: (i) K=3 against the closed form of three_hop; (ii) K=3 and K=5 against brute-force
enumeration of all walks u-j1-v1-j2-...-i, where every repeated edge e traversed k times contributes C_e^k * W_e
(the idempotent replacement of W_e^k C_e^k). Small random dense matrices, float64."""
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.khop import khop  # noqa: E402
from drup.propagation import three_hop  # noqa: E402

torch.manual_seed(0)
m, n = 5, 6
W = torch.rand(m, n, dtype=torch.float64)
C = torch.rand(m, n, dtype=torch.float64)


def brute(K):
    """Sum over walks with K edges from user u to item i; users and items alternate."""
    out = torch.zeros(m, n, dtype=torch.float64)
    nu = (K - 1) // 2          # intermediate users
    for u in range(m):
        for i in range(n):
            tot = 0.0
            # walk: u - j1 - v1 - j2 - v2 ... - i ; K edges, K-1 intermediate nodes alternating item,user,...,item? K odd
            for items in itertools.product(range(n), repeat=nu):
                for users in itertools.product(range(m), repeat=nu):
                    path_u = [u] + list(users)            # users on the walk
                    path_i = list(items) + [i]            # items on the walk
                    edges = []
                    for a in range(nu + 1):
                        edges.append((path_u[a], path_i[a]))
                        if a < nu:
                            edges.append((path_u[a + 1], path_i[a]))
                    cnt = {}
                    for e in edges:
                        cnt[e] = cnt.get(e, 0) + 1
                    val = 1.0
                    for e, k in cnt.items():
                        val *= (C[e] ** k) * (W[e] if k > 1 else W[e])
                    tot += float(val)
            out[u, i] = tot
    return out


res = {}
res["K3_vs_closed_form"] = float((khop(W, C, 3, correct=True) - three_hop(W, C, correct=True)).abs().max())
for K in (3, 5):
    b = brute(K)
    res[f"K{K}_vs_bruteforce_max_abs"] = float((khop(W, C, K, correct=True) - b).abs().max())
    res[f"K{K}_vs_bruteforce_rel"] = float((khop(W, C, K, correct=True) - b).abs().max() / b.abs().max())
print(json.dumps(res, indent=1))
os.makedirs("results/v3", exist_ok=True)
json.dump(res, open("results/v3/khop_check.json", "w"), indent=1)
