"""Diagnostic only (fits on unbiased KuaiRec test entries; never a method, never used for selection):
how much personalised signal does the KuaiRec test matrix contain at all? Split every test user's
unbiased candidates 50/50 at random (A/B). On A fit (i) item means, (ii) item + rank-k residual
(truncated SVD of the user-centred residual matrix, missing = 0), and rank B's candidates.
If (ii) barely beats (i) even with uniform training data, the benchmark cannot reward personalisation."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import load  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

D = load("kuairec")
rng = np.random.default_rng(0)
allt = D["splits"][0][0] + D["splits"][0][1]
users = sorted({u for u, _, _ in allt})
items = sorted({int(i) for _, it, _ in allt for i in it})
ui = {u: k for k, u in enumerate(users)}
ii = {i: k for k, i in enumerate(items)}
R = np.zeros((len(users), len(items)))
M = np.zeros_like(R)
B = []
for u, it, rel in allt:
    msk = rng.random(len(it)) < 0.5
    for i, r in zip(it[msk], rel[msk]):
        R[ui[u], ii[int(i)]] = r
        M[ui[u], ii[int(i)]] = 1
    B.append((ui[u], np.array([ii[int(i)] for i in it[~msk]]), rel[~msk]))
item = (R.sum(0) + 0.5) / (M.sum(0) + 5)
ubar = (R.sum(1) / np.maximum(M.sum(1), 1))
res = M * (R - item[None, :] - (ubar[:, None] - (M * item[None, :]).sum(1, keepdims=True) / np.maximum(M.sum(1, keepdims=True), 1)))
print("item-only (fit on A, test on B)", round(evaluate(torch.tensor(np.tile(item, (len(users), 1))), B, (20,), None)["ndcg@20"], 4))
U, s, Vt = np.linalg.svd(res, full_matrices=False)
for k in (4, 16, 64):
    L = (U[:, :k] * s[:k]) @ Vt[:k]
    for g in (0.5, 1.0, 2.0):
        S = item[None, :] + g * L
        print(f"item + rank-{k} residual x{g}", round(evaluate(torch.tensor(S), B, (20,), None)["ndcg@20"], 4))
