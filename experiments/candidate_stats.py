"""Candidate-set statistics and a random-ranking reference per dataset."""
import json
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.data import load_coat, load_yahoo  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

out = {}
for name, K in (("coat", 5), ("yahoo", 5), ("kuairec", 20)):
    d = load_coat() if name == "coat" else load_yahoo() if name == "yahoo" else \
        torch.load("data/raw/kuairec.pt", weights_only=False)
    test = d["test"]
    nc = np.array([len(it) for _, it, _ in test])
    npos = np.array([rel.sum() for _, _, rel in test])
    users = sorted({u for u, _, _ in test})
    row_of = {u: k for k, u in enumerate(users)}
    n = d["O"].shape[1]
    g = torch.Generator().manual_seed(0)
    rnd = [evaluate(torch.rand(len(users), n, generator=g), test, (K,), row_of)[f"ndcg@{K}"] for _ in range(20)]
    O = d["O"]
    out[name] = {"test_users": len(test), "cand_mean": float(nc.mean()), "cand_median": float(np.median(nc)),
                 "pos_mean": float(npos.mean()), "users_with_pos": int((npos > 0).sum()),
                 f"random_ndcg@{K}": float(np.mean(rnd)), "log_density": float(O.mean()),
                 "log_rows": int(O.shape[0]), "log_cols": int(O.shape[1]), "logged": int(O.sum())}
    print(name, out[name], flush=True)
json.dump(out, open("results/v2/candidate_stats.json", "w"), indent=1)
