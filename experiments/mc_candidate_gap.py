"""Size of the candidate-level residual of Corollary 2 in the Monte-Carlo model.

On an unexposed candidate DRUP estimates F* with Y_ui replaced by Yhat_ui
(DR) or by 0 (IPS). This script reports, for the three-hop term of the
synthetic model of mc_stress.py, the mean absolute gap between that
conditional target and F* relative to the mean target, and the rank agreement
(Kendall tau) between the two within users.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drup.khop import khop  # noqa: E402
from mc_stress import kendall_rows, setup  # noqa: E402


def main():
    out = {}
    for K in (3, 5):
        g, Y, P, Yhat, C = setup(30, 40, 0, 1.0)
        m, n = Y.shape
        target = khop(Y, C, K, correct=False)

        def cond_target(repl):
            T = torch.zeros(m, n, dtype=torch.float64)
            for u in range(m):
                for i in range(n):
                    Yc = Y.clone()
                    Yc[u, i] = repl[u, i]
                    T[u, i] = khop(Yc, C, K, rows=torch.tensor([u]), correct=True)[0, i]
            return T
        scale = target.abs().mean()
        mask = torch.ones_like(Y) > 0
        for name, repl in (("DR", Yhat), ("IPS", torch.zeros_like(Y))):
            ct = cond_target(repl)
            out[f"K{K}/{name}"] = {"rel_gap": float((ct - target).abs().mean() / scale),
                                   "kendall_tau_cond_vs_full": kendall_rows(ct, target, mask)}
            print(K, name, out[f"K{K}/{name}"], flush=True)
    with open("results/v2/mc_candidate_gap.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
