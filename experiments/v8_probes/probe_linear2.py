"""Probe 1: training-free linear (EASE-type) operators on three-state encodings of the log, added to the
exposure-conditional prior. Encodings:
  pos    : X = O*Y                      (implicit positives only; control = "prior + EASE")
  signed : X = O*Y - c*O*(1-Y)          (exposed negatives as negative evidence)
  resid  : X = O*(Y - pi)               (exposure-conditional residual; unexposed = 0 = no evidence)
score = Z + gamma * std(X_u B), B = EASE(X, lambda) (zero diagonal), selected per split on validation.
"""
import sys
import time

import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import Selector, load, report, std  # noqa: E402

ds = sys.argv[1]
OUT = sys.argv[2]
torch.set_num_threads(4)
D = load(ds)
O, Y, pi, Z, rows = D["O"], D["Y"], D["pi"], D["Z"], D["rows"]
lams = {"coat": [10.0, 30.0, 100.0, 300.0, 1000.0], "kuairec": [100.0, 1000.0, 3000.0, 10000.0, 30000.0]}[ds]
gammas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

sel0 = Selector(D, "prior")
sel0.add(Z[rows], {"prior": 1})
report(D, "prior_only", sel0.result(), OUT)

encs = {}
for c in (-0.1, -0.3, -0.5, -1.0):
    encs[f"signed{c}"] = O * Y - c * O * (1 - Y)
for en, X in encs.items():
    t0 = time.time()
    G = (X.T @ X).double()
    ev, Q = torch.linalg.eigh(G)
    Xr = X[rows].double() @ Q
    sel = Selector(D, en)
    for lam in lams:
        inv = 1.0 / (ev + lam)
        diagP = (Q * Q * inv[None, :]).sum(1)                   # diag of (G + lam I)^-1
        XP = (Xr * inv[None, :]) @ Q.T                           # X_rows (G + lam I)^-1
        S = (X[rows].double() - XP / diagP[None, :]).float()     # X_rows B, B = I - P diag(1/diag P)
        Ss = std(S)
        for g in gammas:
            sel.add(Z[rows] + g * Ss, {"enc": en, "lam": lam, "gamma": g})
        # personalised term alone (no prior), for reference
        sel_alone = None
    report(D, f"lin_{en}", sel.result(), OUT)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    del G, Q, Xr
