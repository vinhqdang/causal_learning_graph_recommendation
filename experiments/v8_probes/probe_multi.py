"""Probe 3: three-channel exposure-aware linear scorer (training-free, closed form).

Channels (each an EASE operator on its own encoding of the log, lambda per channel):
  pos   : O*Y           (implicit positives: selection + preference)
  exp   : O             (exposure only: what the logging process chose for the user)
  resid : O*(Y - pi)    (exposure-conditional residual: personalised preference among exposed)
score = Z + g_pos * std(S_pos) + g_exp * std(S_exp) + g_res * std(S_res); all weights chosen on validation.
"""
import itertools
import sys
import time

import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import Selector, load, report, std  # noqa: E402

ds, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
D = load(ds)
O, Y, pi, Z, rows = D["O"], D["Y"], D["pi"], D["Z"], D["rows"]
lam_of = {"coat": {"pos": 300.0, "exp": 300.0, "resid": 100.0},
          "kuairec": {"pos": 3000.0, "exp": 30000.0, "resid": 3000.0}}[ds]
if len(sys.argv) > 4:
    lam_of.update(eval(sys.argv[4]))
encs = {"pos": O * Y, "exp": O.clone(), "resid": O * (Y - pi)}
Ssd = {}
for en, X in encs.items():
    t0 = time.time()
    G = X.T @ X
    G.diagonal().add_(lam_of[en])
    L = torch.linalg.cholesky(G)
    P = torch.cholesky_inverse(L)
    del G, L
    S = X[rows] - (X[rows] @ P) / P.diagonal()[None, :]
    Ssd[en] = std(S)
    del P
    print(f"  channel {en} [{time.time()-t0:.0f}s]", flush=True)
gs = [0.0, 0.1, 0.2, 0.5, 1.0]
sel = Selector(D, "multi")
for gp, ge, gr in itertools.product(gs, gs, gs):
    sel.add(Z[rows] + gp * Ssd["pos"] + ge * Ssd["exp"] + gr * Ssd["resid"], {"gp": gp, "ge": ge, "gr": gr})
report(D, "multi", sel.result(), OUT)
# each channel alone with the prior, for attribution
for en in encs:
    s1 = Selector(D, en)
    for g in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        s1.add(Z[rows] + g * Ssd[en], {"g": g})
    report(D, f"single_{en}", s1.result(), OUT)
