"""Diagnostic only (uses test labels; never used for selection or reported as a method):
how much of the attainable nDCG is item-level? Ranks each test user's candidates by the
leave-this-user-out mean outcome of the item over the other users' unbiased test entries
(a non-personalised oracle), and compares with the log-fitted prior."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import load  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

for ds in sys.argv[1:]:
    D = load(ds)
    agg = {}
    for u, it, rel in D["splits"][0][0] + D["splits"][0][1]:      # all unbiased entries
        a = agg.setdefault(u, ([], []))
        a[0].append(it)
        a[1].append(rel)
    test = [(u, np.concatenate(a[0]), np.concatenate(a[1])) for u, a in agg.items()]
    n = D["O"].shape[1]
    s1, c1 = np.zeros(n), np.zeros(n)
    for u, it, rel in test:
        np.add.at(s1, it, rel)
        np.add.at(c1, it, 1)
    rows, row_of = D["rows"], D["row_of"]
    S = np.zeros((len(rows), n))
    for u, it, rel in test:
        r = row_of[u]
        num, den = s1.copy(), c1.copy()
        np.add.at(num, it, -rel)
        np.add.at(den, it, -1)
        S[r] = (num + 0.5) / (den + 5.0)
    key, ks = D["key"], D["ks"]
    full = [t for t in test]
    print(ds, "item oracle (LOO over test users)", round(evaluate(torch.tensor(S), full, ks, row_of)[key], 4))
    print(ds, "log prior", round(evaluate(D["Z"][rows], full, ks, row_of)[key], 4))
    pr = D["pi"].mean(0).numpy()
    ok = c1 > 0
    print(ds, "corr(item prior mean, test item rate)", round(np.corrcoef(pr[ok], (s1[ok] / c1[ok]))[0, 1], 3))
