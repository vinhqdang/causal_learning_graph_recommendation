"""Paired per-user difference between two probe result JSONs (x1e-3, 95% half-width, p).
python3 experiments/v9_probes/pair.py A.json B.json"""
import json
import sys

import numpy as np
from scipy import stats


def peru(pul):
    a = {}
    for q in pul:
        for u, v in q.items():
            a.setdefault(str(u), []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


a, b = (json.load(open(p)) for p in sys.argv[1:3])
x, y = peru(a["per_user"]), peru(b["per_user"])
us = sorted(set(x) & set(y))
d = np.array([x[u] - y[u] for u in us])
hw = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
print(f"{sys.argv[1].split('/')[-1]} - {sys.argv[2].split('/')[-1]}: {1000*d.mean():+.2f} [{1000*hw:.2f}] "
      f"p={stats.ttest_1samp(d, 0).pvalue:.3g} n={len(d)}")
