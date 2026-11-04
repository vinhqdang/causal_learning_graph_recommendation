"""Paired per-user difference between two probe result JSONs (x1e-3, 95% half-width, p)."""
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/tmp/claude-0/-home-user-causal-learning-graph-recommendation/198950e7-01e5-589f-abc7-a16b1bf09c95/scratchpad/probes")
from harness import peru  # noqa: E402

a, b = (json.load(open(p)) for p in sys.argv[1:3])
x, y = peru(a["per_user"]), peru(b["per_user"])
us = sorted(set(x) & set(y))
d = np.array([x[u] - y[u] for u in us])
hw = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
print(f"{sys.argv[1].split('/')[-1]} - {sys.argv[2].split('/')[-1]}: {1000*d.mean():+.2f} [{1000*hw:.2f}] p={stats.ttest_1samp(d, 0).pvalue:.3g} n={len(d)}")
