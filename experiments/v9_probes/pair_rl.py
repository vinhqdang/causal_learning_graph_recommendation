"""Paired per-user difference between a probe_27k.py JSON (per-user keys = positions in the test list) and a
run_learned JSON method (per-user keys = user row ids), on the same cached setting.
python3 experiments/v9_probes/pair_rl.py PROBE.json RUN_LEARNED.json METHOD data/raw/kuairand_t.pt
"""
import json
import sys

import numpy as np
import torch
from scipy import stats


def peru(pul):
    a = {}
    for q in pul:
        for u, v in q.items():
            a.setdefault(str(u), []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


probe, rl, method, cache = sys.argv[1:5]
test = torch.load(cache, weights_only=False)["test"]
x = peru(json.load(open(probe))["per_user"])
x = {str(test[int(k)][0]): v for k, v in x.items()}
r = json.load(open(rl))["results"][method]
y = peru(r["per_user"])
us = sorted(set(x) & set(y))
d = np.array([x[u] - y[u] for u in us])
hw = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
ref = np.mean([s["ndcg@10"] for s in r["per_split"]])
print(f"{probe.split('/')[-1]} - {method} (ref {ref:.4f}): {1000*d.mean():+.2f} [{1000*hw:.2f}] "
      f"p={stats.ttest_1samp(d, 0).pvalue:.3g} n={len(d)}")
