"""Matched-configuration comparison DRUP - DR on real data, from the per-configuration banks.

For every configuration shared by DR and DRUP (same nuisances, alpha, degrees, beta), the test
score difference per split; reported by beta (degrees from the imputation, as in the main runs).
Banks come from run_filters.py --dump_bank (protocol v2)."""
import json, sys, numpy as np
k = lambda c: tuple(sorted(c.items()))
for ds in ("coat", "yahoo"):
    b = json.load(open(f"results/v2/bank_filters_{ds}.json"))
    dr = {k(x["cfg"]): x for x in b["banks"]["DR"]}
    dp = {k(x["cfg"]): x for x in b["banks"]["DRUP"]}
    print(f"\n{ds} ({b['key']}): configs shared {len(set(dr) & set(dp))}")
    print("beta   mean diff x1000   share of configs < 0   splits with mean<0")
    for beta in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0):
        D = np.array([np.array(dp[c]["test"]) - np.array(dr[c]["test"]) for c in dr
                      if dict(c)["beta"] == beta and dict(c)["deg"] == "Yhat"])
        print(f"{beta:6}  {1000*D.mean():+8.2f}          {(D.mean(1)<0).mean():5.2f}                {(D.mean(0)<0).sum()}/{D.shape[1]}")
