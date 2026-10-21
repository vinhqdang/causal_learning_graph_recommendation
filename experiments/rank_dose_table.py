import json, sys, numpy as np
from scipy import stats
r = json.load(open("results/v3/rank_dose.json"))
betas = ["0.3", "1.0", "3.0", "10.0", "1000.0"]
for pair in (("DRUP", "DR"), ("IPS+WC", "IPS")):
    for met in ("ndcg", "ovl"):
        print(f"\n{pair[0]} - {pair[1]}  {met}  (mean diff x1000 [t-CI half-width], n draws)")
        print("cell".ljust(22), *[f"b={b}".rjust(18) for b in betas])
        for c, v in r.items():
            cells = []
            for b in betas:
                d = np.array([x[f"{pair[0]}|{b}|{met}"] - x[f"{pair[1]}|{b}|{met}"] for x in v["rows"]])
                h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                cells.append(f"{1000*d.mean():+7.2f}[{1000*h:5.2f}]")
            print(c.ljust(22), *[s.rjust(18) for s in cells])
print("\nlevels (DR ndcg x1000):")
for c, v in r.items():
    print(c.ljust(22), *[f"{1000*np.mean([x[f'DR|{b}|ndcg'] for x in v['rows']]):8.1f}" for b in betas])
