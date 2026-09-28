"""Pick, for every trained Yahoo baseline, the propensity model (Naive Bayes
or logistic exposure) with the higher mean *validation* nDCG, and write the
merged results to results/learned_yahoo_sel_tuned.json."""

import json
import re
from collections import defaultdict

RUNS = {"nb": ("results/learned_yahoo_nb_tuned.json", "results/log_learned_yahoo_nb.txt"),
        "pop": ("results/learned_yahoo_pop_tuned.json", "results/log_learned_yahoo_pop.txt")}


def val_scores(log):
    v = defaultdict(list)
    for line in open(log):
        m = re.match(r"\s+(\S+) split \d+: val ([0-9.]+)", line)
        if m:
            v[m.group(1)].append(float(m.group(2)))
    return {k: sum(x) / len(x) for k, x in v.items()}


merged = {"results": {}, "propensity_choice": {}, "val": {}}
cands = defaultdict(dict)
for prop, (res, log) in RUNS.items():
    r = json.load(open(res))["results"]
    vs = val_scores(log)
    for m, x in r.items():
        cands[m][prop] = (vs[m], x)
for m, byp in cands.items():
    prop = max(byp, key=lambda p: byp[p][0])
    merged["results"][m] = byp[prop][1]
    merged["propensity_choice"][m] = prop
    merged["val"][m] = {p: round(v[0], 4) for p, v in byp.items()}
json.dump(merged, open("results/learned_yahoo_sel_tuned.json", "w"), indent=1)
for m in merged["results"]:
    print(f"{m:12s} prop={merged['propensity_choice'][m]:3s} val={merged['val'][m]} "
          f"test ndcg@5={merged['results'][m]['test']['ndcg@5'][0]:.4f}")
