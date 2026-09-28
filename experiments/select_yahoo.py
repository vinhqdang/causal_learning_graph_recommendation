"""Yahoo! R3: choose the propensity model per method on validation data.

Training-free methods: results/filters_yahoo_{pop,nb_cv}.json are merged into
results/filters_yahoo_sel.json, keeping for each method the propensity model
with the higher mean validation nDCG@5 (methods that do not use propensities
come from the 'pop' run). The model chosen for DRUP is written to
results/yahoo_prop.txt and used by the FAT and re-ranking runs.
Trained baselines are merged the same way by merge_yahoo_learned.py.
"""

import json

runs = {"pop": "results/filters_yahoo_pop.json", "nb": "results/filters_yahoo_nb_cv.json"}
res = {p: json.load(open(f))["results"] for p, f in runs.items()}
sel, choice = {}, {}
for m in res["pop"]:
    cands = {p: r[m] for p, r in res.items() if m in r and "val_best" in r[m]}
    if not cands:
        sel[m], choice[m] = res["pop"][m], "pop"
        continue
    p = max(cands, key=lambda q: sum(cands[q]["val_best"]) / len(cands[q]["val_best"]))
    sel[m], choice[m] = cands[p], p
    vals = {q: round(sum(c["val_best"]) / len(c["val_best"]), 4) for q, c in cands.items()}
    print(f"{m:8s} -> {p:3s} val {vals}  test ndcg@5 {cands[p]['test']['ndcg@5'][0]:.4f}")
json.dump({"results": sel, "propensity_choice": choice}, open("results/filters_yahoo_sel.json", "w"), indent=1)
open("results/yahoo_prop.txt", "w").write(choice["DRUP"] + "\n")
print("DRUP propensity:", choice["DRUP"])
