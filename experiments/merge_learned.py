"""Merge the part files of the round-3 trained baselines into one result
file (and one bank file) per dataset."""

import glob
import json
import os

for ds in ("coat_given", "yahoo_pop", "kuairec_pop", "kuairand_pop"):
    parts = sorted(glob.glob(f"results/v3/learned_{ds}_*.json"))
    single = f"results/v3/learned_{ds}.json"
    if not parts:
        continue
    res, banks = {}, {}
    for p in parts + ([single] if os.path.exists(single) else []):
        if p.endswith(".partial.json") or os.path.basename(p).startswith("bank_"):
            continue
        res.update(json.load(open(p))["results"])
        b = os.path.join(os.path.dirname(p), "bank_" + os.path.basename(p))
        if os.path.exists(b):
            bj = json.load(open(b))
            banks.update(bj["banks"])
            key = bj["key"]
    json.dump({"results": res, "parts": parts}, open(single, "w"))
    if banks:
        json.dump({"key": key, "banks": banks}, open(f"results/v3/bank_learned_{ds}.json", "w"))
    print(ds, sorted(res))
