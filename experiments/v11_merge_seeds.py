"""Merge the Stage C seed replicates of DAX into results/v11/c_seeds.json (paper/v7/DESIGN.md, Section 7, Stage C).

Seed 0 is the main run: LEAF-LGN+DAX and LEAF-LGN from results/v11/dax_lgn_post.json, QP family+DAX and QP family from
results/v11/dax_qp.json (the same selection as the replicates: see the notes in experiments/queue_v11.sh, c_seeds_lgn).
Seeds 1, 2, 3 come from results/v11/c_seeds/lgn_s{k}.json and results/v11/c_seeds/qp_s{k}.json (experiments/run_dax.py,
--train_seed k). A seed whose file is missing is left out and reported.

python3 experiments/v11_merge_seeds.py
Output: results/v11/c_seeds.json = {"args": {...}, "seeds": {"0": {method: result}, "1": {...}, ...}}, where result is the
run_dax entry (test, chosen, per_split, val_best, per_user, n_configs, n_evals).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
V11 = "results/v11/"
OUT = V11 + "c_seeds.json"
MAIN = {"LEAF-LGN+DAX": (V11 + "dax_lgn_post.json", "LEAF-LGN+DAX"), "LEAF-LGN": (V11 + "dax_lgn_post.json", "LEAF-LGN"),
        "QP family+DAX": (V11 + "dax_qp.json", "QP family+DAX"), "QP family": (V11 + "dax_qp.json", "QP family")}
REPLICATE = {"LEAF-LGN+DAX": "lgn", "LEAF-LGN": "lgn", "QP family+DAX": "qp", "QP family": "qp"}


def main():
    seeds, args = {}, {}
    seeds["0"] = {}
    for name, (path, key) in MAIN.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing main run: {path}")
        seeds["0"][name] = json.load(open(path))["results"][key]
    args["seed0"] = {p: json.load(open(p))["args"] for p in sorted({v[0] for v in MAIN.values()})}
    for k in (1, 2, 3):
        got = {}
        for name, fam in REPLICATE.items():
            path = f"{V11}c_seeds/{fam}_s{k}.json"
            if os.path.exists(path):
                got[name] = json.load(open(path))["results"][name]
                args.setdefault(f"seed{k}", {})[fam] = json.load(open(path))["args"]
        if len(got) == len(REPLICATE):
            seeds[str(k)] = got
        else:
            print(f"seed {k}: incomplete ({sorted(got)} present), left out")
    with open(OUT, "w") as f:
        json.dump({"args": args, "seeds": seeds}, f)
    print(f"wrote {OUT} with seeds {sorted(seeds, key=int)}")


if __name__ == "__main__":
    main()
