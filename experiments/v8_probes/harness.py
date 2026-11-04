"""Probe harness: same data, splits and selection rule as experiments/run_learned.py.

Selection: per split, the (configuration, epoch) with the highest validation metric; test is read
only for the selected one. Paired per-user comparison against stored result files (cmp of
final_analysis: per-user mean over splits, t half-width, p).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

REPO = "/home/user/causal_learning_graph_recommendation"
sys.path.insert(0, REPO)
from drup.data import load_coat, load_yahoo, split_test  # noqa: E402
from drup.estimation import baseline_imputation  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

CFG = {"coat": dict(ks=(5, 10), key="ndcg@5", by="entry", S=10),
       "kuairec": dict(ks=(10, 20, 50), key="ndcg@20", by="user", S=5),
       "yahoo": dict(ks=(5, 10), key="ndcg@5", by="user", S=5),
       "kuairand": dict(ks=(5, 10, 20), key="ndcg@10", by="user", S=5)}
V4 = "results/v4/"
QP = {"yahoo": [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] +
      [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")],
      "kuairand": [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                   (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")]}

REFS = {
    "coat": {
        "iALS": ("results/v3/learned_coat_given.json", "iALS"),
        "iALS-wide2": ("results/v6/learned_coat_ialswide2.json", "iALS-wide2"),
        "iALS+prior(IPS)": ("results/v5/learned_coat_add_ials.json", "iALS-p"),
        "iALS+prior(noIPS)": ("results/v6/learned_coat_add_noips_ials.json", "iALS-p"),
        "LGN+prior(noIPS)": ("results/v6/learned_coat_add_noips_lgn.json", "LightGCN-p"),
        "prior alone(noIPS)": ("results/v6/learned_coat_alone_add_noips.json", "Prior-alone"),
        "DR-JL": ("results/v3/learned_coat_given.json", "DR-JL"),
    },
    "kuairec": {
        "iALS": ("results/v3/learned_kuairec_pop_A.json", "iALS"),
        "iALS+prior(noIPS)": ("results/v6/learned_kuairec_add_noips_ials_small.json", "iALS-p-small"),
        "prior alone(noIPS)": ("results/v6/learned_kuairec_alone_add_noips.json", "Prior-alone"),
        "DR-JL": ("results/v3/learned_kuairec_pop_A.json", "DR-JL"),
        "MRDR": ("results/v3/learned_kuairec_pop_A.json", "MRDR"),
        "DRUP": ("results/v2/filters_kuairec_pop.json", "DRUP"),
    },
    "yahoo": {
        "iALS": ("results/v3/learned_yahoo_pop_D.json", "iALS"),
        "LightGCN": ("results/v3/learned_yahoo_pop_B.json", "LightGCN"),
        "r-AdjNorm": ("results/v3/learned_yahoo_pop_A.json", "r-AdjNorm"),
        "LightGCN-p": ("results/v6/learned_yahoo_offonly.json", "LightGCN-p"),
        "QP family": "QP",
    },
    "kuairand": {
        "iALS": ("results/v3/learned_kuairand_pop.json", "iALS"),
        "LightGCN": ("results/v3/learned_kuairand_pop.json", "LightGCN"),
        "r-AdjNorm": ("results/v3/learned_kuairand_pop.json", "r-AdjNorm"),
        "LightGCN-p": ("results/v6/learned_kuairand_offonly.json", "LightGCN-p"),
        "QP family": "QP",
    },
}


def load(ds):
    if ds == "coat":
        d = load_coat()
    elif ds == "yahoo":
        d = load_yahoo(os.path.join(REPO, "data/raw"))
    elif ds == "kuairand":
        d = torch.load(os.path.join(REPO, "data/raw/kuairand.pt"), weights_only=False)
    else:
        d = torch.load(os.path.join(REPO, "data/raw/kuairec.pt"), weights_only=False)
    O, Y = d["O"].float(), d["Y"].float()
    c = CFG[ds]
    splits = [split_test(d["test"], 0.3, s, by=c["by"]) for s in range(c["S"])]
    rows_users = sorted({u for u, _, _ in d["test"]})
    rows = torch.tensor(rows_users)
    row_of = {u: k for k, u in enumerate(rows_users)}
    # exposure-conditional additive prior without IPS (the 'add_noips' prior of run_learned.py)
    pi = baseline_imputation(O, Y, torch.ones_like(O), lam=5.0, ips=False).float()
    lg = torch.logit(pi.clamp(1e-3, 1 - 1e-3))
    Z = (lg - lg.mean()) / lg.std()
    return dict(ds=ds, O=O, Y=Y, pi=pi, Z=Z, splits=splits, rows=rows, row_of=row_of, **c)


class Selector:
    """Collects validation/test per split for every (cfg, epoch); test per-user only for the best."""

    def __init__(self, D, name):
        self.D, self.name = D, name
        self.best = [(-1.0, None, None, None) for _ in D["splits"]]
        self.n = 0

    def add(self, S_rows, cfg):
        D = self.D
        self.n += 1
        vals = [evaluate(S_rows, v, D["ks"], D["row_of"])[D["key"]] for v, _ in D["splits"]]
        for s, v in enumerate(vals):
            if v > self.best[s][0]:
                agg, pu = evaluate(S_rows, D["splits"][s][1], D["ks"], D["row_of"], per_user=D["key"])
                self.best[s] = (v, dict(cfg), agg[D["key"]], pu)
        return float(np.mean(vals))

    def result(self):
        return {"test_mean": float(np.mean([b[2] for b in self.best])),
                "test_sd": float(np.std([b[2] for b in self.best])),
                "val_mean": float(np.mean([b[0] for b in self.best])),
                "chosen": [b[1] for b in self.best],
                "per_user": [{str(u): v for u, v in b[3].items()} for b in self.best],
                "n_configs": self.n}


def peru(pul):
    a = {}
    for q in pul:
        for u, v in q.items():
            a.setdefault(str(u), []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


def compare(D, res, out_lines):
    x = peru(res["per_user"])
    for name, fm in REFS[D["ds"]].items():
        if fm == "QP":
            rs = [json.load(open(os.path.join(REPO, f)))["results"][m] for f, m in QP[D["ds"]]]
            r = {"per_user": [], "per_split": []}
            for s_ in range(len(rs[0]["per_split"])):
                j = int(np.argmax([q["val_best"][s_] for q in rs]))
                r["per_user"].append(rs[j]["per_user"][s_])
                r["per_split"].append(rs[j]["per_split"][s_])
        else:
            f, m = fm
            p = os.path.join(REPO, f)
            if not os.path.exists(p):
                continue
            r = json.load(open(p))["results"][m]
        y = peru(r["per_user"])
        us = sorted(set(x) & set(y))
        d = np.array([x[u] - y[u] for u in us])
        ref_mean = np.mean([s[D["key"]] for s in r["per_split"]])
        hw = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        pv = stats.ttest_1samp(d, 0).pvalue
        out_lines.append(f"    vs {name:20s} ref {ref_mean:.4f}  diff {1000*d.mean():+7.2f} [{1000*hw:5.2f}] p={pv:.3g} n={len(d)}")


def report(D, name, res, path):
    lines = [f"{D['ds']} {name}: test {res['test_mean']:.4f} +- {res['test_sd']:.4f} (val {res['val_mean']:.4f}, {res['n_configs']} cfg-epochs)",
             f"    chosen: {res['chosen'][:3]}"]
    compare(D, res, lines)
    txt = "\n".join(lines)
    print(txt, flush=True)
    with open(path, "a") as f:
        f.write(txt + "\n")
    js = path.replace(".txt", f"_{D['ds']}_{name}.json")
    json.dump(res, open(js, "w"))


def std(S):
    return (S - S.mean()) / S.std().clamp_min(1e-12)
