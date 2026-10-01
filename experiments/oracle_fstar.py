"""Ranking quality of the full-exposure target F* itself on the semi-synthetic KuaiRec log:
F* contains C*Y (the label), so it ranks unexposed candidates by their own outcome plus
neighbourhood terms; how good a ranker is the target the estimators try to recover?"""
import json
import sys

import torch

sys.path.insert(0, ".")
from drup.propagation import degree_weights, three_hop  # noqa: E402
from experiments.semisynth_kuairec import DT, load, ndcg_unexposed, propensities  # noqa: E402

Y, ru, ri, pop_big = load()
m, n = Y.shape
out = {}
for name, dens, pmin in (("dense", 0.13, 0.02), ("sparse", 0.03, 0.005)):
    P = propensities(ru, ri, Y, dens, 1.0, pmin)
    C = degree_weights(Y, 0.5, D=Y)
    T3 = three_hop(Y, C, correct=False)
    S1 = C * Y
    F = S1 / S1.abs().mean() + T3 / T3.abs().mean()
    F3 = T3 / T3.abs().mean()
    g = torch.Generator().manual_seed(0)
    r = {"Fstar": [], "three_hop_only": [], "one_hop_only": [], "popularity": []}
    for _ in range(5):
        O = (torch.rand(m, n, generator=g, dtype=DT) < P).to(DT)
        r["Fstar"].append(ndcg_unexposed(F, Y, O))
        r["three_hop_only"].append(ndcg_unexposed(F3, Y, O))
        r["one_hop_only"].append(ndcg_unexposed(S1 + 1e-9 * torch.rand(m, n, generator=g, dtype=DT), Y, O))
        r["popularity"].append(ndcg_unexposed(pop_big[None, :].expand(m, -1) + 1e-9 * torch.rand(m, n, generator=g, dtype=DT), Y, O))
    out[name] = {k: float(sum(v) / len(v)) for k, v in r.items()}
    print(name, out[name], flush=True)
json.dump(out, open("results/v3/oracle_fstar.json", "w"), indent=1)
