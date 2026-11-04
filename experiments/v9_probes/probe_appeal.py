"""Probe A: item appeal as a latent factor measured by several log indicators (policy exposure and
multi-behaviour responses), unsupervised (no uniform data), compared with popularity; then used as the
offset of a personal backbone with beta chosen on validation.
python3 experiments/v9_probes/probe_appeal.py OUT.txt"""
import sys

import numpy as np
import pandas as pd
import torch

from common import Selector, ease, load, report, rowstd, std

torch.set_num_threads(3)
out = sys.argv[1]
D = load(extra_users=True)
d, O, Y, rows = D["d"], D["O"], D["Y"], D["rows"]
n = O.shape[1]
lg = pd.concat([d["log"].assign(src=0), d["aux"].rename(columns={"u_aux": "u"}).assign(src=1)], ignore_index=True)


def indicators(x):
    g = x.groupby("i")
    f = pd.DataFrame({"imp": g.size(), "clk": g.is_click.sum(), "lv": g.long_view.sum(), "like": g.is_like.sum(),
                      "users": g.u.nunique(), "pr": g.play_ratio.mean(),
                      "eng": g.is_profile_enter.sum() + g.is_follow.sum() + g.is_comment.sum() + g.is_forward.sum()})
    f = f.reindex(range(n)).fillna(0.0)
    mu = x.is_click.mean()
    f["ctr"] = (f.clk + 20 * mu) / (f.imp + 20)
    f["lvr"] = (f.lv + 20 * x.long_view.mean()) / (f.imp + 20)
    L = pd.DataFrame({k: np.log1p(f[k]) for k in ["imp", "clk", "lv", "like", "users", "eng"]})
    L["ctr"], L["lvr"], L["pr"] = f.ctr, f.lvr, f.pr
    Z = (L - L.mean()) / L.std().replace(0, 1)
    return Z


def factor(Z):
    U, s, Vt = np.linalg.svd(Z.values, full_matrices=False)
    v = Vt[0] * np.sign(Vt[0][list(Z.columns).index("clk")])
    sc = Z.values @ v
    return torch.tensor((sc - sc.mean()) / sc.std(), dtype=torch.float32), dict(zip(Z.columns, v.round(3)))


def bcast(v):
    return v[None, :].expand(len(rows), -1)


lines = []
for name, x in [("10K users", lg[lg.src == 0]), ("all 27K users", lg)]:
    Z = indicators(x)
    fac, load_ = factor(Z)
    popc = torch.tensor(Z.clk.values, dtype=torch.float32)
    for nm, v in [("pop(clicks)", popc), ("appeal factor", fac)]:
        sel = Selector(D, f"{nm} [{name}]")
        sel.add(bcast(v), {})
        report(D, f"{nm.replace(' ', '_')}_{name.split()[1]}", sel.result(), out)
    with open(out, "a") as f:
        f.write(f"  factor loadings [{name}]: {load_}\n")
    # personal backbone + offset, beta on validation
    X = O * Y
    for nm, v in [("pop(clicks)", popc), ("appeal factor", fac)]:
        sel = Selector(D, f"EASE+{nm} [{name}]")
        for lam in (1000.0, 3000.0):
            S = rowstd(X[rows] @ ease(X, lam))
            for b in (0.0, 0.25, 0.5, 1.0, 2.0):
                sel.add(S + b * bcast(v), dict(lam=lam, beta=b))
        report(D, f"EASE+{nm.replace(' ', '_')}_{name.split()[1]}", sel.result(), out)
