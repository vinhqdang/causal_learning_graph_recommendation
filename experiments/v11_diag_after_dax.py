"""(1) DIAGNOSTIC (test labels; never a selection input): share of the LEAF-LGN -> item-oracle gap closed by the DAX
item appeal (split-0 configuration of results/v11_probes/probe_pbm.json "LEAF-LGN + DAX full grid"), and the
Spearman of DAX's item appeal with the uniform click rate.
(2) PROBE (validation-selected): DAX on a backbone without lifelong channels, the closed-form policy-channel model
EASE(click + w shown) + beta * popularity (row M2 of paper/v6), to check that DAX's gain does not depend on LEAF.
python3 experiments/v11_diag_after_dax.py     output results/v11_probes/diag_after_dax.txt
"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v11_common import OUTDIR, Ctx, Out, bundle, dense_OY, fmt, pair, stored  # noqa: E402
from v11_diag_gap import AllUsers  # noqa: E402
from v11_probe_items import GAMMAS, PerSplit, item_covariates, user_strata  # noqa: E402
from v11_probe_pbm import fit_dax, lifelong_rate  # noqa: E402
from drup.leaf import collective_gram, collective_solve, compact_base  # noqa: E402

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")


def main():
    out = Out(os.path.join(OUTDIR, "diag_after_dax.txt"))
    B = bundle()
    C = Ctx(B)
    m, n = B["shape"]
    cov = item_covariates(B)
    strata = user_strata(B, m)
    life = lifelong_rate(B)
    ST = stored()
    a, _, _ = fit_dax(B, cov, strata, prior=80.0, centre="tag", k_life=100.0, tab=True, life=life)
    Z = C.rowstd(C.item(a))
    tl = B["test_log"]
    N_i = np.bincount(tl.i, minlength=n).astype(float)
    C_i = np.bincount(tl.i, weights=tl.is_click, minlength=n)
    q = (C_i + 10 * C_i.sum() / N_i.sum()) / (N_i + 10)
    ok = N_i >= 50
    pop = torch.load(os.path.join(SC, "pop.pt"))
    out("DIAGNOSTIC (test labels)")
    out(f"  Spearman with the uniform rate (items with >= 50 random impressions): DAX {stats.spearmanr(a.numpy()[ok], q[ok])[0]:.3f}, "
        f"pre-cut clicks (popularity) {stats.spearmanr(pop.numpy()[ok], q[ok])[0]:.3f}")
    A = AllUsers(C)
    leaf = C.rowstd(torch.load(os.path.join(SC, "lgn_s0.pt")))
    v_leaf = A(leaf)
    v_dax = A([l + 0.5 * z for l, z in zip(leaf, Z)])
    import pandas as pd
    own = tl.groupby(["u", "i"]).is_click.agg(["sum", "count"])
    mu = C_i.sum() / N_i.sum()

    def loo(rows, cand):
        idx = pd.MultiIndex.from_arrays([np.repeat(rows.numpy(), cand.shape[1]), cand.numpy().ravel()])
        o = own.reindex(idx).fillna(0.0)
        c = cand.numpy().ravel()
        r = (C_i[c] - o["sum"].values + 10 * mu) / (N_i[c] - o["count"].values + 10)
        return torch.as_tensor(r.reshape(cand.shape), dtype=torch.float32)
    orc = C.table(loo)
    v_orc = A(orc)
    out(f"  all users: LEAF-LGN {v_leaf.mean():.4f}, LEAF-LGN + 0.5 z(DAX) {v_dax.mean():.4f}, item oracle {v_orc.mean():.4f}; "
        f"share of the oracle gap closed by DAX {(v_dax - v_leaf).mean() / (v_orc - v_leaf).mean():.2f}")
    zo = C.rowstd([torch.logit(o.clamp(1e-4, 1 - 1e-4)) for o in orc])
    best = max((A([l + 0.5 * z + g * o for l, z, o in zip(leaf, Z, zo)]).mean(), g) for g in (0.5, 1.0, 1.5, 2.0, 3.0))
    out(f"  LEAF-LGN + DAX + best oracle term: {best[0]:.4f} (weight {best[1]}); remaining item headroom above LEAF+DAX "
        f"{1000 * (best[0] - v_dax.mean()):+.2f}e-3")

    out("\nPROBE: DAX on the policy-channel closed-form backbone (no lifelong channels), validation-selected")
    O, Y = dense_OY(B)
    bases = {}
    for w in (0.5, 1.0):
        G, Cc = collective_gram(O, Y, w)
        for lam in (3000.0, 10000.0):
            Bm, _ = collective_solve(G, Cc, n, lam)
            bases[(w, lam)] = [compact_base(O, Y, w, Bm, bk["rows"], bk["cand"]) for bk in C.layout.buckets]
            del Bm
        del G, Cc
    zp = C.rowstd(C.item(pop))
    for name, gammas in (("EASE-policy + pop", (0.0,)), ("EASE-policy + pop + DAX", GAMMAS)):
        sel = PerSplit(C)
        for (w, lam), bs in bases.items():
            for beta in (0.0, 0.25, 0.5):
                for g in gammas:
                    S = [b + beta * p + g * z for b, p, z in zip(bs, zp, Z)]
                    sel.add([S] * 5, None, dict(w=w, lam=lam, beta=beta, gamma=g))
        rec = sel.record()
        out(f"  {name:28s} test {rec['test']:.4f} chosen {rec['chosen'][0]}")
        for ref in ("QP family", "LEAF-LGN", "r-AdjNorm"):
            out(f"      - {ref:10s} {fmt(pair(rec, ST[ref]))}")
        json.dump(rec, open(os.path.join(OUTDIR, f"dax_on_ease_{'base' if len(gammas) == 1 else 'dax'}.json"), "w"))


if __name__ == "__main__":
    main()
