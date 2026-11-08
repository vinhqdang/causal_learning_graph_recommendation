"""v11 probe: controls for DAX on LEAF-LGN (paper/v7/DESIGN.md Section 3 / gate B2), validation-selected as in
v11_probe_pbm.py with the best DAX configuration (prior 80, tag centre, k_life 100, tab factor):
  tag_only        alpha replaced by its tag mean (is DAX only a tag effect?)
  shuffle_within  alpha permuted across items of the same tag (keeps the tag part, destroys the item part)
  shuffle_all     alpha permuted across all items (must lose the whole gain)
  no_tab_no_life  DAX with neither the tab factor nor the lifelong theta prior
python3 experiments/v11_probe_dax_ablate.py     output results/v11_probes/probe_dax_ablate.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v11_common import OUTDIR, Ctx, Out, bundle, fmt, pair, stored  # noqa: E402
from v11_probe_items import GAMMAS, PerSplit, item_covariates, user_strata  # noqa: E402
from v11_probe_pbm import fit_dax, lifelong_rate  # noqa: E402

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")


def main():
    out = Out(os.path.join(OUTDIR, "probe_dax_ablate.txt"))
    B = bundle()
    C = Ctx(B)
    m, n = B["shape"]
    cov = item_covariates(B)
    strata = user_strata(B, m)
    life = lifelong_rate(B)
    ST = stored()
    base = [C.rowstd(torch.load(os.path.join(SC, f"lgn_s{s}.pt"))) for s in range(5)]
    a, _, _ = fit_dax(B, cov, strata, prior=80.0, centre="tag", k_life=100.0, tab=True, life=life)
    a0, _, _ = fit_dax(B, cov, strata, prior=80.0, centre="tag", k_life=0.0, tab=False, life=life)
    tag = cov["tag"].numpy()
    an = a.numpy()
    rng = np.random.default_rng(0)
    tag_mean = np.zeros(n)
    within = an.copy()
    for t in np.unique(tag):
        idx = np.flatnonzero(tag == t)
        tag_mean[idx] = an[idx].mean()
        within[idx] = an[rng.permutation(idx)]
    arms = {"DAX best (reference)": a, "tag_only": torch.as_tensor(tag_mean, dtype=torch.float32),
            "shuffle_within_tag": torch.as_tensor(within, dtype=torch.float32),
            "shuffle_all": torch.as_tensor(an[rng.permutation(n)], dtype=torch.float32),
            "no_tab_no_life (plain DAX, prior 80)": a0}
    for name, v in arms.items():
        Z = C.rowstd(C.item(v))
        sel = PerSplit(C)
        for g in GAMMAS:
            sel.add(base, [g * z for z in Z] if g else None, {"gamma": g})
        rec = sel.record()
        out(f"  LEAF-LGN + {name:40s} test {rec['test']:.4f} gammas {[c['gamma'] for c in rec['chosen']]}")
        for ref in ("LEAF-LGN", "QP family"):
            out(f"      - {ref:10s} {fmt(pair(rec, ST[ref]))}")


if __name__ == "__main__":
    main()
