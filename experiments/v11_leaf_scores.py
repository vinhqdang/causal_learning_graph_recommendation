"""Reproduce the score matrices of the stored LEAF-LGN and LEAF-tied selections (results/v10), so that v11 probes can
be combined with LEAF ("complementary" probes) and diagnostics can attribute LEAF's remaining gap to the oracle.

For every split the configuration chosen on validation in results/v10/leaf_{lgn,tied}.json is re-run (same code as
experiments/run_leaf.py) and its bucketed candidate scores are saved to $V11_SCORES (default /tmp/v11_scores/).
The test nDCG@10 of each split is compared with the stored value (printed; small float differences are possible for
LEAF-LGN because training is multithreaded).
python3 experiments/v11_leaf_scores.py [lgn] [tied]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v11_common import V10, KEY, Ctx, Out, OUTDIR, bundle, dense_OY  # noqa: E402
from drup.leaf import (CHANNELS, LEAFLGN, LifelongTables, collective_gram, collective_solve, compact_base,  # noqa: E402
                       fit_bpr_policy, make_lightgcn)

SC = os.environ.get("V11_SCORES", "/tmp/v11_scores")
LIFE = os.path.join(os.path.dirname(V10), "..", "data", "raw", "lifelong_27k.npz")


def tables(B, s):
    L = np.load(os.path.abspath(LIFE), allow_pickle=True)
    npz = {k: L[k] for k in L.files if "_pre" in k or k.endswith("_values")}
    return LifelongTables(npz, B["users_raw"], B["item_feat"], window="pre", s=s)


def main():
    what = sys.argv[1:] or ["lgn", "tied"]
    os.makedirs(SC, exist_ok=True)
    out = Out(os.path.join(OUTDIR, "leaf_scores.txt"))
    B = bundle()
    C = Ctx(B)
    O, Y = dense_OY(B)
    X = O * Y
    pop = torch.log1p(X.sum(0))
    pop = (pop - pop.mean()) / pop.std().clamp_min(1e-12)
    torch.save(pop, os.path.join(SC, "pop.pt"))
    if "lgn" in what:
        rec = json.load(open(os.path.join(V10, "leaf_lgn.json")))["results"]["LEAF-LGN"]
        chosen = rec["chosen"]
        tabs = tables(B, 5.0)
        runs = {}
        for s, c in enumerate(chosen):
            runs.setdefault((c["lr"], c["wd"], c["w"]), {}).setdefault(c["epoch"], []).append(s)
        for (lr, wd, w), eps in runs.items():
            t0 = time.time()
            base = make_lightgcn(C.m, C.n, Y, 64, 2)
            model = LEAFLGN(base, tabs, pop, np.ones(6))
            last = max(eps)

            def on_epoch(mod, ep, eps=eps, last=last):
                if ep in eps:
                    emb = mod.embeddings()
                    S = [mod(bk["rows"][:, None], bk["cand"], emb) for bk in C.layout.buckets]
                    sel = C.selector()
                    sel.add(S, {})
                    r = sel.result()
                    for s in eps[ep]:
                        torch.save([x.float() for x in S], os.path.join(SC, f"lgn_s{s}.pt"))
                        out(f"LEAF-LGN split {s} lr {lr} wd {wd} w {w} epoch {ep}: test {r['per_split'][s][KEY]:.5f} "
                            f"(stored {rec['per_split'][s][KEY]:.5f}) gains {mod.gain.data.numpy().round(3)} "
                            f"beta {float(mod.beta):.3f} [{time.time() - t0:.0f}s]")
                return ep >= last
            fit_bpr_policy(model, O, Y, w, lr, wd, last + 1, 8192, on_epoch, seed=0)
            del model, base
    if "tied" in what:
        rec = json.load(open(os.path.join(V10, "leaf_tied.json")))["results"]["LEAF-tied"]
        for s, c in enumerate(rec["chosen"]):
            t0 = time.time()
            G, Cc = collective_gram(O, Y, c["w"])
            Bm, _ = collective_solve(G, Cc, C.n, c["lam"])
            del G, Cc
            base = [compact_base(O, Y, c["w"], Bm, bk["rows"], bk["cand"]) for bk in C.layout.buckets]
            del Bm
            tabs = tables(B, c["s"])
            gw = torch.tensor([float(c.get(ch, 0.0)) for ch in CHANNELS])
            S = [bs + tabs.compact(bk["rows"], bk["cand"]) @ gw + c["beta"] * pop[bk["cand"]]
                 for bs, bk in zip(base, C.layout.buckets)]
            sel = C.selector()
            sel.add(S, {})
            r = sel.result()
            torch.save([x.float() for x in S], os.path.join(SC, f"tied_s{s}.pt"))
            out(f"LEAF-tied split {s} {c}: test {r['per_split'][s][KEY]:.5f} (stored {rec['per_split'][s][KEY]:.5f}) "
                f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
