# DRUP: Doubly Robust, Walk-Unbiased Graph Propagation for Recommendation

DRUP is a training-free graph recommender. It propagates preferences over the
counterfactual full-exposure user-item graph instead of the exposure-biased
logged graph. Its properties are proved in `docs/THEORY.md`:

| property | statement | check |
|---|---|---|
| Unbiasedness | Edge-wise doubly robust; exactly unbiased for multi-hop propagation on the full-exposure graph, for nuisances that do not depend on the log (Thm 1) | `experiments/mc_unbiasedness.py`, `experiments/mc_protocol.py` |
| Bias lower bound | Inverse-propensity or DR adjacencies without the walk correction carry an item bias that is Ω(1/τ) (Thm 2) | same |
| Variance / concentration | Efron–Stein variance bound, clipping-bias bound and McDiarmid ranking bound (Thm 3) | – |
| Fairness (causal, item side) | Exposure elasticity 0: E[score \| do(exposure)] does not depend on exposure (Thm 4) | `experiments/mc_elasticity.py`, `run_fat.py --sections intervene` |
| Transparency | Each score is affine in the user's own interactions: exact Shapley attributions and optimal minimal counterfactual explanations (Prop 5, Thm 6) | `run_fat.py --sections explain` |
| Accountability | Certified bound on the effect of any F injected fake profiles (Thm 7) | `run_fat.py --sections attack` |
| Privacy | One public item operator plus local scoring gives (ε,δ)-joint DP (Thm 8) | `run_fat.py --sections privacy` |

## Main results (protocol v2)

All nuisances are cross-fitted over ten folds of pairs (except DRUP-split), every method (trained or not) is
tuned per split on the same validation data, and differences are tested per
user with a Holm correction (`experiments/significance.py`).

| | Coat nDCG@5 | Yahoo!R3 nDCG@5 | KuaiRec nDCG@20 |
|---|---|---|---|
| best trained model | r-AdjNorm (BPR) 0.575 | r-AdjNorm (BPR) 0.671 | LightGCN (pointwise) 0.628 |
| linear LightGCN, logged graph | 0.557 | 0.659 | 0.604 |
| GF-CF, logged graph / DR graph | 0.551 / 0.560 | 0.657 / 0.631 | 0.614 / 0.637 |
| DR adjacency (no walk correction) | 0.546 | 0.657 | 0.638 |
| DRUP | 0.547 | 0.657 | 0.638 |
| DRUP-split (sample split, Assumption 2 holds) | 0.558 | 0.593 | 0.634 |

- The walk correction does not change top-K accuracy (DRUP vs DR adjacency:
  n.s. on all three datasets; identical on Yahoo!R3, where validation selects
  IPS edges). Training-free propagation over the DR graph is the most accurate
  method on KuaiRec (+0.010 over the best trained model, p < 1e-3); graph models
  trained with BPR are better on Yahoo!R3; nothing is significant on Coat after
  Holm correction.
- The guarantees need nuisances that do not depend on the log. In simulation
  (`experiments/mc_protocol.py`) DRUP is unbiased to MC error with nuisances
  from an independent log, but with nuisances cross-fitted over pairs of the
  same log it is not less biased than the uncorrected operator. DRUP-split
  fits the nuisances on a random 20% of the pairs and imputes those pairs; it
  satisfies the assumptions exactly.
- Monte Carlo: uncorrected IPS/DR propagation is biased by 2.3/1.8 times the
  target at 3 hops and 15/8 at 5 hops; corrected estimators stay within MC
  error, also with fixed-size slates (DR) and misspecified propensities
  (exact imputation).
- Exposure invariance under a real do(p <- p/2): DRUP eta = -0.012 (KuaiRec) and
  -0.008 (Coat), also with re-fitted nuisances; logged graph +1.00. On Yahoo!R3
  the selected clip binds for 82% of pairs and DRUP inherits eta = +0.96; small
  clips reduce it to about +0.2 at a cost of 0.044 nDCG@5.
- Certificates (frozen nuisances): 100% of Coat and KuaiRec users certified
  against 1-2 fake profiles, 93-95% against 5; no violation, also with
  re-fitted nuisances.
- Joint DP with public nuisances (analytic Gaussian): KuaiRec nDCG@20 0.596 at
  epsilon = 16 (non-private 0.637, popularity 0.577).

## Layout

```
drup/propagation.py   edge estimators, walk-corrected 3-hop operator, public operator form
drup/khop.py          exact walk correction for any odd number of hops
drup/rerank.py        exposure-capped allocation (exact min-cost flow, dual certificate)
drup/estimation.py    propensity model, additive and low-rank outcome imputation
drup/fat.py           explanations, counterfactuals, robustness certificates, DP release, fairness metrics
drup/pipeline.py      build a propagation recommender from a configuration
drup/learned.py       trained baselines (MF, IPS-MF, DR-MF, LightGCN, NAVIP, DR-LightGCN)
drup/data.py          Coat, Yahoo!R3 and KuaiRec (MNAR training log, unbiased test)
experiments/          all experiments (run_all.sh runs one dataset end to end)
results/              JSON outputs
docs/THEORY.md        method, theorems and proofs
```

## Data

Put the raw data under `data/raw/`:

- Coat: `https://www.cs.cornell.edu/~schnabts/mnar/coat.zip`. Unzip to `data/raw/coat/`. Its test set is a random (MAR) sample.
- Yahoo! R3: `datasets/yahooR3/{user,random}.txt` as distributed with the
  AutoDebias code (`https://github.com/DongHande/AutoDebias`). Place them in
  `data/raw/yahooR3/`. The official source is Yahoo! Webscope (R3).
- KuaiRec 2.0: `https://zenodo.org/records/18164998`. Unzip to `data/raw/KuaiRec 2.0/`. The big matrix is used as the MNAR log and the fully observed small matrix as the test set. Build the cache with
  `python3 -c "from drup.data import load_kuairec; import torch; torch.save(load_kuairec(), 'data/raw/kuairec.pt')"`.

## Running

```
pip install torch numpy scipy pandas
python3 experiments/mc_unbiasedness.py
python3 experiments/mc_elasticity.py
experiments/run_all.sh coat given
experiments/run_all.sh kuairec pop
```
