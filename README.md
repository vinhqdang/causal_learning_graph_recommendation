# DRUP: Doubly Robust, Walk-Unbiased Graph Propagation for Recommendation

DRUP is a training-free graph recommender. It propagates preferences over the
counterfactual full-exposure user-item graph instead of the exposure-biased
logged graph. Its properties are proved in `docs/THEORY.md`:

| property | statement | check |
|---|---|---|
| Unbiasedness | Edge-wise doubly robust; exactly unbiased for multi-hop propagation on the full-exposure graph (Thm 1) | `experiments/mc_unbiasedness.py` |
| Bias lower bound | Inverse-propensity or DR adjacencies without the walk correction carry an item bias that is Ω(1/τ) (Thm 2) | same |
| Variance / concentration | Efron–Stein variance bound, clipping-bias bound and McDiarmid ranking bound (Thm 3) | – |
| Fairness (causal, item side) | Exposure elasticity 0: E[score \| do(exposure)] does not depend on exposure (Thm 4) | `experiments/mc_elasticity.py`, `run_fat.py --sections intervene` |
| Transparency | Each score is affine in the user's own interactions: exact Shapley attributions and optimal minimal counterfactual explanations (Prop 5, Thm 6) | `run_fat.py --sections explain` |
| Accountability | Certified bound on the effect of any F injected fake profiles (Thm 7) | `run_fat.py --sections attack` |
| Privacy | One public item operator plus local scoring gives (ε,δ)-joint DP (Thm 8) | `run_fat.py --sections privacy` |

## Main results

Accuracy on unbiased test data (mean over splits; on KuaiRec the trained
baselines are tuned and paired with DRUP on the same 5 splits):

| | Coat nDCG@5 | Yahoo!R3 nDCG@5 | KuaiRec nDCG@20 |
|---|---|---|---|
| MF / IPS-MF / DR-MF | 0.462 / 0.457 / 0.536 | 0.507 / 0.512 / 0.509 | 0.632 / 0.629 / 0.619 |
| LightGCN / NAVIP / DR-LightGCN | 0.508 / 0.499 / 0.556 | 0.588 / 0.591 / 0.561 | 0.630 / 0.630 / 0.625 |
| imputation only (additive) | 0.566 | 0.524 | 0.622 |
| linear LightGCN on the logged graph | 0.557 | 0.659 | 0.604 |
| DR adjacency (no walk correction) | 0.559 | 0.662 | 0.632 |
| DRUP, additive imputation | 0.553 | **0.662** (λ = 0) | 0.633 |
| DRUP, low-rank imputation, 5 hops | 0.555 (additive) | – | **0.640** |

DRUP (no gradient training) significantly beats every tuned trained
baseline on KuaiRec and Yahoo!R3 (paired t-test, p < 0.01). On Coat it ties
with the best trained model but is below the additive imputation alone
(`paper/tables/significance.tex`). On Yahoo!R3 (2% dense), validation picks the
IPS end (λ = 0) of the control-variate family.

Guarantees and their checks:

- **Unbiasedness, any number of hops.** The correction is exact for any odd
  K (Möbius inversion over walk-index coincidence patterns, `drup/khop.py`).
  Uncorrected IPS/DR propagation has relative bias 2.4 / 1.9 at 3 hops and
  78 / 70 at 5 hops. The corrected estimators stay within Monte-Carlo error.
- **Causal item fairness.** Under a real do(p ← p/2) intervention with frozen
  nuisances, the mean-score elasticity of the logged graph is +1.00 on every
  dataset. DRUP's is −0.02 (Coat) and −0.01 (KuaiRec). On Yahoo!R3 it is +0.95
  at the validated clip τ = 0.2 and +0.006 at τ ≤ 5e-4, at an nDCG cost of
  0.04 (`paper/tables/tau.tex`). In Monte-Carlo, exposure elasticity is +0.005 (logged graph +1.78,
  DR adjacency −0.19). Under a real exposure intervention on KuaiRec, DRUP's rank
  shift is +0.0005 ± 0.0008, against −0.134 for the logged graph. The
  exposure-conditional bias drops from 0.83 to 0.42, and to 0.29 with low-rank
  imputation.
- **Exposure caps.** An exact min-cost-flow re-ranker enforces per-item
  exposure caps (certified gap < 5e-6). Under every cap on KuaiRec, DRUP-LR is
  the most accurate operator (e.g. 0.260 vs 0.228 for the logged graph at
  c = 10).
- **Transparency.** Explanations are exact (error 1e-15). Minimal
  counterfactual explanations exist for 18% (Coat) and 55% (KuaiRec) of top
  recommendations.
- **Accountability.** No attack with up to 100 fake users placed the target in
  any KuaiRec top-20. On Coat, 100% of users are certified against one fake
  profile.
- **Privacy.** Joint DP: nDCG@20 is 0.613 at ε = 8 and 0.629 at ε = 16 on
  KuaiRec (non-private 0.633).

Limitations:

- At equal Gini, logged-graph propagation keeps slightly more accuracy under
  re-ranking.
- Certificates are vacuous for KuaiRec-sized catalogs.
- A re-fitted low-rank imputation leaves a small exposure sensitivity
  (+0.008), because it violates the fixed-nuisance assumption.
- The number of correction terms grows with the Bell numbers (2,790 at 7 hops).

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
