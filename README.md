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

Accuracy on unbiased test data (mean over splits):

| | Coat nDCG@5 | KuaiRec nDCG@20 |
|---|---|---|
| MF / IPS-MF / DR-MF | 0.462 / 0.457 / 0.536 | 0.625 / 0.623 / 0.610 |
| LightGCN / NAVIP / DR-LightGCN | 0.508 / 0.499 / 0.556 | 0.629 / 0.625 / 0.616 |
| imputation only | 0.566 | 0.622 |
| DR adjacency (no walk correction) | 0.559 | 0.632 |
| **DRUP (training-free)** | 0.553 | **0.633** |

DRUP is accuracy-competitive with trained models, but it is not the most
accurate method on Coat (see `paper/tables/significance.tex`). Its contribution
is the guarantees:

- The corrected estimators are unbiased (relative |bias| 0.02 against 1.9–2.4
  without the correction), and exposure elasticity is +0.005 (logged graph
  +1.78, DR adjacency −0.19).
- Under a real exposure intervention on KuaiRec, DRUP's rank shift is
  +0.0005 ± 0.0008, against −0.134 for the logged graph. The exposure-conditional
  bias drops from 0.83 to 0.42.
- Explanations are exact (error 1e-15). Minimal counterfactual explanations
  exist for 18% (Coat) and 55% (KuaiRec) of top recommendations.
- No attack with up to 100 fake users placed the target in any KuaiRec
  top-20. On Coat, 100% of users are certified against one fake profile.
- Joint DP: nDCG@20 is 0.613 at ε=8 and 0.629 at ε=16 on KuaiRec (non-private 0.633).

Limitations: the item-side guarantee is causal, not distributional (DRUP's
top-K lists are concentrated when quality is). Certificates are vacuous for
KuaiRec-sized catalogs. Nuisances are assumed fixed or cross-fitted.

## Layout

```
drup/propagation.py   edge estimators, walk-corrected 3-hop operator, public operator form
drup/estimation.py    propensity model, outcome imputation
drup/fat.py           explanations, counterfactuals, robustness certificates, DP release, fairness metrics
drup/pipeline.py      build a propagation recommender from a configuration
drup/learned.py       trained baselines (MF, IPS-MF, DR-MF, LightGCN, NAVIP, DR-LightGCN)
drup/data.py          Coat and KuaiRec (MNAR training log, unbiased test)
experiments/          all experiments (run_all.sh runs one dataset end to end)
results/              JSON outputs
docs/THEORY.md        method, theorems and proofs
```

## Data

Put the raw data under `data/raw/`:

- Coat: `https://www.cs.cornell.edu/~schnabts/mnar/coat.zip`. Unzip to `data/raw/coat/`. Its test set is a random (MAR) sample.
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
