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
