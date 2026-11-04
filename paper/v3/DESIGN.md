# v3 method design: leave-edge-out doubly robust graph priors

Status: design document, 2026-11-04. Nothing in `drup/` or `experiments/` has been changed. The two
small feasibility probes in Section 6 were run from a scratch directory. Their numbers are
indicative only: one split seed, a small grid, and in the Coat probe one initialisation.

Working name of the method: **DRGP** (doubly robust graph prior). The full model is
**DRGP + residual learner**, written DRGP-iALS or DRGP-LGN depending on the backbone.

---

## 1. Diagnosis

### 1.1 Why QP-LightGCN reads as "not novel"

1. **Every part already exists in some form.** Weighting sampled negatives by a false-negative
   probability appears in CW (Yin et al. 2026), ICPNS, PDNS and IPS-BPR on LightGCN (Raja and
   Vats). An additive offset from a fixed imputation is a standard GLM offset. The backbone is
   unchanged LightGCN with BPR. Taken together, the contribution reads as "two reweighting
   tricks on a stock model".
2. **Our own ablations show the named contribution (Q) does very little.**
   - Coat: BPR + offset with no Q weights scores 0.6020. The full method scores 0.6015.
   - Coat: Q built from the imputation alone (0.5937) is as good as the full Q (0.5912 to 0.5932).
   - In the paper's title idea, the part that carries the result is the imputation offset. That
     offset is the least novel part.
3. **Q is misspecified for these datasets.** On all four datasets the exposure O is observed:
   - Coat: 6,960 rated pairs, of which 5,055 are explicit negatives.
   - KuaiRec: 10.3 M exposures, of which 9.5 M are explicit negatives.
   - Yahoo! R3: 311,704 exposures, with 186,627 explicit negatives.
   - KuaiRand: 511,863 exposures, with 270,507 explicit negatives.

   Q = Yhat(1-p)/(1-p Yhat) is the posterior P(Y=1 | not a logged positive), which marginalises
   over O. When O is known, the correct posterior is 0 on exposed negatives and Yhat on
   unexposed pairs. A careful reviewer would ask why a quantity that is known gets marginalised
   away. This is also a plausible reason Q helps so little.
4. **The method has no structural graph component.** Nothing in the method interacts with the
   propagation operator. The earlier DRUP work did (walk-corrected DR adjacency), and its
   training-free operator is still our best KuaiRec result. QP-LightGCN dropped the one
   ingredient with a technical identity behind it.

### 1.2 Where we stand, using the stored numbers

Seeds 1 to 3 are the honest reference because they are free of seed-0 selection luck.

| Dataset (metric) | Imputation alone | Best training-free | Strongest trained competitor | QP-LightGCN (seeds 0-3) | Gap |
|---|---|---|---|---|---|
| Coat (nDCG@5) | 0.553 | BSPM 0.573; DRUP 0.547 | iALS 0.5850 (0.5862 on seeds 1-3) | 0.5937 (0.5911 on seeds 1-3) | +0.003 on seeds 1-3, p = 0.75 |
| Yahoo! R3 (nDCG@5) | 0.517 | Obs filter 0.6585; DRUP 0.6567 | r-AdjNorm 0.6738 | 0.6877 | +0.015, p < 1e-4 |
| KuaiRand (nDCG@10) | 0.364 | GF-CF 0.4502; DRUP 0.4492 | r-AdjNorm 0.4582 | 0.4606 | +0.002, p = 2e-4 |
| KuaiRec (nDCG@20) | 0.619 | **DRUP 0.6378**, DR 0.6378 | DR-JL 0.6307, MRDR 0.6310 | 0.6289 | -0.009 to DRUP; -0.002 to MRDR |

Stored per-split references used for the pilot gates in Section 4:
- Coat iALS (seed 0, 10 splits): 0.5828, 0.5896, 0.5651, 0.5945, 0.5868, 0.5641, 0.5924,
  0.5740, 0.5841, 0.5783. Mean 0.5812. The selected configuration is always alpha = 30 (the
  top of the grid), reg = 100, with 1 to 11 sweeps.
- Coat final method (seed 0): 0.6017, 0.6074, 0.5781, 0.5965, 0.5853, 0.6185, 0.6153, 0.5769,
  0.6260, 0.6091.
- KuaiRec DRUP (`ensemble_kuairec.json`, DRUP-repro, 5 splits): 0.6369, 0.6350, 0.6396,
  0.6372, 0.6402. Mean 0.6378.
- KuaiRec DR-JL: 0.6324, 0.6286, 0.6367, 0.6340, 0.6341. MRDR: 0.6311, 0.6306, 0.6347, 0.6319,
  0.6357.

### 1.3 What the evidence says works, and the headroom

- **L1. A debiased prior drives Coat and KuaiRec.**
  - Adding the additive-imputation offset is worth about +0.03 on Coat (LightGCN 0.568 to 0.602)
    and +0.02 on KuaiRec (0.6085 to 0.6285).
  - Where the imputation is weak, the offset is useless or harmful. Imputation alone scores
    0.364 on KuaiRand and 0.517 on Yahoo, and beta = 0 was chosen in 4/5 KuaiRand splits.
  - So prior quality is the lever, and the additive mu + b_u + b_i prior has no personalisation
    at all.
- **L2. On KuaiRec, DR propagation beats every trained model.**
  - The training-free DR propagation (0.638) beats DR-JL/MRDR (0.633) and LightGCN-pt (0.627).
  - Pointwise losses beat BPR there (LightGCN-pt 0.627 vs LightGCN 0.6085), because the test
    ranks about 3,300 candidates per user and needs calibrated item quality, which BPR discards.
- **L3. A per-split z-blend cannot combine the two sources.** The blend picks w = 0 on Coat
  (9/10 splits) and w = 1 on KuaiRec (4/5). The trained BPR model adds nothing to DRUP on
  KuaiRec, and DRUP adds nothing to the trained model on Coat. The only way left to combine them
  is to build the graph prior into the learner, and that prior has to be a valid, unbiased
  quantity.
- **L4. DRUP's mixing weight is chosen on the small uniform validation set.** It is not learned
  from the 10 M logged exposures. DRUP picked beta = 10 to 1000 on KuaiRec (3-hop dominant) and
  0.03 on Yahoo/KuaiRand (1-hop dominant). That spread suggests a per-user-group filter could
  help if it could be fitted from the log.
- **L5. Statistical power on Coat is the binding constraint.**
  - With about 10 test candidates per user and 290 users, the seed-0 difference of +0.020 to
    iALS gave Holm p = 0.047.
  - The seeds 1-3 difference of +0.003 gave p = 0.75.
  - A Coat claim against iALS needs a stable mean gain of about +0.015 to +0.02. Otherwise Coat
    has to be reported as "on par" with a power statement.

**Headroom summary.**
- KuaiRec: personalisation on top of the DR propagation. DRUP is about 0.019 above
  imputation-only, and nothing has improved on DRUP yet.
- Coat: a better prior inside the strongest learner (iALS), not inside LightGCN.
- Yahoo and KuaiRand: already won. The new method only needs not to lose there, so the prior
  must be able to switch itself off.

---

## 2. Three candidate methods

Notation:
- O_ui is exposure and Y_ui the outcome. p is a cross-fitted propensity, clipped at a floor.
- Yhat is a cross-fitted IPS-fitted imputation, so Yhat_ui does not use (O_ui, Y_ui).
- W = Yhat + O (Y - Yhat)/p is the DR edge estimate.
- C_ui = d_u^-a d_i^-(1-a) are degree weights computed from Yhat (deg = 'Yhat'), so C does not
  use (O_ui, Y_ui).
- Wt = C ∘ W, and T = corrected 3-hop as in `drup/propagation.three_hop`.

### Candidate A (recommended): leave-edge-out DR graph prior, fitted by DR risk, used as the imputation and offset of a residual learner

**Core idea.** A debiased graph filter can be made *leave-edge-out* (LEO) exactly and at no
extra cost. Its score at (u,i) is then independent of that pair's own exposure and outcome. That
single property unlocks three things that are impossible with ordinary graph filters, or with
the plain imputation offset we used before:

- (i) The filter's coefficients can be fitted by an *unbiased DR risk on the biased log*, with
  no uniform data. Today only the validation set tunes DRUP's beta.
- (ii) The fitted filter is a *valid imputation model*, with no own-outcome leakage. It can
  serve as the imputation inside a DR objective for a second, trained model.
- (iii) The same quantity is used as that model's offset. The learned part then fits only what
  the propagation misses.

**LEO identity.** In the corrected 3-hop, (u,i) appears only in walks with j = i or v = u.
Define:

    sq = C ∘ C ∘ W,   A_ui = Σ_j sq_uj + Σ_v sq_vi - 2 sq_ui

A_ui does not contain W_ui. T_ui is affine in W_ui:

    T_ui = R_ui + (C_ui A_ui + C_ui^3) W_ui

Here R_ui does not depend on W_ui. The LEO 3-hop is T with W_ui replaced by Yhat_ui (the value
the edge would take if the pair were unexposed):

    T^LEO = T - (W - Yhat) ∘ (C ∘ A + C^3)          (exact; O(mn) after T)

I checked this numerically against brute-force replacement on a random 7 x 9 instance: the
maximum error was 1.3e-15. The same formula with W = O∘Y, Yhat = 0 gives the LEO 3-hop of the
observed-positive graph (linear LightGCN). For binary A the walk correction is not needed. The
1-hop LEO term is C ∘ Yhat.

**Stage A: DR-fitted graph prior (closed form, no training).**

Features (dense m x n, each globally standardised):
- phi_1 = logit Yhat_add (the old prior)
- phi_2, phi_3 = T^LEO_DR at a in {0.5, 0.7}
- phi_4 = T^LEO_obs at a = 0.5
- optionally phi_5 = logit Yhat_lr (cross-fitted low-rank imputation)

Every feature is interacted with B user-activity buckets (quantiles of log |O_u|; B in
{1, 2, 4}).

Prior: pi_ui = a_u + Σ_k θ_{k,b(u)} phi_k(u,i). The coefficients minimise the DR squared risk
with user fixed effects:

    min_{θ, a}  Σ_{u,i} [ (s_ui - Yhat_ui)^2 + O_ui/p_ui ((s_ui - Y_ui)^2 - (s_ui - Yhat_ui)^2) ] + ρ ||θ||^2

Because the DR squared loss equals (s - W)^2 up to a constant per pair, this is ridge regression
of the row-demeaned W on the row-demeaned features (Frisch-Waugh). It needs one K x K Gram matrix
with K ≤ 20 and costs O(mnK^2). Two properties matter:
- Demeaning within each user focuses the fit on within-user ordering, which is what nDCG
  measures.
- Given cross-fitted nuisances and the pairwise independence assumption of the DRUP analysis,
  phi(u,i) is independent of (O_ui, Y_ui). So the DR risk of every fixed θ is unbiased for the
  full-exposure risk, and the fitted θ is consistent.

Without LEO, the 1-hop and backtracking walks contain Y_ui/p_ui. The fit would then reward
copying the training label, and the risk estimate would be optimistic.

**Stage B: residual learner anchored at the prior.**

Score: s_ui = β z(pi)_ui + f_ψ(u,i).

Two backbones and two losses, all with the same prior:

- **B1, iALS with offset (DRGP-iALS).**
  - Objective: Σ c_ui (y_ui - β z_ui - x_u^T y_i)^2 + reg, with iALS confidences.
  - It stays closed form: the right-hand side gains -β (Z Y)_u - αβ Σ_{i∈pos} z_ui y_i. One dense
    m x n x d product per half-step costs 5 GFLOP on KuaiRec.
  - This is the learner that wins on Coat.
- **B2, LightGCN with a self-imputing DR loss (DRGP-LGN, loss `drres`).**
  - Uniform-all-pairs part plus exposed-pairs part, with the prior as the imputation:

        L(ψ) = (1/mn) Σ_{u,i} [ ℓ(s_ui, π̃_ui) + O_ui/p_ui (ℓ(s_ui, Y_ui) - ℓ(s_ui, π̃_ui)) ],
        π̃ = clip(pi, 0.01, 0.99)

    where ℓ is BCE on logits.
  - Gradient: ∂L/∂s = (1 - O/p)(σ(s) - π̃) + O/p (σ(s) - Y).
  - At initialisation (f ≈ 0, logit-offset β z ≈ logit π̃) the imputed term vanishes. Learning
    is then driven only by the IPS-weighted residuals of the prior on logged pairs, and the
    unlogged pairs shrink the model back toward the prior. This is the "boosting" view: the DR
    loss with a LEO prior as imputation is a principled residual objective.
  - It makes pointwise DR training (which wins on KuaiRec: DR-JL 0.633) start from DRUP-level
    scores rather than from scratch.
  - Optional hybrid loss: L_BPR + η L_drres with η in {0, 1}. η = 0 is the old offset recipe
    with the new prior, which keeps Yahoo and KuaiRand safe.

**Theory to state in the paper.**
- (P1) The LEO identity.
- (P2) Unbiasedness of the DR risk for any function of LEO features, given correct p or correct
  Yhat. It is double robustness at the level of the filter.
- (P3) Bias of the residual objective: E[L] - L_true = Σ (p - p̂)/p̂ · (ℓ(s,Y) - ℓ(s,π̃)) terms.
  This is the product of the propensity error and the *prior's* error. Since the prior is itself
  DR-fitted, it is the best available imputation in the DR-risk sense.
- (P4) The fixed-point and boosting interpretation of the gradient above.

**Closest prior work and the exact difference.**
- *DR-JL / MRDR / TDR-CL / relaxed-imputation DR (ICML 2024).* These learn the imputation
  jointly or with a correction term. The imputation is a parametric MF trained on observed
  errors, and it is not part of the prediction score. In DRGP, the imputation is a fixed graph
  filter whose coefficients are fitted by DR risk, made admissible by an exact leave-edge-out
  construction, and also used as the prediction offset. None of these works uses a graph
  operator, and none needs or derives LEO, because their imputation is a model, not a
  propagation of the outcomes.
- *DRUP (our unpublished earlier operator).* Training-free, with beta chosen on validation. DRGP
  fits the filter on the log by DR risk, which needs the new LEO identity, and DRGP is used as
  an imputation and offset.
- *GF-CF / BSPM / Turbo-CF / ChebyCF / SGFCF.* Graph filters with fixed or hand-tuned
  coefficients and no debiasing. No graph-filter paper fits coefficients by an unbiased risk
  under MNAR exposure. EASE's zero-diagonal constraint is an item-level leave-one-out for 2-hop
  item models. LEO here is an edge-level leave-one-out for a debiased 3-hop operator with walk
  correction.
- *NAVIP / AdvDrop / DPAA / PPD.* These change aggregation, edges or embeddings with propensity or
  popularity signals. None uses an imputation, a DR risk or a prior.
- *QP-LightGCN (v2).* The offset was the fixed additive imputation, used only as an offset. Here
  the offset is a personalised graph filter, fitted by unbiased risk and used as the imputation
  of the training loss. Q weighting is dropped.
- *DR-learner (Kennedy 2020, causal inference).* It regresses DR pseudo-outcomes on covariates
  with cross-fitting. Stage A is that idea applied to propagation features, where cross-fitting
  alone is not enough because the features are built from the outcomes. Cite it as the
  statistical template.

A further literature check is needed before writing (Section 5.3): "doubly robust" together with
"graph filter" or "leave-one-out propagation", and "DR boosting" in recommendation. My quick web
searches on 2026-11-04 found TDR-CL and the ICML 2024 relaxed-imputation DR paper as the nearest
work. Neither uses a graph prior.

**Why it should beat the strongest competitor on each dataset.**
- *Coat vs iALS.* iALS is the best learner on Coat, so give it a better prior. In the probe
  (Section 6), iALS + DRGP offset scored 0.5937 (0.599 at fixed β = 1) against iALS 0.5845 and
  iALS + additive offset 0.5861. The gain survives a matched grid. The fairness baseline
  iALS + additive prior is part of the design.
- *KuaiRec vs DRUP.* The prior contains DRUP's terms. The epoch-0 checkpoint of every
  residual run is the prior itself, so per-split selection gives ≥ prior on validation by
  construction. The DR-trained residual is the same kind of learner that reaches 0.633 from
  scratch, and here it starts from the prior.
  - Caveat from the probe (Section 6, item 5): the first DR-fitted prior reached only 0.6325.
    That probe used the additive imputation, three features and no buckets. DRUP reaches
    0.6378 with the low-rank imputation and per-user term scaling.
  - Matching DRUP therefore needs the low-rank imputation and DRUP-style per-user scaled
    features. The fallback is to use DRUP's own validation-selected form, made leave-edge-out,
    as the prior.
  - Even with that unrefined prior, a small rank-16 residual added +0.002 (0.6348), which
    supports the residual idea on KuaiRec.
- *Yahoo and KuaiRand.* β = 0 and the BPR path recover the current winning recipe. A DR-fitted
  prior that includes the observed-graph 3-hop (Obs filter 0.6585 Yahoo, GF-CF-like 0.450
  KuaiRand) is a much better prior than the additive imputation (0.517 and 0.364). So the offset
  can now help where it previously hurt.

**Risk.**
1. On Coat the DRGP prior *alone* is not better than the additive prior: 0.559 to 0.563 vs 0.567
   in the probe. The gain appears only as an offset inside iALS. A reviewer may ask why. The
   answer is complementarity: the prior carries propagated co-preference, and iALS supplies
   low-rank taste.
2. The DR squared fit is dominated by unexposed pairs, where W = Yhat. With a weak Yhat (Yahoo,
   KuaiRand), the fit may lean back toward Yhat. Mitigation: within-user demeaning, the obs-graph
   feature, and floors.
3. On KuaiRec the DR-fitted prior can fall *below* the validation-tuned DRUP (probe: 0.6325 vs
   0.6378). The DR squared risk over all pairs is not nDCG@20. The residual may then only
   recover the gap, giving "DRGP + residual ≈ DRUP" with no net gain.
   - Mitigations, in order:
     - (a) imp = 'lr' (cross-fitted) and per-user scaled LEO features, as in DRUP.
     - (b) Select θ's ridge and the floor on validation.
     - (c) Fallback prior: DRUP's selected (1-hop + β·3-hop) form made leave-edge-out, with β
       from validation. The prior is then no longer "fitted without uniform data", so the claim
       narrows to LEO + residual.
4. The residual through sparse uniform-weight SVT of O(Y - π)/p **failed** on Coat (0.5505 vs
   prior 0.5632). It is not part of the design.

**Cost.**
- Stage A: cross-fitting the nuisances on KuaiRec takes about 9 minutes for the propensity plus
  the imputation folds (measured, 4 cores). The full m x n LEO 3-hop takes about 10 s per
  feature: the m x m Gram took 4.6 s and the product 3.5 s on 4 cores. The fit takes seconds.
  Cache the prior per (dataset, floor, lam) to disk.
- Stage B: iALS cost plus one dense matmul per half-step. LightGCN cost is as for existing jobs.

### Candidate B: self-consistent DR propagation (DR label propagation, an EM fixed point)

**Core idea.** Treat the unexposed outcomes as missing data and iterate two steps:
- E-step: impute them from the current graph filter.
- M-step: rebuild the DR adjacency with that imputation and refilter.

Fixed point:

    Yhat^(t+1) = clip( F_θ( Yhat^(t) + O (Y - Yhat^(t))/p ) ),    with F_θ the LEO filter of Candidate A,

using damping Yhat^(t+1) ← (1-η) Yhat^(t) + η · (above).

- If F is linear and the iteration contracts (the 1/p amplification is controlled by the floor
  and η), the fixed point solves a linear system. It is the DR analogue of label propagation.
- The imputation becomes personalised and graph-consistent. The DRUP theory says a more accurate
  imputation lowers the variance of every DR edge estimate.

**Closest work and difference.**
- *DR-JL.* Alternates imputation and prediction with SGD on MF.
- *Missing-data EM for CF (Marlin and Zemel 2009; Hernández-Lobato et al. 2014).* Uses a
  generative MNAR model.
- Candidate B is training-free and uses the DR adjacency, not a likelihood. No paper iterates a
  debiased graph operator to its own imputation fixed point.

**Expected gains and risk.**
- Best case on KuaiRec: +0.002 to +0.005 over DRUP, because better personalisation of W on the
  87% unexposed pairs.
- Coat: unlikely to reach iALS on its own, since all training-free filters are ≤ 0.573 there.
- Risks:
  - Divergence or over-smoothing.
  - Leakage: after one iteration, Yhat at (u,i) depends on (O_ui, Y_ui) through the 3-hop
    terms, unless LEO is applied at every step. It can be, using the same identity.
  - A reviewer may see it as DRUP plus iterations.

**Cost.** Each iteration is one filter pass, about 10 to 30 s on KuaiRec. Ten iterations
times 6 floors comes to a few minutes. Very cheap. **Use it as an optional upgrade of the
Stage A imputation, not as the paper's method.**

### Candidate C: message passing on the doubly robust adjacency (DR-adjacency LightGCN with a low-rank-plus-sparse operator)

**Core idea.** Train LightGCN on the counterfactual full-exposure graph estimated doubly
robustly, rather than on the logged graph or the IPS-weighted logged graph (NAVIP).

The DR adjacency W = Yhat + O(Y - Yhat)/p is dense, but with the additive imputation it
decomposes:

    Yhat = μ 1 1^T + b_u 1^T + 1 b_i^T          (rank 3)
    S    = O(Y - Yhat)/p                        (sparse)

So the propagation W E = (low-rank factors)(1^T E ...) + S E costs O(nnz·d + (m+n)·d). The
degree weights come from Yhat. A learnable control-variate weight c in W_c = O Y/p + c (Yhat -
O Yhat/p) interpolates between NAVIP (c = 0) and DR (c = 1). It can be learned or tuned, and it
keeps E[W] = Y for every c.

**Closest work and difference.**
- *NAVIP:* 1/p weights on observed edges only. It is IPS, with no imputation and no unexposed
  edges.
- *DR-LightGCN (our baseline):* a DR loss on the observed graph, not DR propagation.
- *DRUP:* training-free, linear.

Candidate C is the first trained GNN whose message passing is unbiased for the full-exposure
graph. That holds per edge; for products of edges it needs the walk correction, which cannot be
applied inside a nonlinear trained model. That makes the method less clean.

**Risk.**
- Signed edges: exposed negatives get Ŷ - Ŷ/p < 0. With p = 0.05 an edge reaches -19 Ŷ. The
  signed, high-variance adjacency is likely to destabilise training. NAVIP needed weight
  clipping for the same reason.
- The KuaiRec evidence suggests a trained BPR graph model adds nothing over the training-free
  operator.
- The novelty is incremental over NAVIP + DRUP.

**Cost.** Moderate. A new sparse-plus-low-rank propagation in LightGCN, and full grids on four
datasets. About 1.5 times current LightGCN jobs.

### Comparison

| | Novelty vs listed work | Evidence it addresses | Coat vs iALS | KuaiRec vs DRUP | Risk | Cost |
|---|---|---|---|---|---|---|
| A: DRGP + residual | high (LEO identity, DR-fitted filter, prior-as-imputation) | L1-L5 | probe: +0.009 to +0.015 | open: first prior 0.6325 < DRUP 0.6378 | medium | low |
| B: DR fixed point | medium | L2, L4 | no | small + | medium | very low |
| C: DR-adjacency GNN | medium-low | L2 | unclear | unlikely | high | medium |

---

## 3. Recommended method: DRGP + residual learner (Candidate A)

**Justification.**
- It is the only candidate that (a) has a new exact technical identity (LEO) with a
  statistical consequence (unbiased risk for propagated features, hence learnable filters
  without uniform data), (b) gives a principled role to the prior inside training rather than a
  hand-set offset, and (c) already moved the hardest comparison (Coat vs iALS) in a probe.
- It reuses the DRUP machinery: walk correction, cross-fitting in `drup/pipeline.Nuisance`, and
  degree weights. It also reuses the existing training and selection pipeline. It drops Q, the
  weakest and misspecified part.
- Candidate B can be added later as a one-flag upgrade of the Stage A imputation.

### 3.1 `drup/propagation.py` (add; do not change existing functions)

```python
def loo_three_hop(W, C, base, correct=True):
    """Leave-edge-out corrected 3-hop: T with W_ui replaced by base_ui at (u, i) only.
    T_ui is affine in W_ui: T = R + (C*A + C^3) * W with A = rowsum(sq) + colsum(sq) - 2 sq,
    sq = C^2 W (corrected squares). Requires correct=True (for binary W both coincide)."""
    T = three_hop(W, C, correct=correct)
    sq = C * C * W
    A = sq.sum(1, keepdim=True) + sq.sum(0, keepdim=True) - 2 * sq
    return T - (W - base) * (C * A + C ** 3)
```
- Unit test (add to `experiments/check_khop.py` style, or a new `experiments/check_loo.py`):
  brute force on a random 7 x 9 instance, max abs error < 1e-10. (Already verified once in the
  scratch probe.)
- `gf_features(O, Y, P, Yhat, alphas=(0.5, 0.7), obs=True, Yhat_lr=None)` returns
  `(list_of_dense_tensors, names)`:
  - logit(Yhat clamped to [1e-3, 1 - 1e-3]).
  - `loo_three_hop(W, degree_weights(W, a, D=Yhat), Yhat)` for each a.
  - If `obs`: `loo_three_hop(A, degree_weights(A, 0.5, D=Yhat), zeros)` with A = O * Y.
  - Each feature is globally z-scored. Compute in float32 on KuaiRec/KuaiRand and free W and T
    between features.
- `fit_dr_filter(feats, W, buckets, ridge)`:
  - `buckets` is a LongTensor (m,) of user-bucket ids.
  - Within each user row, demean every feature and W.
  - Accumulate a (B·K) x (B·K) block-diagonal Gram matrix and right-hand side bucket by bucket,
    with float64 accumulators and float32 inputs. Never materialise the m x n x K stack on the
    large data; loop over feature pairs.
  - Solve the ridge system.
  - Return `theta` (B, K) and the prior `pi = rowmean(W) + Σ_k θ_{b(u),k} (phi_k - rowmean(phi_k))`.
- `drgp_prior(d, O, Y, prop, floor, lam, alphas, n_buckets, ridge, K=10, imp="add", loo=True)`:
  - Uses `pipeline.Nuisance(d, O, Y, prop, K=K)`, then `P = clip_propensity(nz.P, floor)` and
    `Yhat = nz.imputation(floor, {"lam": lam, "imp": imp})`, with W as above.
  - Buckets are quantiles of `O.sum(1)`.
  - Returns `pi` (probability scale), `z = standardise(pi)` and `theta`.
  - `loo=False` (ablation) uses plain `three_hop` and the raw 1-hop instead.

### 3.2 `drup/learned.py`

1. `class IALSP(IALS)`: iALS with fixed offset `beta * Z` (Z dense m x n, the standardised prior).
   - In `_solve`, the right-hand side becomes
     `b_u = Σ_{i∈pos}(1+α) y_i − β (Z F)_u − αβ Σ_{i∈pos} z_ui y_i`.
   - Implementation: precompute `ZF = Z @ Fx` (and `Z.T @ Fx` for the item step) once per
     half-step. Subtract `beta * ZF[s:e]` and, through `index_add_` on the positive pairs,
     `alpha * beta * Z[ru, ri][:, None] * f`.
   - `full_scores(rows) = U[rows] @ V.T + beta * Z[rows]`.
   - Constructor: `IALSP(m, n, d, eu, ei, Z, beta, alpha, reg, seed)`. Keep `Z` by reference (no
     copy).
2. `MF.__init__(..., init_std=0.1)`: add the argument and keep the default, so that residual runs
   can start near the prior (`init_std=0.01`).
3. In `train()`:
   - New kwarg `pimp=None`: the probability-scale prior used as imputation.
   - New loss `"drres"` (stratified self-imputing DR):
     - Per batch, draw `b_all = batch` pairs uniformly from all (u, i) and `b_exp = batch`
       pairs uniformly from the exposed pairs `ou, oi = nonzero(O)`. Let N = m·n and
       N_e = |exposed|.
     - Loss:
       `mean_all[ BCE(s, pimp) ] + (N_e / N) * mean_exp[ (1/p) (BCE(s, Y) − BCE(s, pimp)) ]`.
       This is an unbiased estimate of L(ψ)/N.
     - For epoch size, `pairs_per_epoch` counts the all-pairs draws. Default: the number of
       exposed pairs, capped at 2 M on KuaiRec/KuaiRand.
     - The regulariser is as for the `dr` loss.
   - New loss `"bprdr"`: the BPR term of `"bpr"` plus `eta * drres` term on the same batch. The
     new kwarg `eta` defaults to 1.0.
   - Epoch -1 evaluation: when `pimp` is not None, call `on_epoch(model, -1)` once before the
     first epoch with the embeddings zeroed temporarily. A simpler alternative is the separate
     `DRGP` method below. Pick one; the separate method is simpler and is assumed below.

### 3.3 `experiments/run_learned.py`

- New CLI flags:
  - `--prior {add,drgp}` (default `add`, so all old commands are unchanged)
  - `--gf_alphas 0.5 0.7`, `--gf_buckets 4`, `--gf_ridge 1.0`, `--gf_xfit 10`
  - `--gf_obs 1`, `--gf_loo 1`, `--gf_lam` (default = 5 on Coat, 20 elsewhere)
  - `--prior_cache results/v5/cache`
- Globals: keep `PRIOR` and `QMAT`, and add `PIMP = [None]`. With `--prior drgp`:
  - `pi, z, theta = drgp_prior(...)` at the job's `--floor`.
  - Cache `{pi, z, theta, args}` to `prior_cache/drgp_{ds}_{md5(args)}.pt` with the atomic
    `.tmp` rename from `run_ensemble.cache_save`, and reload if present. The KuaiRec cross-fit
    takes about 10+ minutes, and container resets must not repeat it.
  - Set `PRIOR[0] = z.float()` and `PIMP[0] = pi.clamp(0.01, 0.99).float()`.
  - Store `theta` in the output JSON under `args_extra`.
- New `METHODS` entries:

```python
"DRGP": ("prior", None, {}),                                     # prior alone, no training
"iALS-p": ("ialsp", None, {"alpha": [10.0, 30.0, 100.0], "reg": [10.0, 100.0, 300.0], "pbeta": [0.5, 1.0, 2.0]}),
"DRGP-LGN-bpr": ("lgn", "bpr", {"pbeta": [0.5, 1.0, 2.0]}),     # = LightGCN-p with the DRGP prior
"DRGP-LGN-dr": ("lgn", "drres", {"pbeta": [1.0], "init": [0.01]}),
"DRGP-LGN-hyb": ("lgn", "bprdr", {"pbeta": [1.0], "eta": [0.3, 1.0]}),
"iALS-wide": ("ials", None, {"alpha": [10.0, 30.0, 100.0], "reg": [10.0, 100.0, 300.0]}),  # matched iALS grid
```

- The prior used by a job is given by `--prior`. The same `iALS-p` entry run with `--prior add`
  is the fairness baseline "iALS + additive prior", and with `--prior drgp` it is DRGP-iALS.
  Write them to different `--out` files. `final_analysis.combine` then reads `(file, name)`.
- `_build` needs:
  - `kind == "ialsp"`: `IALSP(m, n, d, eu, ei, PRIOR[0], cfg["pbeta"], ...)`. Do not also set
    `model.prior` in `build`; guard with `if kind != "ialsp"`.
  - `kind == "prior"`: return a small object with `full_scores(rows) = PRIOR[0][rows]`.
- `run()` needs:
  - For `kind == "prior"`: call `record(vals, S, 0)` once, with grid keys `["dim"]` and
    dims = [0].
  - For `"ialsp"`: add it to the iALS tuple that dispatches to `train_ials` and to the grid
    branch.
  - Pass `pimp=PIMP[0]` to `train`.
  - `cfg.get("init")` goes to `MF(..., init_std=...)`. Thread it through LightGCN's
    `super().__init__`.
- `seed_check 0` for all new runs; seeds are handled by `run_seeds.py` later.

### 3.4 Grids per dataset, sized to the budget (4 cores, 1 thread per job, up to 4 jobs in parallel)

Nuisance and prior grid. Each value is one cached prior; ridge is fixed at 1.0 after Stage 1.

| Dataset | prop | floors | lam | alphas | buckets | prior cost |
|---|---|---|---|---|---|---|
| Coat | given | 0.05, 0.1, 0.2 | 5 | 0.5, 0.7 | 1, 2 | seconds |
| Yahoo | pop | 0.1, 0.2 | 20 | 0.5, 0.7 | 4 | about 1 minute |
| KuaiRand | pop | 0.02, 0.05 | 20 | 0.5, 0.7 | 4 | about 5 minutes |
| KuaiRec | pop | 0.1, 0.2 | 20 | 0.5 | 4 | about 15 minutes |

Learner grids. lr and wd are as in the v2 queue for the same dataset.

| Dataset | iALS-p | DRGP-LGN-bpr | DRGP-LGN-dr / -hyb |
|---|---|---|---|
| Coat | dims 32 64, α {10,30,100}, reg {10,100,300}, β {0.5,1,2}: 54 cfgs, minutes | lrs 1e-2 3e-3 1e-3, wds 1e-5 1e-4 1e-3, dims 32 64, β {0.5,1,2}: 162 cfgs, under 2 h | lrs 3e-3 1e-3, wds 1e-5 1e-4, dim 64: 4 to 8 cfgs |
| Yahoo | dim 64, α {10,30}, reg {10,100}, β {0.5,1}: 8 cfgs | lr 3e-3, wds 1e-5 1e-4, dims 64 128 (wide), β {0.5,1}: 8 cfgs | lr 3e-3, wd 1e-5: 2 cfgs |
| KuaiRand | dim 64, α {10,30}, reg {10,100}, β {0.5,1}: 8 cfgs | lrs 1e-2 3e-3, wd 1e-4, β {0.5,1}: 4 cfgs | lr 3e-3, wd 1e-4: 1 to 2 cfgs |
| KuaiRec | dim 64, α {10,30}, reg {100,1000}, β {0.5,1,2}: 12 cfgs | lr 3e-3, wds 1e-5 1e-4, β {1,2}: 4 cfgs | lrs 3e-3 1e-3, wd 1e-5, pairs_per_epoch 2 M: 2 to 4 cfgs |

Budget rule: report the number of configurations per method, as v2 did. Give iALS-wide, and
iALS-p under `--prior add`, exactly the same α/reg/dim grid as DRGP-iALS.

### 3.5 Queue (`experiments/queue_v6.sh`, same style as `queue_v5.sh`)

```bash
cd "$(dirname "$0")/.."
C=results/v5/cache
case "$1" in
  coat_ials)   # DRGP-iALS, iALS + additive prior, matched iALS grid
    for pr in drgp add; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.1 --prior $pr --gf_buckets 2 --prior_cache $C --methods iALS-p --out results/v5/learned_coat_${pr}_ials.json || exit 1
    done
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --methods iALS-wide --out results/v5/learned_coat_ialswide.json ;;
  coat_prior)  # prior alone for every floor (cheap)
    for f in 0.05 0.1 0.2; do for b in 1 2; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --seed_check 0 --floor $f --prior drgp --gf_buckets $b --prior_cache $C --methods DRGP --out results/v5/learned_coat_drgp_f${f}_b${b}.json || exit 1
    done; done ;;
  coat_lgn)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --floor 0.1 --prior drgp --gf_buckets 2 --prior_cache $C --methods DRGP-LGN-bpr DRGP-LGN-dr DRGP-LGN-hyb --out results/v5/learned_coat_drgp_lgn.json ;;
  kuairec_prior)  # builds and caches the prior (about 15 minutes per floor), then scores it
    for f in 0.1 0.2; do
      OMP_NUM_THREADS=4 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 4 --seed_check 0 --floor $f --prior drgp --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods DRGP --out results/v5/learned_kuairec_drgp_f${f}.json || exit 1
    done ;;
  kuairec_ials)
    for pr in drgp add; do
      OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 2 --dims 64 --epochs 15 --patience 3 --seed_check 0 --floor 0.2 --prior $pr --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods iALS-p --out results/v5/learned_kuairec_${pr}_ials.json || exit 1
    done ;;
  kuairec_lgn)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 1 --epochs 30 --patience 3 --batch 32768 --pairs_per_epoch 2000000 --lrs 3e-3 1e-3 --wds 1e-5 --seed_check 0 --floor 0.2 --prior drgp --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods DRGP-LGN-dr DRGP-LGN-bpr --out results/v5/learned_kuairec_drgp_lgn.json ;;
  yahoo_all|kuairand_all)
    ds=${1%_all}; fl=$([ $ds = yahoo ] && echo 0.1 || echo 0.02)
    for m in DRGP iALS-p DRGP-LGN-bpr DRGP-LGN-hyb; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset $ds --prop pop --seeds 5 --threads 1 --epochs 60 --patience 5 --batch $([ $ds = yahoo ] && echo 1024 || echo 8192) --lrs 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --floor $fl --prior drgp --gf_buckets 4 --prior_cache $C --methods $m --out results/v5/learned_${ds}_drgp_${m}.json || exit 1
    done ;;
  *) echo "unknown label: $1" >&2; exit 2 ;;
esac
```
Run labels one per background process, at most 4 at a time. Every run checkpoints after each
configuration (existing `.partial.json` mechanism), and the prior cache makes restarts cheap.

### 3.6 Analysis

- Add a v6 analysis script, modelled on `final_analysis.py`.
  - "Final method" per dataset is the combine-over-jobs of {DRGP, DRGP-iALS, DRGP-LGN-*}.
    It is selected per split on validation, as before.
  - Report the per-backbone rows separately as well, so a reader can see which learner carries
    each dataset.
- Holm families per dataset:
  - Coat: vs iALS, iALS + additive prior, iALS-wide, QP-LightGCN (v2 final), DRUP, BSPM.
  - KuaiRec: vs DRUP, DR-JL, MRDR, iALS + additive prior, v2 final.
  - Yahoo and KuaiRand: vs r-AdjNorm, LightGCN, v2 final, GF-CF.

---

## 4. Staged pilot plan with go/no-go gates

**Stage 0 (half a day): code and identity checks.**
- LEO unit test passes.
- With `--prior add`, `iALS-p` at β = 0 reproduces the stored iALS within 0.002 on Coat split 0.
- On Coat, DRGP with `loo=False` has a markedly larger in-sample DR fit (R^2) than with
  `loo=True`. This is a sanity check that the leakage is real and LEO removes it.

**Stage 1 (1 to 2 days): Coat (10 splits, seed 0) and KuaiRec (splits 0 and 1 first, then all 5).**

| Check | Reference (stored) | GO | NO-GO action |
|---|---|---|---|
| Coat DRGP-iALS mean nDCG@5 | iALS 0.5812 (seed 0); matched iALS-wide (new run) | ≥ iALS-wide + 0.008 and ≥ 7/10 splits above stored iALS per split | if ≥ iALS-wide + 0.004, try buckets/alphas once; otherwise Coat is reported as on par and the paper relies on KuaiRec/Yahoo/KuaiRand |
| Coat DRGP-iALS vs iALS + additive prior (same grid) | probe 0.5937 vs 0.5861 | ≥ +0.005 | if ≤ 0, the graph prior adds nothing over the old prior on Coat: the novelty claim for Coat fails |
| KuaiRec DRGP (prior alone), splits 0-1 | DRUP 0.6369, 0.6350; probe prior 0.6325 (5-split mean) | ≥ 0.635 on both (within 0.002 of DRUP with no validation-tuned beta) | first add imp = 'lr', per-user scaled features and 4 buckets; if still < 0.635, use the fallback LEO-DRUP prior (β from validation) for the residual stage |
| KuaiRec best of DRGP-iALS / DRGP-LGN-dr, splits 0-1 | DRUP 0.6369, 0.6350 | ≥ DRUP + 0.003 on both | if within ±0.002: run Candidate B (fixed-point imputation) as the prior; if still flat, KuaiRec is reported as "matches DRUP" |
| Wall clock | | KuaiRec prior < 30 minutes, each learner job < 3 h | reduce pairs_per_epoch or grid |

**Stage 2 (2 to 3 days): full runs.**
- All four datasets with the grids of Section 3.4 and every baseline row.
- GO to Stage 3 if:
  - Coat: final ≥ iALS-wide + 0.010.
  - KuaiRec: final ≥ DRUP + 0.003 with Holm p < 0.05.
  - Yahoo: final ≥ 0.6866 (v2 final).
  - KuaiRand: final ≥ 0.4613.
  - That is, no dataset may lose against v2.
- Also run the ablations:
  - (a) `--gf_loo 0`.
  - (b) buckets = 1.
  - (c) the obs-graph feature removed.
  - (d) θ fixed to DRUP's form (1-hop + β 3-hop, β from validation) instead of DR-fitted.
  - (e) prior as offset only (BPR) vs as offset + imputation (hyb/dr).
  - (f) iALS + additive prior.
- Ablation (a) is the key one: LEO must matter, otherwise the technical claim is decorative.

**Stage 3 (2 days): seeds 1-3.**
- Use `run_seeds.py` with the per-split selected configurations. iALS-wide, iALS + additive
  prior and DRUP are re-run with the same seeds. DRUP is deterministic, so only the split seed
  matters.
- Report means over seeds 0-3, and per-user Holm tests on seeds 1-3, as in `seeds_summary.txt`.
- GO for writing if the Stage 2 claims hold on seeds 1-3 within half the effect.

**Baselines to add for fairness.**
- iALS + the *same* DRGP prior. This is DRGP-iALS itself, so the residual-learner comparison is
  iALS + additive prior vs DRGP-iALS.
- LightGCN + DRGP offset (BPR).
- iALS with the wider α grid. The stored iALS picked α = 30 at the grid edge.
- DRUP with its stored per-split selection.
- BSPM and GF-CF on Coat/KuaiRand.
- DR-JL and MRDR with the DRGP prior as their imputation initialisation, if time allows. This is
  the strongest "same prior to the DR baselines" control on KuaiRec.

---

## 5. Paper storyline

### 5.1 If it works

- Title idea: "Leave-edge-out doubly robust graph priors for recommendation under exposure bias".
- Story:
  1. Graph filters are the strongest training-free estimators of the full-exposure preference
     graph (DRUP on KuaiRec beats every trained model). Yet they cannot be fitted on biased
     logs, and they cannot be handed to a learner as an imputation, because each score contains
     its own training label.
  2. We give an exact leave-edge-out form of the debiased 3-hop operator, at the cost of the
     operator itself (P1).
  3. This makes the doubly robust risk of propagated features unbiased (P2). That gives (a)
     filters fitted on the log without uniform data, and (b) a valid graph imputation that is
     also the offset of any learner (P3, P4).
  4. Experiments on Coat, Yahoo! R3, KuaiRand-Pure and KuaiRec, with uniform data only for
     validation and testing:
     - DRGP + iALS beats iALS and iALS + additive prior on Coat.
     - DRGP (+ residual) beats DRUP and DR-JL/MRDR on KuaiRec.
     - It matches or improves the v2 results on Yahoo/KuaiRand.
  5. Ablations: LEO on/off, fitted vs validation-tuned filter, prior vs additive imputation,
     offset-only vs self-imputing DR loss.
- Contributions: the LEO identity, the DR-fitted graph prior, and the self-imputing residual
  objective. The empirical claim is backbone-agnostic: the gain appears in iALS and in LightGCN.
- Venue: KBS or ACM TORS, which fit the methodological weight better than Information Sciences
  resubmission. Do not reuse the v2 framing (Q weighting).

### 5.2 If it does not work

- If LEO matters but gains are small (≤ +0.003 everywhere): write a shorter analysis paper.
  - Thesis: "When is propagation enough? Doubly robust graph priors versus trained recommenders
    under exposure bias".
  - Report: (i) the LEO identity and DR-fitted filters as a training-free estimator family that
    matches validation-tuned DRUP without uniform data for tuning; (ii) the negative result that
    trained residuals add little beyond a good debiased prior on KuaiRec; (iii) the Coat power
    analysis.
- If DRGP-iALS fails on Coat but KuaiRec works: centre the paper on the large, densely logged
  datasets (KuaiRec, KuaiRand, Yahoo). Report Coat as underpowered, with a minimum detectable
  effect of about 0.015 to 0.02 nDCG@5.
- If nothing beats v2: submit v2 to KBS with the Q-posterior issue fixed (posterior 0 on exposed
  negatives, Yhat on unexposed) and the novelty statement narrowed. That is a weaker option; I
  would not recommend submitting it unchanged.

### 5.3 To do before writing

- Targeted literature search:
  - "leave-one-out" / "leave-edge-out" graph filter or propagation.
  - "doubly robust" together with graph signal processing / learned spectral filter.
  - "DR-learner" in recommendation.
  - "offset" / "residual" on an imputation in DR recommendation (check TDR-CL, ICML 2024
    relaxed-imputation DR, StableDR, DCE in full text).
- Check whether SGFCF (KDD 2024) or JGCF fit filter coefficients from data, and how.

---

## 6. Feasibility probes run for this design (scratch, not part of the repository)

All probes use the repository loaders, `Nuisance(K=10)` cross-fitting, `split_test` with the
stored seeds and fractions, and per-split selection on validation.

1. **LEO identity.** Brute force vs formula on a random 7 x 9 matrix: max error 1.3e-15.
2. **Timing (KuaiRec size, 4 threads, float32).** 7176 x 10728 Gram: 4.6 s. (m x m)(m x n):
   3.5 s. Cross-fitted propensity (10 folds): 539 s.
3. **Coat, floors {0.05, 0.1, 0.2}, 4 base features, no buckets.**
   - DRGP prior alone: 0.5632 (DRUP 0.5467, Impute 0.5532, BSPM 0.5727).
   - Sparse-SVT residual on O(Y - π)/p: 0.5505, which is worse. That residual was rejected.
4. **Coat, iALS with offset (d = 32, α {10,30,100}, reg {10,100,300}, β {0.5,1,2}, 8 sweeps,
   one initialisation).**

   | Model | nDCG@5 |
   |---|---|
   | iALS (β = 0) | 0.5845 |
   | iALS + additive prior | 0.5861 |
   | **iALS + DRGP prior** | **0.5937** |

   - At fixed β: additive 0.5847 / 0.5921 / 0.5934 vs DRGP 0.5929 / 0.5990 / 0.5986 (β = 0.5 / 1 / 2).
   - The DRGP prior alone was 0.559 at that floor.
   - This is the main reason Candidate A is recommended. It is one seed and needs the Stage 1
     confirmation.
5. **KuaiRec, 5 splits, floors {0.1, 0.2}, additive cross-fitted imputation (lam 20), features
   {logit Yhat, LEO DR 3-hop a = 0.5, LEO obs 3-hop}, no buckets, ridge 1.**
   - Fitted θ (standardised): about (0.000, 0.14, 0.002). The DR fit puts nearly all weight on
     the LEO DR 3-hop.
   - DRGP prior alone: 0.6328 / 0.6319 by floor; 0.6325 selected. That is level with DR-JL
     (0.6332) and below DRUP (0.6378, which uses the low-rank imputation).
   - Prior + rank-16 SVT residual with γ = 0.25: 0.6348 at floor 0.1. Larger ranks and γ hurt
     sharply (down to 0.41). Selected: 0.6348.
   - Run time: about 25 minutes in total, mostly the 10-fold cross-fitting of the propensity
     (539 s) and the imputation (about 150 s).
   - Reading: the plain DR-fitted prior is not yet at DRUP's level. The Stage 1 KuaiRec gate
     and the mitigations in Risk 3 address this before any full run.
