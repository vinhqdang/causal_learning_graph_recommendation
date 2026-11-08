# v7 design: where the remaining headroom lives, methods borrowed from other fields, and the next method

Status: design document, 2026-11-08. Follows the failed Stage 1 gate of LEAF (`results/v10/stage1.txt`:
LEAF-LGN 0.4728 vs QP family 0.4634, +9.63e-3 [1.60] per user, Nadeau-Bengio split-level p_Holm 0.189).
Companion: `paper/v7/LITERATURE.md` (cross-field review, verified references). New files only:
`experiments/v11_common.py`, `v11_leaf_scores.py`, `v11_diag_gap.py`, `v11_probe_items.py`, `v11_probe_pbm.py`,
`v11_diag_after_dax.py`, `v11_probe_dax_ablate.py`, `results/v11_probes/*`, `paper/v7/*`. No existing file was edited.

Number provenance:
- "stored": `results/v10/*.json` (QP family = per split, the job with the best validation value among
  DR-BPR-LGN-1 / -p2 / -p, as in `experiments/final_v10.py`).
- "probe": validation-selected per split (5 user-level splits, frac_val 0.3, `drup.leaf.protocol_split`),
  temporal protocol (every input strictly before 2022-04-22 00:00 UTC+8), test read only for the selected
  configuration; per-user paired difference x1e-3 [95% half-width], two-sided t p, Nadeau-Bengio (NB) p over
  split means (`experiments/significance.py::nadeau_bengio`), and the five split differences.
- "diagnostic": uses random-exposure test labels; never a method, never a selection input.
- LEAF scores in the probes are **exact reproductions** of the stored selections: `v11_leaf_scores.py` re-ran
  the chosen configuration of every split; all ten split values (LEAF-LGN and LEAF-tied) match the stored ones to
  5 decimals (`results/v11_probes/leaf_scores.txt`).

---

## 0. Verdict (read this first)

1. **Where the headroom is.** On the temporal all-user setting the leave-one-user-out item oracle reaches
   **0.5068** (all 22,661 evaluated users; 0.4889 was the stored 10K-user setting), LEAF-LGN 0.4735: a gap of
   **+33.3e-3**; LEAF + oracle reaches 0.5208 (+47.3e-3). The gap is (a) **item-level and concentrated in the
   most appealing items** (the top quintile of uniform click rate holds 83% of the recoverable gain), (b) **not
   temporal drift** (the post-cut standard log - forbidden, used only as a diagnostic - predicts the uniform rate
   *worse* than the pre-cut log, Spearman 0.405 vs 0.431; log growth 0.010; recency weighting is never selected;
   the time-local oracle is lower than the whole-period oracle), (c) **spread evenly over users** (every user
   quintile by activity, lifelong history or list length holds 13-26% of it), and (d) **partly explained by
   item content**: a cross-fitted model of the residual on pre-cut item factors would recover +24e-3 (44% of the
   residual gain) *if it had the right target*; tag alone +9.3e-3, upload type +1.8e-3, author out-of-pool
   exposure +3.0e-3. The bottleneck is the target: every item statistic of the log measures response *to the
   policy's choice of audience and scenario*, not to a uniform insertion in tab 1.
2. **Borrowed methods.** The cross-field review (`LITERATURE.md`, 30+ verified references, 10 fields) maps the
   problem onto population estimation from a non-representative sample (MRP, Fay-Herriot), detection vs presence
   (occupancy models; examination vs attractiveness in click models), demand forecasting, two-way fixed effects,
   DML, IV, IRM, test-time adaptation. 12 ideas are ranked; 5 were probed (Section 3).
3. **One idea is strongly complementary: DAX (detection-adjusted attractiveness).** A click-model / occupancy
   factorisation of the pre-cut log, P(click | u, i, tab) = theta_u kappa_tab alpha_i, fitted by EM with a
   tag-centred Beta prior on alpha_i and a lifelong-history prior on theta_u, gives an item appeal alpha_i that
   adds **+9.43e-3 [1.29] to LEAF-LGN** (0.4821) and **+12.9e-3 to the closed-form policy-channel backbone without
   any lifelong signal** (0.4508 -> 0.4637). LEAF-LGN + DAX vs QP family: **+19.05e-3 [1.75] per user, p = 2e-100,
   but NB p = 0.064** (split differences +15.8 +18.7 +17.3 +15.1 +26.6). The additive (logistic) version of the same
   idea, MRP, adds only +2.5e-3; trajectory forecasting adds nothing (-0.3e-3), as the diagnosis predicts.
4. **Gate status: nothing passes the gate as written.** The complementary criterion (LEAF + idea beats QP by
   >= 15e-3 *and* NB significant) is met on the per-user margin (+19.05) but not on NB (0.064 > 0.05). The NB test
   with 5 splits needs split differences with mean/sd > 4.4; the QP family's own split-to-split variability (its
   split 4 is 6e-3 below its other splits) puts every comparison against it near that limit. No probe passes the
   stand-alone criterion (>= 10e-3 over QP without LEAF): DAX on the EASE backbone ties QP (+0.18 [2.27]).
5. **Recommendation (Section 5).** Build **DAX** as the paper's method: an item-appeal estimator that turns the
   policy log into a uniform-insertion appeal by separating examination (user, scenario) from attractiveness
   (item, with content-pooled shrinkage), added to any collaborative backbone; main instantiation LEAF-LGN + DAX,
   with the DAX channel trained jointly. Stage B (about 4 h) must show NB significance against QP with the joint
   selection and against QP + DAX (the fair baseline). **Honest probability** of a paper whose primary gate
   passes: **about 30%** (Section 7). If it fails, write the evaluation paper (Section 6, "what to do instead").

---

## 1. Diagnosis of the remaining headroom (diagnostic; `results/v11_probes/diag_gap.txt`, `diag_after_dax.txt`)

Reference: LEAF-LGN, split-0 selection reproduced, scored for all 22,661 evaluated users (0.4735). G_all = gain of
LEAF + a z(LOO oracle) over LEAF at the best a (+47.26e-3); the "full residual" correction S_LEAF + a r(LOO), with
r = z(uniform rate) minus its projection on LEAF's implied item effect, recovers +53.85e-3. Shares are of that.

### 1.1 Levels

| Quantity | nDCG@10 | vs LEAF-LGN |
|---|---|---|
| popularity (pre-cut clicks) | 0.4199 | -53.6e-3 |
| LEAF-LGN | 0.4735 | - |
| LEAF-LGN + DAX (probe weight 0.5) | 0.4818 | +8.3e-3 |
| item oracle from the first 8 random days only (LOO) | 0.4854 | +11.9e-3 |
| time-local oracle, +-1 day / +-3 days (LOO, shrunk to the period rate) | 0.4949 / 0.4998 | +21.4 / +26.3e-3 |
| **item oracle, LOO uniform click rate, whole random period** | **0.5068** | **+33.3e-3** |
| LEAF-LGN + 2 z(oracle) | 0.5208 | +47.3e-3 |
| LEAF-LGN + DAX + 3 z(oracle) | 0.5194 | DAX closes 25% of the oracle gap; 37.6e-3 of item headroom remains above LEAF + DAX |

Reliability of the target: split-half Spearman of the uniform item rate 0.775 (Spearman-Brown ceiling 0.873).
LEAF's implied item effect explains 32% of the variance of the logit uniform rate (Spearman 0.552).

### 1.2 Item-level statistics vs the uniform rate (Spearman, 7,334 items with >= 50 random impressions)

| Statistic | rho | Note |
|---|---|---|
| LEAF-LGN implied item effect | +0.552 | |
| pre-cut clicks / long views / users / impressions | +0.431 / +0.426 / +0.402 / +0.398 | |
| DAX attractiveness alpha_i | +0.419 | lower than clicks, yet the most complementary to LEAF (Section 3) |
| pre-cut clicks, last 3 days | +0.368 | recency does not help |
| user-standardised click residual (click - user rate) | +0.333 | |
| author out-of-pool impressions / CTR / long-view rate (pre-cut lifelong) | +0.326 / +0.272 / +0.276 | |
| pre-cut CTR / tab-1 CTR | +0.261 / +0.200 | |
| **post-cut standard log clicks / impressions / CTR (forbidden; drift diagnostic)** | **+0.405 / +0.363 / +0.204** | **no better than pre-cut** |
| growth (log post-cut minus log pre-cut impressions) | +0.010 | |
| uniform rate, first 8 days vs later days (7,225 items) | +0.729 | close to the split-half reliability: little within-period drift |

### 1.3 Attribution of the residual to factors (cross-fitted bin means of the residual; nDCG gain when added to LEAF)

| Factor (pre-cut unless marked) | R2 of residual | nDCG gain x1e-3 | share of the full residual gain |
|---|---|---|---|
| **tag (first tag, 44 values incl. missing)** | **0.137** | **+9.33** | **0.17** |
| user-standardised click residual (deciles) | 0.019 | +3.26 | 0.06 |
| author out-of-pool impressions (deciles) | 0.041 | +3.00 | 0.06 |
| author out-of-pool long-view rate | 0.037 | +1.77 | 0.03 |
| upload type (14) | 0.055 | +1.75 | 0.03 |
| pre-cut clicks / impressions / CTR / tab-1 CTR / last-3-day clicks | <= 0.014 | +0.02 to +0.85 | <= 0.02 |
| duration, aspect ratio, music type, video type, upload day | <= 0.010 | <= +0.6 | <= 0.01 |
| post-cut clicks / CTR / growth (forbidden; drift) | <= 0.013 | <= +0.94 | <= 0.02 |
| **all pre-cut item factors jointly (GBM, 5-fold cross-fit)** | **0.312** | **+23.96** | **0.44** |
| all pre-cut + post-cut factors (GBM) | 0.328 | +26.12 | 0.49 (drift adds 0.05) |

Exact-oracle correction restricted to groups (share of the full residual gain; groups are not additive):
- by pre-cut exposure quintile: 0.19, 0.18, 0.20, 0.25, 0.28 (low-exposure items are *not* where the gap is);
- by author out-of-pool exposure quartile: 0.22, 0.22, 0.24, **0.43** (well-known authors);
- by duration quartile: 0.27, 0.28, 0.34, 0.19;
- **by uniform-rate quintile: 0.22, 0.09, 0.04, 0.03, 0.83** - the gap is failing to promote the most appealing
  items;
- by day of the random impression: days 0-3 0.12, 4-7 0.12, 8-11 0.30, 12-16 0.51 - proportional to the share of
  candidates in those days (0.14, 0.15, 0.27, 0.44): no extra loss late in the period (no drift penalty).

User groups (additive, share of LEAF + oracle gain / of oracle - LEAF): pre-cut activity quintiles 0.16-0.24 /
0.16-0.26; lifelong-history quintiles 0.18-0.22 / 0.15-0.24; candidate-list quintiles 0.13-0.22 / 0.13-0.23.
Heavy users have lower nDCG everywhere (LEAF 0.400 vs 0.524 for the lightest lifelong quintile), but the headroom
is evenly spread: no user segment is the lever.

### 1.4 What the diagnosis rules in and out

- Out: popularity / trajectory forecasting (drift is not the bottleneck); cold-start of unexposed items (low
  exposure items are not over-represented in the gap); user segmentation or sequential context.
- In: an item-level appeal estimate whose *target* is the uniform-insertion response, with strength borrowed across
  items through content (tag above all). The diagnosis also prices the protocol: with the validation users' random
  impressions as an item-calibration sample (off-protocol), LEAF-LGN gains +36.6e-3 (Section 3, row U1).

---

## 2. Ideas (from `LITERATURE.md`), novelty and probes

Novelty is judged against the closest verified paper; "probe" refers to Section 3.

| # | Idea (field) | Closest prior (verified) | Novelty | Diagnosis says | Probe |
|---|---|---|---|---|---|
| I1 | **DAX: detection x attractiveness item appeal** - P(click) = theta_u kappa_tab alpha_i by EM, alpha_i shrunk to its tag mean, theta_u with a lifelong prior (click models / occupancy) | Wang et al., WSDM 2018 (regression-EM, position-based examination, personal search); MacKenzie et al., Ecology 2002 (occupancy); ExpoMF, arXiv 1510.07025 | **M**: the factorisation is classical; using user- and scenario-level examination to recover *uniform-insertion item appeal* from a policy log, with content-pooled attractiveness, is not found | targets the item residual and the tag factor | **P3 / DAX: +9.43 [1.29] over LEAF-LGN; +19.05 [1.75] over QP, NB p 0.064** |
| I2 | MRP item appeal: multilevel logistic of clicks with user effects, item effects nested in tag / author, optional user-stratum cells, post-stratified (survey sampling) | Park, Gelman, Bafumi 2004; Wang et al. IJF 2015 | M | same target as I1 (additive instead of multiplicative) | P1: +2.48 [0.87] over LEAF-LGN |
| I3 | Fay-Herriot content prior on the appeal (small-area estimation) | Fay and Herriot, JASA 1979 | M | content explains 31% of the residual with the right target | P2: as the prior of DAX, +5.69 (worse than the tag prior +7.59) |
| I4 | MRP cells inside DAX (alpha per user stratum, shrunk to alpha_i) | Gelman and Little 1997 | M | user segments are not the lever | P4: +7.29 (no gain over +7.59) |
| I5 | Trajectory forecast of item appeal (Szabo-Huberman / Bass) | Szabo and Huberman, CACM 2010; PDA, arXiv 2105.06067 | L | drift is not the bottleneck | P5: -0.33 [0.22]; DAX recency weights never selected |
| I6 | Occupancy posterior as the personal lifelong channel (replace LEAF's lift) | MacKenzie et al. 2002; LEAF | M | personal headroom is small vs item | not probed (a LEAF tweak; out of mandate) |
| I7 | Raking weights matching user marginals for item CTR | Deming-Stephan (raking); Bayes-raking, arXiv 1901.02117 | L-M | a higher-variance version of I2 | not probed |
| I8 | Orthogonalised (DML) item effect | Chernozhukov et al. 2018 | L-M | same information as I2 | not probed |
| I9 | Two-way fixed effects with leave-one-out limited-mobility correction | AKM 1999; Kline, Saggio, Solvsten 2020 | L-M | additive item effect already fails (v6 A2) | not probed |
| I10 | Uniform-sample calibration (learning from a small random sample) | AutoDebias, arXiv 2105.04170; KDCRec (SIGIR 2020) | L | off-protocol (post-cut random data) | U1 (off-protocol, diagnostic): +36.62 [1.79] over LEAF-LGN |
| I11 | IRM / group-DRO across tabs or exposure strata | Arjovsky et al. 2019; InvPref KDD 2022; Sagawa et al. ICLR 2020 | L | tab adjustment failed in v6 CTR models; inside DAX the tab factor helps (+1.4) | via DAX tab factor only |
| I12 | Calibrated re-ranking to the lifelong tag mix | Steck, RecSys 2018 | L | nDCG rewards order, not mix | not probed |

Why DAX and not MRP (an observation the paper should explain): both remove the user's general clickiness, but
the multiplicative form discounts a non-click in proportion to how unlikely the user was to examine anything
(theta_u small) and in proportion to the scenario (kappa_tab: tabs 0, 3, 8, 11 have examination factors 0.0-0.2
relative to tab 1), while the logistic additive form treats a non-click by a low-rate user as almost as negative
as one by a high-rate user once the user effect is absorbed. With 26% of training impressions outside tab 1, the
scenario factor alone adds +1.4e-3 inside DAX (P3 +7.59 -> +9.03); the MRP probe had no tab term, so a fair
"MRP + tab" arm is part of Stage B.

---

## 3. Probe results (`results/v11_probes/probe_items.txt`, `probe_pbm.txt`, `diag_after_dax.txt`)

Per-user paired differences x1e-3 [95% half-width], n = 22,612 users paired with the stored rows. "+ X": the
backbone's row-standardised score plus gamma z(X), gamma in {0, 0.05, ..., 1.0} and X's own hyperparameters chosen
on validation per split.

| Row | Model | Test | vs QP family | vs LEAF-LGN | vs r-AdjNorm |
|---|---|---|---|---|---|
| S0 | QP family (stored) | 0.4634 | - | -9.63 [1.60] | - |
| S1 | LEAF-LGN (stored = reproduced) | 0.4728 | +9.63 [1.60], NB p 0.063 | - | +11.10 [1.71], NB p 0.008 |
| C0 | LEAF-LGN + popularity again (control) | 0.4728 | +9.70 [1.60] | +0.08 [0.11] | |
| P1 | LEAF-LGN + MRP (36 configurations) | 0.4752 | +12.11 [1.58], NB p 0.147 | +2.48 [0.87], NB p 0.471 | +13.58 [1.76] |
| P3 | LEAF-LGN + DAX, tag prior | 0.4802 | +17.22 [1.74], NB p 0.109 | +7.59 [1.27], NB p 0.179 | +18.69 [1.93] |
| P2 | LEAF-LGN + DAX, Fay-Herriot content prior | 0.4783 | +15.32 [1.65], NB p 0.118 | +5.69 [1.09] | +16.79 [1.84] |
| P3b | LEAF-LGN + DAX + lifelong theta prior | 0.4809 | +17.90 [1.78], NB p 0.110 | +8.28 [1.35] | +19.38 [1.97] |
| P3c | LEAF-LGN + DAX + tab examination | 0.4817 | +18.66 [1.74], NB p 0.074 | +9.03 [1.28], NB p 0.090 | +20.14 [1.94] |
| P4 | LEAF-LGN + DAX + MRP stratum cells | 0.4799 | +16.92 [1.72], NB p 0.102 | +7.29 [1.23] | +18.40 [1.91] |
| **P3*** | **LEAF-LGN + DAX full grid (32 x 10 configurations)** | **0.4821** | **+19.05 [1.75], p 2e-100, NB p 0.064** (splits +15.8 +18.7 +17.3 +15.1 +26.6) | **+9.43 [1.29], NB p 0.066** | **+20.53 [1.94], NB p < 0.001** |
| P3r | P3* + recency weights (half-life 2, 4, 8 days or none) | 0.4821 | +19.08 [1.80], NB p 0.062 | +9.45 [1.36] | none is selected |
| P3t | LEAF-tied + DAX full grid | 0.4725 | +8.97 [2.46], NB p 0.211 | -0.66 [2.27] | +10.44 [2.56] |
| P5 | LEAF-LGN + trajectory forecast | 0.4725 | +9.30 [1.59] | -0.33 [0.22] | |
| P1+P3 | LEAF-LGN + MRP + DAX (joint weights) | 0.4804 | +17.43 [1.75] | +7.80 [1.29] | MRP weight 0 in 5/5 splits |
| E0 | EASE(click + w shown) + popularity (no lifelong) | 0.4508 | -12.75 [1.96] | -22.38 [2.07] | -11.27 [2.04] |
| **E1** | **E0 + DAX** | **0.4637** | **+0.18 [2.27], NB p 0.962** | -9.45 [2.36] | +1.66 [2.38] |
| A1 | LEAF-LGN + DAX replaced by its **tag mean** (`probe_dax_ablate.txt`) | 0.4771 | +13.94 [1.72], NB p 0.112 | +4.31 [1.07] | |
| A2 | LEAF-LGN + DAX **permuted within tag** (tag part kept, item part destroyed) | 0.4760 | +12.92 [1.69] | +3.29 [0.99] | |
| A3 | LEAF-LGN + DAX **permuted across all items** | 0.4728 | +9.63 | +0.00 (weight 0 in 5/5 splits) | |
| alone | DAX / MRP / trajectory, item score only | 0.4160 / 0.3874 / 0.3907 | < -46 | | |
| **U1** | **OFF-PROTOCOL** LEAF-LGN + item rate of the *validation users' random impressions* (leave-own-out) | 0.5093 | +46.24 [2.05], NB p 0.003 | +36.62 [1.79] | uses post-cut random data: **not admissible**; prices the protocol |

Reading:
- **DAX is complementary and backbone-independent**: +12.9e-3 on the closed-form backbone without lifelong data
  (E1 - E0), +9.4e-3 on LEAF-LGN (P3*), about -0.7 on LEAF-tied (whose popularity / item channels already move in the
  same direction; LEAF-tied is not the main instantiation).
- **Controls (A1-A3):** the gain is not a tag effect only: the tag mean of alpha gives +4.3, the within-tag
  permutation +3.3, the full item-level alpha +9.4; a global permutation gives 0 (weight 0 selected everywhere).
  Gate B2's item-level condition (drop >= 30% under the within-tag shuffle) already holds at probe scale (-65%).
- **Gate:** LEAF-LGN + DAX meets the complementary per-user margin (+19.05 >= 15) but not NB (0.064). No stand-alone
  probe reaches QP + 10e-3 (E1 ties QP).
- The NB p of LEAF-LGN + DAX vs LEAF-LGN itself is 0.066 despite a per-user interval of [8.1, 10.7]: split 4 of the
  stored LEAF-LGN selection used another configuration (w = 0.25, epoch 22), so its split difference is larger
  (+13.4). Joint selection of the backbone and the DAX weight (Stage B) removes this artefact of post-hoc stacking.
- Rows P1-P5 are the borrowed ideas; only the multiplicative detection model works. The probes chose MAP
  configurations at interior grid points (prior 80 of {40, 80, 160}, k_life 100 of {0, 100}; gamma 0.4-0.5 of
  [0, 1]), so the grid does not cut the optimum.

---

## 4. Novelty (closest work, stated exactly)

- **Click models with examination x attractiveness** (Chapelle and Zhang, WWW 2009; the regression-EM of Wang et al.,
  WSDM 2018): examination depends on *rank position*; the estimand is query-document relevance for web / personal
  search. DAX: examination depends on the *user* (with a lifelong-history prior from 135M out-of-pool impressions)
  and the *scenario* (tab); there is no position; the estimand is the item's response under a *uniform insertion*,
  validated against randomised exposure. A patent (USPTO "Debiasing training data based upon information seeking
  behaviors") conditions examination on position and user group; it does not target item appeal or random
  insertion.
- **Occupancy / detection** (MacKenzie et al., Ecology 2002): the same likelihood structure (presence x detection)
  with repeated surveys; we found no recommendation work that uses it to estimate item appeal from a policy log.
- **ExpoMF** (arXiv 1510.07025): exposure is latent there; in KuaiRand impressions are observed, and DAX's latent
  variable is *examination given an impression*, not exposure.
- **MRP / Fay-Herriot** (survey sampling): our negative result (additive multilevel logistic +2.5e-3 vs
  multiplicative +9.4e-3) is itself informative and belongs in the paper.
- **Popularity debiasing (PDA, TIDE, IPS / QP family)**: they re-weight or remove popularity; DAX *re-estimates* the
  item's appeal under a different audience and scenario mix, and the evidence is on randomised exposure.
- Honest summary: methodologically modest (an EM for a two-factor Bernoulli model with Beta priors). The paper's
  contribution would be (i) the headroom diagnosis (the gap is item appeal under a uniform audience, not drift and not
  cold start), (ii) the identification that a detection-adjusted, content-pooled appeal recovers a quarter of it and
  stacks on lifelong personal channels, (iii) the negative results for MRP, forecasting and the borrowed causal
  estimators on a randomised test. Risk: a reviewer may call DAX "a click model applied to items".

---

## 5. Recommended method: LEAF-LGN + DAX (detection-adjusted attractiveness)

### 5.1 Model

Pre-cut standard log: impressions (u, i, t, y) with tab t and click y, all before T0. Lifelong totals (N_u, C_u)
of each user's out-of-pool impressions before T0 (`data/raw/lifelong_27k.npz`, `tot_pre`).

Generative model of a logged impression:

    E ~ Bernoulli(theta_u * kappa_t)      (the user examines the shown video; tab-specific)
    R ~ Bernoulli(alpha_i)                (the video is attractive to a random viewer)
    y = E * R

Priors (MAP by EM, all closed-form):

    alpha_i ~ Beta(a * c_tag(i), a * (1 - c_tag(i)))           c_tag = pooled attractiveness of the tag (recomputed
                                                                each iteration from the posterior attraction counts)
    theta_u ~ Beta(k * r_u, k * (1 - r_u))                      r_u = (C_u + 1)/(N_u + 2) / mean_u(...) * mean examination
    kappa_t  free, kappa_1 = 1 (the test scenario is tab 1)

E-step for a non-click: P(E=1, R=0) = t(1-a)/(1-ta), P(E=0, R=1) = (1-t)a/(1-ta) with t = theta_u kappa_t,
a = alpha_i (a click sets E = R = 1). M-step: theta_u = (sum of P(E=1) + k r_u)/(sum of kappa_t + k);
kappa_t = sum P(E=1) / sum theta_u over tab t; alpha_i = (sum P(R=1) + a c_tag)/(n_i + a). 40 iterations, 2 s on 4 cores.

Score of a uniform insertion (u, i): s_ui = rowstd_u(LEAF-LGN_ui) + gamma * rowstd_u(logit alpha_i).
Joint variant (main): logit alpha_i enters LEAF-LGN as a seventh fixed channel with a learned weight, trained with
the BPR objective of `drup.leaf.fit_bpr_policy`; post-hoc variant: gamma on validation (as probed).

Hyperparameters (validation): a in {40, 80, 160}; k in {0, 100}; tab factor on/off; prior centre tag or content
(Fay-Herriot ridge); gamma in {0, 0.05, ..., 1.0}; plus LEAF-LGN's lr, wd, w and epoch (joint selection).

### 5.2 Why it should work
The uniform-insertion target is the item's click rate over all users in tab 1. A logged non-click is weak
evidence against an item when the viewer rarely clicks anything (low theta_u) or the scenario rarely yields clicks
(low kappa_t), and the policy places items in such contexts unevenly; DAX removes exactly that. Items have a median
of 29 logged impressions, so attractiveness is pooled within tags, the one content factor that explains a large
share of the residual (Section 1.3). Neither part is personal, so it adds to LEAF's personal lifelong channels
instead of competing with them.

---

## 6. Implementation spec (for a junior engineer)

New files only; do not edit existing ones.

1. **`drup/dax.py`**
   - `fit_dax(log, n_users, n_items, tag_codes, life_rate=None, prior=80.0, k_life=100.0, tab=True,
     centre="tag", content_codes=None, iters=40) -> (logit_alpha (n,), theta (m,), kappa (T,))` - a clean copy of
     `experiments/v11_probe_pbm.py::fit_dax` (float64 tensors, index_add_ accumulations; peak memory < 1 GB).
   - `content_centre(alpha, n_i, content_codes, ridge=10.0)` - Fay-Herriot prior mean (copy of the probe).
   - `class DAXChannel` - holds logit alpha; `pairs(u, i)` and `compact(rows, cand)` with the interface of
     `drup.leaf.LifelongTables`, standardised by the std of logit alpha over pool items.
   - `class LEAFLGNDAX(LEAFLGN)` - adds one parameter `g_dax` (init 0) and `+ g_dax * dax.pairs(u, i)` in `forward`
     and `full_scores`.
   - Unit tests `tests/test_dax.py`: (a) with alpha fixed and no priors, the M-step of theta on simulated data
     recovers the generating theta within 0.02 (10K users x 1K items, 50 impressions per user); (b) `fit_dax` on the
     KuaiRand bundle with prior 80, k 100, tab on reproduces `results/v11_probes/probe_pbm.json` split-0 choice
     (LEAF-LGN + DAX full grid, test of split 0 = 0.48xx to 1e-4); (c) no random-log row is read (assert the log's
     max time < T0).
2. **`experiments/run_dax.py`** (CLI in the style of `experiments/run_leaf.py`, same output format and resumable
   partial file): `--backbone leaf_lgn|ease|lightgcn|qp_posthoc`, `--mode joint|posthoc`, grids above, `--seeds 5`.
   - `leaf_lgn --mode joint`: the LEAF-LGN grid of `queue_v10.sh leaf_lgn` (lr {1e-2, 3e-3}, wd {1e-4, 1e-5}, w {0,
     0.25, 0.5}, epochs 40, patience 3) x DAX grid (prior {40, 80, 160} x k {0, 100}, tab on), the DAX table fitted
     once per DAX configuration (2 s) - 72 trainings of about 6 min: run with `--dax_grid small` (prior 80, k 100,
     tab on) first (12 trainings, 1.2 h).
   - `leaf_lgn --mode posthoc`: LEAF-LGN trained alone; at every epoch, every DAX configuration and gamma is evaluated
     on validation (selection over epoch x DAX x gamma jointly, which removes the split-4 artefact of Section 3).
   - `qp_posthoc`: re-train each QP-family configuration that any split selected (the 5 `chosen` entries of
     `results/v10/learned_kuairand_t_qp.json`, via `experiments/learned_ctx.train_cfg` as in
     `experiments/v9_probes/probe_lgn_leaf.py`), save epoch scores, and add DAX post-hoc with the same grid: **QP + DAX
     is the fair baseline**.
   - `ease`: E0 / E1 of Section 3 with the full EASE grid (lam {1000, 3000, 10000}, w {0, 0.5, 1}, beta {0, 0.25,
     0.5}).
   - Ablations (`--ablate`): `no_tab`, `no_life_prior`, `no_tag_prior` (prior centre = global mean), `additive`
     (MRP with tab effects instead of DAX: the fair MRP arm), `shuffle_items` (alpha permuted within tag: must lose
     most of the gain), `shuffle_tags` (alpha permuted across items: must lose all of it).
3. **`experiments/v11_stage_b.py`**: the analysis of `experiments/final_v10.py` with the families below (copy its
   helpers; output `results/v11/stageB.txt`).

Queue (`experiments/queue_v11.sh`, same slot mechanism as `experiments/queue_v10.sh`: two flock slots; a LEAF-LGN
or QP training takes one slot, about 5-6 GB; `qp_posthoc` takes both slots because it keeps epoch scores of all
splits, about 8 GB):

```
cd "$(dirname "$0")/.."
mkdir -p results/v11 results/v11/.slots
# slot() and slot_excl() copied from experiments/queue_v10.sh with results/v11/.slots
case "$1" in
  dax_ease)        slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone ease --seeds 5 --out results/v11/dax_ease.json "${@:2}" ;;
  dax_lgn_post)    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --seeds 5 --out results/v11/dax_lgn_post.json "${@:2}" ;;
  dax_lgn_joint_s) slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode joint --dax_grid small --seeds 5 --out results/v11/dax_lgn_joint_small.json "${@:2}" ;;
  dax_lgn_joint)   slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode joint --dax_grid full --seeds 5 --out results/v11/dax_lgn_joint.json "${@:2}" ;;
  dax_qp)          slot_excl env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --seeds 5 --out results/v11/dax_qp.json "${@:2}" ;;
  dax_lgn_gcn)     slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone lightgcn --mode posthoc --seeds 5 --out results/v11/dax_lightgcn.json "${@:2}" ;;
  dax_ablate)      slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --ablate no_tab no_life_prior no_tag_prior additive shuffle_items shuffle_tags --seeds 5 --out results/v11/dax_ablate.json "${@:2}" ;;
  stageB)          python3 experiments/v11_stage_b.py > results/v11/stageB.txt ;;
  tests)           python3 -m pytest -q tests/test_dax.py ;;
  *) echo "usage: $0 {dax_ease|dax_lgn_post|dax_lgn_joint_s|dax_lgn_joint|dax_qp|dax_lgn_gcn|dax_ablate|stageB|tests}" >&2; exit 2 ;;
esac
```

Order: tests -> dax_ease (10 min) -> dax_lgn_post and dax_qp in parallel (about 2 h) -> stageB (gate B1) ->
dax_lgn_joint_s -> dax_ablate, dax_lgn_gcn -> dax_lgn_joint only if joint_s beats post-hoc on validation.
Probe intermediates (bundle, LEAF scores) live in the scratch directory; set `V11_CACHE` / `V11_SCORES`; each
probe script re-creates them when missing, and `results/v11_probes/*.json` are written after every block, so a
container restart loses at most one block.

---

## 7. Staged gates and probability of success

Primary family (Holm over t p and over NB p): {LEAF-LGN+DAX - QP family, LEAF-LGN+DAX - (QP+DAX),
LEAF-LGN+DAX - LEAF-LGN, LEAF-LGN+DAX - r-AdjNorm, LEAF-LGN+DAX - LightGCN}.

- **Stage A (this document): done.** Complementary per-user margin met (+19.05 >= 15); NB not met (0.064).
- **Gate B1 (post-hoc, joint selection; about 3 h):** LEAF-LGN+DAX vs QP family >= +15e-3 per user with
  p_Holm < 0.05 **and** NB p_Holm < 0.05. Also report vs QP+DAX (expected about +6 to +9e-3, the LEAF margin).
  - GO -> Stage C. NO-GO -> "what to do instead" below.
- **Gate B2 (ablations):** shuffle_tags removes >= 90% of the DAX gain; shuffle_items within tag keeps the tag
  part only (the gain must drop by >= 30%, else the item-level estimate adds nothing over a tag effect, and the
  method must be reframed as a scenario-and-user-adjusted tag prior); `additive` (MRP with tab) stays >= 4e-3
  below DAX, else the multiplicative claim is dropped.
- **Stage C (robustness):** KuaiRand-1K (842 users; underpowered, direction only), the concurrent protocol
  (expected larger), long-view label, seeds 1-3 of the trained backbones.

Probability (honest):
- B1 passes: about 40% (per-user margin is +4e-3 above the gate; NB p 0.064 must fall below 0.05 after Holm over
  five comparisons, i.e. the raw NB p must reach about 0.01-0.02; the joint selection removes the split-4 artefact,
  but QP's own split 4 dip stays).
- B2 holds: about 70%.
- Accepted novelty at a good venue given B1 and B2: about 45%.
- Overall: **about 15% for a method paper on the strict path; about 30% for a publishable paper** if the empirical
  contribution (diagnosis + the DAX vs MRP vs forecasting comparison on randomised exposure) carries it.

**What to do instead if B1 fails (recommended plan B, in this order):**
1. *Evaluation / analysis paper* with what is already solid: the headroom decomposition (Section 1: item-level, top
   items, not drift, not cold start, content-explainable with the right target), the protocol results (temporal vs
   concurrent, LEAF's lifelong transfer +15e-3 over LightGCN with NB p 0.010), and the cross-field negative results
   (MRP, forecasting, IV/IRM/TTA arguments). The NB test with 5 splits is the binding constraint: state the
   power analysis (to pass NB at alpha 0.05 with J = 5 and test fraction 0.7, the split differences need
   mean/sd > 4.4).
2. *A setting with a small randomised calibration sample* (I10; the AutoDebias / KDCRec line): U1 shows +36.6e-3
   over LEAF-LGN when the validation users' random impressions calibrate item appeal. A method paper there is
   realistic if the protocol is re-defined *before* running (calibration users disjoint from test users, calibration
   impressions strictly earlier than every test impression, e.g. days 0-7 calibrate, days 8-16 test), and the
   contribution is how to transfer a small calibration sample through DAX's content-pooled attractiveness.
3. Do not spend more compute on in-pool item statistics or forecasting: every variant probed in v6 and v11 stays
   within a few thousandths of popularity, and drift is not where the headroom is.
