# v5 design: search for a new method, probe results, and the remaining option

Status: design document, 2026-11-04. No file outside `paper/v5/` was changed. Probe scripts and raw
outputs are in the scratch directory named in Section 9; they are throw-away and must be re-implemented
in `experiments/` (Section 6) before any number enters a paper.

Number provenance:
- "stored": copied from a result file in `results/` (file named where it appears).
- "probe": computed for this design. The data, splits (`split_test`, frac_val 0.3, seeds 0..S-1),
  metrics (`drup.metrics.evaluate`) and per-split validation selection over (configuration, epoch)
  are the same as in `experiments/run_learned.py`. Test values are read only for the selected
  configuration. Paired differences are per-user means over splits (the `cmp` of
  `final_analysis.py`), x1e-3 with the 95% half-width in brackets, unadjusted p.
- "diagnostic": uses unbiased test labels to measure how much a benchmark can reward a method. It is
  never a method and is never used for selection.

---

## 0. Verdict (read this first)

1. **No candidate passed its probe.** Four new directions were probed on Coat and KuaiRec (and, for the
   most promising one, on Yahoo and KuaiRand): three-state (positive / exposed-negative / unexposed)
   weighted ALS, three-state linear autoencoders, an exposure-conditional residual model, an
   exposure-as-confounder channel, and posterior soft labels for unexposed pairs. None beats
   iALS + exposure-conditional prior on Coat, and none moves KuaiRec at all (validation selects the
   zero-weight personal term in every split).
2. **The reason is structural, and it is measurable.** On KuaiRec, a non-personalised ranker that is
   given the *test* item rates (leave-one-user-out) reaches nDCG@20 0.6396; the log-fitted prior alone
   reaches 0.6369 on the same users and DRUP 0.6378 (stored). Even a low-rank model fitted on half of the
   fully observed unbiased matrix improves on item means by only +3.9e-3 nDCG@20 (diagnostic H2). A gate
   of the form "beat DRUP and DR-JL on KuaiRec by a paired margin" therefore cannot be passed by *any*
   method by more than ~2-4e-3, which is at or below the minimum detectable effect. Coat has the opposite
   problem: personalisation matters, but the MDE is 0.012-0.04.
3. **Exposed negatives are not useful negatives in these logs.** Giving logged-but-disliked items more
   weight as negatives hurts monotonically on Coat (linear probe), Yahoo and KuaiRand (ALS probe; on
   KuaiRand these are true impressions without a click); down-weighting them gains at most +1.46e-3
   [1.48] on Yahoo, and the best three-state ALS stays 8-23e-3 below the stored QP family on
   Yahoo/KuaiRand. This rules out the "three-state learning" family as
   a source of gains on these benchmarks.
4. **Best remaining option (Section 5):** a personalised-posterior EM for unexposed pairs
   ("PEP-ALS"), evaluated with gains required on the two datasets that have both personalisation headroom
   and statistical power (Yahoo R3, KuaiRand-Pure) and non-inferiority required on Coat and KuaiRec. It
   builds on the one ingredient with demonstrated, significant gains on those datasets (posterior
   weighting of unexposed pairs: Yahoo QP family vs LightGCN +16.77e-3 [5.82], KuaiRand +3.93e-3 [1.53],
   stored, `results/v6/tests.txt`). Its novelty over ExpoMF, Bayesian negative sampling and QP-LightGCN is
   **incremental** (Section 5.3). Honest probability that it yields a publishable method with real gains:
   **about 10-15%** (Section 8). If its Stage 0 fails, the recommendation is to stop method work on these
   benchmarks.

---

## 1. Literature findings (2024-2026 focus; every entry checked on arXiv or the publisher page)

All arXiv identifiers below were verified through the arXiv API on 2026-11-04 (title, authors, venue
comment). Venues are given only where the arXiv record or the proceedings state them.

### 1.1 Debiased recommendation on Coat / Yahoo R3 / KuaiRec / KuaiRand

| Paper | Venue (verified) | What it does | Protocol relevant to us |
|---|---|---|---|
| Kweon and Yu, *Doubly Calibrated Estimator for Recommendation on Data MNAR*, arXiv 2403.00817 | WWW 2024 | Calibrates imputation and propensity models inside DR | Coat/Yahoo label = rating > 3; KuaiRec label = watch ratio > 1; selection on 10% of the MNAR training data; reports Coat N@5 0.6623 (DCE-TDR) vs DR-JL 0.6382, Yahoo N@5 0.6901 vs 0.6515, KuaiRec N@50 0.8162 vs 0.7689 |
| Song et al., *CDR: Conservative Doubly Robust Learning*, arXiv 2308.08461 | (venue not verified) | Filters unreliable imputations in DR | Coat, Yahoo, KuaiRand |
| Wang et al., *CounterCLR*, arXiv 2402.05740 | ICDM 2023 | Counterfactual contrastive learning without MAR data | Coat, Yahoo, KuaiRec, nDCG@5 |
| Bai et al., *Invariant debiasing learning via biased imputation (KD-Debias)*, arXiv 2412.20036 | IP&M 62(3), 2025 | Invariant + variant preference with distillation; critique of InvPref | unbiased-data-free |
| Li et al., *Be Aware of the Neighborhood Effect: Selection Bias under Interference*, arXiv 2404.19620 | ICLR 2024 | Treats exposure of neighbours as interference in DR estimation | Coat, Yahoo, KuaiRec |
| Zhang et al., *ConAlign*, arXiv 2607.24092 | (venue not stated) | Conditional alignment of biased/unbiased models; uses KD-Debias as baseline | Coat, Yahoo, nDCG@5 |
| Islam, Zheleva, Wang, *Post-hoc Popularity Bias Correction in GNN-based CF (PPD)*, arXiv 2510.12959 | WWW 2026 | Removes a popularity direction from trained GNN embeddings | KuaiRec @20 |
| Wang et al., *Debias Can be Unreliable*, arXiv 2409.04810 | WWW 2025 | Shows randomly-exposed evaluation can reward non-debiasing tricks (hard negatives at large K); proposes an unbiased recall estimator | KuaiRec only fully exposed set |

**Finding L1 (state of the art under honest protocols).** There is no published number on these four
datasets that is comparable with ours: published work binarises differently (rating > 3 vs >= 4; watch
ratio > 1 vs >= 2), ranks all test items including items already in the log, uses other K, and often
selects on (or trains with) part of the uniform data. Within the DR line the 2024-2026 frontier is
calibration (DCE), conservative imputation (CDR), interference (ICLR 2024) and invariance (KD-Debias);
each reports gains of 0.01-0.03 nDCG over DR-JL/MRDR under its own protocol, i.e. of the same size as the
selection-on-test inflation we measured on Coat (+0.0207 on average, `paper/v4/AMENDMENTS.md`). Under our
protocol, DR-JL/MRDR are far below iALS on Coat/Yahoo/KuaiRand and below a non-personalised prior on
KuaiRec (stored), so the DR frontier is not where gains are.

### 1.2 Strong collaborative filtering

| Paper | Venue | Relevance |
|---|---|---|
| Rendle, Krichene, Zhang, Koren, *Revisiting the Performance of iALS*, arXiv 2110.14037 | RecSys 2022 (proceedings) | Tuned iALS is competitive with or better than recent models; our Coat result (wider grid +12.9e-3, `results/v6/prior_anatomy.txt` X4) repeats this |
| Steck, *EASE*, arXiv 1905.03375 | WWW 2019 | Closed-form linear autoencoder; base of probe P2 |
| Park, Shin, Shin, *Turbo-CF*, arXiv 2404.14243 | SIGIR 2024 | Training-free polynomial graph filters |
| Kim et al., *ChebyCF*, arXiv 2505.00552 | SIGIR 2025 | Chebyshev-interpolated non-linear graph filter on raw histories |
| Wang et al., *DiffRec*, arXiv 2304.04971 | SIGIR 2023 | Diffusion recommender |
| Benigni, Ferrari Dacrema, Jannach, *Diffusion Recommender Models and the Illusion of Progress*, arXiv 2505.09364 | ACM TORS 2026 | Four SIGIR 2023-2024 diffusion recommenders are largely unreproducible and lose to tuned simple baselines |
| Sheng et al., *Language Representations Can be What Recommenders Need (AlphaRec)*, arXiv 2407.05441 | ICLR 2025 | Linear map from frozen LM title embeddings; needs item text |

**Finding L2.** Diffusion models are not a credible route to gains over tuned iALS (TORS 2026
reproducibility study). LLM-embedding recommenders need item text: Coat and Yahoo R3 have none usable;
KuaiRec has captions, but KuaiRec is item-saturated (Section 3). Graph filters (Turbo-CF, ChebyCF) are
training-free CF on the logged graph; our GF-CF/BSPM rows and the DR-graph variants (stored) already show
that the filter family is not the bottleneck under exposure bias.

### 1.3 Exposure, impressions, negative feedback

| Paper | Venue | Closest to |
|---|---|---|
| Pérez Maurera, Ferrari Dacrema, Castells, Cremonesi, *Impression-Aware Recommender Systems*, arXiv 2308.07857 | ACM TORS 2025 | survey; defines impressions vs interactions vs non-impressions |
| Liang, Charlin, McInerney, Blei, *Modeling User Exposure in Recommendation (ExpoMF)*, arXiv 1510.07025 | WWW 2016 | idea (b): exposure as latent variable, EM |
| Saito et al., *Unbiased Recommender Learning from MNAR Implicit Feedback (Rel-MF)*, arXiv 1909.03601 | WSDM 2020 | click = exposure x relevance, IPS |
| Ma et al., *ESMM*, arXiv 1804.07931 | SIGIR 2018 | entire-space impression -> click -> conversion |
| Wang et al., *ESCM2*, arXiv 2204.05125 | SIGIR 2022 | counterfactual regulariser on ESMM |
| Lou et al., *Re-weighting Negative Samples for Model-Agnostic Matching (UMA2)*, arXiv 2207.02468 | (venue not stated) | separates exposed-unclicked and unexposed negatives |
| Liu and Wang, *Bayesian Negative Sampling*, arXiv 2204.06520 | (venue not verified) | posterior probability that a sampled negative is a false negative |
| Chen, Lin, Xu, *In-Community Popularity negative sampling (ICPNS)*, arXiv 2602.18759 | preprint 2026 | exposure estimate for reliable negatives |
| Wang, Liang, Charlin, Blei, *The Deconfounded Recommender*, arXiv 1808.06581 (RecSys 2020 version "Causal Inference for Recommender Systems") | RecSys 2020 | exposure model as substitute confounder in the outcome model |
| Hernández-Lobato, Houlsby, Ghahramani, *PMF with Non-Random Missing Data* | ICML 2014 (PMLR 32) | joint rating + missingness model |
| Marlin and Zemel, *Collaborative Prediction and Ranking with Non-Random Missing Data* | RecSys 2009 | same, mixture models |
| Seo et al., *SiReN*, arXiv 2108.08735 | IEEE TNNLS | signed bipartite graph, GNN on positives + MLP on negatives |
| Chen et al., *SIGformer*, arXiv 2404.11982 | SIGIR 2024 | sign-aware graph transformer |
| Wang et al., *NFARec*, arXiv 2404.06900 | SIGIR 2024 | negative-feedback-aware hypergraph model |
| Zheng et al., *DICE*, arXiv 2006.11011 | WWW 2021 | interest vs conformity embeddings |
| Zhao et al., *Popularity Bias Is Not Always Evil (TIDE)*, arXiv 2109.07946 | (TKDE per listings; not verified) | item quality vs conformity; keeps benign quality effect |

**Finding L3.** Every component of ideas (a) and (b) of the brief has a close precedent: three-state
data and separate treatment of exposed vs unexposed negatives (IARS survey, UMA2), joint exposure /
preference latent models with EM (ExpoMF, Hernández-Lobato 2014, Marlin-Zemel 2009), exposure as a
covariate (Deconfounded Recommender), and signed-graph learning from negative feedback (SiReN,
SIGformer, NFARec). TIDE is the closest published analogue of our "additive item-level prior": it keeps a
static item-quality term and removes conformity. What we did **not** find is any of these evaluated under
a uniform-test, uniform-free-training protocol against tuned iALS and an exposure-conditional item
prior. So novelty could only come from the method's mechanism, not from "using impressions".

### 1.4 Open gaps (as they matter for a method here)

- G1. No paper reports how much of a randomly-exposed benchmark is attainable by a non-personalised
  ranker. Our diagnostic (Section 3) shows KuaiRec is nearly saturated by item effects.
- G2. Posterior handling of unexposed pairs is studied either for exposure (ExpoMF) or for sampled
  negatives (BNS), with fixed or non-personalised posteriors; a personalised posterior iterated with the
  ranker and fitted only on the log, under a uniform-test protocol, was not found.
- G3. Calibration/conformal work in recommendation targets coverage or calibration, not top-K accuracy;
  it is not a route to nDCG gains and was not pursued.

---

## 2. Candidate ideas and scores

Scores: novelty vs the closest paper (H/M/L), expected nDCG gain vs the strongest competitor on each
dataset (from the probes where available), risk, compute cost on 4 CPU cores.

| # | Idea | Closest prior work | Novelty | Expected gain (Coat / Yahoo / KuaiRand / KuaiRec) | Risk | Cost | Probe |
|---|---|---|---|---|---|---|---|
| I1 | Three-state ALS: separate confidence w_n and target tau_n for exposed negatives, plus prior offset (brief idea a) | iALS (Hu et al. 2008; Rendle et al. 2022), IARS survey, UMA2 | L | 0 / +1.5e-3 vs iALS, -23e-3 vs QP / 0 / 0 | high | low | **P1: fails** (Coat, Yahoo, KuaiRand) |
| I2 | Three-state linear autoencoder (signed encodings +1 / -c / 0) added to the prior | EASE, SiReN, SIGformer | L-M | <0 / - / - / 0 | high | low | **P2: fails** |
| I3 | Exposure-conditional residual model (personalised P(Y=1 \| O=1)) on top of the prior | ESMM/ESCM2, Marlin-Zemel | L | <0 / - / - / 0 | high | low | **P2 (resid): fails** |
| I4 | Exposure as a substitute-confounder channel | Deconfounded Recommender | L | <0 / - / - / 0 | high | low | **P3: fails** |
| I5 | Joint exposure-preference EM on the propagation graph (brief idea b) | ExpoMF; in-repo Q weighting (v2) and DR-iALS | L | small / + / + / 0 | medium | medium | P4 (first E-step only): neutral on Coat |
| I6 | Learned propagation filters with a DR risk (brief idea c) | in-repo DRGP (v3); ChebyCF | L | 0 / ? / ? / <0 | high | medium | already failed (DRGP, `results/v5/final_v6_stage1.txt`) |
| I7 | Diffusion / generative CF | DiffRec; TORS 2026 reproducibility study | L | <0 expected | high | high (GPU) | not probed (evidence against) |
| I8 | LLM text embeddings of items | AlphaRec | M | n/a / n/a / ? / ~0 | high | high | not applicable (no text on Coat/Yahoo; KuaiRec saturated) |
| I9 | Invariant preference learning | InvPref (KDD 2022), KD-Debias | L | ? small | high | medium | not probed |
| I10 | Graded feedback (ratings 1-5, watch ratio) in place of the binary label | ordinal MF | L (data, not method) | ? | medium | low | not probed; fairness problem vs baselines |
| I11 | Per-user mixing of item prior and personal score learned on the log | stacking / ensembles; in-repo ensemble | L | small | medium | low | not probed (ensemble with DRUP already chooses DRUP) |
| I12 | Personalised-posterior EM for unexposed pairs (PEP-ALS), evaluated where headroom exists | ExpoMF, BNS, QP-LightGCN | L-M | ~0 / + / + / ~0 | medium-high | low-medium | Section 5; Stage 0 pending |

---

## 3. Headroom diagnostics (explain why three attempts failed)

**H1 (diagnostic, KuaiRec).** Rank every test user's unlogged candidates by the item's mean outcome over
the *other* test users (leave-this-user-out, all 1,411 users, all unbiased entries): nDCG@20 **0.6396**.
The log-fitted exposure-conditional additive prior (no IPS) on the same users: **0.6369**; correlation
between the prior's item mean and the test item rate: **0.972**. Stored DRUP 0.6378, prior alone 0.6367
(5-split test means). So the best training-free methods are within ~2-3e-3 of the best possible
non-personalised ranker.

**H2 (diagnostic, KuaiRec).** Split each test user's unbiased candidates 50/50 (A/B). Fit on A item means
and item means + rank-k SVD of the residual (uniform, unbiased training data), rank B:
item only 0.5481; + rank-4 residual 0.5514 (x0.5), **0.5520** (x1.0), 0.5447 (x2.0); rank 16: 0.5495 /
0.5470 / 0.5343; rank 64: 0.5459 / 0.5341 / 0.5078. Even with unbiased training data on the test users
themselves, personalisation adds at most **+3.9e-3**. A log-trained method can only do worse.

**H3 (diagnostic, Coat).** The same leave-one-user-out item oracle reaches nDCG@5 0.4507 against 0.4755
for the log prior (all candidates; correlation 0.514). On Coat item effects are weak and personalisation
carries the signal, but the paired MDE is 0.012-0.04 (`results/v6/tests.txt`).

**Consequence for gates.** "Beats DRUP and DR-JL/MRDR on KuaiRec by a paired margin" is a test the
benchmark cannot pass by more than about the MDE (2-3e-3). KuaiRec must be a non-inferiority dataset for
any method; gains must be demonstrated where both headroom and power exist: Yahoo R3 (n = 2,310 users,
half-widths 3-6e-3) and KuaiRand-Pure (n = 8,321, half-widths 1-1.5e-3).

---

## 4. Probe results

Reference rows are stored per-user results: Coat iALS (`results/v3/learned_coat_given.json`, 0.5812),
iALS-wide2 (`results/v6/learned_coat_ialswide2.json`, 0.5922), iALS + prior IPS
(`results/v5/learned_coat_add_ials.json`, 0.5984), iALS + prior no-IPS
(`results/v6/learned_coat_add_noips_ials.json`, 0.6044), LightGCN + prior no-IPS
(`results/v6/learned_coat_add_noips_lgn.json`, 0.6052), DR-JL 0.4811; KuaiRec iALS 0.6133, iALS + prior
no-IPS (`results/v6/learned_kuairec_add_noips_ials_small.json`, 0.6350), prior alone no-IPS
(`results/v6/learned_kuairec_alone_add_noips.json`, 0.6367), DR-JL 0.6332, MRDR 0.6328
(`results/v3/learned_kuairec_pop_A.json`), DRUP 0.6378 (`results/v2/filters_kuairec_pop.json`).
Sanity: the probe's "prior alone" reproduces the stored 0.5574 (Coat) and 0.6367 (KuaiRec) exactly
(paired difference 0.00).

Notation: O exposure (logged pair), Y outcome, pi = mu + b_u + b_i the exposure-conditional additive prior
(`baseline_imputation(..., ips=False)`, lam 5), Z its standardised logit (the `add_noips` prior of
`run_learned.py`).

### P1. Three-state ALS (I1)
min sum_ui c_ui (t_ui - beta z_ui - x_u'y_i)^2 + reg(|X|^2+|Y|^2); c = 1+alpha (positive), w_n (exposed
negative), 1 (unexposed); t = 1, tau_n, 0. w_n = 1, tau_n = 0 is iALS + prior exactly.

- **Coat** (dim {32,64} x alpha {30,100,300} x reg {100,300,1000} x beta {0.5,1} x w_n {0.25,1,4} x
  tau_n {0,0.3}; 216 configurations, 1,549 (cfg, epoch) rows): test **0.6041**; vs iALS + prior no-IPS
  **-1.59 [10.31]** p=0.76; vs LightGCN + prior no-IPS -2.49 [14.77]; vs iALS-wide2 +8.57 [19.55];
  vs iALS +21.49 [21.47] p=0.050; vs DR-JL +124.50 [33.25]. Selected w_n over 10 splits: 4.0 in 5,
  1.0 in 3, 0.25 in 2: no consistent direction.
- **Yahoo** (first, interrupted grid; validation only): mean validation nDCG@5 by w_n at
  alpha 10: 0.6662 (0.25), 0.6650 (1), 0.6619 (4), 0.6478 (16); at alpha 3: 0.6182, 0.6187, 0.6162,
  0.6126; with the prior (beta 0.5) every cell is lower (alpha 10: 0.6498 / 0.6479 / 0.6430 / 0.6290),
  consistent with the stored finding that the offset does not help on Yahoo. More weight on exposed
  negatives is monotonically worse. Test numbers on a matched grid: Section 4.5.
- **KuaiRec**: one configuration finished before the run was stopped for CPU (w_n 1, val 0.6357); not
  completed. Not needed for the verdict: P2/P3 show that validation rejects every personal term on KuaiRec,
  and H1/H2 bound the attainable gain.

### P2. Three-state linear autoencoders (I2, I3)
score = Z + gamma * std(X_u B), B = EASE(X, lambda) with zero diagonal; gamma in {0,...,2},
lambda per dataset grid, all chosen on validation.

| Coat encoding X | test | vs iALS + prior no-IPS | vs prior alone |
|---|---|---|---|
| positives O*Y (control) | 0.5922 | -14.86 [16.82] p=0.083 | +37.59 [23.62] |
| exposed negatives as negatives, c = 0.1 | 0.5825 | -24.69 [19.13] p=0.012 | +27.76 |
| c = 0.3 | 0.5626 | -45.74 [23.23] | +6.71 |
| c = 1.0 | 0.5541 | -54.99 [23.92] | -2.54 |
| exposed negatives as weak positives, c = -0.1 | 0.5916 | -14.68 [14.13] p=0.042 | +37.77 |
| c = -0.3 / -0.5 / -1.0 (exposure only) | 0.5902 / 0.5856 / 0.5810 | -15.10 / -19.50 / -24.88 | |
| residual O*(Y - pi) (conditional preference) | 0.5478 | -59.91 [25.22] | -7.46 [16.49] |

KuaiRec: positives, exposure and residual channels, each with lambda in {300, 3000, 30000}: validation
selects gamma = 0 in all 5 splits for every channel, so the test value equals the prior alone (0.6367).

### P3. Three-channel exposure-aware scorer (I4)
Z + g_pos S_pos + g_exp S_exp + g_res S_res (125 weight combinations).
Coat 0.5908 (vs iALS + prior no-IPS -14.06 [13.39] p=0.040); single exposure channel 0.5776 (-28.64).
KuaiRec: validation selects (0, 0, 0) in all splits: 0.6367.

### P4. Posterior soft labels for unexposed pairs (first E-step of I5/I12)
iALS with target kappa * Q_ui on unexposed pairs, Q = pi (1-p) / (1 - p pi), p the logistic popularity
exposure model (log only, floor 0.05); kappa = 0 is iALS on the same grid.
Coat (alpha {30,100,300} x reg {100,300} x kappa {0,0.5,1,2}): **0.5989**; vs the same-grid iALS
(0.5882) **+7.94 [14.42]** p=0.28; vs iALS + prior no-IPS -6.42 [11.49]; vs iALS + prior IPS -0.15.
On Coat the non-personalised posterior acts as another way of injecting the item prior, nothing more.

### 4.5 Yahoo and KuaiRand test numbers (P1, matched grids)

The full Yahoo grid was interrupted by a container restart; it was re-run on a reduced grid that
contains the stored iALS choice (dim 64, reg 100, alpha 10) so that w_n is the only varied factor.

- **Yahoo** (dim 64, alpha {10, 30}, reg 100, beta 0, w_n {0.25, 1, 4, 16}; 5 splits, n = 2,310):
  TS-ALS **0.6646** (w_n = 0.25 selected in all 5 splits) vs the same-grid iALS 0.6629:
  **+1.46 [1.48]** p=0.052. Against stored rows: iALS +1.73 [1.56]; LightGCN -6.38 [7.91];
  r-AdjNorm -10.48 [7.89] p=0.009; LightGCN-p -4.94 [8.15]; **QP family -23.14 [8.70]** p=2e-7.
  Validation by w_n at alpha 30: 0.6398 / 0.6380 / 0.6315 / 0.6129.
- **KuaiRand** (dim 64, alpha 10, reg 100, beta 0, w_n {1, 4, 16}; n = 8,321; KuaiRand exposure is the
  platform's impression log, not self-selection): validation 0.4530 / 0.4512 / 0.4439, so w_n = 1 is
  selected in all splits and TS-ALS equals the same-grid iALS: **0.4533**; vs QP family **-8.45 [3.30]**,
  vs r-AdjNorm -6.51 [3.17]. A partial run at alpha 3 shows the same order (w_n 1: 0.4484, 4: 0.4481).

So up-weighting exposed negatives hurts on all three datasets where it was run to completion
(Coat linear, Yahoo, KuaiRand), including the one whose negatives are true platform impressions.
Down-weighting them (w_n = 0.25) gives at most +1.5e-3 on Yahoo. The gap to the posterior-weighted
LightGCN of v2 (QP family) is 8-23e-3: the lever on these datasets is the treatment of *unexposed* pairs,
not of exposed negatives.

### 4.6 What the probes establish
1. On self-selected logs (Coat, Yahoo) exposed negatives are poor negatives: the user chose to look at
   the item, which is itself evidence of interest. Up-weighting them hurts monotonically (P1 Yahoo, P2
   Coat). This is consistent with the stored finding that the gain of the prior comes from an
   item-level conditional rate, not from treating exposed negatives as negatives per user.
2. On KuaiRec no log-derived personal signal survives validation (P2, P3), and H1/H2 show there is
   almost nothing to gain.
3. On Coat everything that adds personal structure to the prior lands within the noise of iALS + prior.

---

## 5. Recommended (best remaining) option: personalised-posterior EM for unexposed pairs (PEP-ALS)

This is recommended **because nothing else is left with evidence of gains**, not because it passed a
probe. It targets the one lever that produced significant gains on the powered datasets (posterior
weighting of unexposed pairs in QP-LightGCN: Yahoo +16.77e-3 [5.82] and KuaiRand +3.93e-3 [1.53] vs
LightGCN, stored) and replaces its weakest part, the non-personalised posterior.

### 5.1 Model

Data: three states per pair, s_ui in {+ (O=1, Y=1), - (O=1, Y=0), 0 (O=0)}.
Generative model (selection on outcome, as in Marlin-Zemel / Hernández-Lobato, but with an implicit
low-rank preference):

- Preference: Y_ui ~ Bernoulli(r_ui), logit r_ui = f_ui = mu + b_u + b_i + x_u'y_i.
- Exposure: P(O_ui = 1 | Y_ui = y) = p_ui * rho_y, with p_ui = sigmoid(a + c1 logit(deg_u) + c2 logit(deg_i))
  (the existing `popularity_propensity`) and a selection ratio rho_1 / rho_0 = gamma >= 1 (gamma = 1 is
  MAR exposure; gamma > 1 is self-selection on liking). gamma is a hyperparameter chosen on validation.

E-step (posterior that an unexposed pair is a positive):

  q_ui = P(Y=1 | O=0) = r_ui (1 - p_ui rho_1) / [ r_ui (1 - p_ui rho_1) + (1 - r_ui)(1 - p_ui rho_0) ],

with rho_0 = 1 / (r_bar gamma + 1 - r_bar), rho_1 = gamma rho_0 (normalised so that the marginal exposure
rate equals p_ui; r_bar the mean of r over the user's row). For gamma = 1 and r = pi this is exactly the
Q of QP-LightGCN, so the v2 ingredient is the special case "one E-step, non-personalised r, MAR".

M-step (closed-form weighted ALS on all pairs, same solver as P1):

  min sum_{s=+} (1+alpha)(1 - g_ui)^2 + sum_{s=-} w_n (0 - g_ui)^2
      + sum_{s=0} [ q_ui (1 - g_ui)^2 + (1 - q_ui) w_0 (0 - g_ui)^2 ] + reg(|X|^2 + |Y|^2),

  g_ui = beta z_ui + x_u'y_i.

The unexposed term is the expected loss under the posterior; it equals a weighted least-squares term with
weight q + (1-q) w_0 and target q / (q + (1-q) w_0), so the solver of P1 applies unchanged with dense
C and T. After each M-step, r_ui = sigmoid(calibrated g_ui) (Platt scaling of g on the exposed pairs of
the log, two parameters) and the E-step is repeated. 2-4 EM rounds, each a few ALS sweeps.

Ranking score: g_ui.

### 5.2 Why it could work where P1-P4 did not
- It acts on the unexposed pairs, which are the test candidates, rather than on exposed negatives
  (shown to be useless as negatives in P1/P2).
- On Yahoo/KuaiRand the stored evidence shows that softening unexposed negatives helps, and the
  personalised posterior is the natural next step that the v2 method did not take.
- It keeps the exposure-conditional prior as an offset (beta), which already carries the Coat/KuaiRec
  gains, so non-inferiority there is plausible.

### 5.3 Novelty, stated exactly
- vs **ExpoMF** (Liang et al. 2016): ExpoMF infers the posterior of *exposure* for unclicked pairs and
  downweights them; it has no known exposed negatives and no outcome-dependent selection. PEP-ALS infers
  the posterior of *preference* for pairs whose exposure is known to be 0, with selection on the outcome.
- vs **Hernández-Lobato et al. 2014 / Marlin-Zemel 2009**: same selection idea, but for explicit
  rating prediction with variational or mixture inference; no implicit-feedback ALS, no ranking under a
  uniform test with uniform-free training.
- vs **Bayesian Negative Sampling** (arXiv 2204.06520): BNS uses a posterior false-negative probability to
  choose sampled negatives in BPR; PEP-ALS uses the posterior as soft targets over all unexposed pairs in
  a closed-form solver, and the posterior is personalised and iterated with the model.
- vs **QP-LightGCN** (our v2): the special case gamma = 1, one E-step, r = additive prior.
Honest assessment: the difference is real but incremental; a reviewer can call it "ExpoMF-style EM with
known exposure". The paper would have to carry it with the uniform-test evidence on Yahoo/KuaiRand.

### 5.4 Implementation spec (for a junior engineer)

New code only; do not modify existing functions.

1. `drup/learned.py`: add class `DenseWALS(m, n, d, C, T, reg, seed)` with `step()` and
   `full_scores(rows)`, using the batched solve of the probe (`A = (F' * c) F + reg I`,
   `b = (c * t) F`, chunks of 64 rows for KuaiRec/KuaiRand, 1,024 for Coat/Yahoo). Unit test
   (`tests/test_dense_wals.py`): with C = 1 + alpha*pos and T = pos - beta*Z it must match `IALSP`
   to 1e-4 after 3 sweeps on Coat.
2. `drup/learned.py`: add `class PEPALS(DenseWALS)` with `e_step(gamma, w0)` (formula 5.1, uses the
   `popularity_propensity` matrix P passed in, floor 0.05) and `calibrate()` (Platt on exposed pairs,
   10 Newton steps). `train_pep(model, rounds, sweeps, on_epoch)` calls on_epoch after every sweep so
   the existing per-split validation selection and patience work.
3. `experiments/run_learned.py`: register methods
   - `"PEP-ALS": ("pep", None, {"alpha": [3.0, 10.0, 30.0], "reg": [100.0, 300.0], "pbeta": [0.0, 0.5], "gamma": [1.0, 2.0, 4.0], "w0": [1.0]})`
   - `"PEP-ALS-1": same with rounds = 1` (ablation: one E-step = QP-style posterior)
   - `"DenseWALS-iALS": ("pep", None, {..., "gamma": [1.0]})` with E-step disabled (matched-grid iALS
     control).
   Add `"pep"` to `IALS_KINDS`; build it in `_build` with C, T from O, Y, Z, and P.
   Coat grid uses alpha {30, 100, 300} and reg {100, 300, 1000} as in P1.
4. Queue `experiments/queue_v8.sh` (style of `queue_v5.sh`), one label per job, at most 2 in parallel
   on KuaiRec/KuaiRand (memory ~4 GB each):
   ```
   pep_coat)     OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --prior add_noips --seeds 10 --threads 1 --dims 32 64 --epochs 16 --seed_check 0 --methods PEP-ALS PEP-ALS-1 DenseWALS-iALS --out results/v7/learned_coat_pep.json ;;
   pep_yahoo)    OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset yahoo --prop pop --prior add_noips --seeds 5 --threads 2 --dims 64 128 --epochs 16 --seed_check 0 --methods PEP-ALS PEP-ALS-1 DenseWALS-iALS --out results/v7/learned_yahoo_pep.json ;;
   pep_kuairand) OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairand --prop pop --prior add_noips --seeds 5 --threads 2 --dims 64 --epochs 12 --seed_check 0 --methods PEP-ALS PEP-ALS-1 DenseWALS-iALS --out results/v7/learned_kuairand_pep.json ;;
   pep_kuairec)  OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --prior add_noips --seeds 5 --threads 2 --dims 64 --epochs 10 --seed_check 0 --methods PEP-ALS DenseWALS-iALS --out results/v7/learned_kuairec_pep.json ;;
   seeds_pep_*)  OMP_NUM_THREADS=2 python3 experiments/run_seeds.py --dataset ${1#seeds_pep_} --seeds 1 2 3 ;;
   ```
   Cost estimates from the probes (2 threads, contention included): Coat ~1 s per sweep; Yahoo ~5 s per
   sweep (~50 s per configuration); KuaiRec ~110 s and KuaiRand ~200-300 s per configuration. Stage 0
   grid (Section 5.5) is 36 configurations on Yahoo (~45 min) and 12 on KuaiRand (~1 h).
5. Analysis: `experiments/v8_tests.py` reusing `v7_common.peru/cmp/holm/paired_stats`; Holm within each
   dataset over the comparisons listed in Section 5.5.

### 5.5 Staged pilot plan with go/no-go thresholds

All comparisons per-user paired, Holm within dataset, seeds 1-3 for the final rows.

**Backbone caveat (from P1).** On Yahoo and KuaiRand the ALS backbone is itself 6-8e-3 below LightGCN
(same-grid iALS 0.6629 vs LightGCN 0.6708; 0.4533 vs 0.4569) and 24.6e-3 / 8.5e-3 below the QP family.
PEP-ALS must therefore close a large backbone gap before it can beat QP. Run the same E-step inside the
existing LightGCN Q-weighted loss as a second arm: add `qmode="pep"` to `drup/learned.train` (recompute Q
from the current model's calibrated scores every 5 epochs with formula 5.1, instead of the fixed
imputation-based Q) and register `"PEP-LGN": ("lgn", "drbpr", {"fn": [1.0], "gamma": [1.0, 2.0, 4.0], "qmode": ["pep"]})`.
The Stage 0 gates apply to whichever arm is better on **validation**.

**Stage 0 (cheap, Yahoo only, about 2 h).** PEP-ALS and PEP-LGN vs their matched controls
(DenseWALS-iALS; DR-BPR-LGN-1 with the fixed Q) and vs stored QP family (0.6866).
- GO if the chosen arm beats its fixed-Q control (PEP-ALS-1, or DR-BPR-LGN-1) by >= +3e-3 with
  p_Holm < 0.05 (the personalised, iterated posterior must add something over the one-step QP posterior;
  otherwise the method is QP-LightGCN under another name and the novelty objection stands) **and** it is
  not below the stored QP family by more than 3e-3.
- NO-GO otherwise: stop method work on these benchmarks (Section 7).

**Stage 1 (all four datasets, about 5 h).**
- Yahoo: PEP-ALS >= QP family (0.6866) - 3e-3 (non-inferior) and > r-AdjNorm (0.6742) with p_Holm < 0.05.
- KuaiRand: PEP-ALS - QP family >= +2e-3 with p_Holm < 0.05 (half-width ~1.5e-3, so this is testable).
- Coat: non-inferiority to iALS + prior no-IPS (0.6044) at margin 0.010 (TOST).
- KuaiRec: non-inferiority to DRUP (0.6378) and DR-JL/MRDR at margin 0.003 (TOST); **no superiority gate**
  (H1/H2: the attainable gain is ~2-4e-3).
- GO to Stage 2 if Yahoo and KuaiRand gates both pass and neither non-inferiority gate fails.

**Stage 2 (robustness, about 3 h).** seeds 1-3; shuffled-q control (q permuted within user: must return
to DenseWALS-iALS); gamma = 1 ablation; PEP inside LightGCN (BPR with soft unexposed labels) on Yahoo only.

---

## 6. If PEP-ALS fails Stage 0

Say plainly that no new method with real gains was found on Coat, Yahoo R3, KuaiRand-Pure and KuaiRec
under this protocol after four method families (v2 QP-LightGCN, v3 DRGP, walk-corrected DRUP, v5 probes
I1-I4/P4). Options, in order:
1. Add the headroom diagnostics H1-H3 and the probe family of Section 4 to the v4 evaluation study
   (`paper/v4/DESIGN.md`); they explain why methods cannot separate on KuaiRec and Coat and are new.
2. Move method work to data with impression logs, graded signals and power (KuaiRand with its 12 feedback
   signals and full impression log; the loader currently keeps only `is_click`), where personalisation
   headroom can first be measured with an H2-style diagnostic before any method is designed.

---

## 7. Risks

| Risk | Likelihood | Handling |
|---|---|---|
| PEP-ALS gains on Yahoo come only from the item prior / one-step posterior (as P4 on Coat) | high | Stage 0 second condition (PEP vs PEP-1) |
| EM posterior collapses (q -> r everywhere, i.e. unexposed pairs treated as unlabeled) | medium | w0 and gamma grids; monitor mean q per round |
| Dense C, T matrices on KuaiRec/KuaiRand (4 x 300 MB) plus O, Y | low | float32, 2 jobs at most |
| Novelty objection (ExpoMF/BNS/QP) | high | Section 5.3 wording; only proceed if the PEP vs PEP-1 effect is clear |
| Container restarts | medium | run_learned.py checkpoints after every configuration (existing `.partial.json`) |

---

## 8. Honest probability of success

- Probability that PEP (either arm) passes Stage 0: about 25% (Yahoo has demonstrated sensitivity to how
  unexposed pairs are treated; the open question is whether personalising the posterior adds over the
  one-step version, which P4 did not show on Coat, and the ALS arm starts 24.6e-3 below QP on Yahoo).
- Probability that it then passes Stage 1 on both Yahoo and KuaiRand: about 50%.
- Probability that the result is accepted as novel at a good venue: about 60-70% conditional on clear gains.
- Overall: **about 10-15%** for a publishable method paper with real gains. The probability that any
  method beats DRUP on KuaiRec by a paired margin is close to zero (Section 3).

---

## 9. Reproducibility of this design

Scratch probe scripts and outputs (to be re-implemented, not cited). They live in a temporary
`probes/` directory outside the repository; copy it to `experiments/v8_probes/` if it is to be kept:
- `harness.py` (data, splits, selection, paired comparison against stored files)
- `probe_tsials.py` (P1), `probe_linear.py`, `probe_linear2.py` (P2), `probe_multi.py`,
  `probe_multi2.py` (P3), `probe_softlabel.py` (P4), `diag_headroom.py` (H1, H3), `diag_personal.py` (H2),
  `pair.py`; result texts `probe_*.txt` and per-run JSONs with per-user values.
