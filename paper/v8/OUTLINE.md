# v8 paper design: LEAF-LGN + DAX (method paper)

Status: design document, 2026-11-09. Journal-agnostic; written for `elsarticle` (Elsevier) and convertible to an
ACM journal template. New files only: `paper/v8/OUTLINE.md`, `paper/v8/refs_new.bib`.

Inputs read: `results/v11/stageB.txt` (gate B1 passed), `results/v10/stage1.txt`, `results/v11_probes/*.txt`,
`results/v9_probes/*.txt`, `results/logs_v11/dax_lgn_post.log`, `results/v11/*.json` (chosen configurations only),
`paper/v6/DESIGN.md` (LEAF), `paper/v7/DESIGN.md` and `paper/v7/LITERATURE.md` (DAX), `paper/v2/lit_search.md`,
`paper/v2/SUBMISSION_NOTES.md`, `paper/v4/AMENDMENTS.md`, `paper/v5/DESIGN.md`.

## 0. Rules for whoever writes the manuscript from this outline

1. **Number provenance.** Every number in the paper is copied from a result file named next to it in this outline.
   A number that is not yet in a file appears as `{TBD:<file> <quantity>}` and stays a placeholder until the file
   exists. Numbers from the design documents are used only where the same number is in a result file (checked
   for every number below). Do not round differently from the source file; paired differences are reported
   x1e-3 nDCG@10 with the 95% half-width in brackets, as in the files.
2. **Three kinds of number, never mixed in one table without a label:** *main* (re-run, per-split validation
   selection, files in `results/v10/`, `results/v11/`); *probe* (post-hoc on stored scores, `results/v11_probes/`,
   `results/v9_probes/`; used for the negative results and the controls of DAX until `dax_ablate.json` exists);
   *diagnostic* (uses random-exposure test labels; never a method, never a selection input).
3. Inherited wording rules (`paper/v4/AMENDMENTS.md` item 4, `paper/v2/OUTLINE.md` Section 0): no "state of the
   art"; "significant" only with p_Holm < 0.05 *and* the test named; no claim that a trained method is unbiased.
4. The QP family (DR-BPR-LGN-1 / -p2 / -p, called "QP-LightGCN" in earlier drafts) is **our own earlier,
   unpublished** method. It enters the paper as a strong baseline that is fully described in an appendix, never
   as a citation to prior publication, and never with wording such as "our previous work [x]". The DRUP
   manuscript (rejected at IPM) and the Information Sciences manuscript (desk-rejected) are not cited either.
5. The reported combination is the **post-hoc** one: LEAF-LGN trained alone, the DAX weight gamma, the DAX
   configuration and the epoch selected jointly on validation (`results/v11/dax_lgn_post.json`). The jointly
   trained variant (`dax_lgn_joint*.json`) does not exist; the paper must not describe the DAX weight as learned
   unless `{TBD:results/v11/dax_lgn_joint_small.json LEAF-LGN+DAX(joint) test mean}` exists and is reported.

---

## 1. Title

**Main:** *Examination before attraction: lifelong affinities and detection-adjusted item appeal for
recommendation under randomised exposure*

Alternatives:
1. *Recommending for a uniform insertion: separating user examination from item attractiveness in a
   short-video log*
2. *What the logging policy hides: lifelong exposure-conditional channels and a detection model of item appeal,
   evaluated on randomly inserted videos*
3. *LEAF and DAX: two log-only signals that transfer to randomly exposed items*

Avoid "debiasing", "unbiased" and "causal" in the title (rule 0.3; the method is not shown to be unbiased).

---

## 2. Thesis, contributions, evidence, and what we do not claim

### 2.1 Thesis (one paragraph for the abstract and the introduction)

When a recommender is evaluated on items inserted uniformly at random into users' feeds, the remaining
accuracy headroom of a strong graph recommender is mostly item-level: it lies in failing to promote the items
that a random viewer finds most appealing, not in temporal drift and not in cold start. The policy log measures
an item's response *to the policy's choice of audience and scenario*. Two log-only signals transfer to the
uniform target: (i) a user's **lifelong, exposure-conditional affinities** for content keys (tag lift, author
familiarity) learned on millions of out-of-pool videos (LEAF), and (ii) an item **attractiveness** estimated by a
click-model / occupancy factorisation that discounts non-clicks by the user's and the scenario's examination
probability (DAX). Both are added to LightGCN; on the temporal all-user KuaiRand-27K setting the combination
is higher than every tuned baseline, including the strongest debiased graph baseline with the same DAX channel.

### 2.2 Contributions, each tied to its evidence

| # | Contribution | Evidence (file, quantity) | Status |
|---|---|---|---|
| C1 | **Headroom diagnosis.** On randomised insertions the leave-one-user-out item oracle reaches 0.5068 against LEAF-LGN 0.4735 (+33.31e-3); the recoverable gain sits in the top uniform-rate quintile (share 0.83); the post-cut log does not predict the uniform rate better than the pre-cut log (Spearman 0.405 vs 0.431), so drift is not the bottleneck; in-pool low-rank personalisation on uniform data adds nothing (item means 0.5850, + rank-64 residual 0.5851) | `results/v11_probes/diag_gap.txt` (oracle, shares, drift rows); `results/v9_probes/headroom_kuairand_pure.txt` (H1 0.5072 all users, H2 rows) | in files (diagnostic) |
| C2 | **LEAF: lifelong exposure-conditional channels** transfer from out-of-pool videos to a disjoint, randomly exposed pool through content keys only. LEAF-LGN vs LightGCN +15.46 [1.53], NB p_Holm 0.049; vs r-AdjNorm +11.10 [1.71], NB p_Holm 0.047; the shuffled-user control removes the lifelong gain (LEAF-tied shuffle_users vs no_lifelong -1.02e-3, design check PASS); the concurrent protocol inflates the gain (+22.17 vs +16.04 temporal) | `results/v10/stage1.txt` Sections 3 and 5; `results/v9_probes/probe_27k.txt`, `probe_27k_shuffled.txt`, `probe_27k_concurrent.txt` | in files; LEAF-LGN own ablations `{TBD:results/v10/leaf_lgn_ablate.json}` missing (Stage 1 used LEAF-tied ablations) |
| C3 | **DAX: detection-adjusted attractiveness.** P(click) = theta_u kappa_tab alpha_i fitted by EM on the pre-cut log, alpha shrunk to its tag mean, theta with a lifelong prior. It is complementary and backbone-independent: LEAF-LGN+DAX - LEAF-LGN +11.96 [1.47], NB p_Holm 5.48e-03; EASE+DAX - EASE +11.98 [1.64], NB p 0.000; QP family+DAX - QP family +10.28 [1.37], NB p 0.105. Probe-scale controls: a global permutation of alpha gets weight 0 in 5/5 splits; within-tag permutation keeps +3.29 [0.99] of +9.45 [1.36] | `results/v11/stageB.txt` Sections 2, 3, 6; `results/v11_probes/probe_dax_ablate.txt` | B1 in file; **B2 ablations `{TBD:results/v11/dax_ablate.json}`** |
| C4 | **Main result under a leakage-safe protocol with split-level inference.** LEAF-LGN+DAX 0.4854 vs QP family 0.4634: +22.32 [1.83], t p_Holm 1.45e-124, NB p_Holm 0.023 (all five split differences positive: +19.8 +22.2 +21.8 +17.8 +28.0); vs the fair baseline QP family+DAX +12.04 [1.63], NB p_Holm 0.017; vs LightGCN +28.15 [1.86], vs r-AdjNorm +23.79 [1.96] | `results/v11/stageB.txt` Sections 1-3 | in file; seeds and KuaiRand-1K `{TBD:results/v11/c_*.json}` |

A secondary, reported (not claimed as a contribution) result: the borrowed estimators that do *not* work on this
target - additive multilevel logistic (MRP) +2.48 [0.87] over LEAF-LGN vs DAX +9.43 [1.29]; Fay-Herriot content
centre +5.69 below the tag centre +7.59; trajectory forecast -0.33 [0.22] (`results/v11_probes/probe_items.txt`,
`probe_pbm.txt`; all probe-scale, post-hoc on stored LEAF-LGN). The fair MRP-with-tab arm is
`{TBD:results/v11/dax_ablate.json arm additive}`.

### 2.3 What we do not claim

- Not unbiased, not causal identification of item quality: DAX's alpha is a model-based uniform-insertion appeal
  under a stated generative model; we test only its ranking value on randomised exposure.
- Not "state of the art" and no comparison with published numbers (different binarisation, candidate sets, K and
  selection; `paper/v5/DESIGN.md` Finding L1).
- Not better than lifelong deep CTR models (SIM, TWIN, DIN-style): no lifelong-aware deep baseline was run
  (`lgnL_t`, `dinL_t` of `paper/v6/DESIGN.md` Section 6 have no result file). The paper says so and positions LEAF
  as a light, exposure-conditional alternative, not as a winner over them.
- Not that the DAX weight is learned end-to-end (rule 0.5) unless the joint run is reported.
- Not that the item-level part of alpha matters beyond its tag mean until B2 (`shuffle_items` must lose >= 30%
  of the gain) is in `dax_ablate.json`; at probe scale the tag mean alone gives +4.31 [1.07] of +9.45.
- Not that the multiplicative form is necessary until the `additive` arm of B2 is >= 4e-3 below DAX.
- Not generality beyond one platform family (KuaiRand-27K and KuaiRand-1K share users, items and platform).
- Not robustness to seeds until `{TBD:results/v11/c_seeds*.json}`; the main table is one training seed per split.
- Not that examination probabilities theta_u, kappa_t are identified in absolute terms (Section 4.6).
- Not that tab factors or theta_u are interpretable user traits; they are nuisance parameters.
- Not gains on Coat, Yahoo! R3 or KuaiRec (the method cannot be run there; Section 5.5).

---

## 3. Novelty positioning (related work section plan)

Order of the related-work section: 3.1 lifelong behaviour modelling; 3.2 click models and examination; 3.3
exposure models in recommendation; 3.4 occupancy / detection and survey estimation; 3.5 impression-aware
recommendation; 3.6 debiased graph recommendation. Each subsection ends with one sentence "Difference:".
Keys in `paper/refs.bib`, `paper/v2/refs_new.bib` (existing) or `paper/v8/refs_new.bib` (new, verified
2026-11-09 against Crossref or the arXiv API).

| Closest work | Key | What it does | Difference (one sentence for the paper) |
|---|---|---|---|
| SIM (CIKM 2020) | `pi2020sim` (new) | retrieves category-matched behaviours from lifelong sequences inside a deep CTR model; trained and evaluated on policy logs, history and target share item ids | LEAF uses exposure-conditional statistics (non-clicks in the denominator) and familiarity, transfers to a *disjoint* pool through content keys only, and is evaluated on randomised insertions under a temporal protocol |
| TWIN (KDD 2023, Kuaishou) | `chang2023twin` (new) | two-stage target attention over lifelong sequences, same platform family as KuaiRand | as SIM; and TWIN is not evaluated on random insertions. We do not claim LEAF beats it (Section 2.3) |
| Examination hypothesis / PBM | `richardson2007predicting`, `craswell2008cascade`, `chuklin2015clickmodels` (new) | P(click) = P(examined at position) x P(attractive) | DAX conditions examination on the *user* (lifelong prior) and the *scenario* (tab), has no position, and targets an item appeal under uniform insertion validated on randomised exposure |
| Cascade model | `craswell2008cascade` | sequential scan of a ranked list | KuaiRand random insertions are single feed slots; no list scan is modelled |
| DBN | `chapelle2009dbn` (new) | perceived vs actual relevance with satisfaction | DAX has no satisfaction state; multi-behaviour signals failed earlier (v6 probe I1) |
| Regression-EM | `wang2018regressionem`, `wang2016selection` (new) | EM estimate of position bias and relevance without randomisation, personal search | closest estimation machinery; their estimand is query-document relevance with position examination |
| Unbiased LTR | `joachims2017unbiased` (new) | IPS with examination propensities | DAX is a generative model fitted by EM, not a weighted loss |
| ExpoMF (WWW 2016) | `liang2016exposure` (existing) | latent exposure variable, EM | in KuaiRand impressions are observed; DAX's latent variable is *examination given an impression* |
| Occupancy (Ecology 2002) | `mackenzie2002occupancy` (new) | presence x detection with repeated surveys | same likelihood structure; we found no recommendation paper that uses it to estimate item appeal from a policy log (searches of `paper/v7/LITERATURE.md`) |
| MRP / Fay-Herriot | `park2004mrp`, `fay1979smallarea` (new) | population estimates from non-representative samples, small-area shrinkage | borrowed and *tested*: the additive version is weaker (C3 negative results) |
| Impression-aware recommenders | `perezmaurera2025iars` (new) | survey; impressions as negatives or features | the policy channel alone is not claimed as new |
| Co-exposure choice models | `krause2024coexposure` (new) | MNL with co-exposure for exposure-bias mitigation | consideration-set view; not evaluated on random insertion |
| ESMM | `ma2018esmm` (new) | entire-space impression -> click -> conversion | multi-task over the policy log; DAX separates examination from appeal |
| cEASE / add-EASE | `jeunen2020cease` (new) | item side information in closed-form models | LEAF adds *user-side* behaviour on other items |
| NAVIP | `kim2022navip` (existing) | IPS-weighted neighbour aggregation in LightGCN | acts on propagation; no lifelong data, no examination model |
| AdvDrop | `zhang2024advdrop` (existing) | adversarial edge dropout, bias-aware views | as NAVIP |
| PPD | `islam2026ppd` (existing) | post-hoc removal of a popularity direction from GNN embeddings | removes popularity; DAX re-estimates appeal under a different audience and scenario mix |
| PDA | `zhang2021pda` (existing) | popularity-drift-aware recommendation | our diagnosis rules drift out as the bottleneck (C1) |
| AutoDebias, KDCRec | `chen2021autodebias`, `liu2020kdcrec` (existing) | learn from a small uniform sample | not admissible under our protocol (random data only for validation / test) |
| KuaiRand users | `gao2022kuairand` (existing), `zhao2023d2co`, `hu2025uncertainty`, `wang2025debiasunreliable`, `zhang2026categorymeans` (new) | other uses of KuaiRand / random exposure | none uses out-of-pool lifelong history for the random test (as far as our searches went) |
| QP family | none (rule 0.4) | our unpublished IPS/DR-weighted BPR LightGCN | baseline, described in Appendix A |

Honest novelty statement for the introduction: each ingredient has a precedent (lifelong behaviour; examination x
attractiveness; Beta shrinkage). The contribution is the identification of *which* log-only signals transfer to a
uniform-insertion target, a two-factor examination model with user and scenario examination whose appeal stacks on
personal lifelong channels, and controlled evidence (temporal protocol, shuffle controls, split-level tests). This
must be said plainly; a reviewer will call DAX "a click model applied to items" (`paper/v7/DESIGN.md` Section 4).

Citations not verified in this pass and therefore not to be added without a check: Gelman and Little 1997 (no DOI
record), Bass 1969, Szabo and Huberman 2010 (verified only in `paper/v7/LITERATURE.md` on 2026-11-08 with links,
not as BibTeX), the USPTO patent mentioned in `paper/v7/DESIGN.md`.

---

## 4. Method section plan

### 4.1 Setting and notation (Section 3.1 of the paper)

- Users u in U, pool items i in I (the random-insertion pool), tabs (scenarios) t in T, cut-off T0
  (2022-04-22 00:00 UTC+8).
- Pre-cut standard log L = {(u, i, t, y, time)} with time < T0, y = click.
- In-pool matrices X (clicked before T0), N (shown, never clicked before T0).
- Out-of-pool lifelong log: each user's impressions before T0 on videos outside I; content-key families
  K = {tag, author, music}, key map kappa_k: I -> V_k.
- Test target: random insertions after T0 (almost all in tab 1); each user's candidate list is the set of randomly
  inserted pool items; label = click; metric nDCG@10 per user.
- Leakage statement (in the notation subsection): max training time 1650556797718 ms < T0 = 1650556800000 ms <
  min test time 1650595332831 ms (`results/logs_v11/dax_lgn_post.log`, line "temporal protocol").

### 4.2 LEAF channels (Section 3.2)

- Lifelong base rate r_u = (C_u + 1)/(N_u + 2) over all out-of-pool impressions before T0.
- Exposure-conditional lift l^k_uv = (c^k_uv + s r_u)/(n^k_uv + s) - r_u (Beta-binomial shrinkage s);
  familiarity f^k_uv = log(1 + n^k_uv). Six channels (tag, author, music) x (lift, familiarity), each divided by
  the standard deviation of its non-zero entries (`drup/leaf.py`, `CHANNELS`).
- Why exposure-conditional: the uniform insertion measures P(click | shown); conditioning on impressions removes
  how often the policy chose a key. Probe evidence: `results/v9_probes/diag_outpool_1k.txt`.
- Policy channel: shown-not-clicked pairs sampled as positives with probability proportional to w in BPR.
- Coverage figures for the text: from `results/v9_probes/lifelong_27k_summary.txt` (26,904 users with pre-cut
  history; 135,219,168 out-of-pool impressions; 44 tag, 6,510 author and 7,202 music values that occur on pool
  videos).

### 4.3 LEAF-LGN (Section 3.3)

s^LEAF_ui = <e_u, e_i>_LightGCN + beta pi_i + sum_k [gamma_k l~^k_{u,kappa_k(i)} + eta_k f~^k_{u,kappa_k(i)}],
with gamma, eta (6 scalars) and beta trained jointly with the embeddings by BPR on the pre-cut in-pool log; lr, wd,
w and the epoch selected on validation. Hyperparameter grid: lr {1e-2, 3e-3}, wd {1e-4, 1e-5}, w {0, 0.25, 0.5},
dim 64, 2 layers, 40 epochs, patience 3 (`experiments/run_dax.py` defaults; 12 trainings, `dax_lgn_post.log`).
State the closed-form LEAF-tied variant only as an ablation backbone (Stage 1 ablations ran on it).

### 4.4 DAX: detection-adjusted attractiveness (Section 3.4)

Generative model of a logged impression (u, i, t):
E ~ Bernoulli(theta_u kappa_t) (examination), R ~ Bernoulli(alpha_i) (attractiveness), y = E R.

Priors (MAP-EM, all M-steps closed form, `drup/dax.py::fit_dax`):
- alpha_i ~ Beta(a c_tag(i), a(1 - c_tag(i))), c_tag re-estimated each iteration from posterior attraction counts
  (tag-mean shrinkage; alternative centre: Fay-Herriot-style ridge regression on content codes);
- theta_u ~ Beta(k r~_u, k(1 - r~_u)), r~_u the lifelong click rate rescaled to the in-pool mean examination
  (lifelong user prior; k = 0 gives a weak Beta(2, 2));
- kappa_t free, kappa_1 = 1 (tab 1 is the test scenario);
- optional user-stratum cells alpha_{i,s}, s = quintile of the lifelong click rate, shrunk to alpha_i with strength
  prior_s (MRP-style cells inside DAX; prior_s = 80 is selected in 5/5 splits of the main run, `dax_lgn_post.json` "chosen").

E-step: a click sets E = R = 1; for a non-click, with tau = theta_u kappa_t,
P(E=1, R=0) = tau(1-alpha)/(1 - tau alpha), P(E=0, R=1) = (1-tau)alpha/(1 - tau alpha).
M-step: theta_u = (sum P(E=1) + k r~_u)/(sum kappa_t + k); kappa_t = sum_{tab t} P(E=1) / sum_{tab t} theta_u;
alpha_i = (sum P(R=1) + a c_tag(i))/(n_i + a). 40 iterations. Only rows with time < T0 are read (`pre_cut`).

Intuition paragraph (from `paper/v7/DESIGN.md` Section 2): a non-click is weak evidence against an item when the
viewer rarely clicks anything or the scenario rarely yields clicks; the policy places items in such contexts
unevenly. Contrast with the additive logistic (MRP) form, which treats a non-click by a low-rate user almost like
one by a high-rate user once the user effect is absorbed. Tab factors as an illustration: `results/v11_probes/
probe_pbm.txt` lines "tab factors" (several tabs below 0.1 relative to tab 1); report as a figure, not a claim.

### 4.5 Combination with LEAF-LGN (Section 3.5)

s_ui = rowstd_u(s^LEAF_ui) + gamma rowstd_u(logit alpha_{i(, s(u))}), standardisation over the user's candidate
list; gamma in {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0} (`GAMMAS` in `experiments/run_dax.py`), selected
jointly with the backbone epoch and the DAX configuration on validation (post-hoc variant; 3,840 configurations
per split, `stageB.txt` Section 1). Joint variant (DAX as a seventh channel with a BPR-trained weight,
`LEAFLGNDAX`) described in one paragraph and reported only if `{TBD:results/v11/dax_lgn_joint_small.json}`
exists.

The same post-hoc rule applies to every backbone in the paper (EASE, QP family, LightGCN), which makes "+DAX" a
backbone-agnostic module.

### 4.6 Identifiability and what alpha means (Section 3.6, short)

- From y alone, (theta_u c, kappa_t, alpha_i / c) gives the same likelihood: the factorisation is identified only up
  to a scale shared by theta and alpha; kappa_1 = 1, the bounds (all factors in [0,1]) and the two Beta priors fix
  it. The *order* of alpha is invariant to the scale, but the logit transform is not linear in the scale, so the
  standardised channel (and therefore the combined score) depends on the prior strengths; this is why a and k are
  hyperparameters selected on validation.
- Users with no examination signal and items with few impressions are pulled to their priors (tag mean, lifelong
  rate). State the median of logged impressions per item `{TBD:data statistics script, pre-cut impressions per pool item, median}`
  (the value 29 in `paper/v7/DESIGN.md` is not in a result file).

### 4.7 Complexity (Section 3.7)

- LEAF tables: dense user x key-value float32 tables (|U| x |V_tag|, |U| x |V_author|, |U| x |V_music|); a pair is
  a gather, O(1); building them is one streaming pass over the lifelong log (O(lifelong rows), done once).
- LEAF-LGN epoch: LightGCN cost O(L |E| d) plus O(6) per sampled pair.
- DAX: each EM iteration is O(|L| + |U| + |I| + |T|) with index-add accumulations; 40 iterations; wall time about
  2 s per configuration on 2 threads (`results/logs_v11/dax_lgn_post.log`, "DAX fit ... [1.8s]-[2.8s]").
- Post-hoc selection: per split, (epochs) x (DAX configurations) x (gammas) evaluations on validation users;
  86,400 evaluations for the main run, 19,628 s wall time including the 12 backbone trainings, peak RSS 6,032 MB
  (`dax_lgn_post.log`, last lines). Inference adds one standardised item vector per user list.

### 4.8 Algorithm box

Algorithm 1 (DAX-EM) and Algorithm 2 (training and selection of LEAF-LGN + DAX). Pseudocode from `drup/dax.py`
and `experiments/run_dax.py::run_leaf_post`.

---

## 5. Experimental protocol section plan

### 5.1 Data and temporal protocol

- KuaiRand random insertions (`gao2022kuairand`): each feed slot was replaced with small probability by a video
  drawn uniformly from the pool; the random log is test/validation only and never enters training.
- Temporal protocol: every training signal (in-pool log, lifelong tables, DAX fit, popularity) strictly before T0;
  boundary times in Section 4.1. Pairs impressed in both logs are not ranked (loader rule).
- Concurrent protocol as a robustness row only (it lets policy feedback after a random impression leak in; for
  LEAF it inflates the lifelong gain: +22.17 concurrent vs +16.04 temporal, `results/v9_probes/probe_27k_concurrent.txt`,
  `probe_27k.txt`); for the full model `{TBD:results/v11/c_concurrent.json}`.

### 5.2 Splits and selection

- 5 user-level splits of the random-exposure users (frac_val 0.3, `drup.leaf.protocol_split`; validation and test
  users disjoint, asserted). Test users per split: 15,809 / 15,892 / 15,840 / 15,913 / 15,869; 22,612 users pooled
  over splits (`stageB.txt` Section 2).
- Per split, the configuration (including the epoch, the DAX configuration and gamma) with the best validation
  nDCG@10 is chosen; test is read only for the chosen configuration.
- Search budgets are reported per method (n_configs column of `stageB.txt` Section 1) together with a sentence that
  budgets differ (3,840 for LEAF-LGN+DAX vs 20 for the QP family) and that QP family+DAX (3,200) is the
  budget-matched fair baseline; QP family+DAX searches only the QP configurations selected on QP alone, which is
  conservative for our method.

### 5.3 Statistical tests

- Per-user paired difference (pooled test users of the five splits), mean x1e-3 with 95% half-width, two-sided
  t p; Holm within a pre-declared primary family of fixed size 5: {vs QP family, vs QP family+DAX, vs LEAF-LGN,
  vs r-AdjNorm, vs LightGCN} (`stageB.txt` Section 2).
- Split-level Nadeau-Bengio corrected resampled t test (`nadeau2003`) over the five split means: variance
  multiplied by (1/J + n_test/n_val) with J = 5 and the ratio taken as (1 - 0.3)/0.3 (as implemented in
  `experiments/significance.py::nadeau_bengio`; state this choice and its conservativeness). Holm over the same
  family for the NB p values (`holm1979`).
- Decision rule pre-declared before Stage B (`paper/v7/DESIGN.md` Section 7): gate B1 = vs QP family >= +15e-3 with
  both p_Holm < 0.05 (passed). Report the gate in the paper as the pre-declared criterion, with its date.
- Power remark: with J = 5 the NB test needs split differences with mean/sd above about 4.4 (from
  `paper/v7/DESIGN.md` Section 7; derive the constant in the paper from the t quantile rather than copying it).
- Ablation family (B2): Holm within the ablation family, separately from the primary family.

### 5.4 Leakage checks and controls (one subsection, one table)

| Check | Result | Source |
|---|---|---|
| training times < T0 <= test times | 1650556797718 < 1650556800000 <= 1650595332831 | `results/logs_v11/dax_lgn_post.log` |
| cached test lists equal the loader's | assertion passes ("test lists and users match the loader") | same log |
| lifelong tables contain no pool video; "pre" rows before T0 | builder rule (`experiments/v9_build_lifelong.py`); unit test `{TBD:tests/test_lifelong.py result}` | `paper/v6/DESIGN.md` Section 6 item 1 |
| DAX reads only pre-cut rows | `drup.dax.pre_cut`; unit test in `tests/test_dax.py` (c) `{TBD:pytest output file}` | code |
| user linkage Pure <-> 27K | 1,186,059 random rows joined, ids identical 0.9999 | `results/v9_probes/link_users.txt` |
| shuffled users (LEAF) | lifelong channels on top with shuffled tables: -0.93 [0.45] (probe); LEAF-tied shuffle_users vs no_lifelong: -1.02e-3 (main) | `results/v9_probes/probe_27k_shuffled.txt`; `results/v10/stage1.txt` Section 5 |
| alpha permuted across all items (DAX) | probe: weight 0 in 5/5 splits; main: `{TBD:results/v11/dax_ablate.json shuffle_tags}` | `results/v11_probes/probe_dax_ablate.txt` |
| alpha permuted within tag (DAX) | probe: +3.29 [0.99] vs full +9.45 [1.36]; main: `{TBD:results/v11/dax_ablate.json shuffle_items}` | same |
| reproduction of stored LEAF-LGN | posthoc LEAF-LGN - stored: +0.73 [0.52], NB p 0.660; split-level re-runs match to 5 decimals | `stageB.txt` Section 6; `results/v11_probes/leaf_scores.txt` |
| off-protocol calibration (not a method) | random impressions of validation users as item calibration: +36.62 [1.79] over LEAF-LGN; prices the protocol | `results/v11_probes/probe_items.txt` line "OFF-PROTOCOL LEAF-LGN + validation-user uniform rate" (test 0.5093) |

### 5.5 Datasets

- **Primary: KuaiRand-27K** (temporal all-user setting; Pure random-exposure test with the 27K lifelong
  histories; statistics from `results/v9_probes/stats_pure_10k.txt` and `lifelong_27k_summary.txt`; the all-user
  counts must be regenerated `{TBD:results/v9_probes/stats_pure_all.txt users, items, impressions, test users}`).
- **Secondary: KuaiRand-1K** (1,000 users with long out-of-pool histories; same random insertion; low power).
  Results `{TBD:results/v11/c_1k.json LEAF-LGN+DAX, QP family, QP family+DAX, paired differences}`; statistics from
  `results/v9_probes/stats_1k.txt`. Note in the text that 1K users are a subset of 27K users (same platform,
  overlapping population): it is a replication at lower scale, not an independent dataset.
- **Why not Coat, Yahoo! R3, KuaiRec.** The method needs (a) impressions with click / no-click and a scenario
  field (DAX's examination factors) and (b) each user's history on items outside the test pool (LEAF). Coat and
  Yahoo! R3 have MNAR ratings with no record of what was shown and not rated and no out-of-pool history; KuaiRec has
  a fully observed small matrix and a large log of viewing events with watch ratios, but no history on items
  outside its matrix and no scenario field among the signals listed in our survey (`paper/v6/DESIGN.md` Section 1,
  rows 4, 6, 7; re-check the KuaiRec field list against its documentation before writing). Open Bandit has no persistent user ids.
  Running either component there would test a different method.
- **Saturation paragraph (motivation, one paragraph).** Our earlier diagnostics (`paper/v5/DESIGN.md` Section 3)
  showed that on KuaiRec a non-personalised leave-one-user-out item oracle is within a few thousandths of the best
  training-free prior, and that even low-rank models fitted on the unbiased matrix add almost nothing to item means;
  the classic benchmarks therefore cannot reward a method by more than about the minimum detectable effect. The
  KuaiRec values were produced by throw-away scripts and are not in a result file:
  `{TBD:re-run of paper/v5 diagnostics H1/H2 into results/v12/saturation_kuairec.txt: oracle nDCG@20, prior nDCG@20, H2 best gain}`.
  For KuaiRand the same diagnostics are in files: H1 LOO item oracle 0.5072 vs log prior 0.3884 vs random 0.3329;
  H2 item means 0.5850 vs rank-64 residual 0.5851 (`results/v9_probes/headroom_kuairand_pure.txt`). KuaiRand has
  headroom, and that headroom is item-level, which motivates DAX.

### 5.6 Baselines

LightGCN, r-AdjNorm (`zhao2022radjnorm`), iALS (Stage 1 only), EASE with the policy channel, QP family (Appendix A,
rule 0.4), and each with +DAX (EASE+DAX, QP family+DAX, LightGCN+DAX `{TBD:results/v11/dax_lightgcn.json}`).
State that NAVIP, AdvDrop and PPD are not re-run (no official code was run on this setting; they act on
propagation or embeddings and use no impressions or lifelong data) unless a run is added.

---

## 6. Results section plan (every table and figure, source and columns)

| ID | Content | Source file | Columns / series |
|---|---|---|---|
| T1 | Dataset statistics (27K primary, 1K secondary) | `results/v9_probes/stats_1k.txt`, `lifelong_27k_summary.txt`, `{TBD:stats_pure_all.txt}` | users, pool items, pre-cut in-pool impressions, out-of-pool impressions, test users, candidates per user, test click rate |
| T2 | Headroom diagnosis (diagnostic) | `results/v11_probes/diag_gap.txt`, `diag_after_dax.txt` | quantity, nDCG@10, difference to LEAF-LGN; rows popularity, LEAF-LGN, LEAF-LGN + 0.5 z(DAX) 0.4818, time-local oracles, LOO item oracle 0.5068, LEAF + oracle 0.5208; share of oracle gap closed by DAX 0.25 |
| T3 | **Main results** | `results/v11/stageB.txt` Section 1 | method, mean nDCG@10, sd over splits, n_configs, file; rows LEAF-LGN+DAX 0.4854, LEAF-LGN 0.4736, QP family 0.4634, QP family+DAX 0.4733, r-AdjNorm 0.4620, LightGCN 0.4576, EASE 0.4507, EASE+DAX 0.4623; seed rows `{TBD:c_seeds}` |
| T4 | **Primary family of paired tests** | `stageB.txt` Section 2 | comparison, paired diff x1e-3 [hw], t p, p_Holm, NB p, NB p_Holm, five split differences |
| T5 | DAX is backbone-agnostic | `stageB.txt` Sections 3 and 6; `{TBD:dax_lightgcn.json}` | backbone, without DAX, with DAX, paired diff [hw], NB p; EASE +11.98 [1.64]; QP family +10.28 [1.37] NB p 0.105; LEAF-LGN +11.96 [1.47]; LightGCN `{TBD}` |
| T6 | DAX ablations (gate B2) | `{TBD:results/v11/dax_ablate.json via stageB.txt Section 4}` | arm (full, no_tab, no_life_prior, no_tag_prior, additive, shuffle_items, shuffle_tags), mean, diff to LEAF-LGN [hw], share of the full DAX gain, p_Holm (ablation family) |
| T7 | LEAF ablations | `results/v10/stage1.txt` Section 5 (LEAF-tied); `{TBD:results/v10/leaf_lgn_ablate.json}` | arm, n_configs, mean, full minus arm [hw], p_Holm, NB p_Holm |
| T8 | Robustness (stage C) | `{TBD:results/v11/c_*.json}` | setting (KuaiRand-1K, concurrent, long-view label, seeds 1-3), LEAF-LGN+DAX - QP family, LEAF-LGN+DAX - (QP family+DAX) |
| T9 | Borrowed estimators (probe scale, appendix or Section 6.6) | `results/v11_probes/probe_items.txt`, `probe_pbm.txt`, `probe_dax_ablate.txt` | idea (field), closest reference, diff to LEAF-LGN [hw]; MRP +2.48, DAX tag prior +7.59, Fay-Herriot centre +5.69, MRP cells +7.29, trajectory -0.33, full grid +9.43 (copy each from the files, not from `paper/v7/DESIGN.md`) |
| T10 | Leakage checks and controls | Section 5.4 table | check, result, source |
| F1 | Protocol timeline: pre-cut log, lifelong log, T0, random insertions, user-level validation/test split | schematic (no numbers except T0) | - |
| F2 | Model diagram: LEAF tables -> LightGCN score; DAX EM on pre-cut log -> alpha -> standardised sum | schematic | - |
| F3 | Where the headroom is: share of recoverable gain by uniform-rate quintile and by pre-cut exposure quintile | `results/v11_probes/diag_gap.txt` | bars per quintile (0.22, 0.09, 0.04, 0.03, 0.83 and 0.19 ... 0.28; copy from file) |
| F4 | Estimated tab examination factors kappa_t | `results/v11_probes/probe_pbm.txt` lines "tab factors" (prior 80); for the main run `{TBD:export of kappa from drup.dax.fit_dax at the chosen configuration}` | kappa_t per tab, tab 1 = 1 |
| F5 | Per-split differences of LEAF-LGN+DAX vs each comparator | `stageB.txt` Section 2 split lists | dot plot, five dots per comparison, zero line |
| F6 | Validation and test nDCG@10 vs gamma at the chosen backbone epoch | `{TBD:extraction from results/v11/dax_lgn_post.json or a new curve file}` | gamma on x; mean +- sd over splits |
| F7 | alpha vs uniform item rate (diagnostic scatter, items with >= 50 random impressions) | `{TBD:new diagnostic output}`; Spearman values exist in `diag_after_dax.txt` (DAX 0.419, clicks 0.433) | hexbin; annotate both rho |

Narrative order of Section 6: 6.1 headroom (T2, F3); 6.2 main result (T3, T4, F5); 6.3 complementarity across
backbones (T5); 6.4 what in DAX matters (T6, T9, F4); 6.5 what in LEAF matters (T7); 6.6 robustness (T8); 6.7
controls (T10). Point out explicitly that DAX alone ranks worse than popularity-aware CF (item-only score far below,
`probe_items.txt` line "P3 PBM-EM alone", test 0.4160 vs popularity 0.4197) and that its Spearman with the uniform rate (0.419) is below raw clicks (0.433,
`diag_after_dax.txt`): its value is complementarity, not stand-alone accuracy.

Selected configurations sentence (from `results/v11/dax_lgn_post.json`, LEAF-LGN+DAX "chosen"): prior strength 80,
k_life 100, tab factor on, stratum cells prior_s 80 and gamma 0.5 in 5/5 splits; tag centre in 4/5 splits (content
centre in split 1); backbone lr 3e-3, wd 1e-4, w 0.25 or 0.5, epochs 14-27. Prior 80 (lowest of {80, 160}) and
k_life 100 (highest of {0, 100}) sit at grid edges (limitation 5).

---

## 7. Limitations (Section 7 of the paper)

1. **Single platform family.** KuaiRand-27K and KuaiRand-1K come from one platform, one period and overlapping
   users; no other public dataset found has random insertion, impressions and lifelong history together
   (`paper/v6/DESIGN.md` Section 1).
2. **Split-level inference with J = 5.** The NB test with five splits has low power and the split differences
   share training data; NB p_Holm values of 0.017-0.023 against the QP comparators are close to the threshold.
3. **Novelty is a combination.** Each component has a precedent (Section 3); the contribution is the
   identification and the controlled evidence.
4. **Post-hoc combination.** gamma is a validation-selected hyperparameter; the jointly trained variant is not
   reported (or `{TBD}`).
5. **Grid sizes and edges.** Search budgets differ by two orders of magnitude between LEAF-LGN+DAX (3,840) and the
   QP family (20); QP family+DAX (3,200) mitigates but restricts QP to its own chosen configurations; DAX prior
   strength and k_life were chosen at grid edges.
6. **DAX identifiability up to scale** (Section 4.6); theta and kappa are not interpretable in absolute terms.
7. **Single training seed per split** in the main table until `{TBD:c_seeds}`.
8. **No lifelong-aware deep baseline** (SIM/TWIN/DIN-style) was run.
9. **Click label only** in the main table; long view `{TBD:c_longview}`.
10. **Remaining headroom.** DAX closes 25% of the oracle gap; 37.62e-3 of item headroom remains above LEAF+DAX
    (`diag_after_dax.txt`).
11. **Content is coarse.** KuaiRand ships no titles, covers or frames; the tag is the main content key.
12. **Test scenario is tab 1**; the tab factor anchors at kappa_1 = 1, so transfer to other scenarios is untested.
13. **Item-level vs tag-level** part of DAX pending B2 (`dax_ablate.json`).

---

## 8. Venue shortlist (method paper)

Checked on 2026-11-09 with web search; publisher author-guideline pages of ACM DL and ScienceDirect returned HTTP 403
from this environment, so journal-specific limits below are **not verified** and must be read on the official pages
before formatting.

| Venue | Type | Fit | What it requires (verified / not) | Notes |
|---|---|---|---|---|
| ACM TORS | journal | best topical fit (recommender methods with evaluation methodology) | follow the TORS author guidelines on ACM DL (could not be fetched; page limit not verified); ACM template; extended conference versions follow ACM's prior-publication policy | earlier notes say an article processing charge applies (`paper/v2/SUBMISSION_NOTES.md`; not re-verified). Strong evaluation culture: the protocol and controls are an asset |
| ACM TOIS | journal | good (IR-flavoured: click models and examination are core TOIS topics) | ACM template; ACM venues commonly require >= 25% new material for conference extensions (seen for TOMM, not verified for TOIS) | DAX's click-model lineage helps; expect a request for more datasets |
| Information Processing & Management | journal (Elsevier, elsarticle) | fits | Elsevier guide could not be fetched; highlights and data-availability statement are standard Elsevier items (not verified for IPM) | a different manuscript (DRUP) was rejected at IPM (`paper/v2/SUBMISSION_NOTES.md`); a new method is admissible but the cover letter must state the difference |
| Knowledge-Based Systems | journal (Elsevier, elsarticle) | acceptable | Elsevier guide could not be fetched (not verified) | was the named fallback in earlier notes |
| Information Sciences | journal (Elsevier) | - | - | an earlier, different manuscript was desk-rejected for "lack of sufficient novelty" (2026-11-04); not recommended for this paper given that its novelty is a combination |
| ACM RecSys 2027 (main track) | conference | very good | RecSys 2026 rules (verified on recsys.acm.org/recsys26/call): long papers 8 pages including figures, tables, appendices; ACM double-column `sigconf,anonymous`; mutually anonymous, self-citation in third person; code sharing encouraged, not required. 2027 deadline not published (RecSys 2026 had 21 April 2026) | needs heavy compression of the outline (Sections 4.6, 4.7, T9 to supplementary) |
| SIGIR 2027 (full papers) | conference | good (click-model lineage) | deadline 21 January 2027 AoE, abstract 14 January (third-party trackers; official page not read) | anonymity; ACM format |
| WSDM 2027 | conference | good | closed (full papers 24 August 2026, official call); WSDM 2028 later | - |
| KDD 2027 (research track) | conference | moderate | cycle 1 closed (26 July 2026); a February 2027 cycle 2 is reported by secondary sources only | applied data science track is an option if framed around Kuaishou-style feeds |

Recommendation: ACM TORS (journal, full outline as is) or RecSys 2027 (if the conference route is preferred,
waiting for B2, seeds and KuaiRand-1K first). For any venue, submit only after B2 and seeds are in files; a reviewer
will ask for them first.

---

## 9. Paper skeleton (elsarticle)

1. Introduction (problem: uniform-insertion target vs policy log; headroom finding; two signals; contributions
   C1-C4 with one number each; what we do not claim, one sentence)
2. Related work (Section 3 of this outline)
3. Method (Section 4)
4. Experimental protocol (Section 5)
5. Results (Section 6)
6. Discussion (why multiplicative beats additive; why drift is not the lever; what the remaining headroom would need)
7. Limitations (Section 7)
8. Conclusion
- Appendix A: QP family description and grids (rule 0.4)
- Appendix B: EM derivation and convergence trace
- Appendix C: probe-scale borrowed estimators (T9)
- Appendix D: hyperparameter grids and chosen configurations per split (from the JSON "chosen" fields)
- Data and code availability statement; no author-contribution statement mentioning tools.
