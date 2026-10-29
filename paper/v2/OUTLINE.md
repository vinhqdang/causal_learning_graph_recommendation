# Manuscript v2: design document

Status: design only. Nothing here has been typeset. Every number below is copied from a file in this
repository, and its source is given. If a number is not in this document and not in a listed file,
it must not appear in the paper. Entries marked TODO need an action (new analysis script, new run or
bibliographic check) before the text that depends on them is written.

Target: a full-length research article, journal-agnostic, to be typeset later with `elsarticle`
(candidate venues: Information Sciences, Knowledge-Based Systems). Main text length about
8,500-9,500 words without references, 6 tables and 4-5 figures in the main text; the rest goes into
the Supplementary Material.

---------------------------------------------------------------------------------------------------

## 0. Ground rules for the writer

1. **Storyline (do not deviate).**
   (i) Exposure-aware negative weighting plus a fixed propensity-imputation prior offset is a simple
   way to correct graph collaborative filtering (CF) for exposure bias. The prior is fitted on the
   biased log only, before training, so no uniformly collected data enter training.
   (ii) The method beats every graph baseline (LightGCN, r-AdjNorm, NAVIP, DR-LightGCN, SimGCL,
   AdvInfoNCE re-implementation) on all four datasets and is the best method on Yahoo! R3 and
   KuaiRand. It does **not** beat iALS on Coat, and it does **not** beat DR-JL / MRDR or the
   training-free DRUP operator on KuaiRec.
   (iii) Ablations with a shuffled-Q control show where the gain comes from.
   (iv) DRUP's idempotent walk correction is a secondary component. It gives no real-data accuracy
   benefit (matched configurations), only under oracle semi-synthetic conditions. Training-free DR
   propagation (DRUP) is the best method on KuaiRec.
2. **Forbidden wording:** "state of the art", "SOTA", "significantly outperforms all baselines",
   "novel framework", "first to" (unless the gap statement in Section 9 has been verified), and any
   claim of unbiasedness for the trained method. Say "consistently better than the graph baselines",
   "best among the methods we ran", "on par with".
3. **Significance language:** "significant" only with p_Holm < 0.05 from the files listed. Always
   give the paired difference (x1e-3) and its 95% CI half-width in brackets, as in the summary files.
   The paired difference is a per-user mean difference, not the difference of the two table means.
   Copy it from the files, do not subtract.
4. **Main run vs seed runs.** The main table is the "seed 0" run with full grid search and per-split
   selection (`results/v4/final_summary.txt`). The robustness check retrains the selected
   configurations with seeds 1-3 (`results/v4/seeds_summary.txt`). When the two disagree (Coat vs
   iALS; KuaiRec vs DR-JL/MRDR), the text must report both and base the claim on the weaker one.
5. **Naming.** In the paper the method is called **QP-LightGCN** (Q-weighted negatives + Prior
   offset). The writer may rename it, but the same name must be used everywhere. Code-name mapping:

   | Paper name | Code name / file | What it is |
   |---|---|---|
   | QP-LightGCN (final method) | "DR-BPR-LGN + offset", per-split combination of jobs in `experiments/final_config.py` FINAL_JOBS | LightGCN + Q-weighted BPR + offset, beta in {0, 0.5, 1, 2} |
   | Q-LightGCN (no offset) | `DR-BPR-LGN`, `DR-BPR-LGN-1` | beta = 0 only |
   | P-LightGCN (offset only) | `LightGCN-p` (`results/v4/learned_coat_prior_bpr.json`) | plain BPR + offset, beta in {0.5, 1} |
   | Q(Yhat)-LightGCN | `DR-BPR-yhat` | Q replaced by Yhat (no propensity) |
   | Q(pop)-LightGCN | `DR-BPR-pop` | Q replaced by rescaled item popularity |
   | Q(shuffled)-LightGCN | `DR-BPR-shuf` | Q values randomly permuted (negative control) |
   | QP-LightGCN, low-rank imputation | `results/v4/learned_coat_lrQ*.json` | Q and offset from rank-16 low-rank imputation |
   | Q-iALS | `DR-iALS` | iALS whose zero-entry confidence is 1 - fn*Q |
   | DRUP | `DRUP` in `paper/tables/accuracy.tex`, `results/v4/log_new_coat.txt` | training-free DR propagation with walk correction |
   | DR adjacency | `DR` | the same operator without the walk correction |
   | AdvInfoNCE (re-impl.) | `AdvInfoNCE-LGN` | simplified re-implementation, see Section 5.4 |

6. Do not mention how the manuscript or code was produced. No acknowledgement of tools.

---------------------------------------------------------------------------------------------------

## 1. Title

**Proposed:** *Which negatives are really negative? Exposure-aware negative weighting and an
imputation prior for graph collaborative filtering under exposure bias*

Alternatives:
1. *Propensity-imputation priors for LightGCN on exposure-biased logs: negative weighting, score
   offsets and an evaluation on four unbiased-test benchmarks*
2. *Correcting graph collaborative filtering for exposure bias at the negative-sampling step*
3. *Doubly robust ingredients for graph recommenders: what helps, what does not, and when*

(The proposed title states the mechanism; alternative 3 is the most modest and suits a reviewer
who expects an evaluation paper.)

---------------------------------------------------------------------------------------------------

## 2. Abstract draft (199 words)

> Graph recommenders such as LightGCN learn from logs in which an item can be chosen only if shown,
> so most sampled negatives are unexposed items whose preference is unknown.
> We propose a simple correction that uses no uniformly collected data for training. An
> inverse-propensity-weighted additive imputation and an exposure propensity give, in closed form, the
> probability Q that a pair absent from the log is a positive. Each sampled BPR negative is
> down-weighted by 1 - Q, and the standardised logit of the imputation is added to the score as a
> fixed offset whose weight is chosen on validation data. On Coat, Yahoo! R3, KuaiRand-Pure and
> KuaiRec, with uniform-exposure data used only for validation and testing and per-user
> Holm-corrected tests, the method beats all graph baselines, including a re-implemented
> AdvInfoNCE, and is the best method we ran on Yahoo! R3 (nDCG@5 0.6866 vs 0.6742) and KuaiRand
> (nDCG@10 0.4613 vs 0.4590). Over three retraining seeds it is on par with iALS on Coat and with
> DR-JL and MRDR on KuaiRec, where training-free doubly robust propagation remains best (0.6378).
> Ablations with a shuffled-Q control show that the gain requires an informative Q and that the
> offset dominates on Coat and KuaiRec.

Sources: Yahoo/KuaiRand means `results/v4/final_summary.txt` lines 12, 14, 20, 22; KuaiRec DRUP
0.6378 `paper/tables/accuracy.tex` (DRUP row) and `results/v4/ensemble_summary.txt` line 41
("DRUP-repro test 0.6378"); seeds `results/v4/seeds_summary.txt` lines 10, 40-41; ablations Section
6.3 below. Before submission, check the word count against the journal limit (IS: 250 max; KBS:
check the guide for authors, TODO verify).

Keywords (6): recommender systems; exposure bias; graph collaborative filtering; doubly robust
estimation; negative sampling; missing-not-at-random feedback.

Highlights (for elsarticle, max 85 characters each):
- Unexposed items are weighted by their posterior probability of being positive.
- A fixed imputation prior is added to LightGCN scores; its weight is validated.
- Better than all graph baselines on four benchmarks with unbiased test data.
- On par with iALS (Coat) and DR-JL/MRDR (KuaiRec) across retraining seeds.
- A shuffled-Q control shows the gain needs an informative exposure model.

---------------------------------------------------------------------------------------------------

## 3. Contributions (to appear at the end of the Introduction)

1. **Method.** A closed-form posterior Q_ui = Yhat_ui (1 - p_ui) / (1 - p_ui Yhat_ui), the
   probability that a pair that is not a logged positive is a positive, obtained from an
   IPS-fitted additive imputation and an exposure propensity. It is used twice inside an
   otherwise unchanged LightGCN: to down-weight sampled BPR negatives by 1 - Q, and as a fixed
   standardised offset beta * z(logit Yhat) added to the score. Training uses the biased log only.
   The method adds one selected hyper-parameter (beta in {0, 0.5, 1, 2}) and costs one extra
   lookup per sampled negative.
   *Supported by:* `drup/learned.py` lines 290-302 (Q and ablation modes), 404-415 (weighted BPR), 25/33/38 (offset);
   `experiments/run_learned.py` lines 151-160 (nuisances, offset standardisation).
2. **Evaluation.** A comparison on four public benchmarks with an unbiased test set (Coat,
   Yahoo! R3, KuaiRand-Pure, KuaiRec), uniform data only for per-split validation and testing,
   per-user paired t-tests with a Holm correction within each dataset, and retraining with three
   additional seeds. The method is better than all six graph baselines on all four datasets, best
   overall on Yahoo! R3 and KuaiRand, on par with iALS on Coat and with DR-JL/MRDR on KuaiRec, and
   below training-free DRUP on KuaiRec.
   *Supported by:* `results/v4/final_summary.txt`, `results/v4/seeds_summary.txt`.
3. **Mechanism.** Ablations that replace Q by the imputation alone, by item popularity or by a
   random permutation of itself show that the weighting needs an informative Q (the shuffled
   control returns to LightGCN level), that the propensity term in Q matters little on Coat, that
   the additive imputation is better than a low-rank one, and that the relative contribution of
   weighting and offset depends on the dataset (offset on Coat and KuaiRec, weighting on Yahoo! R3
   and KuaiRand).
   *Supported by:* Section 6.3 sources.
4. **A negative result on walk correction, kept in proportion.** The idempotent walk correction of
   DRUP, which removes the repeated-edge bias of multi-hop DR propagation, does not improve
   accuracy on real data at matched configurations (Coat, Yahoo! R3), and improves nDCG only in a
   semi-synthetic setting with true propensities and fixed degree weights. Training-free DR
   propagation is nevertheless the strongest method on KuaiRec, and a validated score ensemble with
   it adds nothing beyond it.
   *Supported by:* `results/v3/matched_config_real.txt`, `experiments/rank_dose_table.py` output,
   `results/v4/ensemble_summary.txt`.

---------------------------------------------------------------------------------------------------

## 4. Section-by-section outline

Word targets are for the main text. "Reuse" means the text can be adapted (shortened, re-framed);
it must not be pasted unchanged, because the old manuscript was framed around DRUP.

### 1 Introduction (~1,100 words)

Paragraph plan:
1. *Context (150).* Graph CF (LightGCN) is a standard backbone; it learns from implicit logs in
   which a positive needs exposure. Reuse `paper/sections/intro_related.tex` lines 5-10 (logged
   graph is not the preference graph). Cite he2020lightgcn, marlin2009collaborative,
   chen2023bias.
2. *The specific problem (200).* BPR samples negatives uniformly from non-logged items. Under
   exposure bias most of them were never shown, and some would have been liked: these are false
   negatives, and they concentrate on rarely exposed items. Point to liang2016exposure (ExpoMF:
   exposure as a latent variable) and saito2020unbiased (Rel-MF) as the matrix-factorisation
   precedents. Loss-level debiasing (IPS/DR: schnabel2016, wang2019drjl, guo2021mrdr) is pointwise
   and does not combine naturally with the pairwise, sampled losses used to train graph models; on
   our data DR-LightGCN (pointwise DR loss on LightGCN) is below plain BPR-LightGCN on Yahoo! R3
   (0.5528 vs 0.6708, `results/v4/dr_bpr_summary.txt` line 7 and `paper/tables/accuracy.tex`).
3. *Idea (200).* Use the two nuisances that DR estimation already fits (propensity p, imputation
   Yhat) to compute Q, the posterior probability that a non-logged pair is positive, by Bayes'
   rule. Down-weight each sampled negative by 1 - Q; add the imputation as a fixed prior offset.
   No uniform data in training; nuisances fitted once on the log.
4. *Why an evaluation-heavy paper (200).* Progress claims in top-N recommendation often fail
   against tuned simple baselines (dacrema2019progress, rendle2019baselines, anelli2023myth). Many
   reported debiasing gains use protocols that train on uniform data or test on biased data. We
   therefore use uniform data only for validation and testing, select per split, test per user with
   Holm correction, and retrain with extra seeds. State the results honestly here: best on two
   datasets, on par on two, below a training-free operator on one.
5. *Walk correction (150).* The authors' earlier operator, DRUP, corrects multi-hop DR propagation
   for repeated edges. We include it as a baseline and component and report that the correction
   itself does not improve accuracy on real data. Do not describe the rejected submission.
6. *Contributions (150).* The four bullets of Section 3 above, shortened.
7. *Outline (50).*

### 2 Related work (~1,300 words)

Use one paragraph per subsection; close each with one sentence that positions this work. A summary
table (Table 1, see Section 5) replaces the old `tab:related`.

- **2.1 Graph collaborative filtering (200).** LightGCN, NGCF, UltraGCN, SGL, SimGCL; closed-form
  EASE/GF-CF/BSPM; tuned baselines (iALS) remain strong. Reuse `intro_related.tex` lines 97-111
  (drop the DRUP sentences). Keys: he2020lightgcn, wang2019ngcf, mao2021ultragcn, wu2021sgl,
  yu2022simgcl, steck2019ease, shen2021gfcf, choi2023bspm, hu2008ials, anelli2023myth,
  wu2022gnnsurvey. New: Rendle et al. 2022 iALS revisited (TODO-C9).
- **2.2 Debiasing under missing-not-at-random feedback (300).** IPS, SNIPS, DR, DR-JL, MRDR, GDR,
  StableDR, CVIB, AutoDebias/CausE (need uniform data in training), MACR, PDA, DICE. Reuse
  `intro_related.tex` lines 113-129. Keys: marlin2009collaborative, schnabel2016,
  swaminathan2015snips, saito2020unbiased, wang2019drjl, guo2021mrdr, dai2022gdr, li2023stabledr,
  wang2020cvib, chen2021autodebias, bonner2018cause, wei2021macr, zhang2021pda, zheng2021dice,
  chen2023bias, gao2024causalsurvey, boratto2021connecting, okamura2024flexibly, bai2025invariant.
  New: Steck 2010 (TODO-C8).
- **2.3 False negatives and negative weighting in implicit feedback (300).** This is the closest
  literature and was missing in the old manuscript. ExpoMF (liang2016exposure) infers exposure
  posteriors for unclicked items in MF; Rel-MF (saito2020unbiased) and unbiased pairwise learning
  (TODO-C4) re-weight by propensity; SRNS (TODO-C5) and debiased contrastive learning (TODO-C6)
  treat false negatives in sampling; BC loss (TODO-C7) and AdvInfoNCE (TODO-C3) re-weight or
  adversarially weight negatives in contrastive CF. Position: Q is the posterior of ExpoMF type,
  but computed in closed form from DR nuisances, applied to a graph model with a pairwise loss, and
  combined with a prior offset.
- **2.4 Debiasing inside graph models (250).** NAVIP (kim2022navip), r-AdjNorm (zhao2022radjnorm),
  popularity amplification by graph convolution (chen2024gcnpop), APDA (zhou2023apda), CAGED
  (que2025caged), DPAA / debiased message passing (dpaa2026). Reuse `intro_related.tex` lines
  131-142, drop the theorem reference. Position: these change the adjacency or aggregation; we
  change which pairs count as negatives and add a prior, and keep LightGCN's propagation unchanged.
- **2.5 Score offsets and priors (100).** The fixed offset resembles logit adjustment for long-tail
  classification (TODO-C10) and a GLM offset; here the offset is a debiased population-level
  preference estimate rather than a class prior.
- **2.6 Evaluation protocols and the gap (150).** See Section 9 for the gap statement; cite
  dacrema2019progress, rendle2019baselines, anelli2023myth, nadeau2003, and Krichene and Rendle
  2020 on sampled metrics (TODO-C11).

### 3 Problem setting and notation (~600 words)

- Users u in [m], items i in [n], potential outcome Y_ui in {0,1}, exposure/observation O_ui,
  log reveals O and O*Y. Reuse `paper/sections/method_theory.tex` lines 5-19 (setting and the
  remark that on Coat/Yahoo "exposure" is self-selection). Keep Assumption 1 (conditionally
  independent exposure) only as the modelling assumption behind Q; **drop** Assumption 2 and
  Remark `rem:xfit` (they belong to DRUP's theory, moved to the supplement).
- Define a "logged positive" O_ui Y_ui = 1 and the non-logged set N_u = {i : O_ui Y_ui = 0}, which
  is what BPR samples from (`drup/learned.py` lines 405-407: negatives are resampled once if they
  hit a logged positive).
- State the evaluation target: ranking of each user's unexposed test candidates (items the user
  never interacted with in the log), measured on uniformly exposed data.
- Edge estimates W^IPS, W^DR are needed only for the DRUP baseline: reuse `method_theory.tex`
  lines 74-80 in Section 4.6.

### 4 Method (~1,600 words)

Figure 1 (method schematic) sits here.

- **4.1 Nuisances (300).**
  - Propensity: Coat uses the propensities shipped with the dataset ("given"); Yahoo! R3, KuaiRand
    and KuaiRec use a logistic exposure model p_ui = sigmoid(a + b logit(r_u) + c logit(r_i)) on
    user and item exposure rates, fitted by Newton's method (`drup/estimation.py` lines 20-30,
    219-231; `experiments/final_config.py` line 12 PROP). Propensities are clipped below at 0.05
    (`experiments/run_learned.py` lines 113, 151-152).
  - Imputation: additive model mu + b_u + b_i fitted by IPS-weighted alternating ridge
    (lambda = 5, 20 iterations), clipped to [0,1] (`drup/estimation.py` lines 97-113;
    `run_learned.py` line 157). Fitted once on the whole log, not cross-fitted (state this).
  - Note honestly: the Coat propensities were estimated by the dataset authors with the uniform
    sample (schnabel2016) that we also use for validation and testing (old
    `paper/sections/discussion.tex` lines 68-70). This is a possible leak for Coat only.
- **4.2 Posterior of a non-logged pair (350).** Derivation in three lines: with O independent of Y
  given the covariates of p, P(Y=1 | OY=0) = P(Y=1, O=0) / P(OY=0) = Yhat(1-p) / (1 - p Yhat).
  Properties to state: Q in [0, Yhat]; Q -> Yhat as p -> 0 (an unexposed pair keeps its prior);
  Q -> 0 as p -> 1 with Yhat < 1 (an exposed but unchosen pair is a true negative). Code:
  `drup/learned.py` line 293, clipped to [0,1]; ablation modes lines 294-302. Mention the relation to ExpoMF's exposure
  posterior. Do not claim unbiasedness of the resulting training loss.
- **4.3 Q-weighted BPR (300).** For a logged positive (u,i) and one sampled negative j,
  L = - sum w_uj log sigma(f(u,i) - f(u,j)) / mean(w), w_uj = max(1 - fn Q_uj, 1e-3)
  (`drup/learned.py` lines 404-415; normalisation by the batch mean of w keeps the effective
  learning rate). fn = 1 in all final jobs except the Coat job `DR-BPR-LGN`, whose grid is
  fn in {0.5, 1} (`experiments/run_learned.py` lines 46-50). L2 regularisation on the batch
  embeddings as in LightGCN. f is LightGCN with 2 layers (Coat: layers/dims searched; see
  Supplementary grid table).
- **4.4 Imputation prior offset (250).** s(u,i) = f_theta(u,i) + beta z(logit Yhat_ui), where z
  standardises over all pairs (mean 0, sd 1) and the logit uses Yhat clipped to [1e-3, 1-1e-3]
  (`run_learned.py` lines 158-159; `drup/learned.py` lines 33, 38). The offset is present during
  training and at inference, so f learns a residual on top of the prior. beta is selected on
  validation per split from {0, 0.5, 1, 2}: jobs with beta = 0, beta in {0.5, 1}, beta in {1, 2}
  are trained separately and each split keeps the job with the best validation score
  (`experiments/final_analysis.py` lines 12-18). Report the chosen beta per dataset (Table 2
  footnote): Coat {1.0: 5, 2.0: 2, 0.5: 1, 0.0: 2}; Yahoo {1.0: 4, 0.0: 1}; KuaiRand
  {0.0: 4, 0.5: 1}; KuaiRec {1.0: 4, 2.0: 1} (`results/v4/final_summary.txt` lines 2, 12, 20, 29).
  (KuaiRec has 5 splits; the line prints 5 choices.)
- **4.5 Algorithm and cost (150).** Algorithm 1: fit p, Yhat; compute Q and z(logit Yhat) once;
  train LightGCN with Q-weighted BPR and offset; select beta and epoch per split on validation.
  Cost: Q and the offset are dense m x n matrices in the current implementation (memory O(mn)),
  which limits it to the catalogue sizes used here (largest: KuaiRec 7,176 x 10,728); the
  per-step overhead is one lookup per negative. Say that Q can be computed on the fly from
  (mu, b_u, b_i) and p for large catalogues (not implemented; flag as future work).
- **4.6 Training-free DR propagation and the walk correction (DRUP) (250).** Short description
  only: DR edge estimates, three-hop propagation, idempotent replacement of repeated factors.
  Reuse `method_theory.tex` lines 74-92 and Eq. (eq:drup) at lines 131-142; Figure `fig:walks`
  (lines 94-129) goes to the supplement. One sentence: under fixed nuisances the correction makes
  the three-hop operator unbiased for propagation over the full-exposure graph (old Theorem
  `thm:unbiased`), proved in the supplement. State that we use DRUP (a) as a baseline, (b) in a
  validated score ensemble (Section 6.5).

### 5 Experimental setup (~1,300 words)

- **5.1 Datasets (350).** Reuse `paper/sections/exp_setup.tex` lines 7-54 and
  `paper/tables/datasets.tex` (Table 3, unchanged). Positives: rating >= 4 (Coat, Yahoo), click
  (KuaiRand), watch ratio >= 2 (KuaiRec). Keep the random-ranking levels (lines 48-51). Drop the
  sentence on the outcome-dependent propensity model for Yahoo (lines 30-33) unless it also applies
  to the trained runs (it does not: PROP = pop, `final_config.py` line 12; rewrite as "a logistic
  exposure model").
- **5.2 Baselines (350).** Only methods that appear in the final analysis or the main table:
  LightGCN (BPR), LightGCN-pt (pointwise, KuaiRec), r-AdjNorm, NAVIP, DR-LightGCN (pointwise DR
  loss with the same fixed imputation), SimGCL, AdvInfoNCE (re-impl.), BPR-MF, MACR, iALS, DR-JL,
  MRDR, DRUP and DR adjacency. Reuse the descriptions in `drup/learned.py` lines 1-5, 87-135 and
  the docstring of `train` (lines 259-274). Justify omissions (StableDR, DICE, CausE, AutoDebias)
  with `exp_setup.tex` lines 77-85. **Delete** the LLM paragraph (lines 87-94) or move to
  limitations in one sentence.
- **5.3 Protocol (400).** Reuse `exp_setup.tex` lines 98-117, rewritten:
  - Uniform data split into validation (30% of each user's candidates on Coat; 30% of test users
    otherwise) and test, 10 random splits on Coat, 5 otherwise (`drup/data.py` lines 109-129;
    `run_learned.py` line 110 `--frac_val 0.3`).
  - Every configuration is trained once on the log; after every epoch it is evaluated on every
    split; each split selects its own (configuration, epoch) by validation nDCG with its own early
    stopping (patience 5 by default) (`run_learned.py` docstring lines 1-8 and lines 172-176).
  - Metrics: nDCG@5 (Coat, Yahoo), nDCG@10 (KuaiRand), nDCG@20 (KuaiRec) (`final_config.py`
    line 15). Report mean +- sd (ddof = 0) over splits.
  - Tests: per user, average the user's test nDCG over the splits in which the user is a test user,
    pair with the baseline, one-sample t-test on the differences, 95% CI half-width,
    Holm-adjusted within dataset over all baselines listed for that dataset
    (`experiments/final_analysis.py` lines 21-39). Holm family sizes: Coat 8, Yahoo 6, KuaiRand 7,
    KuaiRec 9 (`final_config.py` SETS). Users in the tests: Coat 225, Yahoo 2,310, KuaiRand 8,321,
    KuaiRec 1,410 (computed from the per-user entries of the final-method files with
    `final_config.FINAL_JOBS`; Yahoo and KuaiRec also printed in `results/v4/dr_bpr_summary.txt`
    lines 3, 11). Note that splits overlap, so these tests treat the log and fitted nuisances as
    fixed (nadeau2003); keep that caveat sentence from `exp_setup.tex` lines 110-113.
  - Seeds: the selected configuration of each split is retrained with seeds 1, 2, 3; epochs
    re-selected on validation; compared against the two strongest baselines per dataset, Holm over
    2 tests (`experiments/run_seeds.py` docstring lines 1-10; `results/v4/seeds_summary.txt`).
  - Tuning budgets: report configurations searched per method (Table S-grid). Final method: Coat
    324, Yahoo 84, KuaiRand 30, KuaiRec 120 configurations in total over its jobs (sum of
    `n_configs` of the FINAL_JOBS files; per job: Coat 108+108+108; Yahoo 12+12+12+24+24;
    KuaiRand 6+12+12; KuaiRec 12+12+24+24+24+24). Baseline counts are in
    `results/v4/final_summary.txt` (the "cfgs" column).
- **5.4 AdvInfoNCE re-implementation (200).** Must be disclosed in the main text, not only in the
  supplement. Facts (`drup/learned.py` lines 75-84, 277-281, 312-353 (adversary rounds from line 330), 392-396, 484;
  `experiments/queue_v5.sh` lines 19-28):
  - LightGCN encoder with cosine scores / tau; per-negative hardness weights from a separate
    adversary embedding (Pu, Pi, dim 64) trained by Adam to maximise the InfoNCE loss; 64 negatives.
  - Simplifications vs. the original: the adversary budget eta is implicit (number of adversarial
    rounds, eta in {4, 12}, one round = max(one pass over positives, 100 steps)); no KL-ball
    projection (KL is only logged); embedding-based adversary only (no other variants).
  - Grids differ: 216 configurations on Coat (lr 3 x wd 3 x dims {32, 64} x layers {1,2,3} x
    tau {0.2, 0.75} x eta {4, 12}), 16 on Yahoo, 8 on KuaiRand, 8 on KuaiRec
    (`results/v4/final_summary.txt` "cfgs" column).
  - State plainly that our numbers are for this re-implementation and are not those of the
    original paper.

### 6 Results (~2,400 words)

- **6.1 Main comparison (700).** Table 2. Key points, in this order:
  1. Graph baselines: the method is better than LightGCN, r-AdjNorm, NAVIP, DR-LightGCN, SimGCL
     and AdvInfoNCE on all four datasets, p_Holm <= 0.0185 in every graph comparison
     (largest p: KuaiRand vs r-AdjNorm 0.0185; Coat vs DR-LightGCN 0.0116; Coat vs r-AdjNorm
     0.0083) (`final_summary.txt` lines 3-10, 13-18, 21-27, 33-38). Exception to state:
     LightGCN-pt on KuaiRec, +1.31e-3 [3.34], p_Holm 0.4409 (line 32): not significant.
  2. Yahoo! R3: 0.6866 vs r-AdjNorm 0.6742, +12.67e-3 [6.01], p_Holm < 1e-4; iALS 0.6625.
  3. KuaiRand: 0.4613 vs r-AdjNorm 0.4590, +1.94e-3 [1.61], p_Holm 0.0185. Small effect: say so.
     beta = 0 chosen in 4/5 splits, i.e. the gain is from the weighting. Note also that the
     beta = 0 job alone has test 0.4623 (`results/v4/learned_kuairand_drbpr.json`), higher than
     the validated combination; the validation choice is not always the test-best.
  4. Coat: 0.6015 vs iALS 0.5812, +20.02e-3 [19.74], p_Holm 0.0469 in this run; but see 6.2.
  5. KuaiRec: 0.6285, below DR-JL 0.6332 (-4.56e-3 [2.24], p_Holm 0.0002) and MRDR 0.6328
     (-4.34e-3 [2.16], p_Holm 0.0002), and below training-free DRUP 0.6378 / DR adjacency
     0.6378 (`paper/tables/accuracy.tex`). Above all graph baselines and iALS.
  6. One sentence on scale: effects are 0.002-0.033 nDCG against the best graph baseline per
     dataset (paired diffs vs r-AdjNorm/DR-LightGCN: Coat +28.27e-3, Yahoo +12.67e-3, KuaiRand
     +1.94e-3, KuaiRec vs DR-LightGCN +7.73e-3).
- **6.2 Robustness to the training seed (400).** Table 4. Coat: seeds 1-3 mean 0.5911 +- 0.0049
  vs iALS 0.5862 +- 0.0029, +3.39e-3 [20.70], p_Holm 0.7470: not significant, so the Coat claim
  is "on par with iALS". DR-LightGCN still beaten (+33.36e-3, p_Holm 0.0004). Yahoo: 0.6881 +-
  0.0005 vs r-AdjNorm 0.6736, +14.76e-3 [5.23], p_Holm < 1e-4. KuaiRand: 0.4604 vs 0.4580,
  +2.31e-3 [1.23], p_Holm 0.0002. KuaiRec: 0.6290 vs DR-JL 0.6298 (-1.03e-3 [1.64], p_Holm
  0.2213) and MRDR 0.6304 (-1.56e-3 [1.47], p_Holm 0.0760): the significant loss of the main run
  becomes a non-significant gap. Emphasise that the method's seed spread on Coat (sd 0.0049,
  ddof 0) is larger than iALS's (0.0029); seed-0 result for the final method on Coat (0.6015) is
  the highest of the four seeds. Source: `results/v4/seeds_summary.txt` lines 3-41.
- **6.3 Where does the gain come from? (700).** Table 5 + Figure 3.
  - Weight ablation, Coat (54 configurations each, fn = 1, beta = 0): full Q-LightGCN 0.5912
    (`learned_coat_drbpr.json`, 108 cfgs with fn in {0.5,1}: flag the grid difference),
    Q(Yhat) 0.5937, Q(pop) 0.5751, Q(shuffled) 0.5689, LightGCN 0.5683
    (`results/v4/log_learned_coat_abl.txt` lines 60, 117, 174; `log_learned_coat_drbpr.txt`
    line 114). **Note:** `results/v4/SUMMARY.md` line 18 says 0.5770 for Coat popularity-only,
    but the stored result file `learned_coat_abl_DR-BPR-pop.json` and the log give 0.5751. Use
    0.5751.
  - Weight ablation, Yahoo (lr 3e-3, 12 configurations): Q-LightGCN 0.6829
    (`learned_yahoo_drbpr_lr3e-3.json`), Q(pop) 0.6550, Q(shuffled) 0.6689, LightGCN 0.6708.
    Q(Yhat) not run on Yahoo.
  - Reading: a shuffled Q, which has the same marginal distribution, removes the gain (Coat 0.5689
    vs LightGCN 0.5683; Yahoo 0.6689 vs 0.6708), so the gain is not a re-scaling effect;
    popularity-only weighting is weaker than Q (and on Yahoo below LightGCN); on Coat, dropping
    the propensity from Q (Q = Yhat) changes nothing (0.5937 vs 0.5912), so the information is in
    the imputation.
  - Offset ablation, Coat (108 configurations each): LightGCN 0.5683; P-LightGCN (offset only,
    beta in {0.5,1}) 0.6020 (`learned_coat_prior_bpr.json`); Q-LightGCN 0.5912; QP-LightGCN with
    beta in {0.5,1} 0.6039 (`learned_coat_prior_dr.json`), beta in {1,2} 0.6046
    (`learned_coat_p2.json`); validated combination 0.6015. Reading: on Coat the offset is the
    main ingredient and the weighting adds little on top of it (0.6020 vs 0.6039).
  - Per-dataset decomposition (no new runs needed): Q-LightGCN (beta = 0) vs QP-LightGCN:
    Yahoo 0.6837 -> 0.6866 (`dr_bpr_summary.txt` line 2; `final_summary.txt` line 12);
    KuaiRand 0.4623 (beta=0 job) and the validated combination 0.4613 with beta = 0 in 4/5 splits;
    KuaiRec 0.6095 -> 0.6285 (`dr_bpr_summary.txt` line 10). Reading: weighting carries Yahoo
    and KuaiRand, the offset carries Coat and KuaiRec. On KuaiRec, Q-LightGCN without offset is
    not significantly better than LightGCN (+1.39e-3, p_Holm 0.0967) and is below NAVIP,
    DR-LightGCN, iALS (`dr_bpr_summary.txt` lines 11-18).
  - Imputation model, Coat: low-rank (rank 16) imputation for Q and offset gives 0.5887 (no
    offset, 54 cfgs) and 0.5897 (with offset, 108 cfgs) vs additive 0.5912 / 0.6039
    (`learned_coat_lrQ.json`, `learned_coat_lrQ_prior.json`). The richer imputation is worse.
  - Transfer to iALS, Coat: Q-iALS 0.5871 vs iALS 0.5812 (48 vs 24 cfgs,
    `learned_coat_driALS.json`); not run on larger datasets (dense solve too slow). Report as a
    one-sentence observation without significance claim (no test was run).
  - TODO-E1 (recommended, not required): run P-LightGCN (offset only) on Yahoo, KuaiRand and
    KuaiRec (`run_learned.py --methods LightGCN-p LightGCN-p2`) so that the offset-vs-weighting
    decomposition rests on matched runs on all four datasets. Until then, write "on Coat" for the
    offset-only claim.
- **6.4 Comparison with AdvInfoNCE (200).** Table 6. Final method vs AdvInfoNCE re-impl.: Coat
  0.6015 vs 0.5222 (+78.50e-3 [27.63]), Yahoo 0.6866 vs 0.6477 (+37.85e-3 [8.48]), KuaiRand
  0.4613 vs 0.4480 (+12.79e-3 [2.50]), KuaiRec 0.6285 vs 0.5776 (+49.32e-3 [6.57]), p_Holm < 1e-4
  everywhere (`final_summary.txt` lines 10, 18, 27, 38). Context: plain InfoNCE-LightGCN on Coat
  0.5290 and DR-weighted InfoNCE 0.5531 (`log_learned_coat_new.txt` line 120,
  `log_learned_coat_dr.txt` line 225), so on these small candidate sets the contrastive family is
  weak in our hands. Repeat the simplification caveat; the unequal grids favour AdvInfoNCE on
  Coat and disfavour it elsewhere.
- **6.5 The walk correction and DRUP (400).** Table 7 + Figure 4.
  - Matched configurations, DRUP minus DR adjacency (degrees from the imputation), mean test diff
    x1e-3, share of configs < 0, splits with mean < 0: Coat (1,440 shared configs) -0.33 at
    beta = 0.03 to -8.17 at beta = 1000, negative in 10/10 splits for beta >= 0.3; Yahoo (512)
    -0.02 to -0.32 (`results/v3/matched_config_real.txt`). The correction does not help accuracy.
  - Semi-synthetic KuaiRec (true propensities, fixed degree weights): DRUP - DR nDCG@20 +0.60e-3
    to +19.59e-3 across the six cells at small beta, reversing to negative (down to -5.35e-3) at
    large beta when gamma = 1.0 (`python3 experiments/rank_dose_table.py`, reads
    `results/v3/rank_dose.json`). With imputation-based degrees the SUMMARY reports +0.002 to
    +0.006, reversing at large beta (`results/v4/SUMMARY.md` line 6; TODO-E2: locate the file
    behind this statement, probably `results/v3/rank_dose_real.json`, before citing numbers).
  - Training-free DRUP is the best method on KuaiRec: 0.6378 (DR adjacency 0.6378, DR adjacency
    5 hops 0.6379; `paper/tables/accuracy.tex`). Its gain over the final method: ensemble file
    reports DRUP-repro 0.6378 vs A-repro 0.6285.
  - Ensemble s = (1-w) z(A) + w z(DRUP), w chosen per split on validation
    (`experiments/run_ensemble.py` docstring): KuaiRec w in {0.8: 1, 1.0: 4}, ensemble 0.6376,
    vs DRUP -0.18e-3 [0.19], p_Holm 0.0664 (no gain over DRUP); Coat w in {0.0: 9, 0.3: 1},
    ensemble 0.6031, vs A +1.79e-3 [2.39], p_Holm 0.1421 (no gain over A); vs iALS +21.81e-3,
    p_Holm 0.0823 (`results/v4/ensemble_summary.txt`). Reading: the two components are not
    complementary; validation picks one of them.
  - On Coat DRUP is weak (0.5467) and on Yahoo/KuaiRand it selects IPS edges and equals IPS
    adjacency (0.6567 / 0.4492, `accuracy.tex`), so the DR half is tested only on Coat and
    KuaiRec.

### 7 Discussion (~1,000 words)

- **7.1 Why it works (300).** Q is near zero for exposed-but-unchosen pairs and near Yhat for
  never-exposed pairs; the weighting therefore removes gradient from likely false negatives
  among rarely exposed items. The offset injects a debiased population preference (user and item
  effects fitted with IPS) that LightGCN's BPR objective cannot learn from a biased log. The
  shuffled control rules out that the gain is from weight variance alone.
- **7.2 When it does not help (300).** KuaiRec: dense log (13.4%) and a fully observed test
  matrix with 3,314 candidates per user; pointwise DR methods (DR-JL, MRDR) and DR propagation are
  stronger. Also: the pointwise variant of the weighting (LightGCN-pn 0.5999, DR-LightGCN-pn
  0.6027 on KuaiRec, `results/v4/SUMMARY.md` line 17; files `learned_kuairec_*-pn_*.json`
  give 0.5992-0.5995 and 0.5992-0.6027 per lr job) does not close the gap. Coat: iALS is as good.
- **7.3 Practical guidance (200).** Fit the additive IPS imputation first; check that Q is
  informative with the shuffled control; select beta on validation including beta = 0; on dense
  logs with a fully observed target, compare against training-free DR propagation, which needs no
  gradient training (old `discussion.tex` lines 41-44, KuaiRec 0.638, adapt).
- **7.4 Limitations (200 in main text; full list in Section 8 below).**

### 8 Conclusion (~250 words)

Restate the method in two sentences; the four-dataset result with the two "on par" cases and the
KuaiRec loss to DRUP; the mechanism; future work: Q computed on the fly for large catalogues,
matched offset-only runs on all datasets, a faithful AdvInfoNCE reproduction, and propensities not
estimated from the test distribution.

### Supplementary Material

S1 Full grids and configuration counts (from `run_learned.py` METHODS and the queue scripts).
S2 Recall@K and AUC for all methods (logs contain `recall@5`, `auc`, e.g.
`results/v4/log_learned_coat_*.txt`; older baselines `paper/tables/accuracy_recall.tex`,
`accuracy_auc.tex`).
S3 DRUP theory (old `method_theory.tex` sections Problem setting, DRUP, Theory, lines 1-359,
condensed; Figure `fig:walks`; proofs from `paper/sections/appendix.tex`).
S4 Matched-configuration and semi-synthetic walk-correction results (old `exp_semisynth.tex`,
`paper/tables/semisynth.tex`, `results/v3/matched_config_real.txt`).
S5 AdvInfoNCE implementation details and the KL log of the adversary.
S6 Per-split results and chosen beta per split.

---------------------------------------------------------------------------------------------------

## 5. Tables and figures to build

All values for the method rows are "mean +- sd (ddof=0) over splits"; diff = paired per-user
difference x1e-3 with 95% CI half-width in brackets; p_Holm within dataset.

**Table 1 - Positioning** (Related work). Rows: LightGCN/SimGCL; NAVIP; r-AdjNorm/APDA/CAGED/DPAA;
IPS/DR/DR-JL/MRDR; ExpoMF/Rel-MF/UBPR; AdvInfoNCE/BC loss; DRUP; ours. Columns: graph backbone
(yes/no), uses propensity, uses imputation, acts on (loss / negatives / adjacency / score), needs
uniform data in training (yes/no). Qualitative; adapt old `tab:related` in
`paper/sections/intro_related.tex` lines 178-211.

**Table 2 - Main results** (Section 6.1). Source `results/v4/final_summary.txt` (method row: line
"mean ndcg@K"; baselines: name, mean, cfgs, diff, [CI], p_Holm). sd of the method: Coat 0.0162,
Yahoo 0.0034, KuaiRand 0.0024, KuaiRec 0.0022 (std over the per-split values of the validated
combination; reproduce with `final_analysis.combine` and `np.std`, ddof = 0). sd of baselines:
`paper/tables/accuracy.tex`.
Layout: one block per dataset (four column groups would be too wide): rows = methods, columns =
test nDCG (mean +- sd), #cfgs, diff x1e-3 [CI], p_Holm. Bold the best, underline the second.

| Dataset (metric) | Method row | Baseline rows in Holm family |
|---|---|---|
| Coat (nDCG@5) | 0.6015 +- 0.0162, 324 cfgs | LightGCN 0.5683, r-AdjNorm 0.5722, NAVIP 0.5664, DR-LightGCN 0.5724, iALS 0.5812, SimGCL 0.5528, MACR 0.5418, AdvInfoNCE 0.5222 |
| Yahoo! R3 (nDCG@5) | 0.6866 +- 0.0034, 84 cfgs | LightGCN 0.6708, r-AdjNorm 0.6742, NAVIP 0.6701, iALS 0.6625, SimGCL 0.6348, AdvInfoNCE 0.6477 |
| KuaiRand (nDCG@10) | 0.4613 +- 0.0024, 30 cfgs | LightGCN 0.4569, r-AdjNorm 0.4590, NAVIP 0.4565, iALS 0.4528, BPR-MF 0.4475, SimGCL 0.4196, AdvInfoNCE 0.4480 |
| KuaiRec (nDCG@20) | 0.6285 +- 0.0022, 120 cfgs | DR-JL 0.6332, MRDR 0.6328, LightGCN-pt 0.6271, DR-LightGCN 0.6212, NAVIP 0.6186, iALS 0.6133, LightGCN 0.6085, r-AdjNorm 0.6083, AdvInfoNCE 0.5776 |

Extra rows, *outside* the Holm family, marked "not tested" (means +- sd from
`paper/tables/accuracy.tex`): DR-JL (Coat 0.4811, Yahoo 0.5225, KuaiRand 0.3759), MRDR (Coat
0.4869, Yahoo 0.5248), DR-LightGCN on Yahoo (0.5528), DRUP (Coat 0.5467, Yahoo 0.6567, KuaiRand
0.4492, KuaiRec 0.6378), DR adjacency (Coat 0.5459, Yahoo 0.6567, KuaiRand 0.4492, KuaiRec
0.6378). TODO-A1 (analysis only, no training): extend the `names` lists of a copy of
`experiments/final_analysis.py` with DR-JL, MRDR, DR-LightGCN (where present in the v3 files) and
the DRUP/DR rows of the filter files, so that these rows also get diff and p_Holm. If done,
regenerate the Holm-adjusted p values for the whole family (the family size changes) and update
this table and Section 6.1 consistently.
Footnote: beta chosen per split (Section 4.4 list); grid sizes differ (column cfgs).

**Table 3 - Datasets.** `paper/tables/datasets.tex` unchanged.

**Table 4 - Seed robustness** (Section 6.2). Source `results/v4/seeds_summary.txt` lines 3-41.
Columns: dataset, method, seed0, seed1, seed2, seed3, mean +- sd (seeds 1-3, ddof 0), sd (ddof 1);
then per-dataset rows "vs X: diff [CI], p_Holm (2 tests)". Twelve method rows: Coat {final, iALS,
DR-LightGCN}; Yahoo {final, r-AdjNorm, LightGCN}; KuaiRand {final, r-AdjNorm, LightGCN}; KuaiRec
{final, DR-JL, MRDR}. Footnote: seed 0 is the main run of Table 2; seeds 1-3 retrain each split's
selected configuration and re-select only the epoch.

**Table 5 - Ablations** (Section 6.3). Two panels.
Panel A (weights, beta = 0): rows LightGCN, Q-LightGCN, Q(Yhat), Q(pop), Q(shuffled); columns Coat
nDCG@5 (cfgs), Yahoo nDCG@5 (cfgs).
Coat: 0.5683 (54) / 0.5912 (108) / 0.5937 (54) / 0.5751 (54) / 0.5689 (54).
Yahoo: 0.6708 (36) / 0.6829 (12, lr 3e-3) / -- / 0.6550 (12) / 0.6689 (12).
Sources: `results/v4/learned_coat_{drbpr,abl_DR-BPR-yhat,abl_DR-BPR-pop,abl_DR-BPR-shuf}.json`,
`results/v4/learned_yahoo_{drbpr_lr3e-3,abl_DR-BPR-pop,abl_DR-BPR-shuf}.json`, LightGCN from
`final_summary.txt`. sd from the logs (e.g. `log_learned_coat_abl.txt` line 60: +-0.0106).
Note the grid mismatch (Yahoo LightGCN 36 cfgs over 3 lr vs ablations 12 cfgs at lr 3e-3) in a
footnote. No significance tests were run for the ablations (TODO-A2, optional: paired tests from
the per-user entries of these files with `final_analysis.cmp`).
Panel B (offset and imputation, Coat): LightGCN 0.5683; P-LightGCN 0.6020; Q-LightGCN 0.5912;
QP-LightGCN beta {0.5,1} 0.6039; beta {1,2} 0.6046; validated QP-LightGCN 0.6015;
low-rank imputation: Q-LightGCN 0.5887, QP-LightGCN 0.5897; Q-iALS 0.5871 vs iALS 0.5812.
Panel C (per-dataset decomposition): Q-LightGCN vs QP-LightGCN: Coat 0.5912 / 0.6015; Yahoo
0.6837 / 0.6866; KuaiRand 0.4623 (beta = 0 job) / 0.4613; KuaiRec 0.6095 / 0.6285.

**Table 6 - AdvInfoNCE** (Section 6.4). Rows: datasets. Columns: AdvInfoNCE test (cfgs), final
method test, diff [CI], p_Holm, grid description (tau, eta, lr, wd, dims, layers). Values: Coat
0.5222 (216), Yahoo 0.6477 (16), KuaiRand 0.4480 (8), KuaiRec 0.5776 (8); diffs and p from
`final_summary.txt` lines 10, 18, 27, 38; grid flags from `experiments/queue_v5.sh` lines 19-28.
Add Coat-only context rows InfoNCE-LightGCN 0.5290 (108) and DR-InfoNCE 0.5531 (216).

**Table 7 - Walk correction** (Section 6.5). Panel A: matched-configuration DRUP - DR on real data,
`results/v3/matched_config_real.txt` verbatim (beta, mean diff x1000, share < 0, splits < 0; Coat
and Yahoo). Panel B: ensemble, `results/v4/ensemble_summary.txt` (Coat, KuaiRec: ensemble test,
A-repro, DRUP-repro, Ens-fixed-0.5, w distribution, diff vs A, diff vs DRUP, p_Holm (11 tests)).

**Figure 1 - Method schematic** (Section 4). Log (O, OY) -> logistic propensity p and IPS additive
imputation Yhat -> Q = Yhat(1-p)/(1-pYhat) and z(logit Yhat) -> LightGCN with Q-weighted BPR
negatives and offset -> per-split selection of beta and epoch on uniform validation data -> test
on uniform test data. Show that the uniform data never enter training. TikZ.

**Figure 2 - Per-split differences** (Section 6.1). Forest plot: for each dataset, per-split test
difference final - strongest baseline (Coat: iALS; Yahoo, KuaiRand: r-AdjNorm; KuaiRec: DR-JL),
plus the paired mean and CI from Table 2. Data: `per_split` of the combined method
(`final_analysis.combine`) and the baseline's `per_split` in the v3/v4 JSON files. TODO-F2: write
`experiments/plot_forest.py` (read-only on results).

**Figure 3 - Ablation bars** (Section 6.3). Panel A of Table 5 as grouped bars with sd error bars,
LightGCN as a horizontal reference line. Optional: replace Table 5 Panel A by this figure.

**Figure 4 - Walk correction** (Section 6.5). Left: matched-config mean diff vs beta (log x-axis),
Coat and Yahoo (`matched_config_real.txt`). Right: semi-synthetic DRUP - DR nDCG@20 vs b for the
six cells with CI bars (`rank_dose_table.py` output). Zero line in both.

**Figure 5 (optional) - Distribution of Q** on one dataset (histogram of Q for logged negatives
vs never-exposed pairs). Needs a new script (TODO-F5); skip if space is short.

---------------------------------------------------------------------------------------------------

## 6. Keep / cut / move (old manuscript)

| Old material | Decision | Where |
|---|---|---|
| `paper/sections/intro_related.tex` lines 5-17 (motivation: logged graph != preference graph) | Keep, rewrite | Intro para 1 |
| `intro_related.tex` lines 19-27 (repeated-walk bias observation) | Cut from intro; one sentence in Sec 4.6 | 4.6 |
| `intro_related.tex` lines 29-91 (objectives O1-O4, RQs, contributions) | Cut; replace by the new contributions | - |
| `intro_related.tex` lines 97-142 (related work: graph, debiasing, graph debiasing) | Keep, shorten, remove DRUP claims | 2.1, 2.2, 2.4 |
| `intro_related.tex` lines 144-159 (U-statistics, Möbius, polynomial estimation) | Move | Supplement S3 |
| `intro_related.tex` lines 161-176 (fairness, explanation, robustness, privacy) | Cut | - |
| `intro_related.tex` lines 178-211 (`tab:related`) | Replace by new Table 1 | 2 |
| `paper/sections/method_theory.tex` lines 5-19 (setting) | Keep | 3 |
| `method_theory.tex` lines 21-68 (Assumption 2, `rem:xfit`, target F*) | Move | S3 |
| `method_theory.tex` lines 74-92, 131-153 (edge estimates, correction, constants) | Keep condensed (~250 words) | 4.6 |
| `method_theory.tex` lines 94-129 (`fig:walks`) | Move | S3 |
| `method_theory.tex` lines 155-215 (sample split, public operator, Algorithm DRUP, control variates, cost) | Move | S3 |
| `method_theory.tex` lines 217-359 (all theorems, structural properties) | Move theorems on unbiasedness/bias; cut structural properties (attributions, DP, certificates, allocation) | S3 / cut |
| `paper/sections/exp_setup.tex` lines 5-54 (datasets) + `tables/datasets.tex` | Keep | 5.1 |
| `exp_setup.tex` lines 56-85 (baselines) | Keep, prune to the baselines actually in Table 2 | 5.2 |
| `exp_setup.tex` lines 87-94 (LLM paragraph) | Cut (one line in limitations) | 7.4 |
| `exp_setup.tex` lines 96-117 (protocol) | Keep, rewrite for the trained method; drop cross-fitting text (not used by the trained method) | 5.3 |
| `paper/sections/exp_acc.tex`, `tables/accuracy.tex`, `tables/significance.tex` | Use only the baseline rows listed in Section 5 above; cut the long training-free list (EASE/GF-CF/BSPM on DR graph etc.) | 6.1 extra rows / S2 |
| `exp_semisynth.tex`, `tables/semisynth.tex`, `exp_mc.tex`, `tables/mc*.tex` | Move | S4 |
| `exp_fair.tex`, `tables/fat.tex`, `intervene*.tex` (thinning audits), `exp_rerank_text.tex`, `rerank*.tex`, `attack*.tex`, `privacy*.tex`, `explain.tex`, `exp_sparse.tex`, `supp_structural.tex` | Cut (not part of the new story) | - |
| `paper/sections/discussion.tex` lines 41-44 (DR propagation is a strong cheap baseline on KuaiRec) | Keep, adapt | 7.3 |
| `discussion.tex` lines 54-78 (limitations) | Reuse the items on Coat propensities (68-70), static analysis (55-57), catalogue size (72); cut the rest | 7.4 |
| `discussion.tex` lines 12-28 (theoretical implications of the walk correction) | Move, shortened | S3 |
| `paper/ipm/*` (cover letter, response to editor, highlights) | Do not reuse | - |

---------------------------------------------------------------------------------------------------

## 7. Claims vs evidence

| # | Claim (as it may appear in the paper) | Evidence (file, line) | Status |
|---|---|---|---|
| C1 | Better than every graph baseline on all four datasets | `final_summary.txt` 3-6, 8, 10, 13-15, 17-18, 21-23, 26-27, 33-34, 36-38: all p_Holm <= 0.0185 | Supported, except LightGCN-pt on KuaiRec (p_Holm 0.4409): write "every graph baseline trained with a pairwise or contrastive loss" or name the exception |
| C2 | Best method on Yahoo! R3 | `final_summary.txt` 12-18; `seeds_summary.txt` 15-21 | Supported among the methods run (DRUP/DR-JL/MRDR lower in `accuracy.tex`, not tested) |
| C3 | Best method on KuaiRand | `final_summary.txt` 20-27 (+1.94e-3, p_Holm 0.0185); seeds +2.31e-3, p_Holm 0.0002 | Supported, effect small; say "small" |
| C4 | On par with iALS on Coat | seed 0: p_Holm 0.0469; seeds 1-3: +3.39e-3, p_Holm 0.7470 | Supported as "on par"; do NOT claim better |
| C5 | On par with DR-JL/MRDR on KuaiRec | seed 0: worse, p_Holm 0.0002; seeds 1-3: -1.03e-3 (0.2213), -1.56e-3 (0.0760) | Write "below in the main run, within 0.002 and not significantly different over retraining seeds" |
| C6 | DRUP (training-free) is best on KuaiRec | `accuracy.tex` DRUP 0.6378; ensemble DRUP-repro 0.6378 vs A 0.6285 | Supported; DR adjacency equal (0.6378), so credit DR propagation, not the correction |
| C7 | The gain needs an informative Q | Coat shuffled 0.5689 vs LightGCN 0.5683; Yahoo 0.6689 vs 0.6708 | Supported on Coat and Yahoo; no tests (TODO-A2 optional) |
| C8 | Imputation-only Q is as good as full Q | Coat 0.5937 vs 0.5912 | Coat only; grids 54 vs 108 |
| C9 | Popularity-only weighting is weaker | Coat 0.5751, Yahoo 0.6550 (below LightGCN 0.6708) | Supported (use 0.5751, not 0.5770) |
| C10 | Additive imputation better than low-rank | Coat 0.5887/0.5897 vs 0.5912/0.6039 | Coat only |
| C11 | Offset is the main ingredient on Coat | P-LightGCN 0.6020 vs QP 0.6039 vs Q 0.5912 | Coat only; other datasets need TODO-E1 |
| C12 | Weighting carries Yahoo and KuaiRand; offset carries KuaiRec | Yahoo 0.6837 -> 0.6866; KuaiRand beta=0 in 4/5 splits; KuaiRec 0.6095 -> 0.6285 | Supported as a decomposition of selected runs, not as matched ablation |
| C13 | Walk correction gives no real-data accuracy benefit | `matched_config_real.txt` (Coat negative 10/10 splits for beta >= 0.3; Yahoo ~0) | Supported for Coat and Yahoo; KuaiRec: DRUP 0.6378 = DR 0.6378 (`accuracy.tex`) |
| C14 | Walk correction helps under oracle semi-synthetic conditions | `rank_dose_table.py` output; `tables/semisynth.tex` (fixed C, true p: DR 0.097 vs DRUP 0.107 nDCG@20) | Supported, with reversal at large beta when gamma = 1 |
| C15 | Ensemble with DRUP adds nothing | `ensemble_summary.txt` 27-28, 5 | Supported for Coat and KuaiRec only |
| C16 | Beats the AdvInfoNCE re-implementation on all four | `final_summary.txt` 10, 18, 27, 38 | Supported for the re-implementation only; disclose simplifications and grids |
| C17 | No uniform data used in training | `run_learned.py` 147-160 (nuisances from O, Y of the log); uniform data only in `splits` | Supported, but Coat's shipped propensities were estimated with uniform data by the dataset authors: disclose |
| C18 | Method is simple / cheap | Code: one lookup per negative; Q and offset precomputed | Supported qualitatively; no timing comparison in the result files (do not give run times unless taken from logs, e.g. `[230s, 108 cfgs]` in `log_learned_coat_drbpr.txt` line 114) |

---------------------------------------------------------------------------------------------------

## 8. Limitations (all must appear, main text 7.4 or Supplement)

1. **Unequal tuning budgets.** The final method is the union of several jobs: 324 configurations
   on Coat vs 24 (iALS), 54 (LightGCN), 108 (r-AdjNorm); 84 on Yahoo vs 72 (r-AdjNorm); 30 on
   KuaiRand vs 12 (r-AdjNorm); 120 on KuaiRec vs 6 (DR-JL, MRDR). The selection over more
   configurations can favour the method. AdvInfoNCE: 216 configurations on Coat, 16/8/8 elsewhere.
2. **AdvInfoNCE is a simplified re-implementation** (implicit eta via adversary rounds, no KL
   projection, embedding adversary only); its numbers are not those of the original paper.
3. **Small effects.** KuaiRand +1.94e-3 (main run) / +2.31e-3 (seeds); most gains are 0.01-0.03
   nDCG; Coat's CI half-widths are about 0.02.
4. **Main run vs seed runs.** Table 2 is a single training seed with full selection. Seeds 1-3
   retrain the selected configurations only and compare against two baselines per dataset; the
   Coat advantage over iALS and the KuaiRec deficit to DR-JL/MRDR are not significant there.
5. **KuaiRec is lost** to DR-JL/MRDR in the main run and to training-free DRUP / DR adjacency
   (0.6378) in all runs.
6. **One imputation family.** Q and the offset come from one additive model (and one low-rank
   alternative on Coat); other imputations untested.
7. **Ablations mostly on Coat** (offset-only, imputation-only, low-rank, Q-iALS); Yahoo has
   pop/shuffle only; KuaiRand/KuaiRec none. No significance tests for ablations.
8. **Coat propensities** were estimated by the dataset authors using the uniform sample that we use
   for validation and testing (possible leak; Coat only).
9. **Exposure semantics.** On Coat and Yahoo! R3 "exposure" is self-selection, not platform
   exposure; Q's derivation assumes exposure independent of the outcome given p's covariates.
10. **Overlapping splits.** Per-user t-tests over overlapping splits treat the log and nuisances as
    fixed and can be anti-conservative (nadeau2003).
11. **Scale.** Dense m x n matrices for Q and offset; largest catalogue ~10^4 items.
12. **Not compared:** StableDR, DICE, CausE, AutoDebias, LLM-based recommenders; Q-iALS not run on
    larger datasets.
13. **Validation choice is not always test-best** (KuaiRand beta = 0 job 0.4623 vs validated
    combination 0.4613).

---------------------------------------------------------------------------------------------------

## 9. Related-work gap statement and citations

**Gap statement (draft, to go at the end of Section 2; hedge as written):**

> Three lines of work meet in this paper but have rarely been combined. Doubly robust and
> propensity methods (IPS, DR-JL, MRDR) are pointwise and are usually applied to matrix
> factorisation; graph recommenders that address exposure or popularity bias (NAVIP, r-AdjNorm,
> APDA, CAGED) change the adjacency or the aggregation rather than the training signal; and
> false-negative-aware sampling (ExpoMF, SRNS, AdvInfoNCE) does not use the propensity and
> imputation that DR estimation provides. To our knowledge, no previous study uses the DR
> nuisances to weight the negatives of a graph recommender, and few evaluate graph debiasing on
> Coat, Yahoo! R3, KuaiRand and KuaiRec together with uniform data held out from training. Many
> reported gains of debiasing methods come from protocols that use uniform data for training
> (chen2021autodebias, bonner2018cause), evaluate on the biased log, or compare against
> untuned baselines (dacrema2019progress, rendle2019baselines, anelli2023myth).

TODO-G1 (mandatory before submission): a literature search (Scopus / Google Scholar, 2019-2026,
queries "propensity negative sampling graph recommendation", "false negative LightGCN exposure",
"doubly robust graph collaborative filtering", "debiased negative sampling KuaiRand") to confirm the
"to our knowledge" sentence. If a paper combining DR nuisances with negative weighting in a graph
model is found, cite it, add it as a baseline if code exists, and soften the statement. Do not
write "first".

**Citation keys available in `paper/refs.bib`** (use as is): he2020lightgcn, wang2019ngcf,
mao2021ultragcn, wu2021sgl, yu2022simgcl, wu2022gnnsurvey, steck2019ease, shen2021gfcf,
choi2023bspm, hu2008ials, anelli2023myth, dacrema2019progress, rendle2019baselines,
marlin2009collaborative, schnabel2016, swaminathan2015snips, saito2020unbiased, wang2019drjl,
guo2021mrdr, dai2022gdr, li2023stabledr, wang2020cvib, chen2021autodebias, bonner2018cause,
liang2016exposure, wei2021macr, zhang2021pda, zheng2021dice, chen2023bias, gao2024causalsurvey,
boratto2021connecting, okamura2024flexibly, bai2025invariant, kim2022navip, zhao2022radjnorm,
chen2024gcnpop, zhou2023apda, que2025caged, dpaa2026, gao2022kuairec, gao2022kuairand,
nadeau2003, chernozhukov2018dml, lounici2014, chaney2018confounding.
(Check `dpaa2026`: its bib entry is Islam, Faruk, Medya, Zheleva, "Debiasing Message Passing to
Mitigate Popularity Bias in GNN-based Collaborative Filtering", 2026; verify venue and year.)

**New citations needed (all flagged "verify" - check every field against the publisher page
before adding to the .bib):**

- TODO-C1 BPR - Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L. (2009). BPR:
  Bayesian personalized ranking from implicit feedback. In *Proceedings of the 25th Conference on
  Uncertainty in Artificial Intelligence (UAI)*, pp. 452-461. (verify)
- TODO-C2 Holm - Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65-70. (verify)
- TODO-C3 AdvInfoNCE - Zhang, A., Sheng, L., Cai, Z., Wang, X., Chua, T.-S. (2023). Empowering
  collaborative filtering with principled adversarial contrastive loss. In *Advances in Neural
  Information Processing Systems 36 (NeurIPS 2023)*. (verify authors and title)
- TODO-C4 UBPR - Saito, Y. (2020). Unbiased pairwise learning from biased implicit feedback. In
  *Proceedings of the 2020 ACM SIGIR International Conference on Theory of Information Retrieval
  (ICTIR)*, pp. 5-12. (verify pages)
- TODO-C5 SRNS - Ding, J., Quan, Y., Yao, Q., Li, Y., Jin, D. (2020). Simplify and robustify
  negative sampling for implicit collaborative filtering. In *Advances in Neural Information
  Processing Systems 33 (NeurIPS 2020)*. (verify)
- TODO-C6 Debiased contrastive learning - Chuang, C.-Y., Robinson, J., Lin, Y.-C., Torralba, A.,
  Jegelka, S. (2020). Debiased contrastive learning. In *Advances in Neural Information
  Processing Systems 33 (NeurIPS 2020)*. (verify)
- TODO-C7 BC loss - Zhang, A., Ma, W., Wang, X., Chua, T.-S. (2022). Incorporating bias-aware
  margins into contrastive loss for collaborative filtering. In *Advances in Neural Information
  Processing Systems 35 (NeurIPS 2022)*. (verify)
- TODO-C8 MNAR training/testing - Steck, H. (2010). Training and testing of recommender systems on
  data missing not at random. In *Proceedings of the 16th ACM SIGKDD International Conference on
  Knowledge Discovery and Data Mining*, pp. 713-722. (verify pages)
- TODO-C9 iALS revisited - Rendle, S., Krichene, W., Zhang, L., Koren, Y. (2022). Revisiting the
  performance of iALS on item recommendation benchmarks. In *Proceedings of the 16th ACM
  Conference on Recommender Systems (RecSys)*. (verify)
- TODO-C10 Logit adjustment - Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A.,
  Kumar, S. (2021). Long-tail learning via logit adjustment. In *International Conference on
  Learning Representations (ICLR 2021)*. (verify)
- TODO-C11 Sampled metrics - Krichene, W., Rendle, S. (2020). On sampled metrics for item
  recommendation. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge
  Discovery and Data Mining*, pp. 1748-1757. (verify pages)
- TODO-C12 Denoising implicit feedback (optional) - Wang, W., Feng, F., He, X., Nie, L., Chua,
  T.-S. (2021). Denoising implicit feedback for recommendation. In *Proceedings of the 14th ACM
  International Conference on Web Search and Data Mining (WSDM)*, pp. 373-381. (verify pages)

---------------------------------------------------------------------------------------------------

## 10. Open items before drafting (summary)

| ID | Type | Action | Blocks |
|---|---|---|---|
| A1 | analysis | Add DR-JL/MRDR/DR-LightGCN/DRUP/DR rows to the Holm family (copy of `final_analysis.py`, no training) | Table 2 extra rows' p values |
| A2 | analysis (optional) | Paired tests for the ablations | "significant" wording in 6.3 |
| E1 | experiment (recommended) | P-LightGCN (offset only) on Yahoo, KuaiRand, KuaiRec | C11 beyond Coat |
| E2 | lookup | Locate the source of "+0.002 to +0.006 with imputation-based degrees" (`results/v4/SUMMARY.md` line 6) | Figure 4 right panel caption |
| F2 | script | Forest plot of per-split differences | Figure 2 |
| F5 | script (optional) | Q histogram | Figure 5 |
| G1 | literature | Verify gap statement | Section 2.6 |
| C1-C12 | bibliography | Verify and add new references | Sections 2, 4, 5 |
| - | consistency | `results/v4/SUMMARY.md` line 18 (Coat pop-only 0.5770) disagrees with the stored file (0.5751); use 0.5751 | Table 5 |
