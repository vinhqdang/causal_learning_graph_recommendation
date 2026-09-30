# Revision notes (internal pre-submission review, round 1)

This file maps each required revision of the internal review (Editorial
Decision: Major Revision; five reports: Journal-Fit, Methodology, Domain,
Perspective, Devil's Advocate) to the change made. Item numbers follow the
revision roadmap (REV-n).

## Critical issues

- **DA-C1 / REV-1, REV-22 (the correction does not change accuracy).**
  Reframed as an estimator-correctness contribution (route b), with the
  evidence for route (a) reported as far as it goes: the abstract, introduction,
  results and conclusion now state that the walk correction leaves top-K
  accuracy unchanged on all three benchmarks, credit the accuracy of DRUP to
  training-free propagation over the DR graph, and make DRUP vs DR adjacency
  (matched grids, user-level tests) the first rows of the significance table.
  New evidence: Monte-Carlo bias on unexposed candidates (IPS is unbiased there
  at 3 hops, not at 5), ranking agreement with the target (the uncorrected DR
  operator agrees slightly better in simulation; reported and explained), and a
  real-data sparsity sweep on KuaiRec (log thinned to 10% and 3%) with
  top-20 overlap and rank correlation between DR and DRUP. The Discussion
  states when to use the corrected operator (score values, cross-user
  comparisons, allocation, audits, K >= 5) and when the uncorrected one suffices.
- **DA-C2 / REV-26 (degrees from W outside Assumption 2).** Confirmed in the
  result files. All experiments were re-run under a new protocol: every
  nuisance (propensity model, imputation, degree weights) is cross-fitted over
  ten folds of pairs; degree weights come from the imputation or from
  cross-fitted edge estimates, and the degree source is now a disclosed,
  searched hyper-parameter (Appendix B). Raw W-degrees are no longer used for
  any debiased operator. On Yahoo!R3 the previous 0.662 depended on raw
  W-degrees; with imputation degrees DRUP drops to 0.62-0.63, and with
  cross-fitted W-degrees it reaches 0.657.

## Methodology (R1)

- W1 / REV-21: new Corollary 2 (conditional target on unexposed candidates);
  IPS claim restricted (no 3-hop bias on unexposed candidates, bias from 5 hops);
  MC reports bias over all entries and on unexposed candidates.
- W2 / REV-24: Assumption 1 restated as conditional independence given Y with
  outcome-dependent propensities; new Proposition 4 bounds the bias under
  within-user dependence (second order in the imputation error for DR);
  MC stress tests with fixed-size slates and misspecified propensities.
- W3 / REV-25, REV-27: cross-fitting implemented and used everywhere; the
  per-user rescaling of the one- and three-hop terms was replaced by global
  constants (a fixed linear combination), stated in Section 4.
- W4 / REV-9: full proofs of Theorem 3 (K hops, Möbius inversion) and Theorem 12
  (allocation), plus the term-by-term derivation of Eqs. (3) and (4).
- W5 / REV-28, REV-29: analytic Gaussian mechanism (valid for all epsilon);
  Theorem 11 restated as joint DP conditional on public nuisances; experiment
  with population-level nuisances fitted on a public 10% of users, R from the
  public users, fixed denoising ranks, epsilon from 0.5 to 16.
- W6 / REV-30, REV-31: user-level paired tests with 95% CIs and Holm
  correction; Nadeau-Bengio corrected resampled t-test reported in the result
  files; Yahoo!R3 added to the significance table.
- W7 / REV-15: trained baselines re-run with per-split selection of
  configuration and epoch, larger grids, BPR losses and seed checks; the
  number of configurations per method is reported (Table B.1).
- W8: accuracy claims rewritten (see DA-C1); Coat deficits stated.
- W9: interventions now run with frozen nuisances, re-fitted nuisances with the
  known propensity change, and re-fitted nuisances with re-estimated
  propensities; the fraction of treated pairs below the clip is reported.
- W10 / REV-34: frozen-nuisance scope stated in the theorems; certificates
  checked against attacks with nuisances re-fitted on the poisoned log.
- W11 / REV-35: Theorem 5 corrected (worst case Theta(1/tau), attained at p = tau).
- W12: Theorem 6(c) states p_hat = p; t_delta defined; notation clash removed;
  bounds compared numerically with Monte-Carlo (loose by 10^2-10^4, reported).
- W13 / REV-17: every property table states its configuration and data.
- W14: candidate-set sizes and random-ranking references reported.
- W15 / REV-19: code/data availability on the title page; Appendix B lists grids.

## Domain (R2) and Journal fit (EIC)

- New baselines: EASE and GF-CF on the logged and the DR graph, BPR-MF,
  LightGCN (BPR), r-AdjNorm, PDA; linear propagation on the logged graph is
  described as a training-free r-AdjNorm.
- New related work: APDA, r-AdjNorm, CAGED, DPAA, StableDR, GDR, CVIB, SimGCL,
  BSPM, EASE, reproducibility studies, unbiased Gram estimation under
  missingness (Lounici), DOULION, Möbius inversion (Rota), non-backtracking
  operators, DML cross-fitting, doubly robust OPE, and IP&M papers on
  popularity debiasing, fairness measurement, provider fairness, graph-filter
  diversity, DP recommendation and shilling detection.
- One suggested reference (Islam, Zheleva and Wang, WWW 2026, DOI
  10.1145/3774904.3792670) could not be verified (the DOI does not resolve) and
  is not cited.
- Additional tables moved to a separate Supplementary Material file.
- Title and abstract no longer claim "provably fair, transparent, accountable
  and private"; "every guarantee is verified empirically" removed.
- The trust properties that follow from linearity are grouped in one section
  and attributed to linear propagation in general (Table 1 revised).

## Perspective (R3)

- "Causal item fairness" renamed exposure invariance and positioned as a
  necessary, not sufficient, condition for merit-based exposure fairness.
- Stakeholder paragraph (providers, users, platform, auditors, regulators)
  with a mapping to DSA Articles 27, 34-35, 37 and 40; limitations extended
  (static logging, feedback loops, cold start, variance for rare items,
  normative target, scale).
- User-group gaps reported with bootstrap intervals and group sizes.
- Shapley exactness presented as a consequence of linearity with its baseline.

# Round 2 (external pre-submission review of commit 5ca6629, 13 points)

1. **Pair-level cross-fitting does not make products of edge estimates
   unbiased.** Agreed. Remark 1 now states that the guarantees need nuisances
   that do not depend on the log and that the cross-fitted protocol gives no
   guarantee; it also notes that the degree weights sum the imputation over all
   pairs of a user or item and therefore depend on the pair's own exposure.
   New Monte-Carlo study (`experiments/mc_protocol.py`, Table "protocol"):
   with independent-log nuisances DRUP is unbiased to Monte-Carlo error, with
   the cross-fitted protocol it is *not* less biased than the uncorrected
   operator. New sample-split variant DRUP-split (nuisances on a random 20% of
   the pairs, those pairs imputed, degrees and constants from the imputation)
   satisfies Assumption 2 exactly; it is evaluated in simulation and on all
   three datasets.
2. **W-degrees.** Stated as outside Assumption 2 (Remark 1, Appendix B,
   accuracy section); the Yahoo!R3 result with imputation degrees (0.632) is
   reported; W-degrees are included in the Monte-Carlo study.
3. **Data-dependent c1, c3.** Assumption 2 now includes the constants a, b;
   Section 4 states that data-dependent constants only rescale beta for
   within-user rankings but break the score-level statements; constants from
   the imputed graph satisfy the assumption (used by DRUP-split) and are
   compared in the Monte-Carlo study.
4. **Candidate-level target.** Discussion and introduction narrowed; the gap
   between the conditional target of Corollary 2 and F* is quantified
   (12% / 6% of the mean target at 3 / 5 hops, Kendall 0.63 / 0.77;
   `experiments/mc_candidate_gap.py`).
5. **Allocation theorem.** t_delta now uses bounded differences of the full
   score (b c_e plus the one-hop sensitivity a C eps / tau) and a union bound
   over all mn pairs, since the candidate set depends on the log; proof updated.
6. **Certificates.** "exact" replaced by "conservative" in Theorem 10, its
   proof, `drup/fat.py` and `docs/THEORY.md`: the box ignores the coupling of
   D and x_k through the same L entries.
7. **Attribution factor b.** Proposition 8 now has omega_j = b(...); the code
   already applied b / c3.
8. **"Equal potential outcomes receive equal expected scores"** removed;
   replaced by the statement actually proved.
9. **Cost for K > 3.** "Same cost" restricted to three hops; the five-hop cost
   is measured (`experiments/bench_khop.py`).
10. **Table bold.** DRUP labels are no longer bold; the configuration table no
    longer shows duplicate "MF (trained)" / "LightGCN (trained)" labels.
11. **Score-level framing.** Abstract, introduction, discussion and conclusion
    state that the correction changes neither accuracy nor rankings (also under
    caps) and that the contribution is at the level of scores.
12. **p-values conditional on the log.** Stated in the protocol and in
    Appendix B.
13. **Tuning budgets.** Expected test score as a function of the number of
    configurations (`experiments/budget_curve.py`); DRUP at the budget of the
    best trained model is reported (Appendix B).

# Round 3 (internal five-seat review of commit 58d3778; decision Major Revision)

## Critical issues

- **DA-C1 (no evidence that score values matter).** New semi-synthetic test on
  KuaiRec (Section "Score values on a semi-synthetic KuaiRec log"): the fully
  observed small matrix is the outcome matrix, and exposure logs are drawn from
  a known MNAR mechanism fitted to the big matrix (30 draws; a sparser variant
  in the supplement). With true propensities and fixed weights the correction
  lowers the relative RMSE of the three-hop term by a factor of 4 to 12, and the
  full-exposure utility of fixed allocations is overestimated by 18-29%
  without the correction and estimated within 2% with it. Rankings,
  nDCG and exposure-capped allocation are not changed. The limits are reported
  too: estimated propensities, a sparse log, and imputation-indexed weights.
- **DA-C2 (the guarantees do not cover the main experiments).** DRUP-split, the
  sample-split variant that satisfies Assumption 2, is reported next to the
  cross-fitted operators in every table, over five draws of the split-off
  pairs. The text states that the guarantee holds per configuration, not after
  selection. The abstract and introduction name the cross-fitted operators a
  heuristic.

## Methodology (R1)

- Thm 6(a) now gives the general bound eps^2/(4 tau^2) and states eps^2/tau
  when p_hat = p. Parts (b) and (c) assume p_hat = p. The appendix gives a
  counterexample.
- Thm 7 and Definition 6 use post-intervention propensities. The logged-graph
  clause requires fixed C.
- JDP: the configuration is fixed a priori, and the constants c1, c3 come from
  public users. Hyper-parameters, a and b are public. Privacy results were
  rerun.
- Holm is applied over all comparisons of a dataset (all references
  together), for both the t-test and the Wilcoxon test. Within-reference
  values are in the result files.
- Early stopping is per split: each split stops on its own validation score.
  All trained baselines were retrained.
- Budget curves are given per configuration and per evaluation.
- The Monte-Carlo tables report standard errors.
- Cor 2 now states its condition (p_e depends on Y only through Y_e). The
  delta overload is renamed xi. Assumption 2 includes a and b.
- Target dependence on the nuisance (C from Yhat-degrees) is now stated. The
  semi-synthetic test uses weights fixed a priori and quantifies the
  difference.

## Audit (R1-W4, R3-W2, DA-M3)

- New controls for the thinning audit (table "Controls for the thinning
  audit"):
  - a negative control with propensities not updated;
  - popularity and imputation-only rows;
  - DRUP-split rows;
  - rank correlation with Yhat.
- The text states that the audit cannot detect a misspecified propensity model
  on Coat, where the scores are almost a function of the imputation.

## Baselines and budgets (R2, DA-M1/M2, EIC)

- New baselines: iALS, DR-JL, MRDR, MACR, SimGCL, BSPM and BSPM on the DR graph.
- The LightGCN, r-AdjNorm, NAVIP and SimGCL grids were widened with layers and
  dimensions.
- New dataset: KuaiRand-Pure, the only dataset whose log records platform
  exposure.
- New metrics: AUC and Recall, in the supplement.
- DICE, StableDR, CausE and AutoDebias were not run, with the reason stated.

## Scope, positioning, construct validity

- The structural results (Thm 8-12) and the exposure-capped allocation
  experiment moved to the supplement, which has S-numbered sections.
- Table 1 columns are fixed, and a DR-adjacency row is added.
- Related work adds EDLAE, Schnabel & Bennett 2020, non-backtracking and
  self-avoiding walks, gcnpop, DecRS, and multistakeholder and expected-exposure
  fairness.
- The meaning of O (self-selection on Coat and Yahoo!R3) is stated. Fairness
  claims are weakened to "not a fairness criterion". The DSA mapping is
  tightened (first-party audit; Art. 37 per Delegated Regulation 2024/436;
  Art. 40).
