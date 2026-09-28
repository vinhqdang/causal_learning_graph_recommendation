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
  real-data sparsity sweep on KuaiRec (log thinned to 30%, 10% and 3%) with
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
  W-degrees; with imputation degrees DRUP drops to about 0.62, and with
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
