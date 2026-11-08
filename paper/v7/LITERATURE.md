# v7 literature review: methods from other fields for learning under a logging policy and testing under randomised exposure

Status: first complete version, 2026-11-08. Companion of `paper/v7/DESIGN.md` (diagnosis, probes, recommended
method). Every reference below was checked on the web on 2026-11-08: the link is to the publisher, the
proceedings, arXiv or an index record that shows the stated authors, venue and year. Where only a secondary
record was found (the original was not reachable), this is said. Nothing here is cited from memory alone.

## 0. The problem structure, stated so that other fields can recognise it

- **Units.** Users u (27,077), pool videos i (7,583), content keys of the videos (tag, author, music, upload type,
  duration).
- **Training data (before T0 = 2022-04-22).** A *logging policy* (the production recommender) chose which (u, i)
  pairs were shown (impressions, 1.14M in pool). Outcome = click on shown pairs. Off-pool: 135M impressions of the
  same users on 32M other videos (lifelong history).
- **Test data (after T0).** Each feed slot was replaced with small probability by a *uniformly random* pool video
  (1.18M random impressions). The target is the per-user ranking of these random candidates by click.
- **What is shifted.** (a) *Exposure / selection*: the log only shows pairs the policy chose, mostly for users who
  were predicted to like them; the test shows pairs regardless of predicted liking. (b) *Population*: an item's
  log audience is a policy-selected sub-population; its test audience is (close to) the whole user population.
  (c) *Time*: 11-27 days later in the item life cycle (all pool videos were uploaded 2022-04-09..11).
- **What the diagnosis says matters** (DESIGN.md Section 1, diagnostic): the remaining headroom is item-level
  (LEAF-LGN 0.4735 vs LOO item oracle 0.5068; LEAF + oracle 0.5208); it is concentrated in the items with the
  highest uniform click rate (top quintile: 83% of the recoverable gain); *time drift is not the bottleneck* (the
  concurrent post-cut standard log predicts the uniform rate no better than the pre-cut log: Spearman 0.405 vs
  0.431; log growth 0.010); and *item content explains a large share of the residual if the right target were
  available* (cross-fitted GBM on pre-cut item factors fit to the test residual: R2 0.31, +24e-3 nDCG; tag alone
  +9.3e-3). The question for transfer is therefore: **which methods estimate a population-level (uniform-audience)
  item response from policy-selected audiences, with strength borrowed across items through content?**

Transfer potential: H = high (maps onto the measured headroom and is cheap to probe), M = medium (maps, but the
diagnosis limits the gain or the cost is high), L = low (maps onto a part of the problem the diagnosis rules out,
or is already exhausted in recommendation).

## 1. Survey sampling and small-area estimation

| Method (verified reference) | Problem it solves | Mapping onto ours | Already in recommendation? | Transfer |
|---|---|---|---|---|
| **Multilevel regression and post-stratification (MRP)**: Gelman and Little, "Poststratification into many categories using hierarchical logistic regression", *Survey Methodology* 23:127-135, 1997 ([record](https://en.wikipedia.org/wiki/Multilevel_regression_with_poststratification)); Park, Gelman, Bafumi, "Bayesian multilevel estimation with poststratification: state-level estimates from national polls", *Political Analysis* 2004 ([Cambridge](https://www.cambridge.org/core/journals/political-analysis/article/bayesian-multilevel-estimation-with-poststratification-statelevel-estimates-from-national-polls/22A5EF78D027E76C782B3280D400FCC9)); Wang, Rothschild, Goel, Gelman, "Forecasting elections with non-representative polls", *Int. J. Forecasting* 2015 (Xbox panel) | a non-representative sample (self-selected respondents) must give a population quantity: fit a multilevel response model on cells (demographics x region), then re-weight cell predictions by population cell counts | the logged audience of item i is a non-representative sample (selected by the policy); the uniform-insertion click rate is the *population* quantity. Cells = user strata (activity, base click rate, lifelong tag affinity); multilevel item effects nested in tag and author; post-stratify to the test-user population | not found as such (closest: propensity re-weighting by item popularity, e.g. Schnabel et al. 2016, below; it weights pairs, not user cells) | **H** - the Xbox-poll setting is structurally our setting (huge biased sample, population target) |
| **Fay-Herriot small-area model**: Fay and Herriot, "Estimates of income for small places: an application of James-Stein procedures to census data", *JASA* 74(366a):269-277, 1979 ([DOI 10.1080/01621459.1979.10482505](https://scilit.com/publications/6714384cbd8122347ad2be41930b50e9)) | many small areas with noisy direct estimates; shrink each towards a regression on area covariates, weight by sampling variance | areas = pool items (median 29 pre-cut impressions); direct estimate = the item's log-based appeal; covariates = content (tag, author lifelong statistics, upload type, duration); shrinkage weight = item log sample size | content-based cold-start models use covariates, but not as an EB prior on a selection-corrected direct estimate | **H** |
| **Raking / iterative proportional fitting** (Deming and Stephan, 1940 Census; description via [SAGE encyclopedia](https://sk.sagepub.com/ency/edvol/embed/survey/chpt/raking)); **Bayes-raking**, Si, Trangucci, Gabry, Gelman, arXiv 1901.02117 ([arXiv](https://arxiv.org/pdf/1901.02117)) | weights matching known marginals when the joint is unknown | re-weight each item's logged impressions so that the users' marginal distributions (activity, base rate, tab) match the population | no | M (a weighting version of MRP; higher variance) |

## 2. Ecology: occupancy and detection

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Site-occupancy model**: MacKenzie, Nichols, Lachman, Droege, Royle, Langtimm, "Estimating site occupancy rates when detection probabilities are less than one", *Ecology* 83(8):2248-2255, 2002 ([USGS](https://pubs.usgs.gov/publication/5224176)) | a species can be present but undetected; repeated surveys separate occupancy psi from detection p: P(no detection in K visits) = (1 - psi) + psi (1 - p)^K | site = (user, content key) or item; visit = an impression; detection = click. psi = the user truly likes the author/tag; p = probability that a liked video is clicked in one impression (user- and context-specific). Lifelong history gives *repeated surveys* of the same (user, author) and (user, tag) "sites" | **yes in spirit**: ExpoMF, Liang, Charlin, McInerney, Blei, WWW 2016, arXiv 1510.07025 ([arXiv](https://arxiv.org/pdf/1510.07025)) separates exposure from preference with a latent exposure variable; but with *observed* impressions the occupancy form (repeated detection trials) is not used | M-H for the personal channel (replaces LEAF's lift with posterior occupancy); M for item appeal |

## 3. Economics of demand: discrete choice and consideration sets; labour-economics fixed effects

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Consideration-set choice**: Akchen and Mitrofanov, "Consideration set" choice models (UCL repository, [PDF](https://discovery.ucl.ac.uk/10218717/1/akchen_mitrofanov_consideration_sets.pdf)); the classical two-stage form (consider, then choose) | choice is made from an unobserved subset; demand estimates are biased if all alternatives are assumed considered | the policy chooses the "consideration set" (impressions); the uniform test is a *forced* consideration of a random item. Attractiveness given consideration = the test estimand | **yes**: Krause et al., "The relevance of item-co-exposure for exposure bias mitigation", arXiv 2409.12912 ([arXiv](https://web3.arxiv.org/abs/2409.12912)) use multinomial-logit choice models with co-exposure to mitigate exposure bias | M (the impressions already *are* the consideration sets; the gain over CTR-given-shown models is small, probe C0 of v6) |
| **Two-way fixed effects (AKM)**: Abowd, Kramarz, Margolis, *Econometrica* 67:251-, 1999; limited-mobility bias and its leave-one-out correction (Kline, Saggio, Solvsten 2020) (both via [NBER w26653](https://www.nber.org/papers/w26653.pdf), secondary record) | worker and firm effects from a sparse, non-randomly matched bipartite panel; sorting (high-wage workers in high-wage firms) biases the firm effect | user effect + item effect from a sparse, policy-matched user-item log; "sorting" = the policy matches clicky users with some items; the item effect net of user effect is the uniform-audience appeal | the additive user + item model is a standard baseline (v6 probe I3 / A2: additive item effect + control function, 0.377-0.381, failed) | M-L (the additive version is already probed; the bias-correction view motivates hierarchical shrinkage) |

## 4. Causal inference and econometrics

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Heckman selection**: Heckman, "Sample selection bias as a specification error", *Econometrica* 47(1):153-161, 1979 ([Econometric Society](https://www.econometricsociety.org/publications/econometrica/1979/01/01/sample-selection-bias-specification-error)) | outcome observed only for a selected sample; model selection and outcome jointly | selection = impression; outcome = click | yes: Heckman-rank (Ovaisi et al., WWW 2020, arXiv 2001.11358; cited in v6) | L (v6 probe I3 failed) |
| **Double/debiased ML**: Chernozhukov et al., "Double/debiased machine learning for treatment and structural parameters", *Econometrics Journal* 21(1):C1-C68, 2018 ([NBER](https://www.nber.org/papers/w23564)) | low-dimensional target with high-dimensional nuisances; orthogonal scores + cross-fitting | target = item appeal under the uniform audience; nuisances = user propensity to click and the policy's exposure model; cross-fitting over users | DML for creator-side experiments under recommender interference: Zhan et al., arXiv 2406.14380 ([arXiv](https://arxiv.org/html/2406.14380v3)) | M (an orthogonalised version of the MRP item effect; same information) |
| **Causal forests**: Wager and Athey, "Estimation and inference of heterogeneous treatment effects using random forests", *JASA* 113(523):1228-1242, 2018 ([arXiv 1510.04342](https://arxiv.org/abs/1510.04342)) | heterogeneous effects under unconfoundedness | "treatment" = exposure of a content key to a user segment; effect heterogeneity by user features | uplift modelling in recommendation exists; not for uniform-insertion appeal | L (the target is not an effect of exposure but a response given exposure) |
| **Synthetic control**: Abadie, Diamond, Hainmueller, *JASA* 105(490):493-505, 2010 ([MIT DSpace](https://dspace.mit.edu/handle/1721.1/59447)) | counterfactual trajectory of a treated unit as a weighted combination of donors | item = unit; its uniform appeal as a weighted combination of out-of-pool "donor" videos of the same author / tag whose trajectories are seen pre-cut | Kang and Kumar, arXiv 2301.12571, synthetic controls as an oracle in bandit recommendation ([arXiv](https://arxiv.org/pdf/2301.12571)) | L-M (no donor has a uniform-audience outcome either) |
| **Instrumental variables**: IV4Rec, Si et al., WWW 2022, arXiv 2202.04514 ([arXiv](https://arxiv.org/abs/2202.04514v2)) (search behaviour as an instrument) | endogenous treatment | an instrument for exposure would be exogenous variation in being shown (e.g. tab, time of day, server load); KuaiRand ships tab but tab exposure is policy-driven | yes (IV4Rec) | L (no credible instrument in the pre-cut log; tab adjustment failed in v6) |
| **IPS / DR estimation for MNAR ratings**: Schnabel et al., "Recommendations as treatments", *ICML* 2016 ([PMLR](https://proceedings.mlr.press/v48/schnabel16.html)); Wang, Zhang, Sun, Qi, "Doubly robust joint learning ... missing not at random", *ICML* 2019 ([ICML](https://icml.cc/virtual/2019/poster/3717)); Saito et al., "Unbiased recommender learning from MNAR implicit feedback", *WSDM* 2020 ([arXiv 1909.03601](https://arxiv.org/pdf/1909.03601)) | unbiased loss under known or estimated propensities | the QP family (our strongest baseline) is in this family | yes (it is the baseline) | L (already the baseline) |

## 5. Domain adaptation and distribution shift

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Importance weighting under covariate shift**: Shimodaira, *J. Stat. Plan. Inference* 2000 (secondary record via [arXiv 2410.14315](https://arxiv.org/pdf/2410.14315v2)) | train and test covariate distributions differ, p(y given x) fixed | x = (user, item) pair features; train distribution = policy, test = uniform | IPS in recommendation is this | L (= IPS family) |
| **Label shift (BBSE)**: Lipton, Wang, Smola, *ICML* 2018 ([PMLR](https://proceedings.mlr.press/v80/lipton18a.html)) | p(y) shifts, p(x given y) fixed | click rate shifts from 0.46 (log) to 0.18 (uniform) | no | L (a global shift does not change per-user rankings) |
| **Group DRO**: Sagawa, Koh, Hashimoto, Liang, *ICLR* 2020, arXiv 1911.08731 ([arXiv](https://arxiv.org/abs/1911.08731v2)) | worst-group loss | groups = tags or popularity strata | DRO variants exist in recommendation | L-M |
| **Invariant risk minimisation**: Arjovsky, Bottou, Gulrajani, Lopez-Paz, arXiv 1907.02893, 2019 ([arXiv](https://arxiv.org/pdf/1907.02893)); InvPref, Wang et al., KDD 2022 ([record](https://www.aminer.org/pub/63034ec390e50fcafd7418c0/invariant-preference-learning-for-general-debiasing-in-recommendation)) | predictors stable across environments | environments = tabs / scenarios or exposure-intensity strata of the log | yes (InvPref) | L-M (v6 tab adjustment failed; few environments) |
| **Test-time adaptation (Tent)**: Wang, Shelhamer, Liu, Olshausen, Darrell, *ICLR* 2021, arXiv 2006.10726 ([arXiv](https://arxiv.org/abs/2006.10726)) | adapt on unlabelled test inputs | unlabelled test inputs = the random candidate lists (which items were inserted); no outcomes | no | L (the candidate lists are uniform by design, so they carry no information about appeal) |

## 6. Time series and demand forecasting

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Bass diffusion**: Bass, "A new product growth for model consumer durables", *Management Science* 15(5):215-227, 1969 ([record](https://en.wikipedia.org/wiki/Bass_diffusion_model)) | adoption curve of a new product | each pool video is a new product (uploaded 3 days before the log starts); forecast its appeal at test age | not found for uniform appeal | L (diagnosis: drift is not the bottleneck) |
| **Early popularity predicts late popularity**: Szabo and Huberman, "Predicting the popularity of online content", *CACM* 53(8):80-88, 2010 ([arXiv 0811.0405](https://ar5iv.labs.arxiv.org/html/0811.0405)) | log-popularity at t1 is linear in log-popularity at t0 | pre-cut log popularity -> test-period popularity | used in news/video popularity work | L (it forecasts *policy* popularity, which correlates 0.40 with uniform appeal, not more than the pre-cut log) |
| **Hierarchical forecast reconciliation (MinT)**: Wickramasuriya, Athanasopoulos, Hyndman, *JASA* 114(526):804-819, 2019 ([EconPapers](https://econpapers.repec.org/RePEc:taf:jnlasa:v:114:y:2019:i:526:p:804-819)) | coherent forecasts over a hierarchy | item -> author -> tag hierarchy of appeal estimates | no | M (as a pooling device, equivalent in spirit to Fay-Herriot) |
| **Popularity-drift-aware recommendation**: PDA, Zhang et al., SIGIR 2021, arXiv 2105.06067 ([arXiv](https://arxiv.org/pdf/2105.06067)); PARE, arXiv 2308.08799 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2308.08799)); CausalEPP, arXiv 2505.14310 ([arXiv](https://arxiv.org/html/2505.14310v2)) | forecast item popularity for the next stage | as above | yes | L (same reason) |
| **Survival / hazard models of timeliness**: Huang et al., "Personal or general? A hybrid strategy with multi-factors for news recommendation", TOIS 2023 ([record](https://www.aminer.org/pub/62f7958d90e50fcafdeca551/personal-or-general-a-hybrid-strategy-with-multi-factors-for-news-recommendation)); LIME, arXiv 2508.13064 ([arXiv](https://web3.arxiv.org/pdf/2508.13064)) | user-independent decay of an item's click probability with age | test age 11-27 days | yes (news) | L (all pool items have the same age within 2 days) |

## 7. Off-policy learning and bandits

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Counterfactual risk minimisation**: Swaminathan and Joachims, *JMLR* 16:1731-1755, 2015 ([JMLR](https://jmlr.org/papers/v16/swaminathan15a.html)) | learn a policy from logged bandit feedback with propensities | needs the logging propensities; KuaiRand ships none | yes | L |
| **Doubly robust policy evaluation**: Dudik, Langford, Li, *ICML* 2011, arXiv 1103.4601 ([arXiv](https://www.arxiv.org/abs/1103.4601)) | OPE with a reward model and propensities | as above | yes | L |
| **Slate OPE (pseudoinverse)**: Swaminathan et al., *NeurIPS* 2017, arXiv 1605.04812 ([NeurIPS](https://papers.NeurIPS.cc/paper_files/paper/2017/file/5352696a9ca3397beb79f116f3a33991-Paper.pdf)) | whole-slate metrics from per-slot rewards | the random slot is one slot of a feed | yes | L |
| **Learning from a small uniform sample**: AutoDebias, Chen et al., SIGIR 2021, arXiv 2105.04170 ([arXiv](https://arxiv.org/abs/2105.04170)) | meta-learn debiasing parameters on a small uniform set | the random-exposure data of other users | yes | **not admissible** under the protocol (no information after the cut, and random data for validation / test only); reported in DESIGN.md as an off-protocol diagnostic |

## 8. Information retrieval: unbiased learning to rank and click models

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Unbiased LTR**: Joachims, Swaminathan, Schnabel, *WSDM* 2017 ([PDF](https://www.cs.cornell.edu/~schnabts/downloads/joachims2017ltr.pdf)) | clicks biased by examination (position) | examination = whether the user actually looked at the shown video (KuaiRand records play time: an impression with almost no play is "not examined") | IPS-type recommendation losses | M |
| **Regression-EM position-bias estimation**: Wang, Golbandi, Bendersky, Metzler, Najork, *WSDM* 2018 ([Google](https://research.google.com/pubs/archive/3bace79f9bcead0b20dec31e2a0878346ad2fb0d.pdf)) | estimate examination and relevance jointly from clicks without randomisation | P(click) = P(examined given user, context) x P(attractive given item): a user-specific examination term in place of position | not with user-specific examination for item appeal | **H** (a multiplicative "detection x attractiveness" item appeal, fitted by EM on the pre-cut log; the IR twin of the occupancy model) |
| **Click models (DBN)**: Chapelle and Zhang, *WWW* 2009 ([proceedings](https://archives.iw3c2.org/www2009/proceedings/pdf/p1.pdf)) | perceived vs actual relevance, cascade scan | click (perceived, cover) vs long view (actual) | partly (multi-behaviour models) | L-M (multi-behaviour inputs failed in v6) |

## 9. Missing-not-at-random matrix completion

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| Marlin, Zemel, Roweis, Slaney, "Collaborative filtering and the missing at random assumption", *UAI* 2007, arXiv 1206.5267 ([arXiv](https://arxiv.org/pdf/1206.5267)); Marlin and Zemel, *RecSys* 2009 ([slides](https://people.cs.umass.edu/~marlin/research/presentations/acmrec2009-MarlinZemel.pdf)); Steck, "Training and testing of recommender systems on data missing not at random", *KDD* 2010 ([video](https://new.videolectures.net/videos/kdd2010_steck_ttrs)); Ma and Chen, "Missing not at random in matrix completion", *NeurIPS* 2019, arXiv 1910.12774 ([NeurIPS](https://papers.neurips.cc/paper_files/paper/2019/file/d828725179d622a56f951e527a966ed7-Paper.pdf)) | joint models of missingness and value; 1-bit completion of the mask for propensities | the impression mask is fully observed; 1-bit completion of the mask = a low-rank exposure model | yes (these are recommendation papers) | L (the propensity route is the QP family) |

## 10. Graph signal processing, Bayesian hierarchical models, calibration, representation learning

| Method | Problem | Mapping | In recommendation? | Transfer |
|---|---|---|---|---|
| **Graph filtering CF (GF-CF)**: Shen et al., "How powerful is graph convolution for recommendation?", *CIKM* 2021, arXiv 2108.07567 ([arXiv](https://arxiv.org/abs/2108.07567v1)) | closed-form low-pass filters on the item graph | an alternative backbone | yes | L (backbone swap; the headroom is item-level, not collaborative) |
| **Empirical Bayes / James-Stein**: Efron and Morris, "Data analysis using Stein's estimator and its generalizations", *JASA* 70(350):311-319, 1975 ([PDF](https://faculty.ucmerced.edu/jvevea/classes/290_21/readings/week%204/Efron%20and%20Morris.pdf)) | shrink many noisy rates (batting averages) to a common mean | item log rates (median 29 impressions) shrunk to tag / author means | Beta-binomial shrinkage is common (LEAF's lift uses it) | H as a component (inside MRP / Fay-Herriot) |
| **Calibrated recommendations**: Steck, *RecSys* 2018 ([DOI 10.1145/3240323.3240372](https://bibbase.org/network/publication/steck-calibratedrecommendations)) | match the genre mix of a list to the user's history | re-rank the random candidates so the list's tag mix matches the user's lifelong tag mix | yes | L-M (a list-level constraint; nDCG on 43 random candidates rewards item order, not mix) |
| **Conformal recommenders**: Kagita et al., *Information Sciences* 2017; Zammali et al., arXiv 2412.12110 ([arXiv](https://arxiv.org/abs/2412.12110v1)); CPFT, arXiv 2402.08976 ([arXiv](https://web3.arxiv.org/abs/2402.08976)) | coverage-guaranteed prediction sets | no set-valued target here | yes | L |
| **Cold-start content models / meta warm-up**: Meta-Embedding (SIGIR 2019), MWUF, GIFT (CIKM 2022), EmerG, arXiv 2407.10112 ([arXiv](https://arxiv.org/pdf/2407.10112)) | embeddings for new items from content | all pool items are new (uploaded 2022-04-09..11); content = tag, author, music, duration, cover size | yes | M (needs a target; trained on the policy log they learn policy CTR) |
| **Feature-level bias in CTR**: arXiv 2402.03600 ([arXiv](https://arxiv.org/pdf/2402.03600)), evaluated on KuaiRand random exposure | CTR models over-use features correlated with exposure | the same mechanism as our item-level miscalibration | yes (on KuaiRand) | M (closest recent work using the same test) |
| **Foundation-model content embeddings** (text / vision) | rich item content | KuaiRand ships **no** titles, covers or frames; only ids, tags, duration, sizes | - | not applicable (no raw content) |

## 11. Ranked transferable ideas (detail and probes in DESIGN.md Section 3)

Ranked by expected gain on the measured headroom x feasibility on 4 CPU cores x novelty:

1. **MRP item appeal** (survey sampling): multilevel logistic model of pre-cut clicks with user strata and item
   effects nested in author and tag, post-stratified to the test-user population. Probe P1.
2. **Fay-Herriot content prior on the MRP item effect** (small-area estimation): item effect shrunk to a content
   regression with weight from the item's log sample size. Probe P2 (part of P1's grid).
3. **Detection x attractiveness item appeal by regression-EM** (IR click models / ecology occupancy): P(click) =
   theta_u x alpha_i on the pre-cut log, alpha_i with a hierarchical Beta prior over tag / author. Probe P3.
4. **Occupancy posterior as the personal channel** (ecology): replace LEAF's lift by P(u likes key v | n_uv
   impressions, c_uv clicks) under the occupancy likelihood with user-specific detection. Probe P4.
5. **Trajectory forecast of appeal** (demand forecasting, Szabo-Huberman / Bass): per-item log-linear daily trend
   of pre-cut clicks extrapolated to the test window. Probe P5 (cheap control: diagnosis predicts no gain).
6. Raking weights (survey sampling): impression weights matching user marginals; weighted item CTR.
7. Orthogonalised (DML) item effect with cross-fitted user-propensity nuisance.
8. Hierarchical reconciliation (MinT) of item / author / tag appeal estimates.
9. Two-way fixed effects with leave-one-out limited-mobility correction (AKM / KSS).
10. Group-DRO over tag strata for the CF backbone.
11. Calibrated re-ranking to the lifelong tag mix (Steck 2018).
12. IRM across tabs / exposure-intensity environments.

Ideas 1-5 are probed; 6-12 are listed with the reason they rank lower (they re-express ideas 1-3 with more
variance, or address parts of the problem the diagnosis rules out).
