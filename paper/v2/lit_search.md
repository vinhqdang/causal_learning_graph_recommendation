# Literature search: test of the related-work gap statement

Search window: 2019 to October 2026. Date of search: 2026-10-29.
Claim tested (s2_related.tex, "Evaluation protocols and the gap"; also s1_intro.tex):

> "We are not aware of a previous study that combines graph-based CF (LightGCN-style) with propensity/doubly-robust/imputation nuisance models, trained without uniform data, and evaluated on these unbiased-test benchmarks."

Method under test: LightGCN with BPR, sampled negatives down-weighted by (1 - fn*Q), Q = Yhat(1-p)/(1-p*Yhat), plus a fixed standardised offset beta*z(logit Yhat); uniform data only for validation and testing; Coat, Yahoo! R3, KuaiRand-Pure, KuaiRec.

Method of checking: for every paper listed as "close" or "partial", I downloaded the full text and read the dataset, split, baseline and implementation sections. Where a detail could not be confirmed from the text, it is marked "not stated".

---

## 1. Queries run

Queries (each run once, results screened by title/abstract, then full text for candidates):

1. doubly robust LightGCN graph collaborative filtering propensity imputation unbiased recommendation Coat Yahoo R3
2. debiased graph collaborative filtering inverse propensity weighting LightGCN missing not at random unbiased test set
3. exposure-aware negative sampling graph collaborative filtering false negative propensity
4. causal graph neural network recommendation KuaiRand KuaiRec Coat Yahoo debiasing GNN
5. training-free graph filter collaborative filtering ChebyCF Turbo-CF BSPM GF-CF
6. doubly robust graph neural network recommendation selection bias imputation graph
7. AutoDebias learning to debias for recommendation uniform data training validation test Yahoo R3 Coat protocol
8. KDCRec knowledge distillation counterfactual recommendation uniform data Yahoo Coat
9. KBDR kernel balancing doubly robust recommendation debiased
10. graph neural network recommendation exposure bias propensity LightGCN KuaiRec unbiased evaluation 2024 2025
11. ChebyCF Chebyshev graph filter collaborative filtering
12. graph-based recommendation missing not at random doubly robust GNN imputation propensity KuaiRand
13. LightGCN doubly robust OR propensity "false negative" recommendation unobserved items posterior positive exposure imputation graph
14. unbiased recommender learning implicit feedback graph neural network BPR propensity negative sampling exposure MNAR Coat Yahoo R3 LightGCN backbone
15. Interpolative Distillation for Unifying Biased and Debiased Recommendation (InterD)
16. LightGCN KuaiRand unbiased test debiasing graph collaborative filtering random exposure videos propensity
17. propensity-aware negative sampling collaborative filtering exposure probability false negatives implicit feedback unbiased evaluation
18. graph filtering collaborative filtering debiasing exposure bias training-free propensity
19. KDCRec "A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data" (bibliographic lookup)

Also read in full or in the relevant sections: the reference lists of AdvDrop, DR-GNN, PPD, DPAA and ConAlign, to catch graph-debiasing papers the search engine did not return.

Limitation: the search was run on a general web search engine, not a bibliographic database. Absence of a hit is not proof of absence. No hit was found for any paper that combines a graph backbone with a posterior built from propensity and imputation to weight negatives.

---

## 2. Relevant papers

Closeness scale: close = shares at least two of {graph backbone, propensity/DR/imputation, unbiased-test benchmarks, no uniform data in training}; partial = shares one; far = related theme only.

### 2.1 Graph recommenders evaluated on unbiased-test benchmarks

**A. AdvDrop** (not currently cited)
- An Zhang, Wenchang Ma, Pengbo Wei, Leheng Sheng, Xiang Wang. "General Debiasing for Graph-based Collaborative Filtering via Adversarial Graph Dropout." WWW 2024. DOI 10.1145/3589334.3645667; arXiv 2402.13769.
- What it does: learns an adversarial edge-dropout that splits the interaction graph into bias-aware and bias-mitigated views, then trains a LightGCN-style model (2 layers) with BPR plus a contrastive objective. Evaluation: Coat and Yahoo (biased train, random-exposure test), KuaiRec (sparse train, fully observed test), Douban (temporal split), Yelp2018 (ID/OOD). Baselines include IPS-CN (propensity from item popularity in the training set, with clipping and normalisation), DR (imputation with "predefined scores" plus IPS), CVIB, InvPref and AutoDebias, reported relative to LightGCN. Protocol: the appendix describes the Coat/Yahoo/KuaiRec test sets as unbiased; the text I could extract does not describe a validation split for Coat/Yahoo/KuaiRec and does not say that AdvDrop itself uses uniform data for training. Only the AutoDebias baseline is described as using "a small set of uniform data". Per-baseline backbone is not stated beyond "number of layers of LightGCN set to 2".
- Closeness: **close** on setting (graph CF + IPS/DR baselines + Coat/Yahoo/KuaiRec unbiased tests). It does not use a propensity/imputation posterior, does not reweight negatives, and its DR uses fixed imputed scores rather than an IPS-fitted imputation model. Its reported DR is weak on KuaiRec (NDCG@20 -88.9% relative to LightGCN), so it is not a strong DR graph baseline.

**B. PPD (post-hoc popularity debiasing)** (not currently cited)
- Md Aminul Islam, Elena Zheleva, Ren Wang. "Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering." WWW 2026. DOI 10.1145/3774904.3792670; arXiv 2510.12959.
- What it does: removes a popularity direction from pre-trained LightGCN embeddings without retraining. Evaluated on KuaiRec, Coat and Yahoo! R3 with unbiased test data; baselines include MACR, PPAC, IPSCN (IPW), SAM-REG and a post-hoc method. Protocol: training on the biased log; for KuaiRec and Coat the unbiased test data are split one-third validation and two-thirds test; for Yahoo! R3 validation is sampled from the training set and the unbiased set is used as test only. So it does use uniform data for model selection on two of three datasets.
- Closeness: **close** on benchmarks and backbone, **partial** on method (popularity-based IPW baseline, no imputation/DR, no negative weighting).

**C. DPAA** (already cited as `dpaa2026`)
- Md Aminul Islam, Ahmed Sayeed Faruk, Sourav Medya, Elena Zheleva. "Debiasing Message Passing to Mitigate Popularity Bias in GNN-based Collaborative Filtering." arXiv 2605.11145 (v2, 5 Oct 2026). Preprint.
- What it does: re-weights message passing with embedding-aware, interaction-level weights and layer-wise weights on LightGCN. Evaluated on Coat, KuaiRec and Yahoo! R3 (unbiased tests) with baselines PPAC, MACR, CVIB, NAVIP, IPS (max-capped normalised IPW), SAM-REG. In its semi-synthetic KuaiRec study, 10% of the unbiased data is used for validation and 20% for testing. For the real-data experiments the text states early stopping on a validation set; the construction of that set is not described in the sections I read.
- Closeness: **partial** (graph backbone, IPS baseline, same three unbiased benchmarks; acts on aggregation, no imputation).
- Note for the paper: the existing bib entry matches the verified authors, title and eprint. The entry is a `@misc` with note "Preprint"; a v2 date of 5 Oct 2026 is visible.

**D. DR-GNN**
- Bohao Wang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yang Gao, Yan Feng, Chun Chen, Can Wang. "Distributionally Robust Graph-based Recommendation System." WWW 2024. DOI 10.1145/3589334.3645598; arXiv 2402.12994.
- What it does: distributionally robust optimisation over graph neighbourhoods plus a graph-structure regulariser, on LightGCN with BPR. The "exposure shift" experiments use Coat and Yahoo (biased train, random-exposure test). Baselines are InvCF, BOD, APDA and others; the text does not use propensity or imputation. Validation protocol not stated in the sections I read. (The abbreviation "DR" here means distributionally robust, not doubly robust; worth a footnote if cited.)
- Closeness: **partial** (graph backbone and Coat/Yahoo unbiased tests; no propensity/imputation).

**E. NAVIP** (already cited as `kim2022navip`)
- Minseok Kim, Jinoh Oh, Jaeyoung Do, Sungjin Lee. "Debiasing Neighbor Aggregation for Graph Neural Network in Recommender Systems." CIKM 2022. DOI 10.1145/3511808.3557576; arXiv 2208.08847.
- What it does: weights LightGCN neighbour aggregation by inverse propensity (propensity from relative item popularity). Datasets: Alexa, Yahoo, MovieLens. Protocol: Yahoo uses the original random-exposure test; the validation set is built by sampling 5% of data inversely to relative item popularity; for Alexa and MovieLens both validation and test are pseudo-unbiased popularity-based samples. It does not train on uniform data.
- Closeness: **close** on backbone, IPS and protocol principle (no uniform data in training, Yahoo unbiased test); **far** on nuisance models (popularity propensity only, no imputation) and datasets (only Yahoo overlaps).

**F. IPS-weighted BPR on LightGCN with SNIPS evaluation**
- Rahul Raja, Arpita Vats. "Counterfactual Risk Minimization with IPS-Weighted BPR and Self-Normalized Evaluation in Recommender Systems." Causality, Counterfactuals and Sequential Decision-Making Workshop at RecSys 2025. arXiv 2509.00333.
- What it does: IPS-weighted BPR on LightGCN (64 dimensions, 3 layers) with a regulariser on large weights; propensities come from simulated logging (softmax over popularity on MovieLens, an analytic policy on synthetic data). Evaluation is SNIPS-based offline, on MovieLens and a synthetic set; no randomised test set. No imputation or DR. The paper has internal inconsistencies (dataset description differs between Section 3.4 and Appendix B).
- Closeness: **partial** (LightGCN + propensity, but simulated, no unbiased benchmarks, no DR). Cite with care; workshop paper.

### 2.2 Doubly robust and propensity methods with non-graph backbones (protocol checks)

**G. AutoDebias**
- Jiawei Chen, Hande Dong, Yang Qiu, Xiangnan He, Xin Xin, Liang Chen, Guli Lin, Keping Yang. "AutoDebias: Learning to Debias for Recommendation." SIGIR 2021. DOI 10.1145/3404835.3462919; arXiv 2105.04170.
- Protocol (verified in text): on Yahoo!R3 and Coat the biased data are the training set and the unbiased data are split 5% uniform set to help training, 5% validation, 90% test. Meta-learning with a bi-level objective; MF-style backbone. Uses uniform data for training and for selection.
- Closeness: **far** (non-graph, uses uniform data). Supports the protocol claim.

**H. CausE**
- Stephen Bonner, Flavian Vasile. "Causal Embeddings for Recommendation." RecSys 2018. arXiv 1706.07639 (DOI not verified).
- Protocol (verified in text): a small sample from a randomised policy enters training; on MovieLens/Netflix the data are split 70% train (60% logged policy plus 10% randomised), 10% validation (all from the logged policy), 20% test (all from the randomised policy). Not evaluated on Coat/Yahoo in this paper. Domain-adaptation-style matrix factorisation.
- Closeness: **far**. Uses uniform data for training; validation from the logged policy.

**I. KDCRec**
- Dugang Liu, Pengxiang Cheng, Zhenhua Dong, Xiuqiang He, Weike Pan, Zhong Ming. "A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data." SIGIR 2020, pp. 831-840. DOI 10.1145/3397271.3401083.
- What it does: distils from a model fitted on a small uniform dataset into a model trained on the biased log (label-, feature-, sample- and model-structure-based variants). The abstract-level description confirms that uniform data enter training; I could not open the full text, so the exact split proportions are not verified here.
- Closeness: **far**. Uses uniform data for training.

**J. KBDR / KBIPS**
- Haoxuan Li, Chunyuan Zheng, Yanghao Xiao, Peng Wu, Zhi Geng, Xu Chen, Peng Cui. "Debiased Collaborative Filtering with Kernel-Based Causal Balancing." ICLR 2024; arXiv 2404.19596.
- Protocol: Coat, Yahoo (called Music) and a Product dataset (KuaiRec-sized); MF/NCF-style models. The experimental section states only that it "follows previous studies"; the exact role of the unbiased ratings (validation versus test) is not stated in the text I read. Training does not obviously use uniform ratings, but I could not confirm it.
- Closeness: **partial** on the nuisance models (propensity + imputation), **far** on graph.

**K. CDR (conservative doubly robust)**
- Zijie Song, Jiawei Chen, Sheng Zhou, Qihao Shi, Yan Feng, Chun Chen, Can Wang. "CDR: Conservative Doubly Robust Learning for Debiased Recommendation." CIKM 2023. DOI 10.1145/3583780.3614805; arXiv 2308.08461.
- Protocol (verified): Coat, Yahoo!R3 and KuaiRand. The biased data are the training set; the unbiased data are used for validation (10%) and evaluation (90%). DR-family baselines (EIB, DR-JL, MRDR-JL, StableDR style), matrix factorisation.
- Closeness: **partial**. This is the same train/validate/test usage of uniform data as ours, with KuaiRand, but a non-graph backbone and a rating-prediction objective.

**L. Doubly Calibrated Estimator (DCE)**
- Wonbin Kweon, Hwanjo Yu. "Doubly Calibrated Estimator for Recommendation on Data Missing Not At Random." WWW 2024. DOI 10.1145/3589334.3645417; arXiv 2403.00817.
- Protocol (verified): Coat, Yahoo!R3 and KuaiRec; "hold out 10% of the training set as the validation set"; the unbiased test set is not used for training or for propensity estimation; methods that use a fraction of the unbiased test set in training (it names [2, 4, 9, 29, 50]) are excluded for fairness. Base architectures are MF or NCF.
- Closeness: **partial**. Shows that the protocol of training without uniform data on Coat/Yahoo/KuaiRec is already used in the DR literature, but with non-graph backbones and explicit-rating DR losses.

**M. StableDR**
- Haoxuan Li, Chunyuan Zheng, Peng Wu. "StableDR: Stabilized Doubly Robust Learning for Recommendation on Data Missing Not at Random." ICLR 2023; arXiv 2205.04701.
- Protocol: Coat and Yahoo!R3 with MNAR training and MAR ratings used for evaluation; MF, SLIM, NCF. The sections I read do not state whether any MAR data are used for validation. Already conceptually close to existing citations (`li2023stabledr`).
- Closeness: **far** for our claim.

**N. BISER**
- Jae-woong Lee, Seongmin Park, Joonseok Lee, Jongwuk Lee. "Bilateral Self-unbiased Learning from Biased Implicit Feedback." SIGIR 2022; arXiv 2207.12660 (DOI placeholder in the arXiv version; not verified).
- What it does: implicit-feedback MF/autoencoder training with a self-inverse-propensity weight and a bilateral objective that corrects exposure bias from the model itself. Protocol (verified): on Coat and Yahoo! R3 "train the model on the training dataset collected under the MNAR assumption and evaluate it on the test set collected under MAR"; 30% of each user's ratings in the training set are held out for validation. So no uniform data in training or selection.
- Closeness: **partial** (implicit feedback, propensity on exposure, Coat/Yahoo protocol identical to ours for validation); non-graph.

**O. ConAlign**
- Jingcheng Zhang, Yihan Wang, Qi Song, Liyin Hong. "ConAlign: Conditional Alignment Framework for Balancing Biased and Unbiased Recommendation." arXiv 2607.24092 (July 2026); preprint.
- Protocol (verified): Coat, Yahoo! R3, KuaiRand-Pure. Coat and Yahoo: 70% of biased and 50% of unbiased data train, 10% each validation, 20% biased and 40% unbiased test. KuaiRand-Pure: biased and unbiased data from the first two weeks train, day 15 validation, last two days test. A dual-tower design with an unbiased tower trained on unbiased data; baselines include MF, MF-IPS, MF-DR; no graph backbone.
- Closeness: **far** (uses uniform data for training). Directly supports the statement that some recent debiasing papers train on KuaiRand's random-exposure data.

**P. InterD**
- Sihao Ding, Fuli Feng, Xiangnan He, Jinqiu Jin, Wenjie Wang, Yong Liao, Yongdong Zhang. "Interpolative Distillation for Unifying Biased and Debiased Recommendation." SIGIR 2022, pp. 40-49. DOI 10.1145/3477495.3532002.
- What it does: distils from a biased and a debiased teacher, using a random dataset alongside biased data (from the repository description and from how ConAlign describes it). Exact split not verified.
- Closeness: **far**.

### 2.3 Negative weighting, false negatives, exposure

**Q. Correct-and-Weight (CW) loss**
- Minglei Yin, Chuanbo Hu, Bin Liu, Neil Zhenqiang Gong, Yanfang Ye, Xin Li. "Correct and Weight: A Simple Yet Effective Loss for Implicit Feedback Recommendation." arXiv 2601.04291 (January 2026). Preprint.
- What it does: a loss correction and a weighting of negatives to reduce the effect of false negatives, tested on MF, LightGCN and XSimGCL. It uses standard random 80/20 splits with 10-core filtering, no propensity or imputation, no unbiased-test benchmarks.
- Closeness: **partial** (graph backbones and false-negative weighting of sampled negatives, but weights are not from propensity/imputation and evaluation is on random splits). Should be cited as the nearest graph-CF false-negative loss in the recent literature.

**R. ICPNS (In-Community Popularity Negative Sampling)**
- Chen Chen, Haobo Lin, Yuanbo Xu. "Towards Reliable Negative Sampling for Recommendation with Implicit Feedback via In-Community Popularity." arXiv 2602.18759 (February 2026). Preprint.
- What it does: treats an unobserved pair as a reliable negative only if the user was exposed, and approximates exposure by in-community item popularity for sampling; shown with LightGCN-style and MF backbones on ML-100K, ML-1M, Yelp, Beauty. No unbiased-test benchmarks, no imputation.
- Closeness: **partial** (exposure-aware negative sampling for graph CF; popularity proxy, not propensity + imputation).

**S. PDNS**
- Kexin Shi, Yun Zhang, Wenjia Wang, Bingyi Jing. "Enhancing Recommender Systems: A Strategy to Mitigate False Negative Impact." arXiv 2211.13912 (v2, March 2024).
- What it does: hard-negative synthesis with false-negative handling, tested with MF and LightGCN. Random splits on MovieLens-100k and Douban per the search summary; no propensity or unbiased-test evaluation found.
- Closeness: **far**.

### 2.4 Training-free and graph-filter CF

**T. Turbo-CF**
- Jin-Duk Park, Yong-Min Shin, Won-Yong Shin. "Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation." SIGIR 2024. DOI 10.1145/3626772.3657916; arXiv 2404.14243.
- Polynomial graph filter, no training and no matrix decomposition. Evaluated on Gowalla, Yelp and similar standard benchmarks (the datasets visible in the text: Gowalla, Yelp); no debiasing, no propensity.
- Closeness: **far**.

**U. ChebyCF**
- Chanwoo Kim, Jinkyu Sung, Yebonn Han, Joonseok Lee. "Graph Spectral Filtering with Chebyshev Interpolation for Recommendation." SIGIR 2025; arXiv 2505.00552. DOI in the arXiv copy is a placeholder (not verified).
- Chebyshev interpolation of a non-linear graph filter over the item-item graph, with an ideal-pass filter and degree normalisation. Datasets visible: Gowalla, Amazon-Book (and one more not read). No unbiased test, no propensity.
- Closeness: **far**.

**V. GF-CF and BSPM** (already cited as `shen2021gfcf` and `choi2023bspm`). The searches for graph-filter CF returned no paper that adds propensity, DR or imputation nuisance models to a graph filter, and no unbiased-test evaluation of graph filters other than the Islam et al. papers above. I did not re-read GF-CF and BSPM.

### 2.5 Other non-graph DR / MNAR items seen but not used

Doubly robust learning variants (MR, relaxed-imputation DR from ICML 2024, doubly calibrated DR, and others) appeared repeatedly in the DR results; none uses a graph backbone. Hidden-confounding MNAR work from 2026 (arXiv 2605.21066) appeared in one search; I did not read it and do not recommend citing it.

---

## 3. Protocol claims

| Method | Uniform data for training | Uniform data for validation/selection | Source of verification |
|---|---|---|---|
| AutoDebias (SIGIR 2021) | Yes, 5% | Yes, 5% (test 90%) | Full text |
| CausE (RecSys 2018) | Yes, 10% of randomised sample injected | No (validation from logged policy) | Full text |
| KDCRec (SIGIR 2020) | Yes (core premise) | Not verified | Abstract-level only |
| ConAlign (arXiv 2026) | Yes, 50% of unbiased data (Coat, Yahoo); first two weeks of unbiased logs (KuaiRand) | Yes, 10% and day 15 | Full text |
| InterD (SIGIR 2022) | Yes (uses a random dataset alongside biased data) | Not verified | Repository and secondary description |
| CDR (CIKM 2023) | No | Yes, 10% of unbiased data (test 90%) | Full text |
| KBDR (ICLR 2024) | Not stated; likely no | Not stated | Full text, section silent |
| DCE (WWW 2024) | No (explicitly excludes methods that use it) | No (10% of biased train) | Full text |
| BISER (SIGIR 2022) | No | No (30% of biased train per user) | Full text |
| StableDR (ICLR 2023) | No | Not stated | Full text, section silent |
| NAVIP (CIKM 2022) | No | Popularity-sampled pseudo-unbiased validation, not uniform | Full text |
| PPD (WWW 2026) | No | Yes, one-third of unbiased test on KuaiRec/Coat; biased-train sample on Yahoo | Full text |
| DPAA (2026) | No | Validation present; construction not confirmed for real data | Full text, partial |
| AdvDrop (WWW 2024) | No for AdvDrop (AutoDebias baseline yes) | Not stated | Full text |

Consequences for the paper:
- "Some debiasing methods use a small uniform sample for training, as in AutoDebias and CausE" is accurate. KDCRec, InterD and ConAlign can be added. The statement that CausE uses uniform data for training holds, but CausE was not evaluated on Coat or Yahoo in its original paper; if the paper implies it was, rephrase.
- Using unbiased data only for validation and testing is not unique to this paper. CDR (validation 10%, evaluation 90%, includes KuaiRand) and PPD (partial validation from the unbiased test) do it. DCE and BISER avoid uniform data for validation as well. The paper's wording "Some reported debiasing gains rely on protocols that train on uniformly collected data" is correct; it should not suggest that validation on uniform data is unusual.
- Coat note: the shipped propensities were estimated by the dataset authors from a uniform sample; DCE explicitly states it does not use the unbiased test set for propensity estimation. The paper already discloses this.

---

## 4. Verdict on the gap statement

As worded, the statement does **not** hold. The sentence has four conjuncts, and each of them separately has a counterexample:

1. Graph CF with propensity weighting: NAVIP (LightGCN, IPS), IPS-weighted BPR on LightGCN (Raja and Vats), and the IPS baselines in AdvDrop, PPD and DPAA.
2. DR/imputation nuisance with a graph backbone: AdvDrop reports a DR baseline next to LightGCN (imputation with predefined scores plus IPS). It is a weak, fixed-imputation variant, but it is a graph-CF-plus-DR comparison.
3. Trained without uniform data: NAVIP, PPD, DPAA and (as far as the text shows) AdvDrop's own method.
4. Evaluated on unbiased-test benchmarks: AdvDrop (Coat, Yahoo, KuaiRec), PPD and DPAA (Coat, Yahoo! R3, KuaiRec), DR-GNN (Coat, Yahoo), NAVIP (Yahoo).

What I did **not** find, after the searches above: any paper that (a) builds a posterior probability that a non-logged pair is positive from a propensity and an IPS-fitted imputation and uses it to down-weight sampled negatives in a graph recommender's pairwise loss, (b) adds a standardised imputation logit as a score offset to a graph model, or (c) evaluates a graph model on KuaiRand-Pure together with Coat, Yahoo! R3 and KuaiRec. Nor did I find a debiasing method built on training-free graph filters (GF-CF, BSPM, Turbo-CF, ChebyCF) beyond the DRUP-type operator in this paper. These absences hold only for the queries run.

Also, the two sentences of the intro and related work ("few evaluate graph debiasing on Coat, Yahoo! R3, KuaiRand and KuaiRec together with uniform data held out from training") need correction: three of the four datasets (Coat, Yahoo! R3, KuaiRec) are already used for graph-CF debiasing by AdvDrop, PPD and DPAA; I found none using KuaiRand with a graph backbone.

Recommended wording (true given what was found), replacing the current "We are not aware..." sentence:

> Graph recommenders have been debiased with inverse propensities in neighbour aggregation \citep{kim2022navip}, with adversarial graph dropout and baselines including IPS and DR \citep{zhang2024advdrop}, and with post-hoc or message-passing corrections evaluated on Coat, Yahoo!\,R3 and KuaiRec \citep{islam2026ppd,dpaa2026}. These methods change the aggregation, the graph or the embeddings. We are not aware of a previous study that combines a propensity and an IPS-fitted imputation model into a posterior for non-logged pairs and uses it to weight the sampled negatives of a graph recommender's pairwise loss, or that adds a debiased imputation prior to a graph model's score. We are also not aware of a study that evaluates such a graph model on KuaiRand-Pure together with Coat, Yahoo!\,R3 and KuaiRec, with uniform data used only for validation and testing. This statement is based on a literature search up to October 2026.

Shorter variant for the introduction:

> To our knowledge, no earlier graph recommender uses doubly-robust-style nuisance functions to decide which sampled pairs count as negatives, and none has been evaluated on all four unbiased-test benchmarks used here.

Suggested companion edits (not applied; no existing file was changed):
- Table 1: add rows for AdvDrop (graph yes; propensity: popularity-based IPS-CN baseline only; imputation: fixed-score DR baseline only; acts on edges/graph; uniform data: no for AdvDrop) and for PPD (graph yes; no propensity; acts on embeddings post hoc; uniform data: validation only). Note that the current NAVIP row says "Uniform data: no", which matches the verified protocol.
- Remove "TODO verify literature search before submission" once the paper cites A, B and the CW loss (Q).
- In the CausE sentence, keep "trains on uniformly collected data" but avoid implying a Coat/Yahoo evaluation.
- Do not describe DR-GNN as "doubly robust"; it is distributionally robust.
- Cite CW (Yin et al.) and ICPNS next to ExpoMF/SRNS in "False negatives" as the graph-CF false-negative and exposure-aware sampling work of 2026.

---

## 5. BibTeX (new keys; no clash with keys already in refs.bib or refs_new.bib)

Fields not checked against the publisher are marked UNVERIFIED inside the entry. Every field without a marker was read in the paper itself or on the arXiv page.

```bibtex
@inproceedings{zhang2024advdrop,
  author    = {Zhang, An and Ma, Wenchang and Wei, Pengbo and Sheng, Leheng and Wang, Xiang},
  title     = {General Debiasing for Graph-based Collaborative Filtering via Adversarial Graph Dropout},
  booktitle = {Proceedings of the ACM Web Conference 2024 (WWW '24)},
  year      = {2024},
  address   = {Singapore},
  doi       = {10.1145/3589334.3645667},
  eprint    = {2402.13769},
  archivePrefix = {arXiv}
}

@inproceedings{islam2026ppd,
  author    = {Islam, Md Aminul and Zheleva, Elena and Wang, Ren},
  title     = {Post-hoc Popularity Bias Correction in {GNN}-based Collaborative Filtering},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  address   = {Dubai, United Arab Emirates},
  doi       = {10.1145/3774904.3792670},
  eprint    = {2510.12959},
  archivePrefix = {arXiv}
}

@inproceedings{wang2024drgnn,
  author    = {Wang, Bohao and Chen, Jiawei and Li, Changdong and Zhou, Sheng and Shi, Qihao and Gao, Yang and Feng, Yan and Chen, Chun and Wang, Can},
  title     = {Distributionally Robust Graph-based Recommendation System},
  booktitle = {Proceedings of the ACM Web Conference 2024 (WWW '24)},
  year      = {2024},
  address   = {Singapore},
  doi       = {10.1145/3589334.3645598},
  eprint    = {2402.12994},
  archivePrefix = {arXiv}
}

@inproceedings{kweon2024dce,
  author    = {Kweon, Wonbin and Yu, Hwanjo},
  title     = {Doubly Calibrated Estimator for Recommendation on Data Missing Not At Random},
  booktitle = {Proceedings of the ACM Web Conference 2024 (WWW '24)},
  year      = {2024},
  address   = {Singapore},
  doi       = {10.1145/3589334.3645417},
  eprint    = {2403.00817},
  archivePrefix = {arXiv}
}

@inproceedings{song2023cdr,
  author    = {Song, Zijie and Chen, Jiawei and Zhou, Sheng and Shi, Qihao and Feng, Yan and Chen, Chun and Wang, Can},
  title     = {{CDR}: Conservative Doubly Robust Learning for Debiased Recommendation},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23)},
  year      = {2023},
  address   = {Birmingham, United Kingdom},
  doi       = {10.1145/3583780.3614805},
  eprint    = {2308.08461},
  archivePrefix = {arXiv}
}

@inproceedings{li2024kbdr,
  author    = {Li, Haoxuan and Zheng, Chunyuan and Xiao, Yanghao and Wu, Peng and Geng, Zhi and Chen, Xu and Cui, Peng},
  title     = {Debiased Collaborative Filtering with Kernel-Based Causal Balancing},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2404.19596},
  archivePrefix = {arXiv}
}

@inproceedings{lee2022biser,
  author    = {Lee, Jae-woong and Park, Seongmin and Lee, Joonseok and Lee, Jongwuk},
  title     = {Bilateral Self-unbiased Learning from Biased Implicit Feedback},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)},
  year      = {2022},
  address   = {Madrid, Spain},
  eprint    = {2207.12660},
  archivePrefix = {arXiv},
  note      = {DOI UNVERIFIED (the arXiv copy carries a placeholder)}
}

@inproceedings{ding2022interd,
  author    = {Ding, Sihao and Feng, Fuli and He, Xiangnan and Jin, Jinqiu and Wang, Wenjie and Liao, Yong and Zhang, Yongdong},
  title     = {Interpolative Distillation for Unifying Biased and Debiased Recommendation},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)},
  year      = {2022},
  pages     = {40--49},
  doi       = {10.1145/3477495.3532002},
  note      = {Author list, pages and DOI taken from the official code repository and an open bibliographic record; paper full text not read. UNVERIFIED against the proceedings}
}

@inproceedings{liu2020kdcrec,
  author    = {Liu, Dugang and Cheng, Pengxiang and Dong, Zhenhua and He, Xiuqiang and Pan, Weike and Ming, Zhong},
  title     = {A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20)},
  year      = {2020},
  pages     = {831--840},
  doi       = {10.1145/3397271.3401083},
  note      = {Full text not read; bibliographic data from the ACM reference format quoted by a secondary source. UNVERIFIED against the proceedings}
}

@inproceedings{raja2025ipsbpr,
  author    = {Raja, Rahul and Vats, Arpita},
  title     = {Counterfactual Risk Minimization with {IPS}-Weighted {BPR} and Self-Normalized Evaluation in Recommender Systems},
  booktitle = {Causality, Counterfactuals and Sequential Decision-Making Workshop at RecSys 2025},
  year      = {2025},
  eprint    = {2509.00333},
  archivePrefix = {arXiv},
  note      = {Workshop paper}
}

@misc{yin2026cw,
  author    = {Yin, Minglei and Hu, Chuanbo and Liu, Bin and Gong, Neil Zhenqiang and Ye, Yanfang and Li, Xin},
  title     = {Correct and Weight: A Simple Yet Effective Loss for Implicit Feedback Recommendation},
  year      = {2026},
  eprint    = {2601.04291},
  archivePrefix = {arXiv},
  note      = {Preprint; venue UNVERIFIED}
}

@misc{chen2026icpns,
  author    = {Chen, Chen and Lin, Haobo and Xu, Yuanbo},
  title     = {Towards Reliable Negative Sampling for Recommendation with Implicit Feedback via In-Community Popularity},
  year      = {2026},
  eprint    = {2602.18759},
  archivePrefix = {arXiv},
  note      = {Preprint; venue UNVERIFIED}
}

@misc{zhang2026conalign,
  author    = {Zhang, Jingcheng and Wang, Yihan and Song, Qi and Hong, Liyin},
  title     = {{ConAlign}: Conditional Alignment Framework for Balancing Biased and Unbiased Recommendation},
  year      = {2026},
  eprint    = {2607.24092},
  archivePrefix = {arXiv},
  note      = {Preprint; venue UNVERIFIED}
}

@inproceedings{park2024turbocf,
  author    = {Park, Jin-Duk and Shin, Yong-Min and Shin, Won-Yong},
  title     = {{Turbo-CF}: Matrix Decomposition-Free Graph Filtering for Fast Recommendation},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)},
  year      = {2024},
  address   = {Washington, DC, USA},
  doi       = {10.1145/3626772.3657916},
  eprint    = {2404.14243},
  archivePrefix = {arXiv}
}

@inproceedings{kim2025chebycf,
  author    = {Kim, Chanwoo and Sung, Jinkyu and Han, Yebonn and Lee, Joonseok},
  title     = {Graph Spectral Filtering with {Chebyshev} Interpolation for Recommendation},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)},
  year      = {2025},
  address   = {Padova, Italy},
  eprint    = {2505.00552},
  archivePrefix = {arXiv},
  note      = {DOI UNVERIFIED (placeholder in the arXiv copy)}
}
```

Notes on existing entries:
- `dpaa2026` (refs.bib): authors, title and eprint 2605.11145 match what I read; keep as `@misc`.
- `chen2021autodebias` (AutoDebias) and `bonner2018cause` (CausE) are already in the bibliography; DOI 10.1145/3404835.3462919 is verified for AutoDebias, arXiv 1706.07639 for CausE.
- Entries kept as optional: `kim2025chebycf` and `park2024turbocf` only if the training-free graph-filter paragraph in Related Work is extended; they are not needed for the gap statement.
