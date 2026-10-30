# Verified references (checked 2026-10-29)

Scope: the ten entries of `refs_new.bib`, the iALS item and the dpaa2026 item of `TODO_citations.md`, and the eight `refs.bib` entries whose description in `sections/s2_related.tex` and in Table 1 (`tab:positioning`) depends on the content of the cited paper.

Sources used: ACM DOI metadata (Crossref records), arXiv abstract pages, NeurIPS/NIPS proceedings pages, PMLR, the full text of the arXiv or author versions of the Rel-MF (WSDM 2020), AutoDebias, CausE, APDA and BPR papers, and secondary indexes where the publisher page could not be loaded. Fields that could not be confirmed are marked UNVERIFIED. No field was filled from memory.

Status codes: V = confirmed from at least one authoritative record; V* = confirmed from a secondary source only (reason given).

## (a) Corrected BibTeX blocks

All ten keys of `refs_new.bib` are kept. Authors (full list and order), titles and years of all ten were confirmed and need no change. Changes are limited to added identifiers (DOI, arXiv id, URL), the volume of the NeurIPS proceedings, and the booktitle wording of the ICTIR entry. One new entry (`rendle2022ials`) is added for the TODO item.

```bibtex
@inproceedings{rendle2009bpr,
  author    = {Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  title     = {{BPR}: Bayesian Personalized Ranking from Implicit Feedback},
  booktitle = {Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI)},
  pages     = {452--461},
  year      = {2009},
  eprint    = {1205.2618},
  archivePrefix = {arXiv}
}
% Status V* for pages: 452--461 is the page range given by the Hildesheim repository record and by an
% independent bibliographic index; the PMLR reprint (vol. R7) is paginated 460--469, so do not mix the two.
% DOI: none exists for the UAI proceedings (UNVERIFIED if one is required).

@article{holm1979,
  author  = {Holm, Sture},
  title   = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics},
  volume  = {6},
  number  = {2},
  pages   = {65--70},
  year    = {1979},
  url     = {https://www.jstor.org/stable/4615733}
}
% Status V*: volume, issue, pages and JSTOR id 4615733 agree across the reference list of a standard
% encyclopedic article and an index record. The JSTOR page itself did not load. No DOI is registered.

@inproceedings{zhang2023advinfonce,
  author    = {Zhang, An and Sheng, Leheng and Cai, Zhibo and Wang, Xiang and Chua, Tat-Seng},
  title     = {Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023},
  eprint    = {2310.18700},
  archivePrefix = {arXiv},
  url       = {https://neurips.cc/virtual/2023/poster/71149}
}
% Status V (authors, title, venue: arXiv page states "Accepted to NeurIPS 2023"; NeurIPS poster page lists it).
% Proceedings page numbers: not applicable (NeurIPS has none). Proceedings URL/hash: UNVERIFIED.

@inproceedings{saito2020ictir,
  author    = {Saito, Yuta},
  title     = {Unbiased Pairwise Learning from Biased Implicit Feedback},
  booktitle = {Proceedings of the 2020 ACM SIGIR on International Conference on Theory of Information Retrieval (ICTIR)},
  pages     = {5--12},
  year      = {2020},
  publisher = {ACM},
  doi       = {10.1145/3409256.3409812}
}
% Status V (DOI metadata: single author, pages 5-12, year 2020).

@inproceedings{ding2020srns,
  author    = {Ding, Jingtao and Quan, Yuhan and Yao, Quanming and Li, Yong and Jin, Depeng},
  title     = {Simplify and Robustify Negative Sampling for Implicit Collaborative Filtering},
  booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS 2020)},
  year      = {2020},
  eprint    = {2009.03376},
  archivePrefix = {arXiv},
  url       = {https://papers.nips.cc/paper/2020/hash/0c7119e3a6a2209da6a5b90e5b5b75bd-Abstract.html}
}
% Status V (proceedings abstract page: title, five authors in this order, volume 33; no page numbers given).

@inproceedings{chuang2020dcl,
  author    = {Chuang, Ching-Yao and Robinson, Joshua and Lin, Yen-Chen and Torralba, Antonio and Jegelka, Stefanie},
  title     = {Debiased Contrastive Learning},
  booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS 2020)},
  year      = {2020},
  eprint    = {2007.00224},
  archivePrefix = {arXiv},
  url       = {https://papers.nips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html}
}
% Status V (proceedings page; the arXiv id comes from the BibTeX published by the authors' lab, V*).

@inproceedings{zhang2022bc,
  author    = {Zhang, An and Ma, Wenchang and Wang, Xiang and Chua, Tat-Seng},
  title     = {Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year      = {2022},
  eprint    = {2210.11054},
  archivePrefix = {arXiv},
  url       = {https://proceedings.nips.cc/paper_files/paper/2022/file/334da4cbb76302f37bd2e9d86f558869-Paper-Conference.pdf}
}
% Status V (authors and title from the proceedings PDF record and the arXiv listing; the volume number 35
% for the 36th conference is the standard NeurIPS numbering and was not read off the page: V*).

@inproceedings{steck2010mnar,
  author    = {Steck, Harald},
  title     = {Training and Testing of Recommender Systems on Data Missing Not at Random},
  booktitle = {Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {713--722},
  year      = {2010},
  publisher = {ACM},
  doi       = {10.1145/1835804.1835895}
}
% Status V (DOI metadata: single author, pages 713-722, year 2010).

@inproceedings{menon2021logit,
  author    = {Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  title     = {Long-tail Learning via Logit Adjustment},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  eprint    = {2007.07314},
  archivePrefix = {arXiv}
}
% Status V (arXiv page: six authors in this order, "Published as a conference paper in ICLR 2021").
% The arXiv title is "Long-tail learning via logit adjustment" (lower-case); the capitalisation above is a style choice.
% OpenReview id: UNVERIFIED.

@inproceedings{krichene2020sampled,
  author    = {Krichene, Walid and Rendle, Steffen},
  title     = {On Sampled Metrics for Item Recommendation},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages     = {1748--1757},
  year      = {2020},
  publisher = {ACM},
  doi       = {10.1145/3394486.3403226}
}
% Status V (DOI metadata: authors in this order, pages 1748-1757, year 2020).

@inproceedings{rendle2022ials,
  author    = {Rendle, Steffen and Krichene, Walid and Zhang, Li and Koren, Yehuda},
  title     = {Revisiting the Performance of {iALS} on Item Recommendation Benchmarks},
  booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
  pages     = {427--435},
  year      = {2022},
  publisher = {ACM},
  doi       = {10.1145/3523227.3548486},
  eprint    = {2110.14037},
  archivePrefix = {arXiv}
}
% New entry. Status V: DOI metadata and the arXiv record both give these four authors in this order.
% The TODO file lists the fourth author correctly as Koren. "Anderson" is not an author of this paper.

@misc{dpaa2026,
  author        = {Islam, Md Aminul and Faruk, Ahmed Sayeed and Medya, Sourav and Zheleva, Elena},
  title         = {Debiasing Message Passing to Mitigate Popularity Bias in {GNN}-based Collaborative Filtering},
  year          = {2026},
  eprint        = {2605.11145},
  archivePrefix = {arXiv},
  note          = {Preprint}
}
% Status V: no change needed. arXiv lists the same title and four authors in this order. v1 submitted 11 May 2026,
% v2 revised 5 Oct 2026. The page shows no journal reference, no comments field and no acceptance note, so the
% paper is a preprint with no published venue as of 2026-10-29. primaryClass: UNVERIFIED.
```

Metadata check of the `refs.bib` entries used below (all match the DOI records, no change needed): zhou2023apda (SIGIR 2023, pp. 7--17), zhao2022radjnorm (SIGIR 2022, pp. 50--59), que2025caged (CIKM 2025, pp. 2471--2481, arXiv 2510.04502, states "Accepted by CIKM 2025"), kim2022navip (CIKM 2022, pp. 4128--4132), chen2021autodebias (SIGIR 2021, pp. 21--30), bonner2018cause (RecSys 2018, pp. 104--112), liang2016exposure (WWW 2016, pp. 951--961), saito2020unbiased (WSDM 2020, pp. 501--509). Author lists and order agree for all eight.

What each of the eight papers does (one sentence each) and whether the paper's description is correct:

| Entry | What the paper does | s2 sentence / Table 1 |
|---|---|---|
| APDA (zhou2023apda) | Replaces the aggregation of GNN collaborative filtering by a learned popularity-debiasing aggregator (per-edge weights) and proposes an inverse-popularity score to measure popularity in the representation space; it targets the aggregation rather than the loss. | Correct (s2 line 65, "learn aggregation weights"). Table "Propensity no, Uniform data no, Acts on aggregation" is supported; the full text uses no uniform sample. |
| CAGED (que2025caged) | Treats graph aggregation as a backdoor adjustment and estimates unbiased aggregation weights with an encoder-decoder trained on an evidence lower bound, refined by a momentum update. | Correct for "learn aggregation weights". "Propensity no" is defensible (weights approximate a historical interaction likelihood, not an inverse propensity). "Uniform data no": not read in full, UNVERIFIED. |
| ExpoMF (liang2016exposure) | Models exposure as a latent variable in a matrix factorisation model, infers its posterior for unclicked items, can condition exposure on covariates, and contains weighted matrix factorisation as a special case. | s2 lines 44--46 correct. It is not a sampling method (see item 2 below) and uses no uniform data, so "depends" in Table 1 is wrong (item 4 below). |
| UBPR (saito2020ictir) | Pairwise learning from implicit feedback with an unbiased estimator of the ideal pairwise loss (an inverse-propensity-weighted extension of BPR). | s2 lines 46--48 consistent with the description of the paper in later work. The ICTIR paper text was not available, so how its propensities are estimated is UNVERIFIED. |
| Rel-MF (saito2020unbiased) | Unbiased pointwise estimator of the ideal loss under MNAR implicit feedback, with a clipped variant; the experiments estimate propensities from relative item popularity (exposure proportional to a power of item popularity). | s2 lines 46--47 correct. No uniform data enters training, so "depends" is wrong for this row. |
| AutoDebias (chen2021autodebias) | Learns debiasing parameters (sample weights w1 and w2, and an imputation function m) by bi-level meta-learning on a small uniform sample; it generalises inverse-propensity and imputation methods. | Uniform data "yes" correct, "learned weights" correct, but "Imputation no" is wrong: the method learns imputation values. |
| CausE (bonner2018cause) | Matrix factorisation with separate control (large biased sample) and treatment (small randomised sample) embeddings, tied by a regulariser on their discrepancy. | s2 "small uniform sample for training" correct. Table "learned weights" under Propensity is wrong: CausE learns no weights and uses no propensity. |
| NAVIP (kim2022navip) | Derives a propensity for every user-item edge and applies inverse-propensity weights with Laplacian normalisation inside neighbour aggregation of a GNN recommender. | s2 lines 61--62 correct. Table row "Propensity yes, Imputation no, adjacency" supported. "Uniform data no": the abstract does not say how the propensity is obtained, UNVERIFIED. |
| r-AdjNorm (zhao2022radjnorm) | Shows that symmetric neighbourhood aggregation amplifies popularity bias with propagation depth and proposes a plug-in that tunes the strength of the adjacency normalisation to trade accuracy against novelty. | s2 lines 62--64 correct. Table row (no propensity, no imputation, adjacency) correct. |
| DPAA (dpaa2026) | Adds adaptive, representation-aware interaction weights and layer-wise weights to message passing to counter popularity amplification. | s2 line 65 correct ("debiased message passing"); it is an arXiv preprint with no venue. |

## (b) Entries and fields that could not be verified

| Item | What is missing |
|---|---|
| holm1979 | Publisher (JSTOR) page did not load. Volume, issue, pages and JSTOR id confirmed only from secondary records. No DOI exists. |
| rendle2009bpr | No DOI for the UAI proceedings. Page range 452--461 confirmed from a repository record and an index, not from the proceedings volume itself. |
| zhang2023advinfonce | NeurIPS 2023 proceedings page (paper-file hash) not retrieved. Accepted status confirmed from the arXiv page and the NeurIPS poster page. |
| zhang2022bc | NeurIPS volume number 35 not read off the proceedings page. |
| chuang2020dcl | arXiv id 2007.00224 taken from the authors' lab BibTeX only. |
| menon2021logit | OpenReview id and ICLR proceedings URL not retrieved. |
| saito2020ictir | Content of the paper (propensity estimation, form of the estimator) not read; only the bibliographic record and later descriptions of it. |
| dpaa2026 | arXiv primary category; any later publication. |
| steck2010mnar | Abstract available only as a paraphrase. Whether the paper relies on a uniformly sampled subset is not established (see item 1 of part (c)). |
| Table 1, row "IPS, SNIPS, DR, DR-JL, MRDR", column "Uniform data = no" | Not checked against the original papers (propensity estimation in schnabel2016 can use a randomised sample). |
| Table 1, cells "Uniform data" for NAVIP, CAGED, DPAA | Papers' full texts not read. |
| Gap statement, s2 lines 126--129 | No literature search was run for the claim that no earlier study uses DR nuisances to weight negatives of a graph recommender, nor for the claim about evaluation on the four datasets. |
| Earlier DRUP work (TODO) | Not in scope of this check; still missing. |

## (c) Sentences in `sections/s2_related.tex` and Table 1 that are wrong or unsupported

1. Lines 26--28. "Steck \citep{steck2010mnar} studied how training and evaluation should account for missingness when a uniformly sampled subset is available." Unsupported. The abstract says that missing ratings carry information, that two test measures can be estimated without bias under MNAR, and that surrogate training objectives account for observed and missing ratings; it does not state that a uniform subset is required. Corrected wording: "Steck \citep{steck2010mnar} showed that missing ratings carry information, proposed test measures that can be estimated without bias when ratings are missing not at random, and derived training objectives that account for the missing entries."

2. Lines 124--126. "False-negative-aware sampling (ExpoMF, SRNS, AdvInfoNCE) does not use the propensity and imputation that DR estimation provides." ExpoMF is a latent-exposure factorisation model, not a sampling method, and AdvInfoNCE is a loss. Corrected wording: "Exposure-aware and false-negative-aware methods (ExpoMF, SRNS, DCL, AdvInfoNCE) do not use the propensity and imputation that DR estimation provides." Keep the lead-in of line 48 ("Sampling-based methods ...") limited to SRNS and DCL, as it is.

3. Lines 51--53. "the bias-aware margins of the BC loss and the adversarial re-weighting of negatives in AdvInfoNCE change how hard negatives are weighted." BC loss adds a margin per interaction that depends on the popularity-bias degree of the interaction; it does not re-weight negatives. Corrected wording: "the BC loss \citep{zhang2022bc} adds a margin that depends on the popularity-bias degree of each interaction, and AdvInfoNCE \citep{zhang2023advinfonce} assigns each negative an adversarially learned hardness weight."

4. Table 1, line 100 (row "ExpoMF, Rel-MF, UBPR"), column "Uniform data = depends". None of the three uses uniform data in training; ExpoMF learns exposure from clicks and Rel-MF estimates propensities from relative item popularity. Corrected cell: "no". If no row keeps "depends", delete "(``depends'': the original propensity estimation can use one)" from the caption (lines 84--85); keep it only if the IPS/SNIPS row is changed to "depends" after the check listed in part (b).

5. Table 1, line 102 (row "AutoDebias, CausE"), cells "Propensity = learned weights" and "Imputation = no". Split the row.
   - AutoDebias: Propensity "learned weights", Imputation "learned", Acts on "loss (weights and imputation)", Uniform data "yes".
   - CausE: Propensity "no", Imputation "no", Acts on "embeddings (discrepancy regulariser)", Uniform data "yes".

6. Table 1, line 98, cell "Uniform data = no" for CAGED, DPAA and NAVIP (line 97): not confirmed from the papers (see part (b)). Either read the three papers or add "(as described)" to the caption sentence at lines 85--87, which already says that the entries for APDA, CAGED and DPAA follow their published descriptions.

7. Lines 126--129 (gap statement). Unsupported until the literature search of TODO-G1 is run. Do not write "first". Keep "We are not aware of ..." only after that search.

8. `TODO_citations.md` line 7. The iALS paper is confirmed: Rendle, Krichene, Zhang, Koren; RecSys 2022 (16th ACM Conference on Recommender Systems), pp. 427--435, DOI 10.1145/3523227.3548486. Entry `rendle2022ials` above can be added; the paper cites iALS via `hu2008ials` at line 13 and may cite the revisit paper there as a baseline reference.

Lines 44--46 (ExpoMF), 46--48 (Rel-MF, with the pairwise sentence), 49--51 (SRNS, DCL), 61--65 (NAVIP, r-AdjNorm, APDA, CAGED, DPAA), 73--74 (Menon) and 117--118 (Krichene) were checked and are supported.
