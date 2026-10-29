# Citation and sourcing TODOs for sections 1-3

Scope: `sections/s1_intro.tex`, `sections/s2_related.tex`, `sections/s3_setting.tex`.

## Not added to refs_new.bib (authors, venue or year not certain)

- **iALS revisited (OUTLINE TODO-C9).** The outline lists Rendle, Krichene, Zhang and Koren (RecSys 2022). The author list is not confirmed, so no entry was added and the text does not cite it. Add once the publisher page is checked.
- **Earlier DRUP work.** The introduction says "DRUP, which we introduced in earlier work". No bib entry for that work exists in `paper/refs.bib`. Add its key when the reference is confirmed.

## Added to refs_new.bib from memory: verify each against the publisher page before submission

rendle2009bpr, holm1979, zhang2023advinfonce, saito2020ictir (pages), ding2020srns, chuang2020dcl, zhang2022bc, steck2010mnar, menon2021logit, krichene2020sampled (pages). Keys are used in the sections as follows: rendle2009bpr and holm1979 in the introduction, the other keys in Section 2.

## Claims without a verified source

1. **Gap statement (Section 2.6, outline Section 9, TODO-G1).** "To our knowledge, no previous study uses the DR nuisances to weight the negatives of a graph recommender" and the sentence on the evaluation protocols of other work. The literature search has not been run. Do not submit before G1 is done. Do not write "first".
2. **Table 1, cells not stated in the outline.** These are based on the method families and were not checked against the original papers:
   - the "Uniform data" entries for NAVIP, r-AdjNorm, APDA, CAGED and DPAA (set to "no");
   - "depends" for ExpoMF, Rel-MF and UBPR;
   - "learned weights" for AutoDebias and CausE;
   - the "Acts on" column for AdvInfoNCE and BC loss.
3. **Descriptions of APDA, CAGED and DPAA** (Section 2.4) and of ExpoMF, SRNS, DCL, BC loss and Steck (2010) (Sections 2.2 and 2.3) are based on the titles and the outline. Check each against the paper.
4. **Assumption 1 in Section 3 (exposure independent of the outcome given covariates).** The outline does not state it. It is inferred from the closed form Q = Yhat(1-p)/(1-pYhat) (outline Section 4.2, code in `drup/learned.py` lines 290-293). It is stronger than the DRUP-theory assumption. The authors should confirm the wording.
5. **Coat propensities.** Section 3 says they were estimated by the dataset authors from a uniform sample and cites schnabel2016. Outline Section 4.1 attributes the estimation to schnabel2016, but the outline does not say which passage of that paper or dataset page it refers to. Confirm before submission.
6. **dpaa2026** is an arXiv preprint (2605.11145) per `refs.bib`. Its venue is not settled, so the text cites it as "debiased message passing" without a venue.

# Sections 4 and 5 (method and experimental setup)

Scope: `sections/s4_method.tex`, `sections/s5_setup.tex`.

## Build and consistency

1. **Bibliography.** `main.tex` passes only `../refs` to `\bibliography`. The keys that sections 4 and 5 use from `refs_new.bib` (rendle2009bpr, holm1979, zhang2023advinfonce, menon2021logit, krichene2020sampled, steck2010mnar) are not in that list. Add `refs_new` to the bibliography before typesetting. These entries are the ones marked "from memory" above.
2. **Assumption behind Q (Section 4.2).** The closed form Q = Yhat(1-p)/(1-pYhat) needs exposure independent of the outcome given the covariates of the propensity model. Section 3 of the old `paper/sections/method_theory.tex` (Assumption 1) lets the propensity depend on Y. The two must be aligned in Section 3 before submission. Section 4.2 states the independence explicitly and says it is not tested.
3. **Table 3.** Section 5.1 includes `../tables/datasets.tex` (i.e. `paper/tables/datasets.tex`, unchanged). Its "Evaluated users" column agrees with the loaders (225, 2,315, 8,347, 1,411). Keep it in sync if the file changes.
4. **Figure 1.** Only a comment marks its place at the start of Section 4. The TikZ schematic is not written.
5. **Overfull lines.** `pdflatex` compiles both sections cleanly (checked in a scratch wrapper, not in the repository). Overfull boxes are below 20 pt.

## Facts to confirm or source

6. **Per-baseline grids (Section 5.6, supplement S1).** The exact grid of each baseline was not reconstructed. The recorded arguments of a merged result file do not describe every method in it (for example `learned_coat_given.json` lists layers [2], yet LightGCN there has 54 configurations, which implies a wider grid). Configuration counts come from `results/v4/final_summary.txt`. Build S1 from `run_learned.py` METHODS and the queue scripts, and check each count.
7. **Random-ranking levels (old `exp_setup.tex` lines 48-51) are not reused.** A recomputation with the loaders gave nDCG@5 0.311 on Coat (old text: 0.306), 0.352 on Yahoo!\,R3, 0.327 on KuaiRand at @10 (old text: 0.33) and 0.046 on KuaiRec at @20. The Coat value needs a check before any level is quoted.
8. **Test-user counts (Section 5.1 and 5.4).** The final-method files give 225, 2,310, 8,321 and 1,410 users with a test entry. The difference from the table column is explained as a user who is in the validation part of every split. The per-user cause was not checked individually.
9. **Coat propensities (Section 4.1).** Section 4.1 repeats the claim, from the old `discussion.tex` lines 68-70, that the shipped Coat propensities were estimated from the uniform sample (schnabel2016). It is the same item as item 5 in the file above and still needs the exact passage.
10. **Nadeau-Bengio and equivalence tests (old `exp_setup.tex` lines 110-112).** Not carried over. No corrected resampled test or equivalence test exists for the final method. Add them only if they are run.
11. **DRUP unbiasedness (Section 4.6).** The sentence "designed to make the three-hop operator unbiased ... argument given in the supplement" depends on supplement S3, which is not written. The earlier-work citation for DRUP is still missing (see the first part of this file).
