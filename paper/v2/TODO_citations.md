# Citation and sourcing TODOs for sections 1-3

Scope: `sections/s1_intro.tex`, `sections/s2_related.tex`, `sections/s3_setting.tex`.

## Still to verify by the author

Each item below is a field or claim that the checks on 2026-10-29 (`refs_verified.md`, `lit_search.md`) did not confirm. The field is omitted from the entry in `refs_new.bib`, or the claim is hedged in the text.

- **rendle2009bpr:** the UAI proceedings have no DOI. Page range 452--461 is from a repository record and an index; the PMLR reprint is paginated 460--469.
- **holm1979:** the JSTOR page did not load; volume, issue and pages are from secondary records. No DOI.
- **zhang2023advinfonce:** NeurIPS 2023 proceedings hash and page not retrieved; acceptance confirmed from the arXiv and poster pages.
- **zhang2022bc:** NeurIPS volume 35 not read off the proceedings page.
- **chuang2020dcl:** arXiv id 2007.00224 is from the authors' lab BibTeX only.
- **menon2021logit:** OpenReview id and ICLR proceedings URL not retrieved.
- **saito2020ictir:** paper content (propensity estimation) not read; bibliographic record only.
- **liu2020kdcrec (KDCRec):** pages and DOI omitted; full text not read and bibliographic data not confirmed against the proceedings. Abstract-level description only (uniform data used for training).
- **lee2022biser (BISER):** DOI not in the entry (the arXiv copy has a placeholder).
- **zhang2026conalign (ConAlign), yin2026cw (Correct-and-Weight):** arXiv preprints; venue not established.
- **dpaa2026:** arXiv primary category and any later publication.
- **steck2010mnar:** the abstract was available only as a paraphrase; the corrected sentence in Section 2 is based on it.
- **Table 1, IPS, SNIPS, DR, DR-JL, MRDR, "Uniform data = no":** not checked against the original papers (propensity estimation in schnabel2016 can use a randomised sample).
- **Table 1, Uniform data for CAGED, DPAA, APDA:** based on published descriptions (caption says so); full texts not read. NAVIP was read in full text.
- **AdvDrop and PPD (table rows and text):** IPS and DR baseline details, and the validation protocol of DR-GNN, are as described in the search summary and not separately checked.
- **InterD (ding2022interd) and KBDR (li2024kbdr):** not added to the text. The search did not verify their protocols (InterD split not verified; KBDR section silent on the role of unbiased ratings).
- **Earlier DRUP work.** The introduction says "DRUP, which we introduced in earlier work". No bib entry for that work exists in `paper/refs.bib`. Add its key when the reference is confirmed.
- **Gap statement (Section 2, "Evaluation protocols and the gap").** The search used general web search engines only, not bibliographic databases. The claim is hedged accordingly; a database search may find further papers.

## Claims without a verified source

1. **Gap statement (Section 2.6, outline Section 9, TODO-G1).** Literature search run on 2026-10-29 (`lit_search.md`); the text now uses the narrower wording and states the search limits. Do not write "first".
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
