# Results, discussion, conclusion, abstract and tables: open items

Scope: `sections/s6_results.tex`, `sections/s7_discussion.tex`, `sections/s8_conclusion.tex`,
`sections/abstract.tex`, `tables/tab_{main,extra,seeds,ablation,decomp,advinfonce,walk}.tex`,
`figs/make_figs.py` with `figs/fig_{forest,ablation,walk}.pdf`.

## 1. Numbers that could not be sourced (left out of the text)

- LightGCN-pn on KuaiRec, 0.5999 (`results/v4/SUMMARY.md` line 17). No stored file gives this value. The text
  uses the per-job values in `results/v4/learned_kuairec_{LightGCN,DR-LightGCN}-pn_lr*.json`
  (0.5992 to 0.5995 and 0.5992 to 0.6027). Action: check which run SUMMARY.md refers to.

## 2. Corrections to the outline (applied in the text)

- Coat popularity-only ablation: 0.5751 (stored JSON), not 0.5770. Yahoo! R3 shuffled control: 0.6689 (JSON),
  not 0.6690 (SUMMARY).
- Outline 6.1, item 6 ("effects 0.002 to 0.033"): against the strongest graph baseline by mean, the paired
  differences are +32.68 (Coat, DR-LightGCN), +12.67 (Yahoo! R3, r-AdjNorm), +1.94 (KuaiRand, r-AdjNorm) and
  +1.31 (KuaiRec, LightGCN-pt, not significant). The KuaiRec comparison uses LightGCN-pt, not DR-LightGCN.
- Outline 6.5 (semi-synthetic DRUP minus DR): "+0.60 to +19.59 at small beta" is not the right description.
  At beta = 0.3 all six cells are positive (+12.5 to +19.6); the sign reverses at beta = 10 and 1000 in the two
  cells with gamma = 1.0. The value +0.60 is the beta = 1000 cell with gamma = 0.
- Outline storyline (ii) "beats every graph baseline on all four datasets": the KuaiRec comparison with
  LightGCN-pt is not significant (p_Holm = 0.4409). Abstract and Results say so.
- Outline storyline (ii) "does not beat iALS on Coat": in the main run the method is above iALS (p_Holm = 0.0469).
  Following ground rule 4, the text reports both results and calls the two "on par" (seed runs, p_Holm = 0.747).
- DR-LightGCN "beaten on all four datasets": the final method was tested against DR-LightGCN only on Coat and
  KuaiRec. On Yahoo! R3 and KuaiRand the only DR-LightGCN comparison is the beta = 0 job in `dr_bpr_summary.txt`
  (Yahoo, 0.5528). The text makes no claim there.
- Abstract (199 words kept): KuaiRec caveats added (LightGCN-pt not significant; main run below DR-JL and MRDR).
  "the offset dominates on Coat and KuaiRec" is replaced by "on Coat the offset alone recovers most of it",
  because the KuaiRec evidence is a decomposition of selected runs, not a matched ablation (C12).
- Table numbering is automatic. Table 1 (positioning, Section 2) and Table 3 (datasets, Section 5) are in other
  writers' sections. The non-family rows of the outline's Table 2 moved to a separate table `tab:extra`, and
  the outline's Table 5 was split into `tab:ablation` (Panels A and B) and `tab:decomp` (Panel C), to keep the
  tables within the text width.

## 3. Open items from the outline (not done)

- TODO-E1 (offset-only runs on Yahoo! R3, KuaiRand, KuaiRec): not run. The text restricts the offset-only claim
  to Coat.
- TODO-A1 (Holm family extended with DR-JL, MRDR, DR-LightGCN, DRUP, DR rows): not done. The rows in
  `tab:extra` are marked as not tested.
- TODO-A2 (paired tests for the ablations): not run. The text states that no significance tests were run.
- TODO-E2 (source of "+0.002 to +0.006 with imputation-based degrees"): resolved. It is the stored cell
  `k1.5_g1.0_d0.13_xfit_est_yhat` in `results/v3/rank_dose_real.json`: DRUP minus DR is +2.07, +6.25 and +2.90
  (x1e-3) at beta = 0.3, 1 and 3, and -5.20 and -9.36 at beta = 10 and 1000. Action: this cell uses
  cross-fitted nuisances (the key says `xfit`). The trained method does not cross-fit, so the supplement (S4)
  must describe the cell accordingly, or the cell must be rerun without cross-fitting.
- TODO-F5 (histogram of Q): not created.
- Figure 1 (method schematic, TikZ, Section 4): not created here; it belongs to the Section 4 writer.
- TODO-G1 (literature check behind the gap statement): not in these sections. No "first to" or novelty
  wording is used here.
- Bibliography keys used in these sections (all present in `paper/refs.bib`): zhao2022radjnorm, kim2022navip,
  yu2022simgcl, liang2016exposure, schnabel2016, nadeau2003. AdvInfoNCE (TODO-C3) and Holm (TODO-C2) are not
  cited here.

## 4. Build notes (other writers' files, not changed here)

- `main.tex` now cites `\bibliography{../refs,refs_new}`, but `paper/refs_new.bib` does not exist. BibTeX fails
  on this line.
- `sections/s1_intro.tex` uses an undefined `assumption` environment, and `../tables/datasets.tex` does not
  resolve from `paper/v2`. The full build therefore fails at those files. My files compile in a scratch build
  with stubs for s1 to s5: no undefined references or citations, and no overfull boxes in the tables.
