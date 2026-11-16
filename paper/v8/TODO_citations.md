# TODO: citations and source checks for paper/v8 (updated 2026-11-16)

Open items are listed first. Closed items follow, each with the change that closed it.

## Open

1. **Source descriptions read from titles or outlines (verify against the papers, then keep or correct).**
   `richardson2007predicting` (the first page is 87 in the bib record; end page missing),
   `craswell2008cascade` ("compare cascade and position-bias models on search logs"),
   `chapelle2009dbn` (satisfaction state), `chuklin2015clickmodels` (Springer re-issue DOI, not the 2015 Morgan & Claypool printing),
   `wang2016selection`, `wang2018regressionem` (position bias without randomisation, personal search),
   `joachims2017unbiased`, `saito2020unbiased`, `liang2016exposure`, `ma2018esmm`, `krause2024coexposure`, `jeunen2020cease`,
   `mackenzie2002occupancy`, `fay1979smallarea`, `park2004mrp`, `kim2022navip` (title only), `chen2021autodebias`.
2. **Claims from the authors' earlier searches.** "We found no recommendation paper that uses the occupancy likelihood..." (s2)
   and "as far as our searches went, none of these works uses the out-of-pool history..." (s2, evaluation subsection; novelty
   paragraph; table row). Both rest on `paper/v7/LITERATURE.md` and `paper/v6/DESIGN.md`, not on a search in this pass.
3. **Sources to check.** `hu2025uncertainty` (IEEE TKDE record not returned by Crossref; check IEEE Xplore),
   `zhao2023d2co`, `wang2025debiasunreliable` (description read from the title; the bib note says Companion, short paper),
   `zhang2026categorymeans` (arXiv 2611.05006; "accepted at WSDM 2027" only in the arXiv comment), `gao2022kuairec`
   ("fully observed matrix", from `paper/v6/DESIGN.md`), `schnabel2016` and `marlin2009collaborative` (wording of the
   random test sets).
4. **zhang2021pda description.** The bib title is "Causal Intervention for Leveraging Popularity Bias in Recommendation". The
   outline table and `paper/v7/LITERATURE.md` Section 6 describe it differently. Read the paper; correct the outline and the
   LITERATURE entry, or the bib entry.
5. **Pytest output.** The 15 tests (`tests/test_lifelong.py`, `tests/test_dax.py`) passed on 2026-11-13, but the output is only in
   the session scratchpad. Save it under `results/` (for example `results/v12/pytest_leakage.txt`); the leakage table cites it.
   The file does not exist yet.
6. **KuaiRand-1K paired difference against LEAF-LGN.** Not reported. From the files, LEAF-LGN+DAX is 0.4590 (`c_1k_lgn.json`) and
   the 1K LEAF-LGN base is in the same file. Compute the paired difference with its interval (`experiments/v13_tables.py` helpers)
   before the 1K replication is described against LEAF-LGN.
7. **EM closed-form M-steps (decision for the authors).** `drup/dax.py` updates theta and kappa with closed forms whose denominators
   count every impression. Appendix B now discloses this and the clamps. A synthetic check gave theta 0.755 for the implemented
   update against 0.747 for the exact stationary point (one user, 114 of 400 impressions clicked). Decide whether to keep the
   implemented updates or to fix the code and re-run the DAX fits; a re-run changes every DAX number.
8. **Joint variant.** `LEAFLGNDAX` in `drup/dax.py` has no result file (`results/v11/dax_lgn_joint*.json` missing). The paper
   reports no number for it.
9. **Extended DAX grid.** `results/v12/daxref_*.json` and `results/v12/daxwide_*.json` are missing. The grid question in the
   limitations stays open.
10. **Long-view label and concurrent protocol for the full model.** `results/v11/c_longview*.json` and `results/v11/c_concurrent.json`
    are missing. The limitations say so.
11. **Authors' note on the QP family.** It is never cited (Appendix A). Keep it that way unless the authors decide to publish it.

## Closed (2026-11-16)

- Citation keys: `holm1979`, `rendle2009bpr`, `zhang2024advdrop`, `islam2026ppd`, `zhou2018din`, `saito2021obd`, `pi2020sim`
  are present in `refs_new.bib` and cited (s1 to s4, app A). `liu2020kdcrec` is in `refs_new.bib` and not cited; remove it or
  cite it before submission. The Open Bandit Dataset key `saito2021obd` is the dataset, not `saito2020unbiased`.
- DIN (1K and 27K) and SIM (27K and 1K) are run and reported (`results/v12/din_27k.json`, `din_1k.json`, `sim_27k.json`,
  `sim_1k.json`). The "SIM did not finish" and "SIM not run" statements were removed.
- LightGCN+DAX is run (`results/v11/dax_lightgcn.json`, 960 configurations). The LightGCN row of the backbone table uses the
  retrained base of that run (3 configurations), with the stored base disclosed in the table note.
- Holm families: fixed and listed in Section 4 (gate B1 five; main table nine; sensitivity eleven; 1K three; ablations seven and
  six; stage 1 eight; LEAF controls eight). Ablation and control p Holm values are computed by `experiments/v13_tables.py` and
  written with their family sizes to `results/v13_tables_numbers.txt`.
- Two LEAF-LGN means (0.4735 diagnostic, 0.4728 stored, 0.4736 post-hoc) and the two oracle values (0.5068 on 22,661 users in
  `diag_gap.txt`; 0.5072 on 22,847 users in `headroom_kuairand_pure.txt`) are named where used (Section 5).
- Ablation base (0.4706) explained, with the gain difference (3.14e-3) and the caveat for the Kept percentages (Section 5, table note).
- Seeds: stated in the text (reduced grid; base 0.4714, 0.4708, 0.4716 against 0.4736). The QP seed run is not used; the partial file
  `results/v11/c_seeds/qp_s1.json.partial.json` exists and is not summarised.
- KuaiRec saturation values are in `results/v13_saturation_kuairec.txt`; the leave-one-user-out oracle for KuaiRec is not stored
  and is not reported.
- Data-statistics TBDs (median impressions per pool item, candidate counts, 1K per-split test users, pre-cut impressions) are
  filled from `results/v13_data_stats.txt`. The 27K lifelong history count appears as 26,904 (tables) and 26,723 (DIN and SIM
  sequences), both with their source.
- The stage 1 gate (LEAF-LGN against QP family, +9.63, NB Holm 0.189) is stated with its number (s1, s5).
- Items with a result file that the outline listed as TBD (dax_ablate, the 1K QP comparisons, the 27K SIM and LightGCN rows) are
  filled from the JSON files through the generator.
- All TBD placeholders in the tables, figures and sections were removed. The generator writes no TBD cells for present inputs, and a
  missing input is reported as "not available" in the numbers file.
- The ablation family p values, the NB Holm values of the sensitivity family and the DIN and SIM DAX increments (+12.53 and
  +11.72) are generated by `experiments/v13_tables.py`; the table notes name the JSON files.
