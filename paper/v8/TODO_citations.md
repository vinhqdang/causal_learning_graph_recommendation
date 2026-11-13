# TODO: citations and source checks for paper/v8/sections/s1_intro.tex and s2_related.tex

Written 2026-11-13 with the sections. Items are grouped by what must happen before submission.

## A. Citations that cannot be made from the allowed bibliographies

1. **AdvDrop** (`zhang2024advdrop`) and **PPD** (`islam2026ppd`). The keys exist only in `paper/v2/refs_new.bib`, which
   is not an allowed bibliography for v8. Placeholder left in `s2_related.tex`, subsection "Debiasing for graph
   recommenders". Action: verify the entries and add them to `paper/v8/refs_new.bib`, then replace the placeholder.
2. **DIN** (Deep Interest Network). No key in `paper/refs.bib` or `paper/v8/refs_new.bib`. Placeholder in
   `s1_intro.tex`, paragraph after the not-claim sentence. Action: verify the entry (Zhou et al., KDD 2018) and add it.
3. **Liu 2020 KDCRec** (`liu2020kdcrec`), **Holm 1979** (`holm1979`), **Rendle 2009 BPR** (`rendle2009bpr`): exist only in
   `paper/v2/refs_new.bib`. Not cited in s1 or s2; the Methods and Protocol writers need them.

## B. Facts stated in s1 and s2 without a source file (check the paper, then keep or remove)

4. SIM: selects category-matched behaviours from a lifelong sequence (`pi2020sim`). Description comes from
   `paper/v8/OUTLINE.md` Section 3, not from the paper.
5. TWIN: two-stage target attention on Kuaishou (`chang2023twin`). Same origin.
6. Examination hypothesis attributed to `richardson2007predicting`; `craswell2008cascade` described as comparing cascade and
   position-bias models "on search logs". The bib record has only the first page (87); the end page is missing.
7. `chapelle2009dbn`: satisfaction state; `chuklin2015clickmodels`: Springer re-issue DOI, not the 2015 Morgan & Claypool printing.
8. `wang2016selection`, `wang2018regressionem`: estimation of position bias without randomisation in personal search.
9. `joachims2017unbiased`, `saito2020unbiased`: IPS or propensity-weighted losses. `liang2016exposure`: EM for latent exposure.
10. `ma2018esmm`, `krause2024coexposure` (MNL with co-exposure), `jeunen2020cease` (item side information, closed form).
11. `mackenzie2002occupancy`: repeated non-detections identify occupancy and detection. Also the statement "We found no
    recommendation paper that uses the occupancy likelihood to estimate item appeal from a policy log". This rests on
    searches recorded in `paper/v7/LITERATURE.md` only, so it is phrased as the authors' own search result.
12. `fay1979smallarea`, `park2004mrp`: descriptions of the methods. The MRP and Fay-Herriot results are from probe-scale files.
13. `kim2022navip`: title only. `chen2021autodebias`: small uniform sample (title plus outline).
14. `hu2025uncertainty`: propensity calibration on KuaiRand random exposure (from `paper/v6/DESIGN.md` Section 1 table).
    Its bib note says the IEEE TKDE record was not returned by Crossref; check the IEEE Xplore page.
15. `zhao2023d2co`: watch-time debiasing on random exposure (from `paper/v6/DESIGN.md` Section 1 table).
16. `wang2025debiasunreliable`: the description "analysis of whether the evaluation of debiasing methods is itself biased"
    is read from the title. Read the paper before submission (the bib note says Companion, short paper).
17. `zhang2026categorymeans`: arXiv 2611.05006, "accepted at WSDM 2027" per the arXiv comment only. Re-check the proceedings
    record before submission; the description "audit of category means in logs with random insertions" is from the title.
18. `schnabel2016`, `marlin2009collaborative`: "a small randomly exposed sample from Coat and Yahoo! R3". Confirm the
    wording against the papers.
19. `gao2022kuairec`: described as "a fully observed matrix"; from `paper/v6/DESIGN.md` Section 1.
20. Claim "as far as our searches went, none of these works uses the out-of-pool history of each user in the random test"
    (s2, evaluation subsection; also the novelty paragraph and the table row). Based on `paper/v6/DESIGN.md` Section 1 and the
    outline, not on a fresh search in this pass.
21. Thesis sentence in the outline: "not in cold start". No result file supports a cold-start statement. Omitted from s1.

## C. Conflicts between sources (to resolve in the Results section, or fix in the outline)

22. **zhang2021pda description.** The bib title is "Causal Intervention for Leveraging Popularity Bias in Recommendation".
    The outline table (Section 3) and `paper/v7/LITERATURE.md` Section 6 call it "popularity-drift-aware recommendation" and
    describe it as a forecast of popularity. s2 uses the bib title ("applies a causal intervention to the popularity effect").
    Action: read the paper and correct the outline table and the LITERATURE entry, or the bib entry if it is wrong.
23. **Full DAX gain against LEAF-LGN.** `results/v11/stageB.txt` Section 2 gives LEAF-LGN+DAX - LEAF-LGN = +11.96 [1.47]
    (the post-hoc run, 0.4736). Section 4 (gate B2) gives "full DAX gain (LEAF-LGN+DAX - LEAF-LGN) +15.10 [1.42]" against
    the backbone of the ablation run. Same label, different backbone run, different value. s1 uses +11.96 and says where it
    comes from. Action: report both in T6 with a clear label of the backbone, or reconcile the two backbone runs.
24. **Holm family for the NB p values.** `stageB.txt` (family of five) gives NB p_Holm 0.023 for LEAF-LGN+DAX vs QP family;
    `final12.txt` Section 3b (family of seven, adding DIN and DIN+DAX) gives 0.046. Both are below 0.05 but the second is
    close. The Protocol section must state which family is primary and why. s1 gives no p value for C4.
25. **Deep baselines.** The outline (Section 2.3) says no lifelong-aware deep baseline was run. `final12.txt` reports DIN
    (27K: 0.4675; DIN+DAX 0.4796) and reads `sim_1k.json` and `din_1k.json`, but reports no SIM number. The DIN code
    (`experiments/run_din.py`) is lifelong-aware (last L events of the lifelong sequence). s1 now reports DIN (+5.73 [1.64]
    vs DIN+DAX). SIM 1K and DIN 1K results are not in any summary file; add them or drop the claims.
26. **Seeds.** The outline says `results/v11/c_seeds.json` is missing. The directory `results/v11/c_seeds/` exists with
    `lgn_s1.json`, `lgn_s2.json`, `lgn_s3.json` and `qp_s1.json.partial.json`. No summary has been written. The main table
    still uses one training seed per split; s1 says so.
27. **Stage 1 gate.** `results/v10/stage1.txt` Section 4: LEAF-LGN vs QP family +9.63 [1.60] fails the pre-declared gate
    (required >= 10e-3; NB p_Holm 0.189). s1 says LEAF alone does not pass the gate, with no number.
28. **Two LEAF-LGN means.** Stage 1 `leaf_lgn.json` gives 0.4728 (197 configurations); the post-hoc run gives 0.4736
    (12 configurations); `diag_gap.txt` uses 0.4735 (split-0 configuration). s1 uses 0.4735 for C1 (matching the diagnostic)
    and +15.46 for C2 (stage 1). The Results section must state which LEAF-LGN each number refers to.
29. **Number of users.** `diag_gap.txt` evaluates 22,661 users (C1 oracle); the stage files use 22,612 (C2, C3, C4). The
    headroom numbers in s1 are diagnostics on the larger set. State this in T2.
30. **Outline stale entries.** Section 2.2 and Section 5.4 give `{TBD:results/v11/dax_ablate.json}`, but the file exists
    (28.5 MB) and its summary is in `stageB.txt` Section 4. The joint variant (`dax_lgn_joint*.json`), `dax_lightgcn.json`
    and `daxref_*`/`daxwide_*` (extended grid) are still missing. KuaiRand-1K results exist in `final12.txt` Section 5
    (+37.58 [11.10] vs QP family), which the outline lists as TBD.
31. **KuaiRec saturation numbers** (outline Section 5.5) are in `paper/v5/DESIGN.md` Section 3 only, not in a result file.
    They were not used in s1 or s2. Keep the TBD placeholder in Section 5.5 until `results/v12/saturation_kuairec.txt` exists.

## D. Numbers used in s1 (provenance)

| Number | Where in s1 | File |
|---|---|---|
| 7,583 pool videos | intro, para 1 | `results/v11_probes/diag_gap.txt` header |
| LOO item oracle 0.5068; LEAF-LGN 0.4735; gap +33.31e-3 | intro, para 2; C1 | `results/v11_probes/diag_gap.txt` (Section 1) |
| Spearman 0.405 (post-cut) vs 0.431 (pre-cut) | intro, para 2 | `results/v11_probes/diag_gap.txt` Section 2 rows post_clk, pre_clk |
| Item-only 0.5850 and rank-64 residual 0.5851 | intro, para 2 (in the source comment) | `results/v9_probes/headroom_kuairand_pure.txt` H2 |
| LEAF-LGN +15.46 [1.53] vs LightGCN | C2 | `results/v10/stage1.txt` Section 3 |
| LEAF-LGN+DAX - LEAF-LGN +11.96 [1.47] | C3 | `results/v11/stageB.txt` Section 2 (see item 23) |
| LEAF-LGN+DAX 0.4854 vs QP family 0.4634; +22.32 [1.83] | C4 | `results/v11/stageB.txt` Sections 1, 2 |
| +12.04 [1.63] vs QP family+DAX; 3,840 vs 3,200 vs 20 configurations | C4 paragraph | `results/v11/stageB.txt` Sections 1, 3 |
| +5.73 [1.64] vs DIN+DAX | not-claim paragraph | `results/v12/final12.txt` Section 3b |

No number in s1 is a placeholder. Numbers in s2: none.

# Added by the writer of sections/s3_method.tex and sections/s4_protocol.tex (2026-11-13). Appended; earlier items unchanged.

## E. Citations needed by s3 and s4 (not in refs.bib or refs_new.bib; no key was invented)

32. **holm1979.** Step-down Holm procedure, used for every adjusted p-value in s4 (Section "Holm adjustment"). Placeholder
    `\{TBD:citation key holm1979 ...\}` in s4. The v8 refs_new.bib header lists holm1979 as in paper/v2/refs_new.bib, which is
    not an allowed bibliography. Action: verify Holm (1979, Scandinavian Journal of Statistics) and add the entry.
33. **rendle2009bpr.** BPR loss, named in s3 (LEAF-LGN training) without a citation. Same origin as item 3. Action: verify and add.
34. **DIN (Zhou et al., KDD 2018).** Deep interest network; named in s4 (Table of baselines and "Baselines" paragraph) with a
    placeholder `\{TBD:citation key ...\}`. No key in either bibliography. Action: verify the entry and add it; the s1 placeholder
    (item 2) refers to the same entry.
35. **Open Bandit Dataset.** Named in s4 without a citation (no persistent user identifiers). The key `saito2020unbiased` in
    paper/refs.bib is a different paper ("Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback"); it must
    not be used for the dataset. The dataset reference (Saito et al., NeurIPS 2021 Datasets and Benchmarks, arXiv 2008.07146, from
    paper/v6/DESIGN.md) is not in an allowed bibliography. Action: verify and add under a new key.
36. **EM, expectation-maximisation.** Named in s3 without a citation (standard; no key requested).

## F. Numbers and facts in s3 and s4 that still need an action

37. **Placeholders that stay TBD** (rule 0.1; each names the file that would supply it): median pre-cut impressions per pool item
    (s3, identifiability); temporal pre-cut training impressions for 27K and 1K, 1K pre-cut out-of-pool impressions, temporal
    candidates per user for 27K and 1K, 1K evaluated test users per split (s4, Table of data); KuaiRec saturation values
    (`results/v12/saturation_kuairec.txt`, s4 Datasets paragraph).
38. **Pytest output.** The 15 tests (`tests/test_lifelong.py`, `tests/test_dax.py`) passed on 2026-11-13 in this environment; the output
    was written only to the session scratchpad (`pytest_dax_lifelong.txt`), not to `results/`. Action: save it under `results/` (for
    example `results/v12/pytest_leakage.txt`) so that the statement in s4 (Leakage checks) has a file behind it.
39. **EM closed-form M-steps (method, decision for the author).** `drup/dax.py` updates theta and kappa with closed forms whose
    denominators count every impression (sum of kappa, sum of theta) instead of the exact stationarity terms (1 - P(E=1)) kappa /
    (1 - theta kappa). The paper (s3, DAX section) describes the updates as implemented and does not claim monotone increase of the
    penalised likelihood. A synthetic check (scratchpad, `mstep_check2.py`, one user, 114 of 400 impressions clicked) gave theta 0.755
    for the implemented update against 0.747 for the exact stationary point. Action: decide whether to keep the implemented updates
    (and say so, as s3 does) or to fix the code and re-run the DAX fits; a re-run changes every DAX number.
40. **Model description.** The LEAF-LGN score in `drup/leaf.py` and `drup/learned.py` includes learned user, item and global biases
    (MF base) that the outline (Section 4.3) omits. s3 states the score as implemented.
41. **Scenario factors above one.** The probe tab factors include values above 1 (for example tabs 9 and 10 near 1.4 at prior 80,
    `results/v11_probes/probe_pbm.txt`). s3 says the factors are relative to tab 1 and not bounded by one.
42. **SIM at KuaiRand-27K is incomplete.** `results/v12/sim_27k.json.partial.json` holds three of four trainings
    (`sim|lr0.003|wd1e-05` is missing), and no `sim_27k.json` exists. s4 reports SIM on 1K only. Action (Results writer): do not
    report a 27K SIM number unless the run is finished and summarised.
43. **Gate B2 family.** `stageB.txt` Section 4 and `final12.txt` Section 4 print raw paired p-values for the ablation arms and no Holm
    adjustment. s4 says so. Action: either compute the ablation-family Holm values in a script or drop the claim of an adjusted family.
44. **KuaiRand-1K: DAX increment on the 1K replication.** From the files: LEAF-LGN+DAX 0.4590 (`c_1k_lgn.json`) and LEAF-LGN 0.4579
    (same file), so the difference of the means is about 1.1e-3. `stageB.txt` Section 5 reports only the 1K comparisons against the QP
    family and the QP family with DAX, not this comparison. Action (Results writer): report the 1K paired difference against LEAF-LGN
    with its interval, as a file-based number, before the 1K replication is described.
45. **Stored LEAF-LGN count.** `results/v10/leaf_lgn.json` gives n_configs 197 and no n_evals field. The count is not a configuration
    count (`drup/leaf.py` BucketSelector.load_state comment). s4 does not use it.
46. **Ablation gains are against the ablation run's own base** (0.4706 in `dax_ablate.json`, not the main-run 0.4736). The full DAX
    gain is +15.10 [1.42] there and +11.96 [1.47] against the main-run backbone (item 23 of section C). s4 says which base applies.
