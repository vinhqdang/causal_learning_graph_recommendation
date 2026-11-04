# Review 2 of the v4 manuscript: required fixes (apply all; line numbers approximate)

BLOCKING
1. abstract.tex is about 290 words; must be <= 200. Structure: one thesis sentence first; protocol; the four findings C1-C4 with the main numbers only (per-split test-oracle inflation up to +0.030 on Coat; third-party debiasers point-estimate gains of at most +4.10 [+5.26 in seed re-runs]; additive item prior +31.98 Coat / +21.22 KuaiRec with the Nadeau-Bengio caveat; walk correction no ranking benefit); scope sentence (four benchmarks, Coat/Yahoo/KuaiRec for the real-log walk claim). Move the QP numbers (+16.77, +3.93), the 5.26 re-runs and Coat half-widths to s1/s5.
2. s5 ~l.76, s6 ~l.49, conclusion say semi-synthetic runs contain no nDCG / ranking 'not measured'. False: results/v3/semisynth_kuairec_dense.json stores ndcg20 per method (DRUP above DR in 8 of 10 cells: indep/true/fixed 0.1071 vs 0.0973, split 0.0479 vs 0.0303, est 0.0379 vs 0.0294; not in the yhat cells 0.3697 vs 0.3769). Read the file, add a small generated appendix table (script in experiments/, e.g. v7_tables.py or a new v7_semisynth.py; true-label semi-synthetic world) and fix the text in s5/s6/s7 accordingly (RQ4).
3. s1 ~l.66 and C2 ~l.96 still say 'change ... by at most'; use 'improve ... by point-estimate gains of at most +4.10' (MACR -48.31, SimGCL -36.49, AdvInfoNCE -21.08 on Yahoo are strongly negative).
4. File paths still printed: s5 ~l.60 (\path{results/v4/ensemble_summary.txt}), ~l.76 (mc_stress and semisynth paths), ~l.82 (filters_kuairec_pop.json). Remove; cite 'the stored synthetic-study outputs' or a table.

MAJOR
5. s3 Nadeau-Bengio paragraph (~l.168-180): state n1 and n2 explicitly (selection part 30%, test part 70%; code significance.py uses (1-frac_val)/frac_val = 7/3 = n_test/n_sel) or drop 'conservative'; report the J=10 value for Coat (0.1+2.33) next to J=5.
6. NB disclosure one-sided: these survive NB: NAVIP KuaiRec +10.41 (NBHolm 0.034, T12), LightGCN-pt KuaiRec (0.030), QP vs LightGCN-p on Yahoo +18.20 (NBHolm 0.0023), DR-BPR-LGN-1 (0.031). Add one clause each in s5 ~l.40 and ~l.58-60; qualify 'none of the gains survives' in abstract and s7 as 'on Coat, Yahoo! R3 and KuaiRand-Pure'.
7. s5 ~l.21: 'Of the 14 conclusions, 7 are significant ... signs survive in 12, 11, 13, 11' - counts are over all 14 (T7); rewrite.
8. s5 ~l.78 says T8 gives selected-run values but T8 has only the KuaiRec row; add Coat (+1.40 [3.86]) and Yahoo (+0.00 [0.00], identical per-user rows: walk correction inactive in the selected Yahoo runs, uninformative for RQ4) to T8 via the generator.
9. s6 ~l.15 'the backbone matters as much as the debiaser' and 'can produce a gap of that size from budget alone' are untested; duplicate of s5 ~l.68. Reword to 'grid choice moved iALS by +12.92 [8.01] (pHolm 0.003)'.
10. s1 ~l.47-50 and s6 ~l.26 present pHolm=0.047 as 'significant after Holm' though baselines were test-chosen: say 'nominally'. s5 ~l.58, s6 ~l.11 'the gain there comes from the weights' on KuaiRand (no control): say 'is not from the offset (beta = 0 in 4 of 5 splits)'.
11. s5 ~l.60: add '5 of 5 jobs (KuaiRec), 10 of 10 (Coat), single run, no seed repeats' for DRGP.
12. Thesis early + RQ table: one thesis sentence at the top of the abstract and the first sentence of s1; replace the roadmap at s1 ~l.127-132 by a 4-row table 'RQ -> answer -> section' (labels from the \ref's).

MINOR
13. Jargon: 'bank', 'replay', 'unsplit legacy result files' (s3 ~l.115-124, s6 ~l.43), 'the DRGP queue' (s4 ~l.197), 'earlier versions of this work' (s4 ~l.11), 'stage 1/A/B' -> 'stored result files', 'earlier runs', 'grid'.
14. s6 ~l.9 'significantly better' without pHolm: add or cite s5.
15. T5 prints 'yes (p=<0.001)': 'yes (p<0.001)'; define the T6 TOST 'yes'/'no' in the note (generators).
16. Delete filler: s7 'Four findings stand out'; s6 'in ways that matter at the scale of'; 'Each item rests on a result above'; 'The check is cheap'; s1 'Our reading of the evidence is as follows'.

SHORTENING (do after the fixes; target about 40 pages including the appendix, main text about 25; keep every number's source; moved material goes to appendix sections in main.tex with their tables)
A. s4_setup.tex ~l.190-342 (Q-weighting grids, QP, DRGP stages, ensemble): move to an appendix 'Refinement family' with the configuration-count table; keep one paragraph in the main text.
B. s6 'What the evidence supports' and the Implications numbers: replace with the RQ-answer table plus 5 bullets with no numbers repeated from s5; merge limitations from 17 to about 8 (merge 8/10/11/13/14/15/16).
C. s1 ~l.64-125: collapse to the thesis paragraph and the RQ table; C1-C4 as a short list with at most one number each.
D. s5 RQ1 paragraphs: keep 3 sentences and the key tables (T5, T9); move T7, T10, T11 and the rank-distribution figure to the appendix.
E. s5 RQ4 (mc_stress, semi-synthetic, training-free ranking) and the prior-alone/DRGP/ensemble paragraph: appendix with generated tables (incl. the new semi-synthetic ndcg20 table); compress s2 positioning.
After all edits: abstract <= 200 words (count with a script), no 'state of the art', no new numbers without a source in results/ or generated tables, no file paths in the manuscript, \ref to all moved material.
