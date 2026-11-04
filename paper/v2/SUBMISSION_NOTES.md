# Submission record

- Manuscript: "Which negatives are really negative? Exposure-aware negative weighting and an imputation prior for graph collaborative filtering under exposure bias"
- Journal: Information Sciences (Elsevier)
- Submitted by the author: 2026-10-30 (as a new submission, not an Elsevier transfer of IPM-D-26-07816)
- Previous submission: Information Processing & Management, DRUP manuscript, rejected; the present manuscript has a different method, evaluation and conclusions.
- Files submitted: `paper/v2/submission/` (manuscript, title page, highlights, cover letter), source in `paper/v2/`.
- Abstract submitted: the 198-word version (no DR-LightGCN sentence, no "DR adjacency" clause); `paper/v2/sections/abstract.tex` still holds the earlier ~200-word text.
- Results frozen at repository state of this commit: main run (seed 0), seeds 1-3, AdvInfoNCE re-implementation, KuaiRec ensemble, offset-only ablation on Coat and KuaiRec.
- Open at submission: offset-only ablation on Yahoo! R3 and KuaiRand not completed (partial files `results/v4/learned_{yahoo,kuairand}_offonly.json.LightGCN-p.partial.json`); references listed in `paper/v2/TODO_citations.md` under "Still to verify by the author".
- Fallbacks if rejected: Knowledge-Based Systems, then ACM Transactions on Recommender Systems (article processing charge applies).

## Decision (received 2026-11-04)
- Information Sciences: declined by the Editor-in-Chief; within scope, but "lack of sufficient novelty". No reviewer reports were reported to us beyond the editor's message (the Editorial Manager folder "Submissions with a Decision" may contain comments; the author has not yet read them here).
- Consequence: the narrowed novelty (Q-posterior negative weighting plus imputation-logit offset in a graph model's pairwise loss) was judged insufficient. Close prior work found in the literature search: NAVIP, AdvDrop, PPD, DPAA, DR-GNN, Correct-and-Weight, IPS-weighted BPR on LightGCN.
- Next candidates (not yet decided): Knowledge-Based Systems; ACM Transactions on Recommender Systems; or strengthen the method contribution first.
