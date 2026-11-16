# Rewrite brief: Sections 2 and 4 to 8, appendices, generated tables and figures

Read `STYLE.md` first; it is binding. The abstract, `sections/s1_intro.tex` and `sections/s3_method.tex` have been
rewritten and are the reference for tone, names and notation. A new appendix subsection `app:impl`
(implementation details: cut-off times, LEAF and LightGCN settings, DAX initialisation and clipping, the gamma and
DAX grids, the median number of impressions per item) was added to `sections/app_d_supp.tex`; material moved out of
the method section now lives there, so do not re-add it to Section 3.

Keep every `\label` and every citation key. Do not add citations that are not in `paper/refs.bib` or
`paper/v8/refs_new.bib`. Do not change any number; only move, drop or rephrase. After editing, rebuild the tables
(`python3 experiments/v13_tables.py`) and the figures (`python3 experiments/v13_figs.py`), then compile
(pdflatex, bibtex, pdflatex, pdflatex) and check that `main.log` has no undefined references. Finally run

    grep -nE 'texttt|results/|\.json|\.txt|\.py|\.log|[Ss]tage|[Gg]ate|B1|B2|probe|stored|v1[0-3]|v9|bank|partial|arm\b|DRUP|result file|is missing' sections/*.tex tables/*.tex

The only acceptable hits are the word "partial" in an ordinary mathematical sense (none expected) and the URL in
the availability statement.

## 0. Global replacements (apply everywhere, including tables)

| Internal term | Write instead |
|---|---|
| gate B1, primary family of the gate | "the primary comparison family, fixed before the main experiments were run" |
| gate B2 conditions | "the criteria for the ablations, fixed in advance" (state the three criteria in words) |
| stage 1, stage 1 run, stage 1 family | "the LEAF development experiment" (one sentence in Section 4 defines it: a first comparison of two LEAF variants with four baselines on the same splits) |
| stage B, stage C | drop; say "the main experiment" and "the robustness experiments" |
| probe, probe scale, probe number | "preliminary analysis on fixed backbone scores" (only in the appendix; remove from main text) |
| stored, stored run, (stored, v10) | "LEAF-LGN (independent tuning)" for the 197-configuration LEAF-LGN; "QP family" for the QP baseline (it is the reference version, no qualifier) |
| post-hoc run, main run | "the main experiment" or nothing; LEAF-LGN of the main experiment is simply "LEAF-LGN" |
| QP family (retrained) | "QP family, re-trained on its selected configurations" |
| arm, full arm, full-minus-arm | "variant", "full DAX", "full DAX minus variant" |
| shuffle_tags / shuffle_items labels | "attractiveness permuted across all items" / "attractiveness permuted within tags" |
| no_tab / no_life_prior / no_tag_prior | "without scenario factors $\kappa_t$" / "without lifelong prior on $\theta_u$" / "without tag prior on $\alpha_i$" |
| additive (MRP-style) | "additive logit model with user and scenario effects" (cite park2004mrp once, in Section 6 or 2) |
| prior, k\_life, prior\_s, tab=True, centre=tag, wd, lr, w, gamma | $\nu$, $\lambda$, user-group strength, scenario factors on, tag centre, $\rho$, learning rate, $w$, $\gamma$ |
| n\_configs | "configurations searched" (defined once in Section 4.3) |
| design check | "a control" |
| split-0 configuration | "the configuration selected on the first split" |
| DRUP | remove entirely (unpublished, not cited) |
| "X has no result file", "X.json is missing" | state the scope in words in Section 7 ("we did not evaluate ...") or drop |

## 1. Section 2, Related work (`sections/s2_related.tex`)

Mostly clean. Changes:
- l.12 "two-stage target attention" is fine (TWIN's own term), keep.
- `tab:positioning`, SIM row: "SIM run here (SIM-hard, 27K and 1K)" -> "SIM-hard is a baseline in our experiments". DIN row:
  "DIN run here on 27K and 1K" -> "baseline in our experiments".
- `sec:rel-occupancy` "Difference": keep the occupancy analogy but match the intro wording (impression = survey, click =
  detection, attractiveness = presence). Replace "the borrowed MRP and Fay-Herriot forms, which we tested on this target, are
  weaker" by "an additive alternative of the MRP kind is weaker in our ablations (Section~\ref{sec:res-ablate})"; Fay-Herriot is
  only a centre option inside DAX now (Appendix `app:fh`), so do not claim it is weaker unless a table shows it.
- `sec:rel-position`: shorten the last paragraph to three sentences; avoid "controlled evidence" as a slogan.

## 2. Section 4, Experimental protocol (`sections/s4_protocol.tex`)

- l.5: "the stage 1 controls" -> "the LEAF controls".
- `tab:data` (l.36 to 60): delete the "Source" column entirely; delete the table note about "780 difference is not traced"
  and "8 more" (move nothing to the main text). If a reconciliation note is wanted, put one sentence in `app:impl`: "Counts of
  out-of-pool impressions differ by less than 0.02\% between the raw logs and the aggregated tables." Replace "(tables)" and
  "(DIN and SIM sequences)" with "(LEAF statistics)" and "(sequence models)".
- l.62 to 68 (why not Coat, Yahoo! R3, KuaiRec): keep the argument, delete "against 0.6378 for DRUP
  (\texttt{results/v13\_saturation\_kuairec.txt})". Write: "on KuaiRec, a training-free prior fitted to the log already reaches an
  nDCG@20 of 0.6367 on 1,410 users, close to that of learned recommenders, which leaves little room to detect an improvement."
- l.72 to 79: remove "(\texttt{results/v9\_probes/stats\_1k.txt})". Say how splits are formed in words (random permutation of
  users, 30\% validation, 70\% test, five seeds) without `frac_val`.
- l.85 to 90: define "configurations searched" once, in words: "the number of backbone trainings times the number of DAX
  configurations times the number of values of $\gamma$". Keep the counts. Point to `sec:res-fair` for the fairness discussion.
- l.94 to 123 (tests and families): keep the mathematics. Rewrite the family list as a short itemised list with plain names:
  primary family (5, fixed in advance), main-table family (9), sensitivity family (11), KuaiRand-1K family (3), ablation
  families (7 gains; 6 differences), LEAF development family (8) and LEAF controls (8). Delete l.122 to 123 ("computed by
  \texttt{experiments/v13\_tables.py} ... \texttt{results/v13\_tables\_numbers.txt}").
- l.125 to 131 ("Gate and reading"): rename the paragraph "Pre-specified criteria". Delete "\texttt{paper/v7/DESIGN.md} (Section 7)";
  say "fixed before the main experiment was run". State the 15e-3 margin and the three ablation criteria in words.
- l.135 to 147 (baselines): fine; keep "The QP family is the authors' own earlier method, not published" and the pointer to
  `app:qp`. Write SIM-hard once with a one-line explanation, then "SIM" in tables is acceptable only if the caption says
  "SIM denotes SIM-hard".
- l.149 to 159 (ablation design): replace "arm" by "variant". Explain in one sentence why the ablation base differs from the main
  LEAF-LGN (a reduced search around the selected configuration, to keep cost manageable) and why comparisons use
  "full DAX minus variant".
- `sec:protocol-kinds` (l.161 to 167): rewrite without directories: "We report three kinds of number. Main results are selected on
  validation users and measured on test users. Diagnostic quantities use random-exposure labels to locate headroom; they are never
  used to select or train a method and are marked as diagnostic. Preliminary analyses on fixed backbone scores appear only in the
  appendix." Keep the label.

## 3. Section 5, Results (`sections/s5_results.tex`)

Restructure every subsection as: **one or two sentences stating the question and the answer in words**, then the key numbers,
then the table or figure, then caveats. Suggested openings (keep the numbers exactly as in the current text):

| Subsection | Opening question and answer |
|---|---|
| `sec:res-headroom` | "Where is the remaining accuracy of a strong graph recommender on random insertions? Mostly in item-level appeal: an oracle that knows each item's click rate under random insertion is far ahead, and drift does not explain the gap." |
| `sec:res-methods` | "How do the methods rank, and how much did each search? LEAF-LGN+DAX has the highest mean; budgets differ widely, which the next subsections address." |
| `sec:res-main` | "Is LEAF-LGN+DAX more accurate than each baseline? Yes on the per-user test against all nine; on the split-level test against all nine after Holm correction, with two borderline comparisons." |
| `sec:res-fair` | "Is the advantage explained by the DAX module or by search budget? No: with the same module and a comparable budget, the QP family remains behind." |
| `sec:res-backbone` | "Does DAX help backbones other than LEAF-LGN? Yes, its increment is positive on all six; the split-level test supports it on five." |
| `sec:res-ablate` | "Which parts of DAX matter? All three priors contribute; the multiplicative form beats an additive one; permuting attractiveness across items removes the gain, within tags keeps about two fifths of it." |
| `sec:res-1k` | "Does the result hold on the smaller release? The advantage over the QP family, with and without DAX, holds." |
| `sec:res-seeds` | "Is the result specific to one training seed? No; three further seeds give the same level." |
| `sec:res-deep` | "Do lifelong deep models do as well? No; DIN and SIM-hard, with or without DAX, stay below, with one borderline comparison." |
| `sec:res-controls` | "Do the signals depend on what they are supposed to capture? Shuffling users' histories removes the LEAF gain; shuffling attractiveness across items removes the DAX gain." |

Specific edits:
- Opening paragraph: keep the definition of differences and the borderline rule; move the per-user / split-level test
  definitions to a single sentence pointing to `sec:protocol-stats`.
- `sec:res-headroom` l.11 to 34: delete all file names (l.14, 16, 22, 24, 28). Delete the "second file gives an oracle of 0.5072"
  sentence or rephrase as "a second computation with a different user set gives 0.5072". Replace "split-0 configuration" (l.13,
  l.32) as in Section 0. Delete the last paragraph (three LEAF-LGN values) from the main text; put a single clarifying sentence
  in the caption of `tab:methods` instead. Consider a small table for the headroom numbers (popularity, LEAF-LGN, LEAF-LGN with
  DAX blend, oracle) marked "diagnostic".
- `sec:res-main` l.60 to 71: figure caption `fig:forest`: delete "Source: ..." and say what a dot and a diamond are and that
  positive values favour LEAF-LGN+DAX. Rewrite l.68 to 71 without "gate B1" and without `stageB.txt`: "The pre-specified criterion
  of the primary family, a margin of at least $15\times10^{-3}$ over the QP family with both Holm-adjusted p values below 0.05, is met
  ($+22.32$ [1.83]; split-level Holm-adjusted $p=0.023$ in the family of five)."
- `sec:res-fair` l.79 to 81: "stored QP family", "stage 1 family of eight" -> "QP family", "LEAF development family of eight".
- `sec:res-backbone` l.90 to 92: delete `dax_lightgcn.json`; write "the LightGCN row uses a LightGCN re-trained within the same
  experiment as LightGCN+DAX; against the four-configuration LightGCN of Table~\ref{tab:main} the increment is $+13.73$ [1.46]".
- `sec:res-ablate` l.98 to 125: replace "arms" and "ablation run"; delete "They are listed in
  \texttt{results/v13\_tables\_numbers.txt}" (l.119): either report the range of raw split-level p values in the text, as now, or
  add a column to `tab:ablate`. Caption of `fig:ablate`: delete "Source: \texttt{results/v11/dax\_ablate.json}".
- `sec:res-seeds` l.140 to 158: caption of `fig:seeds`: delete both file paths. Say "Seeds 1 to 3 use a reduced backbone search
  (two trainings)" instead of describing it via the files.
- `sec:res-deep`: fine apart from "SIM" vs "SIM-hard" consistency.
- `sec:res-controls` l.177 to 179: "stage 1 run on the LEAF-tied backbone" -> "the LEAF development experiment, which used the
  closed-form variant LEAF-tied". "The LEAF-LGN ablations have no result file" -> "We did not repeat these controls on LEAF-LGN."
- Do not end the section with a summary that repeats all numbers; three sentences in words suffice.

## 4. Section 6, Discussion (`sections/s6_discussion.tex`)

- l.4 "against the full arm" -> "against full DAX".
- Use the running example once: explain the multiplicative versus additive point with the cooking clip (non-clicks from
  inattentive viewers barely lower its attractiveness under DAX, but lower its additive item effect as much as any other
  non-click).
- "Remaining headroom": keep as interpretation, mark it as such (already done).

## 5. Section 7, Limitations (`sections/s7_limitations.tex`)

- l.19 to 20: replace the file names by "We did not evaluate a variant in which $\gamma$ is learned jointly with the backbone."
- l.24 to 26: replace by "The prior strengths $\nu$ and $\lambda$ were selected at the edges of their grids, and we did not
  test whether a wider grid would select other values."
- l.34: "the LEAF-LGN ablations have no result file" -> "the LEAF controls were run on LEAF-tied, not on LEAF-LGN".
- l.39 to 41: "We used the click label only; we did not evaluate a long-view label or a protocol in which the training log
  extends past $T_0$ for the full model."
- l.13 to 15: delete "A reviewer may describe DAX as a click model applied to items"; say instead "DAX applies the structure of
  examination-based click models to a feed without positions; its novelty lies in the user and scenario examination factors, the
  priors and the evaluation target, not in the factorisation itself."

## 6. Section 8, Conclusion (`sections/s8_conclusion.tex`)

Rewrite as two short paragraphs in words: what problem, what the two ideas are, what the evaluation shows (at most two numbers,
e.g. 0.4854 against 0.4634), what remains open (one platform; joint training; richer content). Remove the p-value sentence. Then
add, after the conclusion and before the bibliography in `main.tex`, an unnumbered section:

    \section*{Data and code availability}
    KuaiRand is publicly available \citep{gao2022kuairand}. The code that builds the lifelong statistics, fits DAX,
    trains all models and produces every table will be released upon publication.

This is the only place where code is mentioned.

## 7. Appendices

### `sections/app_a_qp.tex` (QP family)
- l.5: delete "(the \texttt{pop} option of \texttt{drup/estimation.py}, \texttt{get\_propensity})"; write the propensity
  model: "$P_{ui}=\sigma(a+b\log(1+\mathrm{pop}_i))$ fitted by logistic regression on the policy log" (check the form in the
  code before writing; if it is not exactly this, describe it in words: "a logistic model of exposure on item popularity").
- l.7: delete "\texttt{baseline\_imputation}"; describe the imputation as "an additive model of user and item effects fitted with
  inverse-propensity weights".
- l.17: delete "(\texttt{results/v10/learned\_kuairand\_t\_qp.json})".
- l.19: delete the paragraph "Controls on the same family" (it only points to a script and reports nothing).

### `sections/app_b_em.tex` (EM)
- l.3: "implemented in \texttt{drup/dax.py} (\texttt{fit\_dax})" -> "This appendix gives the complete model and updates."
- Align notation with Section 3: prior strengths are $\nu$ and $\lambda$ (not $a$ and $k$); lifelong centre $\tilde c_u$;
  tag centre $c_g$. "With $k=0$ the code uses" -> "With $\lambda=0$ we use".
- l.33 to 40 ("Approximation and clamps"): keep the mathematics, but replace "the implemented sum" by "the update"; delete
  "(\texttt{drup/dax.py}, lines 105 to 133)". Add the Poisson-approximation reading of the $\theta$ and $\kappa$ updates from
  Section 3.4 so the two places agree.
- l.42: "$\texttt{prior\_s}=80$ (\texttt{results/v11/dax\_lgn\_post.json}, chosen configurations)" -> "strength 80, the value
  selected in every split".
- Several clamps are now also stated in `app:impl`; keep them in one place (preferably here) and shorten `app:impl` accordingly.

### `sections/app_c_grids.tex` and `tables/tab_grid_*.tex`
- Text: replace `\texttt{prior\_s}` by "the user-group strength"; "for every method with a result file" -> "for every method".
- The 18 grid tables print raw configuration dictionaries in typewriter font. See Section 8 below for the generator change:
  render each table with one column per hyperparameter, headed by the symbol ($\mathrm{lr}$, $\rho$, $w$, epoch, $\gamma$, $\nu$,
  centre, $\lambda$, scenario factors, user groups). Values: "tag"/"content", "on"/"off", "none" for no user groups.
- Captions: "LEAF-LGN+DAX (main, post-hoc)" -> "LEAF-LGN+DAX"; "LEAF-LGN (post-hoc base)" -> "LEAF-LGN"; "QP family (stored,
  per-split best variant)" -> "QP family (best of its three variants per split)". Consider merging the 1K tables into one.

### `sections/app_d_supp.tex`
- l.6: delete "Every table here is generated by \texttt{experiments/v13\_tables.py} from the result files named in its note."
- l.4 to 5: "the stage 1 LEAF controls" -> "the LEAF controls".
- `app:cost`: delete "(\texttt{results/logs\_v11/dax\_lgn\_post.log})"; keep the measured times as "measured on a machine with two
  CPU threads".
- `app:controls` and `tab:controls`: this table must be rewritten from scratch. Delete every unit-test name, every file name,
  the dates and the bracketed kind tags. Group the rows under three headings and phrase each as a check in words:
  (a) *Leakage checks*: all training inputs precede $T_0$; the random-exposure log is never a training input; lifelong
  statistics and the DAX fit are unchanged when post-cut rows or validation and test labels are altered; sequence models see
  only out-of-pool events before $T_0$; user identifiers of the two releases agree for 0.9999 of the joined events.
  (b) *Correctness of the estimators on simulated data*: LEAF statistics equal Equations (2) and (3); the EM likelihood does not
  decrease with negligible priors; EM recovers the ordering of $\alpha$ and $\theta$ and the scenario factors; the within-tag
  permutation keeps tags and the all-item permutation is a bijection.
  (c) *Controls on real data*: shuffled users ($+18.90$ [1.99] and $-1.02$, LEAF-tied; $-0.93$ [0.45] on LEAF-LGN scores);
  attractiveness permuted across items (gain $-0.27$ [0.36]); tag mean alone, within-tag permutation, full ($+4.31$, $+3.29$,
  $+9.45$, preliminary analysis on fixed scores); reproduction of LEAF-LGN ($+0.73$ [0.52]); off-protocol calibration
  ($+36.62$ [1.79], diagnostic, not a method).
  Replace the intro sentence about "main", "probe", "diagnostic", "code", "unit" rows by a one-line legend using the kinds of
  number defined in Section 4.
- `tab:baselines`: delete the "Source" column. "(Stage 1 only)" -> "(LEAF development experiment only)". "``Not run'' marks a
  row that has no result file" -> delete.
- `app:leafctl`: title "LEAF controls"; replace "stage 1 run", "design check of that run" -> "control", and delete the sentence
  with `leaf_lgn_ablate.json`; say "These controls were run on LEAF-tied."
- `app:sens`: fine.
- `alg:dax`, `alg:leaflgn`: notation already matches Section 3; change "fit (once per configuration) the DAX table" -> "fit DAX
  once per configuration (Algorithm~\ref{alg:dax})", and rename $\operatorname{rstd}$ to $\operatorname{std}_{C_u}$ as in
  Equation~\eqref{eq:combine} of Section 3.

## 8. Table generator (`experiments/v13_tables.py`) and figure generator (`experiments/v13_figs.py`)

The generated `.tex` files must obey `STYLE.md`. Edit the generator, not the generated files.

| Location in `v13_tables.py` | Current text | Change |
|---|---|---|
| `DISPLAY` (l.66) | "LEAF-LGN (stored, v10)", "LEAF-LGN (post-hoc run)" | "LEAF-LGN (independent tuning)", "LEAF-LGN" |
| `ABL_LABEL` (l.71 to 73) | "no examination factor kappa_t", "no lifelong prior on theta_u", "no tag prior on alpha_i", "additive logit (MRP-style), tab effects", "alpha shuffled across items", "alpha shuffled within tag" | "without scenario factors $\kappa_t$", "without lifelong prior on $\theta_u$", "without tag prior on $\alpha_i$", "additive logit, user and scenario effects", "$\alpha$ permuted across items", "$\alpha$ permuted within tags" (do not pass math through `esc`) |
| `V10_LABEL` (l.77) | "shuffle users" etc. | "users' lifelong statistics shuffled", "without tag channels", ... |
| methods table (l.224 to 240) | "Source" column with JSON names; note "The stored LEAF-LGN counts another unit"; "stored QP variants" | drop the Source column; note: "Configurations searched: backbone trainings times DAX configurations times values of $\gamma$. LEAF-LGN (independent tuning) is a separate search of the backbone alone, counted per training and epoch setting, and is not comparable. The QP family is, in each split, the best of its three variants on validation." Caption: drop "and the source file". |
| main table note (l.282 to 290) | "Sources: results/v11/... results/v12/..."; "stored four-configuration run of the stage 1 file" | delete the Sources sentence; "LightGCN is the four-configuration LightGCN baseline (a LightGCN re-trained with DAX appears in Table~\ref{tab:backbone})". Keep the test definitions. |
| sensitivity caption (l.306) | "counts the comparators whose files exist" | "counts the comparators in the family" |
| fair table (l.324 to 342) | row "LEAF-LGN (stored, v10) minus QP family"; caption "the stored LEAF-LGN"; note "the stored v10 comparison is part of a different family (eight comparisons in the stage 1 run)" | row "LEAF-LGN (independent tuning) minus QP family"; caption "LEAF-LGN without DAX"; note "... is part of the LEAF development family of eight comparisons" |
| backbone table (l.353 to 373) | "QP family (stored)"; note "\texttt{dax\_lightgcn.json} (3 base configurations ...)", "stored four-configuration LightGCN", "stored QP family" | "QP family"; "LightGCN: base and DAX variant come from the same experiment (3 base configurations; 960 with DAX). Against the four-configuration LightGCN of Table~\ref{tab:main} the increment is ..."; drop "stored" |
| ablation table (l.389 to 446) | note "Source: results/v11/dax\_ablate.json", "Arm", "ablation run", "main-run LEAF-LGN", "(family sizes computed here, not in final12.txt)", "pooled arm"; header "Arm", "Full minus arm"; caption "(post-hoc)" | delete the Source sentence and the final12 parenthesis; header "Variant", "Full minus variant"; "ablation run" -> "ablation experiment"; "main-run LEAF-LGN" -> "LEAF-LGN of Table~\ref{tab:main}"; "pooled arm" -> "the variant above, which selects over the three permutation seeds"; caption without "(post-hoc)" |
| 1K table note (l.485 to 490) | "Sources: results/v11/c\_1k\_lgn.json, ..." | delete the Sources sentence |
| seeds table (l.507 to 535) | row "QP family: seeds 1 to 3 not run (partial file incomplete)"; note "Sources: results/v11/dax\_lgn\_post.json (seed 0) and results/v11/c\_seeds/...", "the main run" | delete the QP row (say in the note "The QP family was not re-trained under seeds 1 to 3"); delete Sources; "the main run" -> "the main experiment" |
| deep table note (l.571) | "Sources: results/v12/din\_27k.json, ..."; "lr {1e-2, 3e-3} times wd {1e-4, 1e-5}" | delete Sources; "learning rate $\{10^{-2},3\times10^{-3}\}$ times weight decay $\{10^{-4},10^{-5}\}$" |
| LEAF controls (l.586 to 610) | note "Source: results/v10/leaf\_tied.json ...", "(results/v10/leaf\_lgn\_ablate.json is missing)", "design check"; caption "(stage 1 stored run, LEAF-tied backbone)"; "arm" | delete Source and the missing-file clause; "control" for design check; caption "LEAF controls on KuaiRand-27K (closed-form LEAF-tied backbone): means and full minus variant" |
| grid tables (l.617 to 666) | `fmt_cfg` prints `key=value` in `\texttt`; labels "(main, post-hoc)", "(post-hoc base)", "(stored, per-split best variant)" | one column per hyperparameter with symbol headers (see Section 7); map keys: `lr`->learning rate, `wd`->$\rho$, `w`->$w$, `gamma`->$\gamma$, `prior`->$\nu$, `k_life`->$\lambda$, `tab`->scenario factors on/off, `centre`->tag/content, `prior_s`->user groups (none or strength), `method`->QP variant name, `pbeta`->offset weight, `r`->normalisation exponent; drop `dim`, `layers`, `fn` (constant, stated in `app:impl`) |
| `tbd(...)` cells (l.148 and calls) | print file names when an input is missing | keep for internal builds, but the submitted PDF must contain no `{TBD:...}` cell; check with grep before submission |

`v13_figs.py`: in `fig_forest` rename "LEAF-LGN (post-hoc run)" to "LEAF-LGN" and fix the literal "95\%" in the axis label (use
"95% half-width" with matplotlib's text, not a LaTeX escape); in `fig_ablate` use the variant names above
("without $\kappa_t$", "without lifelong prior", "without tag prior", "additive", "permuted across items", "permuted within tags").

## 9. Final checks for the other writer

1. Every Results subsection starts with the question and the answer in words.
2. The grep in the header returns nothing.
3. The abstract has at most two numbers; the introduction has numbers in one paragraph only (already true; do not add).
4. Names: LEAF, LEAF-LGN, DAX, LEAF-LGN+DAX, QP family, SIM-hard defined once.
5. No em dashes (`---`) and no en dashes used as dashes (` -- `) in any `.tex` file.
6. Compile with no undefined references or citations.
7. The `elsarticle` class adds a full stop after a `\paragraph{}` title, so titles must not end with a period
   (`\paragraph{Split-level test}` not `\paragraph{Split-level test.}`). The intro, method and `app:impl` already follow
   this; fix the remaining ones (Sections 4 and 6, Limitations, appendices A and B).
