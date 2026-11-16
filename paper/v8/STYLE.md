# Style guide for the v8 manuscript (binding)

Applies to every file that ends up in the PDF: `sections/*.tex`, `tables/*.tex`, figure captions, and the
captions and notes written by the table generator. The abstract, `s1_intro.tex` and `s3_method.tex` already
follow it; use them as the reference.

## 1. The manuscript is self-contained

A reviewer has never seen the repository. Nothing in the PDF may point to it.

- **Forbidden anywhere in the PDF** (text, captions, table notes, algorithm boxes, axis labels):
  file names and paths (`results/...`, `*.json`, `*.txt`, `*.log`, `*.py`, `tests/...`, `paper/v7/DESIGN.md`),
  script, module, class and function names (`run_dax`, `fit_dax`, `LEAFLGNDAX`, `protocol_split`, ...),
  configuration keys as written in code (`prior_s`, `k_life`, `tab=True`, `centre=tag`, `n_configs`, `frac_val`),
  `\texttt{}` for anything other than a URL in the availability statement,
  line numbers of code, unit-test names, dates on which a test was run.
- **Forbidden internal vocabulary**: bank, partial, stage 1 / stage B / stage C, gate, B1, B2, probe, probe
  scale, stored (as in "stored run"), v9, v10, v11, v12, v13, "the main run", "the post-hoc run", "arm" (say
  "variant" or "ablation"), "design check", "result file", "has no result file", "is missing", split-0
  configuration, DRUP, "the code", "the implementation".
- **Describe procedures in words and mathematics.** Instead of "computed by `v13_tables.py`" write what is
  computed ("Holm's step-down procedure over the seven ablation variants"). Instead of "the run asserts" write
  "we verified that".
- **Experiments that were not run are not mentioned as missing files.** Either omit them or state the scope in
  words in the Limitations ("we did not train the combination jointly", "we did not evaluate the long-view
  label").
- **Reproducibility** goes in one place: a short "Data and code availability" statement after the conclusion
  (data: KuaiRand release, cited; code: "will be made available upon publication" or an anonymised link). Nowhere
  else.

## 2. Where numbers go

- **Abstract**: at most one or two headline numbers. It states problem, gap, idea, method, evaluation setting,
  main finding in words, and what it means. No p values, no half-widths, no configuration counts.
- **Introduction**: at most one paragraph with a few numbers (the "Main findings" paragraph). The motivation,
  gap and ideas are stated without numbers.
- **Method**: no experimental numbers. Hyperparameter values, grids, run times and data counts go to the
  appendix (`app:impl`, `app:grids`, `app:cost`).
- **Results and tables** carry the evidence. Each Results subsection opens with the question and the answer in
  one or two sentences of words, then gives the numbers and the table.
- Paired differences: per-user mean difference of nDCG@10 times 10^3, with the 95% half-width in brackets,
  defined once in Section 4 and then used without re-explanation. Copy numbers exactly; do not re-round.
- "significant" only when the Holm-adjusted p value of the named test is below 0.05. A Holm-adjusted value
  between 0.04 and 0.05 on the split-level test is called "borderline". Never "state of the art", "remarkable",
  "substantial" without a number, "clearly", "dramatically".

## 3. Names and symbols

- Method names, always written exactly so: **LEAF**, **LEAF-LGN**, **DAX**, **LEAF-LGN+DAX**; for other backbones
  "X+DAX" (EASE+DAX, LightGCN+DAX, DIN+DAX, SIM+DAX, QP family+DAX). "LEAF-tied" is the closed-form variant used only
  for the LEAF controls. The baseline is "the QP family" (our own unpublished baseline, fully described in
  Appendix A, never cited and never called "our previous work").
- Data: KuaiRand-27K (primary), KuaiRand-1K (secondary); "policy log", "lifelong log", "random-exposure log",
  "pool", "candidate list", "cut-off $T_0$", "scenario" (with "tab" allowed once as an explanation).
- Symbols are those of Section 3 and must not be redefined: $\mathcal U,\mathcal I,\mathcal T,T_0,\mathcal D,
  X_{ui},N_{ui},C_u,v_k(i),r_u,\ell^k_{uv},f^k_{uv},\eta,z_{ui},g,\beta,q_i,\rho,w,\theta_u,\kappa_t,\alpha_i,\tau_r,
  p_{E,r},p_{R,r},\nu,\lambda,c_g,\tilde c_u,\zeta_i,\gamma$. In tables use the same symbols ($\nu$ not "prior",
  $\lambda$ not "k_life", $\rho$ not "wd"). Every symbol is defined before its first use in a section that a reader
  can reach without the appendix.
- One running example only: the viewer who watches cooking videos and the cooking clip by a little-known author
  inserted at random. Reuse it (Discussion may refer back to it); do not introduce a second example.

## 4. Prose

- Plain academic English, British spelling (as in the existing text: "randomised", "normalisation").
- No marketing ("novel", "powerful", "significantly outperforms"), no rhetorical questions in the Results, no
  bold labels inside running text, no one-line closing slogans.
- No em dashes (and no en dashes used as dashes); use commas, colons, parentheses or two sentences.
- Do not attribute work to tools; there is no author-contribution statement that mentions software assistants.
- Every table and figure is referenced in the text before it appears, and its caption says what is shown and how
  to read it (what a positive value means, which test, which family) without naming any source.
