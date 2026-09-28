"""Build the anonymised IP&M manuscript (elsarticle) from paper/main.tex.

The body (introduction to discussion, plus the proof appendix) is taken from
main.tex; the front matter, citation style and closing sections are IP&M
specific. Run from paper/:  python3 ipm/build_manuscript.py
"""

import re

src = open("main.tex").read()
body = src[src.index("\\section{Introduction}"):src.index("\\bibliographystyle")]
appendix = src[src.index("\\appendix"):src.index("\\end{document}")]

# textual citations ("survey of X", "surveyed by X") and parenthetical ones
body = re.sub(r"\b(of|by|in|with)~\\cite\{", r"\1~\\citet{", body)
body = re.sub(r"~?\\cite\{", "~\\\\citep{", body)
appendix = re.sub(r"~?\\cite\{", "~\\\\citep{", appendix)
# elsarticle appends its own full stop to run-in paragraph titles
body = re.sub(r"\\paragraph\{([^}]*?)\.\}", r"\\paragraph{\1}", body)
appendix = re.sub(r"\\paragraph\{([^}]*?)\.\}", r"\\paragraph{\1}", appendix)
# elsarticle's \ref to an appendix section already prints "Appendix A"
body = body.replace("Appendix~\\ref{", "\\ref{")
# tables live one directory up
body = body.replace("\\input{tables/", "\\input{../tables/")

preamble = r"""\documentclass[preprint,review,12pt,authoryear]{elsarticle}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{adjustbox}
\usepackage[hidelinks]{hyperref}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{assumption}{Assumption}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}

\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\tW}{\tilde W}
\newcommand{\tY}{\tilde Y}

\journal{Information Processing \& Management}

\begin{document}
\begin{frontmatter}

\title{Walk-unbiased doubly robust graph propagation: provably fair,
transparent, accountable and private recommendation from exposure-biased logs}

% Anonymised for double-anonymised review: author details are given on the
% separate title page (title_page.tex).

\begin{abstract}
Graph-based recommenders propagate preferences along multi-hop walks of a
logged user--item graph. That graph has been filtered by the platform's own
exposure policy, so propagation amplifies exposure bias. Existing corrections
re-weight edges by inverse propensities or impute missing edges with doubly
robust (DR) estimates. We show that both remain biased beyond one hop: a walk
that revisits an edge squares its estimate, which inflates exactly the rarely
exposed edges that debiasing targets. The resulting bias grows with the
inverse of the propensity clip. We propose DRUP (Doubly Robust, walk-Unbiased
Propagation), which replaces every repeated edge by its idempotent unbiased
counterpart. Through Möbius inversion over the coincidences of walk indices,
DRUP is exactly unbiased and edge-wise doubly robust for propagation over the
counterfactual full-exposure graph at any number of hops. It costs no more than
the uncorrected operator and needs no training. The same structure yields
provable properties beyond accuracy: variance and concentration bounds;
exposure invariance as a causal item-fairness guarantee; exact Shapley
attributions and optimal counterfactual explanations; closed-form certificates
against injected profiles; joint differential privacy through a single public
item operator; and exposure-capped allocation with a causal utility guarantee.
On unbiased test data from Coat, Yahoo!\,R3 and KuaiRec, DRUP significantly
outperforms tuned MF, LightGCN, NAVIP and DR baselines on Yahoo!\,R3 and
KuaiRec, and it ties with the best trained model on Coat. Every guarantee is
verified empirically. On very sparse logs, exposure invariance needs a small
clip and costs accuracy.
\end{abstract}

\begin{keyword}
Causal recommendation \sep Graph neural networks \sep Doubly robust estimation
\sep Exposure bias \sep Fairness \sep Explainability \sep Differential privacy
\end{keyword}

\end{frontmatter}

"""

closing = r"""
\section{Implications}
\paragraph{For research}
Debiasing a graph recommender is not only a matter of re-weighting its loss or
its edges. Once propagation spans several hops, the propagation operator
itself must be made unbiased, and repeated walks are the precise place where
standard estimators fail. Treating recommendation scores as walk
polynomials of edge estimates also makes their fairness, explanations,
robustness and privacy analysable in closed form. This offers a way to
state and test trustworthiness claims rather than only measure them.

\paragraph{For practice}
DRUP needs no gradient training and runs at the cost of one item Gram matrix,
so it can serve as a strong, auditable baseline or as a production scorer.
Its exact attributions show which of a user's own interactions drive a
recommendation, or certify that none can overturn it. The public-operator form
lets a platform publish one privatised item operator while users score
locally. The exposure-capped allocation enforces hard provider-side exposure
limits with a certified optimality gap. The choice of the clip $\tau$ and of
the control-variate weight should follow the density of the log. On very
sparse logs, exact exposure invariance requires small clips and costs accuracy
(Table~\ref{tab:tau}).

\section{Conclusion}
We identified the repeated-walk bias of inverse-propensity and doubly robust
graph propagation, gave its exact form and a lower bound, and removed it with
an idempotent walk correction. The correction is exact for any number of hops
and costs no more than the uncorrected operator. The resulting estimator is
unbiased and edge-wise doubly robust for propagation over the full-exposure
graph. Its structure yields bounds on variance and clipping bias, causal
exposure invariance, exact and optimal explanations, certified robustness,
joint differential privacy and certified exposure-capped allocation. On three
benchmarks with unbiased test data, DRUP is at least as accurate as tuned
trained baselines and significantly more accurate on two of them. Future work
includes privatising the nuisance models, tighter certificates for large
catalogs, and combining the walk correction with learned propagation weights.

"""

bib = r"""\bibliographystyle{elsarticle-harv}
\bibliography{../refs}

"""

out = preamble + body + closing + bib + appendix + "\\end{document}\n"
open("ipm/manuscript.tex", "w").write(out)
print("wrote ipm/manuscript.tex", len(out.split()), "words (approx.)")
