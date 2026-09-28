"""Build the anonymised IP&M manuscript (elsarticle) from paper/main.tex.

The body, abstract and keywords come from paper/main.tex and the files in
paper/sections/; only the front matter and the citation style are IP&M
specific. Run from paper/:  python3 ipm/build_manuscript.py
"""

import re


def expand(text):
    """Inline \\input{sections/...} recursively."""
    def rep(m):
        name = m.group(1)
        if not name.startswith("sections/"):
            return m.group(0)
        path = name if name.endswith(".tex") else name + ".tex"
        return expand(open(path).read())
    return re.sub(r"\\input\{([^}]*)\}", rep, text)


src = expand(open("main.tex").read())
abstract = src[src.index("\\begin{abstract}") + len("\\begin{abstract}"):src.index("\\end{abstract}")].strip()
kw_start = src.index("\\textbf{Keywords:}") + len("\\textbf{Keywords:}")
keywords = src[kw_start:src.index("\\section{Introduction}")].strip().rstrip(".")
keywords = " \\sep ".join(k.strip() for k in keywords.split(";"))
body = src[src.index("\\section{Introduction}"):src.index("\\bibliographystyle")]
appendix = src[src.index("\\appendix"):src.index("\\end{document}")]


def natbib(t):
    # textual citations ("survey of X", "surveyed by X") and parenthetical ones
    t = re.sub(r"\b(of|by|in|with)~\\cite\{", r"\1~\\citet{", t)
    t = re.sub(r"~?\\cite\{", "~\\\\citep{", t)
    # elsarticle appends its own full stop to run-in paragraph titles
    t = re.sub(r"\\paragraph\{([^}]*?)\.\}", r"\\paragraph{\1}", t)
    # elsarticle's \ref to an appendix section already prints "Appendix A"
    t = t.replace("Appendix~\\ref{", "\\ref{")
    # tables live one directory up
    return t.replace("\\input{tables/", "\\input{../tables/")


body, appendix, abstract = natbib(body), natbib(appendix), natbib(abstract)

preamble = r"""\documentclass[preprint,review,12pt,authoryear]{elsarticle}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{adjustbox}
\usepackage{tikz}
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

\title{Walk-unbiased doubly robust graph propagation for recommendation from
exposure-biased logs}

% Anonymised for double-anonymised review: author details are given on the
% separate title page (title_page.tex).

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{keyword}
""" + keywords + r"""
\end{keyword}

\end{frontmatter}

"""

bib = r"""\bibliographystyle{elsarticle-harv}
\bibliography{../refs}

"""

out = preamble + body + bib + appendix + "\\end{document}\n"
open("ipm/manuscript.tex", "w").write(out)
print("wrote ipm/manuscript.tex", len(out.split()), "words (approx.)",
      "| abstract", len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", abstract).split()), "words")
