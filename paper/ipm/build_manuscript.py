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

# Everything after the SUPPLEMENT-START marker (structural properties with their
# proofs and experiments, additional results) goes to a separate
# Supplementary Material file, numbered S1, S2, ...
supp = ""
if "%SUPPLEMENT-START" in appendix:
    k = appendix.index("%SUPPLEMENT-START")
    supp, appendix = appendix[k:], appendix[:k]
    snum = {}
    cnt = {"sec": 0, "sub": 0, "thm": 0, "tab": 0, "fig": 0}
    tok = re.compile(r"\\(section|subsection)\{[^}]*\}\\label\{([^}]*)\}|"
                     r"\\begin\{(theorem|proposition|corollary)\}(?:\[[^\]]*\])?\\label\{([^}]*)\}|"
                     r"\\begin\{(table|figure)\}|\\label\{(tab:[^}]*|fig:[^}]*)\}")
    for mt in tok.finditer(supp):
        if mt.group(1) == "section":
            cnt["sec"] += 1
            cnt["sub"] = 0
            snum[mt.group(2)] = f"S{cnt['sec']}"
        elif mt.group(1) == "subsection":
            cnt["sub"] += 1
            snum[mt.group(2)] = f"S{cnt['sec']}.{cnt['sub']}"
        elif mt.group(3):
            cnt["thm"] += 1
            snum[mt.group(4)] = f"S{cnt['thm']}"
        elif mt.group(6):
            key = "tab" if mt.group(6).startswith("tab:") else "fig"
            cnt[key] += 1
            snum[mt.group(6)] = f"S{cnt[key]}"

    def to_supp(t):
        t = t.replace("Appendix~\\ref{app:more}", "the Supplementary Material")
        t = t.replace("\\ref{app:more}", "the Supplementary Material")
        for lab, sn in snum.items():
            t = t.replace("\\ref{" + lab + "}", sn)
        return t
    body, appendix, abstract = to_supp(body), to_supp(appendix), to_supp(abstract)
    supp = supp.replace("%SUPPLEMENT-START", "").replace("\\section{Additional results}\\label{app:more}",
                                                         "\\section{Additional results}")

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
\setlength{\emergencystretch}{3em}

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

bib = r"""\section*{Declaration of generative AI and AI-assisted technologies in the writing process}
During the preparation of this work the author used generative AI tools to
support code development and the development of ideas for the paper. After
using these tools, the author reviewed and edited the content as needed and
takes full responsibility for the content of the publication.

\bibliographystyle{elsarticle-harv}
\bibliography{../refs}

"""

out = preamble + body + bib + appendix + "\\end{document}\n"
open("ipm/manuscript.tex", "w").write(out)
if supp:
    sp = preamble[:preamble.index("\\journal")] + r"""
\usepackage{xr}
\externaldocument{manuscript}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetheorem}{S\arabic{theorem}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\begin{document}
\begin{center}{\large\bfseries Supplementary Material\\[2pt]
Walk-unbiased doubly robust graph propagation for recommendation from
exposure-biased logs}\end{center}

Theorem, table, section and equation numbers without the prefix S refer to the
main text.

""" + supp + "\n\\bibliographystyle{elsarticle-harv}\n\\bibliography{../refs}\n\\end{document}\n"
    open("ipm/supplementary.tex", "w").write(sp)
print("wrote ipm/manuscript.tex", len(out.split()), "words (approx.)",
      "| abstract", len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", abstract).split()), "words")
