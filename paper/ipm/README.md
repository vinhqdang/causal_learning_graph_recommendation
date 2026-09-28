# Submission package: Information Processing & Management

| file | content |
|---|---|
| `manuscript.tex` / `.pdf` | anonymised manuscript (elsarticle, review mode, author–year references). Generated from `../main.tex` by `build_manuscript.py`. |
| `title_page.tex` / `.pdf` | title, author, affiliation, declarations (competing interests, funding, CRediT, data availability) |
| `highlights.tex` / `.pdf` | 5 highlights, each ≤ 85 characters |
| `cover_letter.tex` / `.pdf` | cover letter to the editor |

Build, from `paper/`:

```
python3 ipm/build_manuscript.py
cd ipm && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
pdflatex title_page && pdflatex highlights && pdflatex cover_letter
```

Checklist before submitting:

- The abstract has 242 words (limit 250). Keywords: 7.
- Review is double-anonymised. The manuscript has no author names, affiliations,
  acknowledgements or repository link. The code link is on the title page only.
- Confirm the funding statement and the competing-interest statement on the
  title page.
- Elsevier asks every author to add a declaration on the use of generative AI
  tools in the writing process when such tools were used. Decide whether this
  applies and add it to the title page if it does.
- LaTeX sources (`manuscript.tex`, `../refs.bib`, `../tables/*.tex`) are
  uploaded together with the PDF.
