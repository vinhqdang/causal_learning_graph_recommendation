#!/bin/bash
# Build all PDFs; references are re-rendered in APA 7 (journal style) after bibtex.
set -e
cd "$(dirname "$0")"
(cd .. && python3 ipm/build_manuscript.py >/dev/null)
for f in manuscript supplementary; do
  pdflatex -interaction=nonstopmode $f >/dev/null || true
  bibtex $f >/dev/null 2>&1 || true
  python3 apa_bbl.py $f.bbl
  pdflatex -interaction=nonstopmode $f >/dev/null || true
  pdflatex -interaction=nonstopmode $f | grep -E "Output written|Undefined|Citation.*undefined" || true
done
for f in title_page highlights cover_letter response_to_editor; do
  pdflatex -interaction=nonstopmode $f | grep "Output written" || true
done
cp manuscript.pdf ../main.pdf
