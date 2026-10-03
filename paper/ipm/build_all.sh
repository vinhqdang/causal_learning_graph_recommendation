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
# Descriptively named copies for upload (the build names stay fixed because the
# supplement cross-references the manuscript aux file).
mkdir -p submission
cp manuscript.pdf    submission/DRUP_1_Manuscript_anonymised.pdf
cp supplementary.pdf submission/DRUP_2_Supplementary_Material_anonymised.pdf
cp title_page.pdf    submission/DRUP_3_Title_page_with_author_details.pdf
cp highlights.pdf    submission/DRUP_4_Highlights.pdf
cp cover_letter.pdf  submission/DRUP_5_Cover_letter.pdf
cp response_to_editor.pdf submission/DRUP_6_Response_to_editor.pdf
