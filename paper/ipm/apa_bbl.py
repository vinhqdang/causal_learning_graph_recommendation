"""Replace the body of each \\bibitem in a .bbl by its APA 7 rendering.

The labels written by elsarticle-harv (author, year, a/b suffix) are kept so that
natbib citations stay valid; only the formatted text changes. The text is made
by pandoc/citeproc with the APA 7 style from the CSL repository, which is the
parent of the journal's own CSL style.
"""
import re, subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
bbl = sys.argv[1]
src = open(bbl).read()
parts = re.split(r'(?=\\bibitem\[)', src)
head, items = parts[0], parts[1:]
out = [head]
for it in items:
    m = re.match(r'(\\bibitem\[\{.*?\}\]\{([^}]+)\})', it, re.S)
    lab, key = m.group(1), m.group(2)
    tail = ''
    if it.rstrip().endswith('\\end{thebibliography}'):
        tail = '\n\\end{thebibliography}\n'
    md = '---\nnocite: "@%s"\n---\n' % key
    r = subprocess.run(['pandoc', '-f', 'markdown', '-t', 'latex', '--citeproc',
                        '--bibliography', os.path.join(here, '..', 'refs.bib'),
                        '--csl', os.path.join(here, 'apa.csl')],
                       input=md, capture_output=True, text=True)
    body = r.stdout
    body = '\n'.join(l for l in body.split('\n') if not re.match(
        r'\\(begin|end)\{CSLReferences\}|\\hypertarget\{refs\}|\\leavevmode\\vadjust', l))
    body = body.replace('\u0131\u0301', "\\'{\\i}").strip()
    if not body or 'WARN' in r.stderr and key in r.stderr:
        print('PROBLEM', key, r.stderr[:200], file=sys.stderr)
    out.append(lab + '\n' + body + '\n\n' + tail)
open(bbl, 'w').write(''.join(out))
