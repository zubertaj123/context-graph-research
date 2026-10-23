#!/bin/sh
# Build the arXiv source package for the paper. Run on a machine with TeX (arXiv compiles the LaTeX itself
# but does NOT run BibTeX, so main.bbl must be included). Usage: sh make_arxiv.sh  -> arxiv-src.tar.gz
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null
[ -s main.bbl ] || { echo "main.bbl missing: BibTeX did not run"; exit 1; }
grep -q "LaTeX Warning: Citation" main.log && { echo "undefined citations remain, see main.log"; exit 1; } || true
grep -q "LaTeX Warning: Reference" main.log && { echo "undefined references remain, see main.log"; exit 1; } || true
grep -q "TODO" main.pdf 2>/dev/null && echo "WARNING: the PDF still contains TODO marks" || true
rm -rf arxiv-src && mkdir -p arxiv-src/fig
# figures are referenced as ../out/fig/...; copy them next to the source and rewrite the path for arXiv
cp ../out/fig/budget_curve_hit.pdf arxiv-src/fig/
sed 's#\.\./out/fig/#fig/#g' main.tex > arxiv-src/main.tex
cp numbers.tex numbers_online.tex numbers_lme.tex refs.bib main.bbl llncs.cls splncs04.bst aliascnt.sty arxiv-src/
# macOS tar adds ._* AppleDouble files and xattr headers; arXiv flags them as unexpected files
COPYFILE_DISABLE=1 tar --no-xattrs --exclude='._*' -czf arxiv-src.tar.gz -C arxiv-src .
tar -tzf arxiv-src.tar.gz | grep -q '/\._' && { echo "ABORT: AppleDouble files in the archive"; exit 1; } || true
echo "wrote arxiv-src.tar.gz ($(du -h arxiv-src.tar.gz | cut -f1)); upload this archive at arxiv.org/submit, category cs.AI (cross-list cs.CL, cs.DB), licence CC BY 4.0 or arXiv non-exclusive"
