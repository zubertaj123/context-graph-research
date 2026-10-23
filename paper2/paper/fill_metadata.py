"""Fill the author/metadata TODOs in main.tex. Usage:
  python paper2/paper/fill_metadata.py --name "First Last" --affiliation "Independent researcher, City, Country" \
      --email you@example.org --orcid 0000-0000-0000-0000 --repo https://github.com/zubertaj123/context-graph-research \
      --doi 10.5281/zenodo.XXXXXXX --licence "MIT (code), CC BY 4.0 (paper and run artefacts)" [--conference "WORKSHOP NAME"]
Prints every remaining TODO afterwards."""
from __future__ import annotations
import argparse, re
from pathlib import Path
TEX = Path(__file__).resolve().parent / "main.tex"
ap = argparse.ArgumentParser()
for k in ("name", "affiliation", "email", "orcid", "repo", "doi", "licence"):
    ap.add_argument(f"--{k}", required=k not in ("orcid", "doi", "conference"))
ap.add_argument("--conference", default=None)
a = ap.parse_args()
s = TEX.read_text()
orcid = a.orcid
rep = [
    ("Author Name (TODO)", a.name), ("Author (TODO)", a.name.split()[-1]), ("author@example.org", a.email),
    ("Independent researcher (TODO), City, Country", a.affiliation), ("Independent researcher (TODO)", a.affiliation.split(",")[0]),
    ("orcid=0000-0000-0000-0000, ", f"orcid={orcid}, " if orcid else ""),
    ("\\todo{repository URL}", f"\\url{{{a.repo}}}"), ("\\todo{Zenodo DOI}", f"DOI \\url{{https://doi.org/{a.doi}}}" if a.doi else "\\todo{Zenodo DOI}"),
    ("\\todo{licence}", a.licence),
]
if a.conference:
    rep.append(("WORKSHOP NAME (TODO)", a.conference))
for old, new in rep:
    s = s.replace(old, new)
TEX.write_text(s)
left = [m.group(0)[:90] for m in re.finditer(r".{0,40}(TODO|\\todo\{).{0,50}", s)]
print(f"filled; remaining TODO marks: {len(left)}")
for l in left:
    print("  ", l)
