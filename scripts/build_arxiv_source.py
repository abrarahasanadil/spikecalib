#!/usr/bin/env python3
"""Package the paper's LaTeX source for arXiv, and prove the package compiles on its own.

The package holds main.tex, numbers.tex, fusion_section.tex, fusion_appendix.tex, refs.bib,
main.bbl and only the figure files that the source includes. arXiv compiles with pdflatex and
publishes the source, comments included, so the script also lists every comment line for a
final read.

    python scripts/build_arxiv_source.py --out <path to spikecalib_arxiv_source.tar.gz> --check

``--check`` extracts the package into an empty temporary folder, runs pdflatex twice there
(arXiv uses the supplied main.bbl and does not run BibTeX), reports undefined references and
overfull boxes, and deletes the folder.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
SOURCES = ("main.tex", "numbers.tex", "fusion_section.tex", "fusion_appendix.tex", "refs.bib", "main.bbl")


def figures_used() -> list[str]:
    used = []
    for name in ("main.tex", "fusion_section.tex", "fusion_appendix.tex"):
        text = open(os.path.join(PAPER, name)).read()
        text = re.sub(r"(?<!\\)%.*", "", text)
        used += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    for path in used:
        if not os.path.exists(os.path.join(PAPER, path)):
            raise FileNotFoundError(path)
    return sorted(set(used))


def build(out: str) -> list[str]:
    members = list(SOURCES) + figures_used()
    for m in members:
        if not os.path.exists(os.path.join(PAPER, m)):
            raise FileNotFoundError(f"{m}: build the paper first so that main.bbl exists")
    with tarfile.open(out, "w:gz") as tar:
        for m in members:
            info = tar.gettarinfo(os.path.join(PAPER, m), arcname=m)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(os.path.join(PAPER, m), "rb") as fh:
                tar.addfile(info, fh)
    return members


def check(out: str) -> int:
    tmp = tempfile.mkdtemp(prefix="arxiv_check_")
    try:
        with tarfile.open(out) as tar:
            tar.extractall(tmp, filter="data")
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = open(os.path.join(tmp, "main.log"), errors="replace").read()
        errors = re.findall(r"^!.*", log, re.M)
        undefined = re.findall(r".*undefined.*", log)
        overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
        big = [o for o in overfull if float(o) > 1.0]
        pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
        print(f"  compiled in an empty folder: {pages.group(1) if pages else 'NO'} pages; "
              f"{len(errors)} errors, {len(undefined)} undefined, {len(big)} overfull boxes above 1pt")
        return 0 if pages and not errors and not undefined and not big else 1
    finally:
        shutil.rmtree(tmp)


def comments() -> list[str]:
    out = []
    for name in ("main.tex", "fusion_section.tex", "fusion_appendix.tex"):
        for i, line in enumerate(open(os.path.join(PAPER, name)), 1):
            m = re.search(r"(?<!\\)%(.*)", line)
            if m and m.group(1).strip():
                out.append(f"{name}:{i}: {m.group(1).strip()}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--comments", action="store_true", help="print every source comment for review")
    args = ap.parse_args()
    members = build(args.out)
    print(f"  wrote {args.out}: {len(members)} files")
    for m in members:
        print(f"    {m}")
    if args.comments:
        for c in comments():
            print(c)
    return check(args.out) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
