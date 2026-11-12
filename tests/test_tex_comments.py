"""Provenance comments in the paper source must not swallow prose.

LaTeX drops everything after an unescaped ``%`` on a line. A provenance comment
(``% src:``, ``% [DERIVED]``, ``% [LIT]``) that is followed on the same line by more of the
sentence silently removes that text from the PDF. These checks catch the three ways that has
happened: a second provenance marker inside a comment, a LaTeX macro inside a comment, and a
source path followed by an ordinary sentence.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
SOURCES = ["main.tex", "fusion_section.tex", "fusion_appendix.tex"]

FIRST_COMMENT = re.compile(r"(?<!\\)%")
SECOND_MARKER = re.compile(r"(?<!\\)%\s*(src:|\[DERIVED\]|\[LIT\])")
MACRO = re.compile(r"\\[A-Za-z]+")
PATH_THEN_SENTENCE = re.compile(r"\.(json|csv|py|md|npz)\s+[A-Z][a-z]+\s+[a-z]+\s+[a-z]+")


def _inline_comments(name: str):
    for number, line in enumerate((PAPER / name).read_text(encoding="utf-8").splitlines(), 1):
        match = FIRST_COMMENT.search(line)
        if match and line[: match.start()].strip():
            yield number, line[match.end():]


@pytest.mark.parametrize("name", SOURCES)
def test_no_prose_hides_inside_a_comment(name):
    bad = []
    for number, comment in _inline_comments(name):
        if SECOND_MARKER.search(comment) or MACRO.search(comment) or PATH_THEN_SENTENCE.search(comment):
            bad.append(f"{name}:{number}: {comment.strip()[:100]}")
    assert not bad, "comment swallows prose:\n" + "\n".join(bad)
