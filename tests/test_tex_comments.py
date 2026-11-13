"""Provenance comments in the paper source must not swallow prose.

LaTeX drops everything after an unescaped ``%`` on a line. A provenance comment
(``% src:``, ``% [DERIVED]``, ``% [LIT]``) that is followed on the same line by more of the
sentence silently removes that text from the PDF. These checks catch the ways that has
happened: a second provenance marker inside a comment, a LaTeX macro or a dollar sign inside a
comment, and a source path or a locator (Sec. 6.1, Eq. (5), Table 2, ...) followed by an
ordinary sentence.
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
DOLLAR = re.compile(r"(?<!\\)\$")
LOCATOR_THEN_SENTENCE = re.compile(
    r"(Secs?\.|Section|Eqs?\.|Equation|Figs?\.|Figure|Table|Prop\.|Proposition|Appendix|Thm\.|Theorem|pp?\.)"
    r"\s*[\w.()\-]+[,;]?\s+[A-Z][a-z]*\s+[a-z]+\s+[a-z]+"
)

CHECKS = (SECOND_MARKER, MACRO, PATH_THEN_SENTENCE, DOLLAR, LOCATOR_THEN_SENTENCE)


def _inline_comments(name: str):
    for number, line in enumerate((PAPER / name).read_text(encoding="utf-8").splitlines(), 1):
        match = FIRST_COMMENT.search(line)
        if match and line[: match.start()].strip():
            yield number, line[match.end():]


@pytest.mark.parametrize("name", SOURCES)
def test_no_prose_hides_inside_a_comment(name):
    bad = []
    for number, comment in _inline_comments(name):
        if any(check.search(comment) for check in CHECKS):
            bad.append(f"{name}:{number}: {comment.strip()[:100]}")
    assert not bad, "comment swallows prose:\n" + "\n".join(bad)


@pytest.mark.parametrize(
    "comment",
    [
        " [LIT] li2023dynamic, Sec. 6.1 A count over $T$ timesteps is a whole number between $0$ and $T$,",
        " [LIT] wang2017normface, Section 3.2 The bound holds for any scale",
        " src: results/ece_floor.json Every run is listed",
        " [DERIVED] Eq. (5) with $T$",
    ],
)
def test_the_checks_catch_known_failures(comment):
    assert any(check.search(comment) for check in CHECKS)


@pytest.mark.parametrize(
    "comment",
    [
        " [LIT] li2023seenn, Sec. 3.3, Eq. (5) and Table 1",
        " src: results/ece_floor.json runs[*].schemes",
        " [LIT] chen2025knowing, Section I and Figure 1(b)",
    ],
)
def test_the_checks_pass_plain_provenance(comment):
    assert not any(check.search(comment) for check in CHECKS)
