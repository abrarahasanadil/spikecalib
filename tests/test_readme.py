"""Every number the README reports is the number the paper reports.

Each reported value in README.md is followed by an HTML comment naming the macro in
paper/numbers.tex that holds it, ``<!-- n:macroName -->``. This test checks that every such
value equals its macro, and that no decimal number or percentage in the README's prose and
tables escapes the check. Integers that name a setting (2 timesteps, 101 classes, seed 0) are
not results and are not annotated.
"""
from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _macros() -> dict:
    out = {}
    with open(os.path.join(REPO, "paper", "numbers.tex")) as fh:
        for line in fh:
            m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}%", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _plain(value: str) -> str:
    return value.replace("{,}", ",").replace("$-$", "-").replace("$+$", "+").strip()


def _readme() -> str:
    with open(os.path.join(REPO, "README.md")) as fh:
        return fh.read()


def test_every_annotated_number_matches_its_macro():
    macros = _macros()
    text = _readme()
    pairs = re.findall(r"([-+]?[\d,]*\.?\d+)(%?)\**\s*<!-- n:([A-Za-z]+) -->", text)
    assert len(pairs) >= 20
    for number, _pct, name in pairs:
        assert name in macros, f"README names {name}, which numbers.tex does not define"
        assert number == _plain(macros[name]), f"README says {number} for {name}; numbers.tex says {macros[name]}"


def test_no_decimal_or_percentage_escapes_the_check():
    text = _readme()
    body = re.sub(r"```.*?```", "", text, flags=re.S)          # commands, not results
    body = re.sub(r"\(https?://[^)]*\)|https?://\S+", "", body)  # links
    body = re.sub(r"ORCID [\d-]+X?", "", body)                  # identifiers
    for m in re.finditer(r"\d+\.\d+%?|\d+%", body):
        after = body[m.end():m.end() + 40]
        if re.match(r"\.\d", after) or re.match(r"\d", after):
            continue
        token = m.group(0)
        if token in ("1.0.0",) or body[max(0, m.start() - 1)] in "v.":
            continue
        if token.endswith("%") and after.startswith(" interval"):
            continue  # a confidence level is a setting, not a result
        assert re.match(r"\**\s*<!-- n:[A-Za-z]+ -->", after), (
            f"README reports {token} without naming its macro: ...{body[m.start() - 30:m.end() + 30]}..."
        )
