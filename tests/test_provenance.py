"""Guards for the documentation halves of the correctness rules.

The rules are asserted in code where they can be: rule 1 by ``assert_disjoint``, rule 2 by
the metrics object carrying both estimators, rule 4 by the split reproducibility tests.
**Rule 3 is half documentation** -- "every appearance of an energy number describes it as a
design-comparison proxy", and "``E_MAC`` is ``[DERIVED]``, not cited to Horowitz" -- and an
audit found that half unguarded: mutating ``E_MAC_JOULES`` failed a test, but nothing would
have caught the constant being re-attributed to a figure number it does not appear in.

Rule 2 gets the other guard here. ``results/*.csv`` is gitignored, so ``docs/sweep.csv`` is
the only copy of the sweep a reader ever sees; it drifted eleven columns behind the copy
embedded in ``docs/data.js`` and nothing noticed.
"""

from __future__ import annotations

import csv
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts)) as fh:
        return fh.read()


# --------------------------------------------------------------------------- rule 3

ENERGY_FILES = ["NOTES.md", "SOURCES.md", os.path.join("docs", "index.html"),
                os.path.join("spikecalib", "models.py"), os.path.join("spikecalib", "figures.py")]


def test_e_mac_is_never_attributed_to_horowitz() -> None:
    """4.6 pJ appears nowhere in that paper. It is the 0.9 + 3.7 convention."""
    # Checked per file rather than per window: the disclaimer that licenses a mention can
    # sit several paragraphs from the number, and a window narrow enough to be precise was
    # narrow enough to flag figures.py's own prohibition as a violation.
    for rel in ENERGY_FILES:
        text = _read(rel)
        if "4.6" not in text:
            continue
        if "Horowitz" not in text and "Figure 1.1.9" not in text:
            continue
        assert re.search(r"nowhere|not stated|must not|convention|DERIVED|no MAC row", text), (
            f"{rel} mentions 4.6 pJ alongside Horowitz with no disclaimer anywhere in the "
            f"file that the value is the 0.9 + 3.7 convention rather than his"
        )


def test_e_mac_carries_derived_and_e_ac_carries_lit() -> None:
    src = _read("spikecalib", "models.py")
    mac = src[src.index("E_MAC_JOULES ="):][:200]
    ac = src[src.index("E_AC_JOULES ="):][:200]
    assert "[DERIVED]" in mac and "[LIT]" not in mac, "E_MAC must be DERIVED, not cited"
    assert "[LIT]" in ac, "E_AC is confirmed against Figure 1.1.9 and should say so"


def test_energy_is_described_as_a_proxy_wherever_it_is_reported() -> None:
    """Never 'measured power', always a design-comparison proxy."""
    for rel in ("NOTES.md", os.path.join("docs", "index.html")):
        text = _read(rel)
        if "SynOps" not in text and "synops" not in text:
            continue
        assert "proxy" in text, f"{rel} reports energy without calling it a proxy"
    notes = _read("NOTES.md")
    rule3 = notes[notes.index("3. **Energy is a proxy"):][:2400]
    assert "design-comparison proxy" in rule3
    assert "[DERIVED]" in rule3 and "Figure 1.1.9" in rule3
    assert "UNVERIFIED" not in rule3, (
        "rule 3 still marks the energy values unverified; they were resolved 2026-09-12"
    )


def test_no_withdrawn_reference_is_cited_in_live_files() -> None:
    """Withdrawn on verification; they belong in SOURCES.md as a record, nowhere else."""
    for rel in (os.path.join("spikecalib", "calibration.py"), os.path.join("docs", "index.html"),
                "README.md"):
        text = _read(rel)
        for bad in ("AOT-SNN", "Skatchkovsky"):
            assert bad not in text, f"{rel} still cites the withdrawn reference {bad}"


# --------------------------------------------------------------------------- rule 2

def test_the_tracked_csv_matches_the_one_the_page_renders() -> None:
    """results/*.csv is gitignored, so the tracked copy is all a reader ever gets."""
    tracked = os.path.join(ROOT, "docs", "sweep.csv")
    with open(tracked) as fh:
        tracked_cols = next(csv.reader(fh))
    js = _read("docs", "data.js")
    key = "window.SPIKECALIB_CSV = "
    i = js.index(key)
    embedded = json.loads(js[i + len(key):js.index(";\nwindow.SPIKECALIB = ", i)])
    embedded_cols = embedded.splitlines()[0].split(",")
    assert tracked_cols == embedded_cols, (
        "docs/sweep.csv and the CSV embedded in docs/data.js have different columns; the "
        "download would hand a reader a different table from the one the page shows"
    )


def test_the_paired_bootstrap_reports_both_estimators() -> None:
    """The one place rule 2 was violated: an equal-width difference with no counterpart."""
    with open(os.path.join(ROOT, "docs", "sweep.csv")) as fh:
        cols = set(next(csv.reader(fh)))
    for c in ("iso_minus_temp_ece_ew", "iso_minus_temp_ece_em",
              "iso_minus_temp_em_ci_low", "iso_minus_temp_em_ci_high"):
        assert c in cols, f"sweep.csv is missing {c}"


@pytest.mark.parametrize("claimed", ["72"])
def test_the_dashboard_states_the_real_column_count(claimed: str) -> None:
    with open(os.path.join(ROOT, "docs", "sweep.csv")) as fh:
        n = len(next(csv.reader(fh)))
    page = _read("docs", "index.html")
    assert f"carries {n} columns" in page, (
        f"the page claims a column count that is not {n}; sweep.csv has {n}"
    )
    assert claimed == str(n)


# --------------------------------------------------------------------------- energy

def test_dashboard_energy_ranges_match_the_sweep() -> None:
    """The energy note once quoted seed 0's range, 41-65%, as if it covered every run."""
    with open(os.path.join(ROOT, "docs", "sweep.csv")) as fh:
        rows = list(csv.DictReader(fh))
    runs: dict[tuple[str, str], float] = {}
    for r in rows:
        runs.setdefault((r["timesteps"], r["seed"]), float(r["encoder_energy_fraction"]))
    by_t: dict[str, list[float]] = {}
    for (t, _), v in runs.items():
        by_t.setdefault(t, []).append(v)
    lo, hi = round(100 * min(runs.values())), round(100 * max(runs.values()))
    means = [100 * sum(v) / len(v) for v in by_t.values()]
    page = " ".join(_read("docs", "index.html").split())
    assert f"between {lo}% and {hi}% of the modelled energy across the fifteen sweep runs" in page, (
        f"the per-run range in the energy note should read {lo}-{hi}%"
    )
    assert f"between {round(min(means))}% and {round(max(means))}% when the three seeds" in page, (
        "the per-timestep-mean range in the energy note does not match the sweep"
    )
