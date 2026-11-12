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
    """Withdrawn on verification; they belong in SOURCES.md as a record, nowhere else.

    The second was the two-author "Skatchkovsky & Simeone" attribution. The paper it meant,
    Skatchkovsky, Jang and Simeone (2022), was found and is cited under its real record from
    2026-10-17, so only the withdrawn form is barred.
    """
    for rel in (os.path.join("spikecalib", "calibration.py"), os.path.join("docs", "index.html"),
                "README.md", os.path.join("paper", "main.tex")):
        text = _read(rel)
        for bad in ("AOT-SNN", "Skatchkovsky & Simeone", "Skatchkovsky and Simeone"):
            assert bad not in text, f"{rel} still cites the withdrawn reference {bad}"



def test_the_paper_carries_the_disclosure_notes_prescribes() -> None:
    """The Acknowledgements are exactly the latest superseding text recorded in NOTES.md."""
    notes = _read("NOTES.md")
    section = notes[notes.index("## AI disclosure, for the paper"):notes.index("## Status")]
    block = section[section.rindex("**Superseded "):]
    quoted = " ".join(line[2:] for line in block.splitlines() if line.startswith("> "))
    assert quoted.startswith("AI assistance was used in preparing this work.")
    paper = _read("paper", "main.tex")
    section = paper[paper.index("\\section*{Acknowledgements}"):paper.index("\\bibliographystyle")]
    body = " ".join(section.split("\n", 1)[1].split())
    assert body == quoted
    assert "withdrawn" not in body


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
    """The page once typed the column count. Since 2026-10-17 it counts the embedded CSV."""
    with open(os.path.join(ROOT, "docs", "sweep.csv")) as fh:
        n = len(next(csv.reader(fh)))
    page, app = _read("docs", "index.html"), _read("docs", "app.js")
    assert '<span data-n="csv-columns"></span>' in page, "the column count must be rendered, not typed"
    assert 'setN("csv-columns", String(lines[0].split(",").length))' in app
    assert claimed == str(n)


# --------------------------------------------------------------------------- cell counts

def test_count_cells_are_counted_once() -> None:
    """The matched grid reuses twelve models. Adding 15 + 9 + 24 counted them twice, as 48."""
    def rows(name: str) -> list[dict]:
        return [r for r in json.loads(_read("results", name))["rows"] if r["readout"] == "count"]

    def blocks(stem: str) -> int:
        return len(json.loads(_read("results", f"{stem}_record.json"))["model_config"]["channels"])

    cells = set()
    for r in rows("readout_experiment.json"):
        cells.add(("nmnist", blocks(f"nmnist_T{r['timesteps']}_seed{r['seed']}"),
                   r["timesteps"], r["seed"]))
    for r in rows("caltech_experiment.json"):
        cells.add(("ncaltech101", blocks(f"ncaltech101_T{r['timesteps']}_seed{r['seed']}"),
                   r["timesteps"], r["seed"]))
    grid = rows("arch_matched.json")
    for r in grid:
        cells.add((r["dataset"], r["n_blocks"], r["timesteps"], r["seed"]))
    n = len(cells)
    assert n < len(rows("readout_experiment.json")) + len(rows("caltech_experiment.json")) + len(grid)

    assert f"\\newcommand{{\\cellsUnion}}{{{n}}}" in _read("paper", "numbers.tex"), (
        f"numbers.tex does not state {n} distinct count-readout cells; re-run make_numbers.py"
    )
    assert f"**{n} of {n}** distinct" in _read("README.md")
    page = _read("docs", "index.html")
    assert '<span data-n="cells-all"></span>' in page, "the cell count must be rendered, not typed"
    js = _read("docs", "data.js")
    payload = json.loads(js[js.index("window.SPIKECALIB = ") + len("window.SPIKECALIB = "):js.rindex(";")])
    assert payload["summary"]["cells_union"] == n, (
        f"data.js counts {payload['summary']['cells_union']} cells, not {n}; re-run export_dashboard_data.py"
    )


# --------------------------------------------------------------------------- registered predictions

def _result(name: str) -> dict:
    return json.loads(_read("results", name))


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def test_learned_scale_verdicts_follow_from_the_rows() -> None:
    """Predictions 6a, 6b, 7 and 8, re-evaluated against the conditions in NOTES.md."""
    d = _result("scaled_count_experiment.json")
    cell = lambda ds, T: [r for r in d["rows"] if r["dataset"] == ds and r["timesteps"] == T]  # noqa: E731
    nm, c2, c6 = cell("nmnist", 2), cell("ncaltech101", 2), cell("ncaltech101", 6)
    assert [len(x) for x in (nm, c2, c6)] == [3, 3, 3]
    v = d["predictions"]
    assert v["6a"]["held"] == (abs(_mean(nm, "mean_conf_minus_acc")) < 0.05)
    assert v["6b"]["held"] == (_mean(c2, "mean_conf_minus_acc") > 0)
    assert v["7"]["held"] == (_mean(c2, "test_accuracy") >= 0.5184)
    assert v["8"]["held"] == (_mean(c6, "mean_conf_minus_acc") > -0.05)
    for r in d["rows"]:
        # a scaled-count readout is still capped, at the scale it learned, and still on the lattice
        assert r["max_confidence_observed"] <= r["ceiling_at_learned_scale"] + 1e-9
        assert r["n_distinct"] <= r["max_distinct_bound"]
        assert r["scale_exceeds_s_min"] == (r["logit_scale"] > r["s_min"])


def test_learned_scale_runs_used_the_count_runs_splits() -> None:
    """The arm is a like-for-like comparison only on the split the count run was trained on."""
    for r in _result("scaled_count_experiment.json")["rows"]:
        stem = f"{r['dataset']}_T{r['timesteps']}_seed{r['seed']}"
        count = json.loads(_read("results", stem + "_record.json"))
        scaled = json.loads(_read("results", stem + "_scaled-count_record.json"))
        assert r["split_fingerprint"] == count["split_fingerprint"] == scaled["split_fingerprint"]
        assert scaled["batch_size"] == count["batch_size"] and scaled["lr"] == count["lr"]
        assert scaled["model_config"]["channels"] == count["model_config"]["channels"]


def test_resolution_verdict_and_bound_check_follow_from_the_rows() -> None:
    d = _result("resolution_experiment.json")
    assert d["check"]["violations"] == sum(not c["within_bound"] for c in d["check"]["rows"]) == 0
    for c in d["check"]["rows"]:
        assert c["n_distinct_confidences"] <= c["bound"]
        assert c["n_distinct_gap_multisets"] <= c["bound"]
        # one confidence per gap multiset, so there cannot be more confidences than multisets
        assert c["n_distinct_confidences"] <= c["n_distinct_gap_multisets"]
    p9 = d["prediction_9"]
    assert len(p9["comparisons"]) == 12
    for c in p9["comparisons"]:
        assert c["count_below_accumulator"] == (c["count_resolution"] < c["accumulator_resolution"])
    assert p9["held"] == all(c["count_below_accumulator"] for c in p9["comparisons"])
    for r in d["rows"]:
        gres = r["resolution"] - r["within_bin_variance"] + r["within_bin_covariance"]
        assert abs(r["brier"] - (r["reliability"] - gres + r["uncertainty"])) < 1e-9
        assert r["split_fingerprint"]


def test_calibrator_verdicts_apply_the_tie_rule_as_registered() -> None:
    """Differ only if both intervals exclude zero on the same side; otherwise a tie."""
    def side(ci: list[float]) -> int:
        return -1 if ci[1] < 0 else (1 if ci[0] > 0 else 0)

    d = _result("caltech_calibrators.json")
    for p in d["paired"]:
        ew, em = side(p["ew_ci"]), side(p["em_ci"])
        assert p["differ"] == (ew != 0 and ew == em) and p["tie"] == (not p["differ"])
        assert p["favours"] == ((("isotonic" if ew < 0 else "temperature")) if p["differ"] else None)
    count = [p for p in d["paired"] if p["readout"] == "count"]
    assert len(count) == 9
    assert d["prediction_10"]["ties"] == sum(p["tie"] for p in count)
    assert d["prediction_10"]["held"] == (d["prediction_10"]["ties"] >= 7)

    s = _result("calibration_size_experiment.json")
    assert len(s["rows"]) == 3 * 3 * 20
    for r in s["rows"]:
        ew, em = side(r["ew_ci"]), side(r["em_ci"])
        assert r["differ"] == (ew != 0 and ew == em)
    at = lambda n: [x for x in s["summary"] if x["n_cal"] == n]  # noqa: E731
    for x in s["summary"]:
        cell = [r for r in s["rows"] if r["seed"] == x["seed"] and r["n_cal"] == x["n_cal"]]
        assert x["temperature_wins"] == sum(r["favours"] == "temperature" for r in cell)
        assert x["differ"] == sum(r["differ"] for r in cell)
    v = s["prediction_11"]
    assert v["part_100_held"] == (sum(x["temperature_wins"] >= 10 for x in at(100)) >= 2)
    assert v["tie_at_1000"] == all(x["differ"] < 10 for x in at(1000))
    assert v["tie_at_5000"] == all(x["differ"] < 10 for x in at(5000))
    assert v["held"] == (v["part_100_held"] and v["tie_at_1000"] and v["tie_at_5000"])


def test_notes_and_readme_report_each_verdict_as_the_artifact_has_it() -> None:
    """A falsified prediction must read as falsified in the documents, not as something softer."""
    verdicts = {**{k: v["held"] for k, v in _result("scaled_count_experiment.json")["predictions"].items()
                   if isinstance(v, dict)},
                "9": _result("resolution_experiment.json")["prediction_9"]["held"],
                "10": _result("caltech_calibrators.json")["prediction_10"]["held"],
                "11": _result("calibration_size_experiment.json")["prediction_11"]["held"]}
    assert set(verdicts) == {"6a", "6b", "7", "8", "9", "10", "11"}
    notes = _read("NOTES.md")
    table = notes[notes.index("## Outcome of predictions 6a to 11"):][:1800]
    for key, held in verdicts.items():
        word = "held" if held else "falsified"
        assert re.search(rf"^\| {key} \| \*\*{word}\*\*", table, re.M), (
            f"NOTES.md does not record prediction {key} as {word}"
        )
    n_held = sum(verdicts.values())
    words = {3: "Three", 4: "Four"}
    assert f"{words[n_held]} held and {words[7 - n_held].lower()} were falsified" in table
    assert f"{words[n_held]} held and {words[7 - n_held].lower()} were" in " ".join(_read("README.md").split())


def test_fusion_verdicts_follow_from_the_rows() -> None:
    """Predictions 12 to 15, re-evaluated against the conditions in NOTES.md."""
    d = _result("fusion_experiment.json")
    rows = d["rows"]
    cell = lambda readout, T: sorted((r for r in rows if r["readout"] == readout and r["timesteps"] == T),  # noqa: E731
                                     key=lambda r: r["seed"])
    excludes_zero = lambda ci: ci[0] > 0 or ci[1] < 0  # noqa: E731
    assert len(rows) == 5 * 3 + 3 * 3
    assert all(r["pairings"] == 20 and r["dropout_rate"] == d["dropout"]["rate"] for r in rows)
    # the rate is the one fixed on the calibration split before the run, not one chosen after it
    assert d["dropout"]["rate"] == _result("fusion_dropout_rate.json")["rate"]

    v, t2 = d["predictions"], cell("count", 2)
    assert v["12"]["held"] == all(r["delta_ci"][0] > 0 for r in t2)
    for r in t2:
        lo, hi = r["delta_on_s_ci"]
        assert r["predicted_within_on_s_interval"] == (lo <= r["predicted_on_s"] <= hi)
        assert r["drafted_within_on_s_interval"] == (lo <= r["predicted_on_s_as_first_drafted"] <= hi)
    assert v["13"]["held"] == all(r["predicted_within_on_s_interval"] for r in t2)

    mean = {T: _mean(cell("count", T), "delta") for T in (2, 4, 6, 8, 10)}
    ordering = mean[2] > mean[4] > mean[6] and mean[8] < mean[6] and mean[10] < mean[6]
    high = cell("count", 8) + cell("count", 10)
    assert v["14"]["ordering_held"] == ordering
    assert v["14"]["intervals_excluding_zero_at_8_and_10"] == sum(excludes_zero(r["delta_ci"]) for r in high)
    assert v["14"]["held"] == (ordering and not any(excludes_zero(r["delta_ci"]) for r in high))

    accum = [r for T in (2, 6, 10) for r in cell("accumulator", T)]
    assert len(accum) == 9
    assert v["15"]["intervals_excluding_zero"] == sum(excludes_zero(r["delta_ci"]) for r in accum)
    assert v["15"]["held"] == (not any(excludes_zero(r["delta_ci"]) for r in accum))


def test_fusion_rows_are_consistent_with_the_proposition() -> None:
    """What part (b) says must hold of the numbers, whatever the verdicts are."""
    from spikecalib.fusion import excess_error_as_first_drafted, excess_error_on_saturation

    for r in _result("fusion_experiment.json")["rows"]:
        assert abs(r["delta"] - (r["delta_on_s"] + r["delta_off_s"])) < 1e-12
        assert abs(r["delta"] - (r["acc_r1"] - r["acc_r0"])) < 1e-12
        if r["readout"] == "accumulator":
            # an accumulator has no counts, so nothing saturates and nothing of delta is on S
            assert r["p_s"] == 0 and r["delta_on_s"] == 0 and r["predicted_on_s"] == 0
            continue
        assert 0 <= r["p_s_and_disagree"] <= min(r["p_s"], r["p_disagree"])
        if r["p_s"] == 0:
            continue
        a, b = r["accuracy_a_on_s"], r["accuracy_b_on_s"]
        assert abs(r["predicted_on_s"] - excess_error_on_saturation(r["p_s"], a, b)) < 1e-12
        drafted = excess_error_as_first_drafted(r["p_s_and_disagree"], a, b)
        assert abs(r["predicted_on_s_as_first_drafted"] - drafted) < 1e-12
        # the draft is the corrected value times P(disagree | S), so never the larger in size
        assert abs(drafted) <= abs(r["predicted_on_s"]) + 1e-12
        # two voters: the heavier vote decides every disagreement, so R2 is modality A alone
        if r["weight_a"] > r["weight_b"]:
            assert abs(r["acc_r2"] - r["accuracy_a"]) < 1e-12


def test_notes_record_each_fusion_verdict_as_the_artifact_has_it() -> None:
    verdicts = {k: v["held"] for k, v in _result("fusion_experiment.json")["predictions"].items()}
    assert set(verdicts) == {"12", "13", "14", "15"}
    notes = _read("NOTES.md")
    table = notes[notes.index("## Outcome of predictions 12 to 15"):][:1800]
    for key, held in verdicts.items():
        word = "held" if held else "falsified"
        assert re.search(rf"^\| {key} \| \*\*{word}\*\*", table, re.M), (
            f"NOTES.md does not record prediction {key} as {word}"
        )


# --------------------------------------------------------------------------- energy

def test_dashboard_energy_ranges_match_the_sweep() -> None:
    """The energy note once quoted seed 0's range, 41-65%, as if it covered every run.

    The page rebuilt on 2026-10-17 carries no energy note, because energy is not among the
    paper's results. If one returns, it must be rendered from ``encoder_energy_fraction``
    in the embedded CSV and this test rewritten to check that, not typed.
    """
    page = " ".join(_read("docs", "index.html").split())
    assert "modelled energy" not in page and "SynOps" not in page, (
        "the page reports energy again; render the ranges from data.js and restore a range check"
    )