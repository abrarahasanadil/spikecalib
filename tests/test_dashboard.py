"""Static guards on the dashboard, ``docs/index.html``.

The page was rebuilt on 2026-10-17 as a chart-first summary of the paper. These tests guard
what a browser check cannot see once the page loads correctly, and what the rebuild
promised:

1. **No typed numbers.** Every measured value is rendered from ``data.js``. A number span in
   the HTML is empty, and the payload carries the field the page renders from.
2. **The interface does not depend on the data.** The theme toggle is wired before any data
   is read, and each renderer runs inside its own ``try``, so one missing field cannot take
   the page down with it. The old modals failed silently for exactly that reason.
3. **Assets are versioned**, so a browser holding an old ``app.js`` cannot serve it.
4. **The bands are the ones the brief named, in its order**, and detail lives in
   ``<details>`` blocks, not modals.
5. **One colour per readout**, with the hex values the brief fixed, in both themes.

Cheap structural checks, not a substitute for opening the page; the browser check in
``LOG.md`` is that.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def _read(name: str) -> str:
    with open(os.path.join(DOCS, name)) as fh:
        return fh.read()


@pytest.fixture(scope="module")
def index() -> str:
    return _read("index.html")


@pytest.fixture(scope="module")
def app() -> str:
    return _read("app.js")


@pytest.fixture(scope="module")
def css() -> str:
    return _read("style.css")


@pytest.fixture(scope="module")
def payload() -> dict:
    js = _read("data.js")
    key = "window.SPIKECALIB = "
    i = js.index(key) + len(key)
    return json.loads(js[i:js.rindex(";")])


def test_number_spans_are_empty_in_the_html(index: str) -> None:
    """A typed fallback is a number the page did not compute. None may be typed."""
    spans = re.findall(r'<span data-n="([^"]+)">([^<]*)</span>', index)
    assert spans, "the page has no data-driven number spans"
    typed = [(k, v) for k, v in spans if v.strip()]
    assert not typed, f"number spans with typed values: {typed}"


def test_every_number_span_has_a_renderer(index: str, app: str) -> None:
    for key in set(re.findall(r'data-n="([^"]+)"', index)):
        assert f'setN("{key}"' in app, f"nothing in app.js renders data-n={key!r}"


def test_the_page_text_carries_no_measured_number(index: str) -> None:
    """Digits left in the visible text are conditions, identifiers and the ORCID only."""
    body = re.sub(r"<script.*?</script>|<pre>.*?</pre>", " ", index, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body).replace("N-Caltech101", "N-Caltech")    # a dataset's name
    allowed = {"0", "1", "2", "4", "5", "6", "10", "15", "95", "0009", "0006", "9273", "6593"}
    found = set(re.findall(r"\d+(?:\.\d+)?", text))
    assert found <= allowed, f"typed numbers in the page text: {sorted(found - allowed)}"
    assert not re.search(r"\d\.\d", text), "a decimal value is typed into the page"


def test_the_payload_carries_what_the_page_renders(payload: dict) -> None:
    s = payload["summary"]
    for k in ("share_explained_t2", "ece_t2", "auroc_t2", "cells_union"):
        assert k in s, f"data.js summary carries no {k}; re-run export_dashboard_data.py"
    for est in ("ew", "em"):     # both estimators wherever a calibration number is claimed
        assert est in s["ece_t2"]["uncalibrated"] and est in s["ece_t2"]["temperature"]
    led = payload["ledger"]
    assert led["paper"]["total"] == 11 and led["early"]["total"] == 14
    assert led["all"]["total"] == 25 and led["all"]["held"] + led["all"]["falsified"] == 25
    assert led["paper"]["held"] + led["paper"]["falsified"] == led["paper"]["total"]
    ids = [r["id"] for r in led["rows"]]
    assert ids == ["1", "2.1", "2.2", "2.L", "3.1", "3.2", "3.3", "4.1", "4.2", "4.3",
                   "5.1", "5.2", "5.3", "5.4",
                   "6a", "6b", "7", "8", "9", "10", "11", "12", "13", "14", "15"]


def test_the_ledger_matches_the_verdict_blocks(payload: dict) -> None:
    """The paper's eleven come from the result files' verdict blocks, unaltered."""
    def res(name):
        with open(os.path.join(ROOT, "results", name)) as fh:
            return json.load(fh)
    blocks = {**{k: v["held"] for k, v in res("scaled_count_experiment.json")["predictions"].items()
                 if isinstance(v, dict)},
              "9": res("resolution_experiment.json")["prediction_9"]["held"],
              "10": res("caltech_calibrators.json")["prediction_10"]["held"],
              "11": res("calibration_size_experiment.json")["prediction_11"]["held"],
              **{k: v["held"] for k, v in res("fusion_experiment.json")["predictions"].items()}}
    for r in payload["ledger"]["rows"]:
        if r["group"] == "paper":
            assert r["held"] == blocks[r["id"]], f"ledger row {r['id']} differs from its verdict block"


def test_the_early_verdicts_are_the_ones_notes_records() -> None:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from dashboard_summary import RECORDED
    finally:
        sys.path.pop(0)
    assert RECORDED == {"1": False, "2.1": True, "2.2": True, "2.L": False,
                        "3.1": True, "3.2": True, "3.3": False,
                        "4.1": False, "4.2": True, "4.3": False,
                        "5.1": True, "5.2": False, "5.3": True, "5.4": True}
    with open(os.path.join(ROOT, "NOTES.md")) as fh:
        notes = fh.read()
    assert "the prediction as stated is wrong" in notes
    assert "Part 1 is falsified, and that is the result" in notes
    assert "The stated falsification condition is not met, so the prediction stands" in notes
    assert "**Prediction 1 held.**" in notes and "**Prediction 2 held, and it is the sharper" in notes
    assert "**It was tested. It does not hold**" in notes
    assert "Part 2 of the prediction held cleanly" in notes and "### Part 3 was wrong" in notes
    assert "### Part 2 held" in notes and "Part 3 also missed" in notes
    assert "prediction FALSIFIED" in notes and notes.count("prediction HELD") >= 3


def test_interface_is_wired_before_any_data_is_read(app: str) -> None:
    theme = app.index("function setTheme")
    data = app.index("window.SPIKECALIB ||")
    assert theme < data, "the theme toggle must be wired before the payload is read"


def test_each_renderer_is_isolated(app: str) -> None:
    assert re.search(r"try \{ fn\(\); \} catch \(err\)", app), "renderers are not each wrapped in try/catch"


def test_assets_are_versioned(index: str) -> None:
    for asset in ("app.js", "data.js", "style.css"):
        refs = re.findall(rf'(?:src|href)="{re.escape(asset)}([^"]*)"', index)
        assert refs, f"{asset} is not referenced from index.html"
        assert all(r.startswith("?v=") for r in refs), f"{asset} is referenced unversioned"


def test_footer_date_is_generated(index: str, app: str, payload: dict) -> None:
    assert 'id="last-updated"' in index
    assert "D.last_updated" in app
    for key in ("last_updated", "last_updated_source", "n_source_artifacts"):
        assert key in payload, f"data.js carries no {key}; re-run export_dashboard_data.py"


def test_export_records_its_source_artifacts() -> None:
    with open(os.path.join(ROOT, "scripts", "export_dashboard_data.py")) as fh:
        src = fh.read()
    assert "def source(" in src and "sources.append" in src
    assert "max(sources, key=os.path.getmtime)" in src


def test_bands_are_in_the_order_the_brief_set(index: str) -> None:
    order = ["hero", "cap", "calibration", "readouts", "scale", "calibrators", "fusion", "ledger", "limits"]
    found = re.findall(r'<section class="band[^"]*" id="([^"]+)"', index)
    assert found == order, f"bands are {found}"
    assert "Preprint in preparation" in index
    assert "<dialog" not in index and "<details" in index, "detail belongs in <details>, not modals"


def test_the_fusion_band_is_labelled_a_pilot_with_its_limits(index: str) -> None:
    band = index[index.index('id="fusion"'):index.index('id="ledger"')]
    assert "pilot" in band
    for limit in ("one dataset", "not a second modality", "no fusion network trained",
                  "no reference-frame transformation", "no closed loop"):
        assert limit in band, f"the fusion band does not state: {limit}"


def test_one_colour_per_readout_in_both_themes(css: str) -> None:
    light = css[css.index(":root {"):css.index(':root[data-theme="dark"]')]
    dark = css[css.index(':root[data-theme="dark"]'):]
    for var, lo, hi in (("--count", "#2a78d6", "#3987e5"), ("--accum", "#eb6834", "#d95926"),
                        ("--membrane", "#1baf7a", "#199e70"), ("--wcount", "#eda100", "#c98500")):
        assert re.search(rf"{var}:\s*{lo};", light), f"{var} is not {lo} in the light theme"
        assert re.search(rf"{var}:\s*{hi};", dark.split("}")[0]), f"{var} is not {hi} in the dark theme"
    assert 'data-theme="light"' in _read("index.html"), "light is the default theme"


def test_limitations_are_not_labelled_unverified(index: str) -> None:
    band = index[index.index('id="limits"'):]
    heading = re.search(r"<h2>Limitations.*?</h2>", band, re.S)
    assert heading, "limitations heading not found"
    assert "pill-warn" not in heading.group(0)
