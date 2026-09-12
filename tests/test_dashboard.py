"""Static guards on the dashboard's interface wiring.

These exist because a verification failure, not a theory. The card modals were checked by
a one-off headless script that asserted ``dialog.open`` and reported eight of eight
passing, while every button on the rendered page was dead. Two causes, both invisible to
that check:

1. **Ordering.** ``wireModals()`` was the last statement of a single IIFE, after six calls
   that read ``window.SPIKECALIB``. Any failure above it -- an unavailable ``data.js``, one
   absent column -- threw, the IIFE stopped, and all eight buttons went dead with a single
   console error. The interface must not depend on the data rendering succeeding, so the
   wiring runs first and each renderer is isolated.
2. **Caching.** ``app.js`` was referenced with no version, so a browser holding the copy
   from before the modals existed served dead buttons with no error at all.

Neither is detectable from the rendered page once it loads correctly, which is why they are
asserted against the source instead. These are cheap structural checks, not a substitute
for opening the page.
"""

from __future__ import annotations

import os
import re

import pytest

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


@pytest.fixture(scope="module")
def index() -> str:
    with open(os.path.join(DOCS, "index.html")) as fh:
        return fh.read()


@pytest.fixture(scope="module")
def app() -> str:
    with open(os.path.join(DOCS, "app.js")) as fh:
        return fh.read()


def test_every_trigger_has_a_dialog(index: str) -> None:
    """A button pointing at an id that is not in the document is a dead button."""
    triggers = set(re.findall(r'data-modal="([^"]+)"', index))
    dialogs = set(re.findall(r'<dialog[^>]*\bid="([^"]+)"', index))
    assert triggers, "no modal triggers found -- the selector or the markup changed"
    assert triggers <= dialogs, f"triggers with no dialog: {sorted(triggers - dialogs)}"


def test_every_card_has_a_trigger(index: str) -> None:
    """A card whose 'full reasoning' button lost its target loses its content silently."""
    buttons = re.findall(r'<button class="card-more"[^>]*>', index)
    assert buttons, "no card-more buttons found"
    assert all("data-modal=" in b for b in buttons)


def test_modals_are_wired_before_any_data_is_read(app: str) -> None:
    """The regression that made every button dead: wiring placed after the renderers."""
    wire = app.rindex("wireModals();")
    readers = [app.rindex(name) for name in ("updateHero", "renderArch", "render()")]
    assert wire < min(readers), (
        "wireModals() must run before anything that reads window.SPIKECALIB, or a data "
        "error takes the whole interface down with it"
    )


def test_each_renderer_is_isolated(app: str) -> None:
    """One failing renderer must not abort the others, or the page half-renders."""
    tail = app[app.rindex("wireModals();"):]
    assert "try {" in tail and "catch" in tail, "renderers are not individually guarded"


def test_assets_are_versioned(index: str) -> None:
    """Without a version query, a stale app.js is served and the buttons die silently."""
    for asset in ("app.js", "data.js", "style.css"):
        refs = re.findall(rf'(?:src|href)="{re.escape(asset)}([^"]*)"', index)
        assert refs, f"{asset} is not referenced from index.html"
        assert all(r.startswith("?v=") for r in refs), f"{asset} is referenced unversioned"


def test_footer_date_is_generated(index: str, app: str) -> None:
    """It was hardcoded and two days stale. It now comes from the newest artifact."""
    assert 'id="last-updated"' in index, "the footer date has no element to render into"
    assert "D.last_updated" in app, "nothing renders the footer date from the payload"
    with open(os.path.join(DOCS, "data.js")) as fh:
        head = fh.read(400_000)
    for key in ('"last_updated"', '"last_updated_source"', '"n_source_artifacts"'):
        assert key in head, f"data.js carries no {key}; re-run export_dashboard_data.py"


def test_export_records_its_source_artifacts() -> None:
    """The date is resolved at export time -- a clone rewrites every mtime."""
    path = os.path.join(os.path.dirname(DOCS), "scripts", "export_dashboard_data.py")
    with open(path) as fh:
        src = fh.read()
    assert "def source(" in src and "sources.append" in src
    assert "max(sources, key=os.path.getmtime)" in src


def test_limitations_are_not_labelled_unverified(index: str) -> None:
    """A limitation is a known gap, which is the opposite of an unverified claim."""
    heading = re.search(r"<h3>Limitations.*?</h3>", index, re.S)
    assert heading, "limitations heading not found"
    assert "pill-warn" not in heading.group(0)
