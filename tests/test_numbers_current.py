"""The committed paper/numbers.tex is exactly what paper/make_numbers.py writes.

make_numbers.py reads the split files of the results archive, so the test runs only where the
release asset has been unpacked into results/ (see README) and is skipped otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results" / "nmnist_T2_seed0_split.npz").exists(),
                    reason="needs the results archive unpacked into results/")
def test_numbers_tex_is_regenerated(tmp_path):
    out = tmp_path / "numbers.tex"
    env = dict(os.environ, SPIKECALIB_NUMBERS_OUT=str(out))
    subprocess.run([sys.executable, str(ROOT / "paper" / "make_numbers.py")], check=True, env=env,
                   cwd=ROOT, capture_output=True)
    assert out.read_text(encoding="utf-8") == (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8"), \
        "paper/numbers.tex differs from what paper/make_numbers.py writes; rerun it"
