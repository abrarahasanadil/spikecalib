"""The stopping rules of the A7 retraining (added after review, not pre-registered)."""
from __future__ import annotations

import glob
import json
import os

import pytest

from spikecalib.convergence import original_rule_stop_epoch, theta_converged

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_theta_converged_needs_a_full_window():
    # Constant from the start: converged at the first epoch with five epochs behind it.
    assert theta_converged([0.5] * 10, window=5, tol=0.01) == 5
    assert theta_converged([0.5] * 5, window=5, tol=0.01) is None


def test_theta_converged_uses_the_span_of_the_window():
    # Ends equal but a bump inside the window: not settled until the bump leaves it.
    thetas = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert theta_converged(thetas, window=5, tol=0.01) == 9


def test_theta_converged_threshold_is_strict():
    rising = [0.125 * e for e in range(20)]  # exactly representable steps
    # Five steps of 0.125 span exactly 0.625, which is not below a tolerance of 0.625.
    assert theta_converged(rising, window=5, tol=0.625) is None
    assert theta_converged(rising, window=5, tol=0.6251) == 5


def test_theta_converged_rejects_bad_settings():
    with pytest.raises(ValueError):
        theta_converged([0.0], window=0)
    with pytest.raises(ValueError):
        theta_converged([0.0], tol=0.0)


def test_original_rule_patience_and_cap():
    acc = [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    assert original_rule_stop_epoch(acc, patience=5, cap=50) == (7, 2)
    assert original_rule_stop_epoch([0.1 * e for e in range(10)], patience=5, cap=4) == (3, 3)
    with pytest.raises(ValueError):
        original_rule_stop_epoch([0.1, 0.2], patience=5, cap=50)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REPO, "results", "*_scaled-count_record.json"))))
def test_original_rule_replays_every_recorded_scaled_run(path):
    with open(path) as fh:
        rec = json.load(fh)
    acc = [h["val_accuracy"] for h in rec["history"]]
    stop, best = original_rule_stop_epoch(acc, patience=5, cap=rec["epochs_requested"])
    assert stop == rec["epochs_run"] - 1
    assert best == rec["best_epoch"]
