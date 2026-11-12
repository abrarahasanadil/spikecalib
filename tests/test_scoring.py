"""Scoring rules, TOST and Wilson intervals (A3 and A4, added after review)."""
from __future__ import annotations

import numpy as np
import pytest

from spikecalib.calibration import bootstrap_ece_difference, softmax_probs
from spikecalib.scoring import (
    CLIP, multiclass_scores, paired_difference_draws, top_label_scores, tost_equivalent,
    wilson_interval,
)


def test_top_label_scores_match_hand_values_and_count_clips():
    conf = np.array([1.0, 0.5, 0.0, 0.8])
    correct = np.array([1.0, 0.0, 0.0, 1.0])
    s = top_label_scores(conf, correct)
    assert s["brier_top_label"] == pytest.approx((0 + 0.25 + 0 + 0.04) / 4)
    expected = -np.mean([np.log(1 - CLIP), np.log(0.5), np.log(1 - CLIP), np.log(0.8)])
    assert s["log_loss_top_label"] == pytest.approx(expected)
    assert s["n_clipped_top_label"] == 2


def test_multiclass_scores_on_a_known_distribution():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 1])
    s = multiclass_scores(probs, labels)
    assert s["nll_multiclass"] == pytest.approx(-(np.log(0.7) + np.log(0.1)) / 2)
    b0 = 0.3 ** 2 + 0.2 ** 2 + 0.1 ** 2
    b1 = 0.1 ** 2 + 0.9 ** 2 + 0.8 ** 2
    assert s["brier_multiclass"] == pytest.approx((b0 + b1) / 2)
    assert s["n_clipped_multiclass"] == 0


def test_multiclass_scores_clip_a_zero_probability():
    probs = softmax_probs(np.array([[0.0, 2000.0]]))
    s = multiclass_scores(probs, np.array([0]))
    assert s["n_clipped_multiclass"] == 1
    assert s["nll_multiclass"] == pytest.approx(-np.log(CLIP))
    with pytest.raises(ValueError):
        multiclass_scores(np.array([[0.5, 0.4]]), np.array([0]))


@pytest.mark.parametrize("estimator", ["equal_width", "equal_mass"])
def test_draws_reproduce_the_stored_bootstrap_interval_exactly(estimator):
    rng = np.random.default_rng(5)
    n = 3000
    correct = (rng.random(n) < 0.85).astype(float)
    a = np.clip(0.85 + 0.05 * rng.standard_normal(n), 0, 1)
    b = np.clip(0.80 + 0.05 * rng.standard_normal(n), 0, 1)
    ref = bootstrap_ece_difference(a, b, correct, n_bootstrap=200, seed=2, estimator=estimator)
    point, draws = paired_difference_draws(a, b, correct, seed=2, n_bootstrap=200, estimator=estimator)
    assert point == ref["difference"]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert (lo, hi) == ref["ci"]


def test_tost_declares_equivalence_only_inside_the_margin():
    draws = np.linspace(-0.004, 0.004, 1001)
    ok, (lo, hi) = tost_equivalent(draws, 0.005)
    assert ok and lo == pytest.approx(-0.0036) and hi == pytest.approx(0.0036)
    assert not tost_equivalent(draws, 0.003)[0]
    assert not tost_equivalent(draws + 0.01, 0.005)[0]
    with pytest.raises(ValueError):
        tost_equivalent(draws, 0.0)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
