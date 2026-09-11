"""Tests for selective prediction.

The load-bearing property is that **calibration cannot change risk-coverage**. Every
calibrator in this study is a monotone map, and a monotone map cannot reorder samples, so
AURC and the achievable operating points are invariant under it. If that ever stopped
holding, either a calibrator had stopped being monotone or the ranking had been corrupted
somewhere, and both would invalidate the selective numbers silently.

The second property is that ties never split. A spike-count readout produces few distinct
confidences, and reporting an operating point that sits halfway through a tie would be
reporting something nobody can deploy.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikecalib.calibration import IsotonicTopLabel
from spikecalib.selective import (
    achievable_coverage,
    aurc,
    coverage_at_risk,
    n_achievable_coverage_levels,
    normalised_excess_aurc,
    optimal_aurc,
    risk_coverage,
    selective_summary,
    threshold_for_coverage,
)


def graded(n=2000, seed=0):
    """Confidence that genuinely ranks correctness, with continuous values."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf
    return conf, correct


# ---------------------------------------------------------------------------------------
# Ties and achievable operating points
# ---------------------------------------------------------------------------------------


def test_coverage_steps_once_per_distinct_confidence_not_once_per_sample():
    """The result that matters: nine levels give nine operating points, not ten thousand."""
    rng = np.random.default_rng(0)
    n = 10000
    levels = np.array([0.12, 0.20, 0.23, 0.30, 0.32, 0.37, 0.40, 0.45, 0.4509])
    conf = rng.choice(levels, size=n)
    correct = rng.random(n) < conf

    rc = risk_coverage(conf, correct)
    assert rc.n_levels == len(levels)
    assert rc.coverage.size == len(levels), "one achievable point per distinct level"
    assert rc.n_samples == n
    assert n_achievable_coverage_levels(conf) == len(levels)


def test_ties_are_never_split_across_operating_points():
    """Every achievable prefix ends on a confidence boundary."""
    rng = np.random.default_rng(1)
    conf = rng.choice([0.3, 0.6, 0.9], size=900)
    correct = rng.random(900) < 0.8
    rc = risk_coverage(conf, correct)
    for k in rc.n_selected:
        selected = np.sort(conf)[::-1][:k]
        rest = np.sort(conf)[::-1][k:]
        if rest.size:
            assert selected.min() > rest.max(), "a tie was split across the boundary"


def test_full_coverage_is_always_achievable_and_equals_overall_error():
    conf, correct = graded()
    rc = risk_coverage(conf, correct)
    assert rc.coverage[-1] == pytest.approx(1.0)
    assert rc.risk[-1] == pytest.approx(1.0 - correct.mean())


def test_coverage_is_ascending_and_within_the_unit_interval():
    conf, correct = graded()
    cov = achievable_coverage(conf, correct)
    assert np.all(np.diff(cov) > 0)
    assert cov.min() > 0 and cov.max() <= 1.0


# ---------------------------------------------------------------------------------------
# Calibration cannot change any of this
# ---------------------------------------------------------------------------------------


def test_monotone_transform_leaves_aurc_and_coverage_unchanged():
    """A calibrator is a monotone map; it cannot reorder samples."""
    conf, correct = graded(seed=3)
    before = risk_coverage(conf, correct)
    a0 = aurc(conf, correct)

    for f in (lambda c: c ** 3, lambda c: (c + 1) / 2, lambda c: np.sqrt(c),
              lambda c: 0.1 + 0.8 * c):
        after = risk_coverage(f(conf), correct)
        assert np.allclose(after.coverage, before.coverage)
        assert np.allclose(after.risk, before.risk)
        assert aurc(f(conf), correct) == pytest.approx(a0, abs=1e-12)


def test_one_vs_rest_is_not_a_function_of_top_label_confidence_and_does_reorder():
    """The exception to "calibration cannot reorder", asserted rather than assumed.

    One-vs-rest recalibrates every class probability and renormalises, so its output
    depends on the whole vector, not on the top-label confidence. Two samples sharing a
    top-label confidence can therefore come out in either order. This test builds exactly
    that situation: equal top-label confidence, different runner-up mass.
    """
    from spikecalib.calibration import IsotonicOneVsRest, model_outputs

    rng = np.random.default_rng(31)
    n, C = 3000, 6
    counts = rng.integers(0, 5, size=(n, C)).astype(float)
    labels = rng.integers(0, C, size=n)
    out = model_outputs(counts, labels, np.random.default_rng(0))

    fitted = IsotonicOneVsRest().fit(out.conf, out.correct, outputs=out)
    calibrated = fitted.transform(out.conf, outputs=out)

    # Samples with identical top-label confidence receive different calibrated values,
    # which a function of conf alone could never do.
    order = np.argsort(out.conf, kind="stable")
    c_sorted, cal_sorted = out.conf[order], calibrated[order]
    same = np.flatnonzero(np.diff(c_sorted) == 0)
    assert same.size > 0, "fixture needs ties in the raw confidence"
    assert np.any(cal_sorted[same] != cal_sorted[same + 1]), (
        "one-vs-rest gave every tied sample the same output, so it behaved like a "
        "function of conf; the reordering claim would then be wrong"
    )


def test_a_real_fitted_isotonic_calibrator_leaves_aurc_unchanged():
    """Not a synthetic monotone function: the actual calibrator the study uses."""
    rng = np.random.default_rng(5)
    cal_conf = rng.uniform(0.4, 1.0, 4000)
    cal_correct = rng.random(4000) < cal_conf
    test_conf, test_correct = graded(n=4000, seed=6)

    model = IsotonicTopLabel().fit(cal_conf, cal_correct)
    calibrated = model.transform(test_conf)

    before = aurc(test_conf, test_correct)
    after = aurc(calibrated, test_correct)
    # Isotonic collapses distinct values into blocks, which can only merge operating
    # points, never reorder them — so AURC is unchanged or marginally lower.
    assert after <= before + 1e-12
    assert risk_coverage(calibrated, test_correct).n_levels <= \
           risk_coverage(test_conf, test_correct).n_levels


def test_what_calibration_does_change_is_the_threshold_value_not_the_rank():
    """The distinction the module exists to make explicit."""
    rng = np.random.default_rng(7)
    cal_conf = rng.uniform(0.4, 1.0, 3000)
    cal_correct = rng.random(3000) < cal_conf
    test_conf, test_correct = graded(n=3000, seed=8)
    model = IsotonicTopLabel().fit(cal_conf, cal_correct)

    calibrated = model.transform(test_conf)
    raw = threshold_for_coverage(test_conf, test_correct, 0.5)
    cal = threshold_for_coverage(calibrated, test_correct, 0.5)

    # The threshold VALUE moves: that is the point of calibrating.
    assert raw["threshold_confidence"] != pytest.approx(cal["threshold_confidence"], abs=1e-6)

    # The achievable set can only COARSEN, never refine. Isotonic maps distinct raw
    # confidences onto equal calibrated ones, which removes boundaries; it can never
    # create one, because that would require reordering. So every calibrated operating
    # point was already available before calibration.
    raw_cov = set(np.round(achievable_coverage(test_conf, test_correct), 12))
    cal_cov = set(np.round(achievable_coverage(calibrated, test_correct), 12))
    assert cal_cov <= raw_cov, "calibration invented an operating point it cannot create"
    assert len(cal_cov) <= len(raw_cov)


# ---------------------------------------------------------------------------------------
# AURC bounds
# ---------------------------------------------------------------------------------------


def test_a_perfect_ranker_attains_the_optimal_aurc():
    correct = np.array([1] * 800 + [0] * 200, dtype=bool)
    perfect = np.linspace(1.0, 0.0, correct.size)
    assert aurc(perfect, correct) == pytest.approx(optimal_aurc(correct), abs=1e-12)
    assert normalised_excess_aurc(perfect, correct) == pytest.approx(0.0, abs=1e-9)


def test_random_confidence_gives_risk_roughly_independent_of_coverage():
    rng = np.random.default_rng(11)
    n = 20000
    correct = rng.random(n) < 0.8
    conf = rng.random(n)  # carries no information about correctness
    rc = risk_coverage(conf, correct)
    mid = rc.risk[rc.coverage > 0.2]
    assert mid.max() - mid.min() < 0.05, "uninformative confidence should give a flat curve"
    assert normalised_excess_aurc(conf, correct) > 0.3


def test_optimal_aurc_is_a_lower_bound_on_measured_aurc():
    conf, correct = graded(seed=13)
    assert aurc(conf, correct) >= optimal_aurc(correct) - 1e-12


def test_normalised_excess_lies_between_zero_and_one():
    conf, correct = graded(seed=17)
    e = normalised_excess_aurc(conf, correct)
    assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------------------
# Coverage at a risk target
# ---------------------------------------------------------------------------------------


def test_coverage_at_risk_returns_none_when_no_point_meets_the_target():
    """Honest absence, not zero: zero would imply a reachable point with no coverage."""
    rng = np.random.default_rng(19)
    correct = rng.random(500) < 0.5  # never reaches 99.9% accuracy anywhere
    conf = rng.random(500)
    out = coverage_at_risk(conf, correct, targets=(0.001,))
    assert out[0.001] is None


def test_coverage_at_risk_picks_the_largest_qualifying_coverage():
    correct = np.array([1] * 900 + [0] * 100, dtype=bool)
    conf = np.linspace(1.0, 0.0, 1000)
    out = coverage_at_risk(conf, correct, targets=(0.01,))
    hit = out[0.01]
    assert hit is not None
    assert hit["risk"] <= 0.01
    assert hit["coverage"] > 0.85


def test_summary_carries_every_reported_field():
    conf, correct = graded(seed=23)
    s = selective_summary(conf, correct)
    for k in ("aurc", "aurc_optimal", "aurc_excess_normalised",
              "n_achievable_coverage_levels", "achievable_coverage", "achievable_risk"):
        assert k in s
    assert any(k.startswith("coverage_at_risk_") for k in s)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        risk_coverage(np.array([]), np.array([]))
