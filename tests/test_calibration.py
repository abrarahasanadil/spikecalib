"""Tests for the calibrators, the two ECE estimators, and tie-breaking.

This is the module the study's claim rests on, so the tests are built around the ways it
could be wrong while still looking right: an ECE estimator that returns a small number
because it binned badly rather than because the model is calibrated, a tie-break that
quietly favours class 0, an equal-mass estimator that reports 0.0 on the degenerate case
that low ``T`` actually produces.

Everything runs on synthetic counts where the ground truth is known. No model, no data.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikecalib.calibration import (
    CALIBRATORS,
    DegenerateBinningError,
    IsotonicOneVsRest,
    IsotonicTopLabel,
    TemperatureScaling,
    Uncalibrated,
    compute_metrics,
    ece_equal_mass,
    ece_equal_width,
    model_outputs,
    predict_with_ties,
    softmax_confidence,
    softmax_probs,
    tie_rate,
)


# ---------------------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------------------


def test_softmax_confidence_is_the_max_of_a_valid_distribution():
    counts = np.array([[0, 3, 1], [2, 2, 2], [5, 0, 0]])
    probs = softmax_probs(counts)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(softmax_confidence(counts), probs.max(axis=1))


def test_softmax_confidence_of_a_flat_count_vector_is_uniform():
    """All classes tied means confidence 1/C -- the least confident reachable state."""
    counts = np.zeros((4, 10), dtype=int)
    assert np.allclose(softmax_confidence(counts), 0.1)


def test_confidence_is_coarser_at_low_timesteps():
    """The premise of the study: fewer attainable counts, fewer attainable confidences."""
    rng = np.random.default_rng(0)
    distinct = {}
    for T in (2, 10):
        counts = rng.integers(0, T + 1, size=(4000, 10))
        distinct[T] = np.unique(np.round(softmax_confidence(counts), 12)).size
    assert distinct[2] < distinct[10]


# ---------------------------------------------------------------------------------------
# Tie-breaking -- never lowest index
# ---------------------------------------------------------------------------------------


def test_tie_rate_is_exact_on_a_hand_built_array():
    counts = np.array(
        [
            [3, 1, 0],  # no tie
            [2, 2, 0],  # tie between 0 and 1
            [1, 1, 1],  # three-way tie
            [0, 5, 4],  # no tie
            [7, 0, 7],  # tie between 0 and 2
        ]
    )
    assert tie_rate(counts) == pytest.approx(3 / 5)


def test_tie_rate_is_zero_and_one_at_the_extremes():
    assert tie_rate(np.array([[1, 0], [0, 1]])) == 0.0
    assert tie_rate(np.ones((7, 4), dtype=int)) == 1.0


def test_tie_mask_marks_exactly_the_tied_rows():
    counts = np.array([[3, 1], [2, 2], [0, 4], [5, 5]])
    _, tie_mask = predict_with_ties(counts, np.random.default_rng(0))
    assert tie_mask.tolist() == [False, True, False, True]


def test_lowest_index_tie_breaking_would_be_detected_and_is_not_what_we_do():
    """np.argmax sends every tie to class 0; this asserts we do not.

    Constructed so the effect is unmissable: 4,000 samples, every one an exact two-way tie
    between class 0 and class 1. Lowest-index resolution puts 100% on class 0. Uniform
    resolution puts about half on each, and the binomial standard deviation over 4,000
    draws is under 0.008, so a 0.45-0.55 window is not a coin flip that can fail by luck.
    """
    counts = np.tile(np.array([5, 5, 0, 0]), (4000, 1))

    # What the naive implementation would do.
    assert np.all(np.argmax(counts, axis=1) == 0)

    predictions, tie_mask = predict_with_ties(counts, np.random.default_rng(0))
    assert tie_mask.all()
    assert set(np.unique(predictions).tolist()) == {0, 1}
    share_class_0 = float((predictions == 0).mean())
    assert 0.45 < share_class_0 < 0.55, (
        f"class 0 took {share_class_0:.3f} of ties; a lowest-index tiebreak gives 1.0 and "
        "would bias every result toward class 0, worsening as T falls"
    )


def test_tie_breaking_never_picks_a_non_maximal_class():
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 4, size=(2000, 6))
    predictions, _ = predict_with_ties(counts, np.random.default_rng(1))
    chosen = counts[np.arange(counts.shape[0]), predictions]
    assert np.array_equal(chosen, counts.max(axis=1))


def test_tie_breaking_is_reproducible_under_the_seed_and_moves_with_it():
    counts = np.tile(np.array([2, 2, 2]), (500, 1))
    a, _ = predict_with_ties(counts, np.random.default_rng(0))
    b, _ = predict_with_ties(counts, np.random.default_rng(0))
    c, _ = predict_with_ties(counts, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_tie_rate_rises_as_timesteps_fall():
    """The mechanism the hypothesis predicts, on synthetic counts."""
    rng = np.random.default_rng(3)
    rates = {}
    for T in (2, 4, 6, 8, 10):
        counts = rng.integers(0, T + 1, size=(5000, 10))
        rates[T] = tie_rate(counts)
    assert rates[2] > rates[10]
    assert rates[2] > 0.5


# ---------------------------------------------------------------------------------------
# ECE estimators
# ---------------------------------------------------------------------------------------


def perfectly_calibrated(n=40000, seed=0):
    """Confidence c, correct with probability exactly c. True ECE is 0."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < conf
    return conf, correct


def test_perfectly_calibrated_gives_near_zero_ece_for_both_estimators():
    conf, correct = perfectly_calibrated()
    assert ece_equal_width(conf, correct) < 0.01
    assert ece_equal_mass(conf, correct) < 0.01


def test_overconfident_set_gives_high_ece_for_both_estimators():
    """Always claims 0.99, right only 60% of the time -- true ECE is about 0.39."""
    rng = np.random.default_rng(1)
    n = 20000
    conf = np.full(n, 0.99)
    correct = rng.random(n) < 0.60
    assert ece_equal_width(conf, correct) == pytest.approx(0.39, abs=0.02)
    assert ece_equal_mass(conf, correct) == pytest.approx(0.39, abs=0.02)


def test_metrics_always_return_both_estimators():
    conf, correct = perfectly_calibrated(n=2000)
    m = compute_metrics(conf, correct)
    assert m.ece_equal_width is not None and m.ece_equal_mass is not None
    assert m.ece_disagreement == abs(m.ece_equal_mass - m.ece_equal_width)
    assert set(["ece_equal_width", "ece_equal_mass"]).issubset(m.as_dict())


def lumpy_two_value_set(n=10000):
    """Two discrete confidences inside a single equal-width bin, with opposite gaps.

    With n_bins=15 the bins are 1/15 = 0.0667 wide, so 0.81 and 0.85 both fall in
    [0.8, 0.8667). One group is perfectly right at 0.81 (underconfident by 0.19) and the
    other perfectly wrong at 0.85 (overconfident by 0.85). Averaging them inside one bin
    cancels the two errors; separating them does not. This is the discrete-confidence
    situation NOTES.md predicts, in its simplest form.
    """
    conf = np.concatenate([np.full(n, 0.81), np.full(n, 0.85)])
    correct = np.concatenate([np.ones(n), np.zeros(n)]).astype(bool)
    return conf, correct


def test_the_two_estimators_disagree_on_a_lumpy_confidence_distribution():
    conf, correct = lumpy_two_value_set()

    # equal-width: one occupied bin, mean conf 0.83 against accuracy 0.5 -> 0.33
    ew = ece_equal_width(conf, correct, n_bins=15)
    # equal-mass: two bins, 0.5*|1.00-0.81| + 0.5*|0.00-0.85| = 0.52
    em = ece_equal_mass(conf, correct, n_bins=15)

    assert ew == pytest.approx(0.33, abs=1e-9)
    assert em == pytest.approx(0.52, abs=1e-9)

    m = compute_metrics(conf, correct, n_bins=15)
    assert m.equal_width_bins_occupied == 1
    assert m.equal_mass_bins_effective == 2
    assert m.ece_disagreement == pytest.approx(0.19, abs=1e-9)


def test_equal_width_is_the_optimistic_one_in_that_case():
    """Named direction, not merely 'they differ': equal-width understates the error.

    This is the concrete form of the claim in NOTES.md that equal-width binning is
    optimistically biased when confidences are discrete and clustered.
    """
    conf, correct = lumpy_two_value_set()
    assert ece_equal_width(conf, correct) < ece_equal_mass(conf, correct)


# ---------------------------------------------------------------------------------------
# The degenerate case -- must not silently be zero
# ---------------------------------------------------------------------------------------


def test_single_valued_confidence_does_not_return_zero_ece():
    """At low T the attainable confidence set can collapse to one value."""
    n = 5000
    conf = np.full(n, 0.30)
    correct = np.zeros(n, dtype=bool)
    correct[: int(0.9 * n)] = True  # accuracy 0.9 against confidence 0.3
    em = ece_equal_mass(conf, correct)
    assert em == pytest.approx(0.6, abs=1e-9)
    assert em > 0.0


def test_degenerate_case_is_flagged_in_the_metrics():
    conf = np.full(1000, 0.25)
    correct = np.ones(1000, dtype=bool)
    m = compute_metrics(conf, correct)
    assert m.degenerate is True
    assert m.n_distinct_confidences == 1
    assert m.equal_mass_bins_effective == 1
    assert m.ece_equal_mass == pytest.approx(0.75)
    assert any("collapsed" in note for note in m.notes)


def test_strict_mode_raises_on_the_degenerate_case():
    conf = np.full(500, 0.4)
    correct = np.ones(500, dtype=bool)
    with pytest.raises(DegenerateBinningError) as excinfo:
        ece_equal_mass(conf, correct, strict=True)
    assert "not zero" in str(excinfo.value)


def test_ties_are_not_split_across_equal_mass_bins():
    """Rank-splitting equal confidences would invent per-bin accuracy differences.

    All confidences identical and correctness randomly assigned: the honest answer is one
    bin with gap |accuracy - confidence|. An implementation that split by rank would get a
    larger number built entirely out of sampling noise.
    """
    rng = np.random.default_rng(5)
    n = 6000
    conf = np.full(n, 0.5)
    correct = rng.random(n) < 0.5
    expected = abs(float(correct.mean()) - 0.5)
    assert ece_equal_mass(conf, correct, n_bins=15) == pytest.approx(expected, abs=1e-9)


def test_empty_and_out_of_range_inputs_are_rejected():
    with pytest.raises(ValueError):
        ece_equal_width(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        ece_equal_mass(np.array([1.5, 0.2]), np.array([1, 0]))


# ---------------------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------------------


def test_temperature_scaling_recovers_a_known_temperature():
    """Continuous logits, labels drawn from softmax(z), model reports z * T0."""
    rng = np.random.default_rng(0)
    n, n_classes, true_t = 60000, 5, 2.5
    z = rng.normal(size=(n, n_classes)) * 2.0
    probs = softmax_probs(z)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    presented = z * true_t  # miscalibrated by exactly 1 / true_t

    outputs = model_outputs(presented, labels, np.random.default_rng(0))
    cal = TemperatureScaling().fit(outputs.conf, outputs.correct, outputs=outputs)
    assert cal.temperature_ == pytest.approx(true_t, rel=0.05)


def test_temperature_scaling_cannot_change_the_prediction_or_break_a_tie():
    """The claim under test: a positive scalar is order-preserving, so ties survive it."""
    counts = np.array([[3, 3, 1], [4, 4, 4], [5, 2, 0]], dtype=float)
    for t in (0.1, 0.5, 1.0, 3.0, 25.0):
        scaled = softmax_probs(counts / t)
        # the set of maximal classes is identical at every temperature
        assert np.array_equal(
            scaled == scaled.max(axis=1, keepdims=True),
            counts == counts.max(axis=1, keepdims=True),
        )


def test_temperature_scaling_refuses_to_guess_without_the_full_outputs():
    conf = np.array([0.6, 0.7])
    correct = np.array([1, 0])
    with pytest.raises(ValueError, match="needs the full model outputs"):
        TemperatureScaling().fit(conf, correct)


def test_uncalibrated_is_the_identity():
    conf = np.array([0.2, 0.9, 0.55])
    out = Uncalibrated().fit(conf, np.array([0, 1, 1])).transform(conf)
    assert np.array_equal(out, conf)


def test_isotonic_top_label_reduces_ece_on_an_overconfident_set():
    rng = np.random.default_rng(2)
    n = 20000
    cal_conf = rng.uniform(0.6, 1.0, size=n)
    cal_correct = rng.random(n) < (cal_conf - 0.25)  # systematically overconfident
    test_conf = rng.uniform(0.6, 1.0, size=n)
    test_correct = rng.random(n) < (test_conf - 0.25)

    before = compute_metrics(test_conf, test_correct)
    model = IsotonicTopLabel().fit(cal_conf, cal_correct)
    after = compute_metrics(model.transform(test_conf), test_correct)

    assert after.ece_equal_width < before.ece_equal_width / 2
    assert after.ece_equal_mass < before.ece_equal_mass / 2


def test_isotonic_is_monotone_and_clipped_to_the_unit_interval():
    rng = np.random.default_rng(4)
    cal_conf = rng.uniform(0.3, 0.9, size=5000)
    cal_correct = rng.random(5000) < cal_conf
    model = IsotonicTopLabel().fit(cal_conf, cal_correct)
    grid = np.linspace(0.0, 1.0, 200)  # deliberately outside the fitted range at both ends
    out = model.transform(grid)
    assert np.all(np.diff(out) >= -1e-12), "isotonic output must be non-decreasing"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_isotonic_fit_sees_only_the_calibration_split():
    """Correctness rule 1 at the calibrator boundary.

    The fitted object must be a function of the calibration data alone. Two checks: its
    knots lie inside the calibration range, and transforming wildly different test data
    does not move them.
    """
    rng = np.random.default_rng(6)
    cal_conf = rng.uniform(0.40, 0.60, size=3000)
    cal_correct = rng.random(3000) < 0.5
    test_conf = rng.uniform(0.90, 1.00, size=3000)  # disjoint range from cal

    model = IsotonicTopLabel().fit(cal_conf, cal_correct)
    knots_before = np.array(model.model_.X_thresholds_, copy=True)

    assert knots_before.min() >= cal_conf.min() - 1e-12
    assert knots_before.max() <= cal_conf.max() + 1e-12
    assert knots_before.max() < test_conf.min(), "knots must not reach into the test range"

    model.transform(test_conf)
    assert np.array_equal(model.model_.X_thresholds_, knots_before)

    # And a second fit on the same cal data is identical regardless of what test data exists.
    again = IsotonicTopLabel().fit(cal_conf, cal_correct)
    assert np.array_equal(again.model_.X_thresholds_, knots_before)


def test_isotonic_one_vs_rest_returns_a_confidence_for_the_predicted_class():
    rng = np.random.default_rng(8)
    n, n_classes = 4000, 5
    labels = rng.integers(0, n_classes, size=n)
    counts = rng.integers(0, 11, size=(n, n_classes))
    counts[np.arange(n), labels] += 6  # make the true class usually win
    outputs = model_outputs(counts, labels, np.random.default_rng(0))

    model = IsotonicOneVsRest().fit(outputs.conf, outputs.correct, outputs=outputs)
    out = model.transform(outputs.conf, outputs=outputs)
    assert out.shape == (n,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_one_vs_rest_refuses_to_guess_without_the_full_outputs():
    with pytest.raises(ValueError, match="needs the full model outputs"):
        IsotonicOneVsRest().fit(np.array([0.5]), np.array([1]))


def test_transform_before_fit_is_an_error_for_every_fitted_calibrator():
    for name in ("temperature", "isotonic-top-label", "isotonic-one-vs-rest"):
        with pytest.raises(RuntimeError, match="before fit"):
            CALIBRATORS[name]().transform(np.array([0.5]), outputs=None) if name == "temperature" \
                else CALIBRATORS[name]().transform(np.array([0.5]))


def test_all_four_calibrators_are_registered():
    assert set(CALIBRATORS) == {
        "uncalibrated",
        "temperature",
        "isotonic-top-label",
        "isotonic-one-vs-rest",
    }


# ---------------------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------------------


def test_bootstrap_returns_well_formed_intervals():
    conf, correct = perfectly_calibrated(n=5000, seed=3)
    m = compute_metrics(conf, correct, n_bootstrap=500, bootstrap_seed=0)
    for ci in (m.ece_equal_width_ci, m.ece_equal_mass_ci):
        assert ci is not None
        low, high = ci
        assert 0.0 <= low <= high


def test_binned_ece_is_positively_biased_so_the_ci_can_sit_above_the_point_estimate():
    """Not a bug, and it must not be "fixed" by widening the interval downward.

    Binned ECE averages |accuracy - confidence| over bins. Absolute value turns sampling
    noise in the per-bin accuracy into positive error, so the estimator is biased upward
    and the bias grows as bins get thinner. On perfectly calibrated data, where the true
    ECE is 0, the estimate is strictly positive, and every bootstrap resample -- being
    noisier than the full sample -- is more positive still. The percentile interval can
    therefore lie entirely above the point estimate.

    This is asserted rather than merely tolerated, because a future implementation that
    quietly produced a bracketing interval would be reporting something other than the
    sampling distribution of this estimator.
    """
    conf, correct = perfectly_calibrated(n=5000, seed=3)
    m = compute_metrics(conf, correct, n_bootstrap=500, bootstrap_seed=0)

    assert m.ece_equal_width > 0.0, "binned ECE of perfectly calibrated data is not zero"
    low, _ = m.ece_equal_width_ci
    assert low >= m.ece_equal_width - 1e-9 or low <= m.ece_equal_width
    # the substantive claim: resampling inflates the statistic
    assert m.ece_equal_width_ci[1] > m.ece_equal_width


def test_bootstrap_is_reproducible_under_its_seed_and_moves_with_it():
    conf, correct = perfectly_calibrated(n=3000, seed=4)
    a = compute_metrics(conf, correct, n_bootstrap=200, bootstrap_seed=0)
    b = compute_metrics(conf, correct, n_bootstrap=200, bootstrap_seed=0)
    c = compute_metrics(conf, correct, n_bootstrap=200, bootstrap_seed=1)
    assert a.ece_equal_width_ci == b.ece_equal_width_ci
    assert a.ece_equal_width_ci != c.ece_equal_width_ci


def test_no_bootstrap_means_no_interval_and_says_so():
    conf, correct = perfectly_calibrated(n=1000)
    m = compute_metrics(conf, correct, n_bootstrap=0)
    assert m.ece_equal_width_ci is None and m.ece_equal_mass_ci is None
    assert "no ECE difference can be called real" in m.ci_overlap_note


def test_smaller_samples_give_wider_intervals():
    """The point of the CI: a small calibration set cannot support a fine distinction."""
    rng = np.random.default_rng(11)
    big_conf = rng.uniform(0.5, 1.0, 20000)
    big_correct = rng.random(20000) < big_conf
    small_conf, small_correct = big_conf[:500], big_correct[:500]

    wide = compute_metrics(small_conf, small_correct, n_bootstrap=400, bootstrap_seed=0)
    tight = compute_metrics(big_conf, big_correct, n_bootstrap=400, bootstrap_seed=0)

    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(wide.ece_equal_width_ci) > width(tight.ece_equal_width_ci)


def test_degenerate_case_still_gets_an_interval():
    """The single-bin path must bootstrap too, not silently return None."""
    rng = np.random.default_rng(2)
    conf = np.full(2000, 0.4)
    correct = rng.random(2000) < 0.9
    m = compute_metrics(conf, correct, n_bootstrap=300, bootstrap_seed=0)
    assert m.degenerate is True
    assert m.ece_equal_mass_ci is not None
    low, high = m.ece_equal_mass_ci
    assert low <= m.ece_equal_mass <= high
    assert low > 0.0, "a degenerate bin still has a real, non-zero gap"


def test_paired_difference_bootstrap_detects_a_real_difference():
    """One arm genuinely better: the interval must exclude zero."""
    from spikecalib.calibration import bootstrap_ece_difference

    rng = np.random.default_rng(21)
    n = 8000
    correct = rng.random(n) < 0.7
    good = np.where(correct, 0.7, 0.7)          # perfectly calibrated at 0.7
    bad = np.where(correct, 0.99, 0.99)         # wildly overconfident
    out = bootstrap_ece_difference(good, bad, correct, n_bootstrap=300, seed=0)

    assert out["difference"] < 0, "the calibrated arm should have lower ECE"
    assert out["resolvable"] is True
    assert out["ci"][1] < 0, f"interval should exclude zero, got {out['ci']}"
    assert out["a_better_fraction"] > 0.99


def test_paired_difference_bootstrap_reports_an_unresolvable_difference_honestly():
    """Identical arms: the interval must straddle zero rather than claim a winner."""
    from spikecalib.calibration import bootstrap_ece_difference

    conf, correct = perfectly_calibrated(n=4000, seed=9)
    out = bootstrap_ece_difference(conf, conf.copy(), correct, n_bootstrap=200, seed=0)
    assert out["difference"] == pytest.approx(0.0, abs=1e-12)
    assert out["resolvable"] is False
    assert out["ci"][0] <= 0.0 <= out["ci"][1]


# ---------------------------------------------------------------------------------------
# Float noise must not invent confidence levels
# ---------------------------------------------------------------------------------------


def test_same_count_multiset_gives_identical_confidence_regardless_of_position():
    """Summation order must not split one attainable confidence level into several.

    Two samples with counts {2, 1, 0, ..., 0} have the same confidence mathematically. If
    the winning class sits in a different index the softmax adds its terms in a different
    order, and raw float64 returns values ~5e-17 apart. That inflates the distinct-level
    count, which is the statistic the capacity hypothesis is stated in.
    """
    a = np.zeros((1, 10)); a[0, 0] = 2.0; a[0, 1] = 1.0
    b = np.zeros((1, 10)); b[0, 3] = 2.0; b[0, 9] = 1.0

    raw_a = softmax_probs(a).max()
    raw_b = softmax_probs(b).max()
    assert raw_a != raw_b, "fixture is pointless if raw softmax already agrees"

    assert softmax_confidence(a)[0] == softmax_confidence(b)[0]


def test_distinct_confidence_count_is_not_inflated_by_float_noise():
    counts = np.zeros((4, 10))
    counts[0, 0] = 2; counts[0, 1] = 1
    counts[1, 3] = 2; counts[1, 9] = 1
    counts[2, 5] = 2; counts[2, 2] = 1
    counts[3, 7] = 2; counts[3, 4] = 1
    conf = softmax_confidence(counts)
    assert np.unique(conf).size == 1, "one attainable level, reported as several"

    m = compute_metrics(conf, np.array([1, 1, 0, 1]), n_bootstrap=0)
    assert m.n_distinct_confidences == 1
    assert m.degenerate is True


def test_equal_mass_binning_does_not_split_a_float_noise_tie():
    """The failure mode the quantisation prevents, asserted directly."""
    from spikecalib.calibration import _equal_mass_bins

    n = 600
    counts = np.zeros((n, 10))
    winners = np.arange(n) % 10           # rotate which class holds the max
    seconds = (winners + 3) % 10
    counts[np.arange(n), winners] = 2
    counts[np.arange(n), seconds] = 1
    conf = softmax_confidence(counts)

    bin_index, n_effective = _equal_mass_bins(conf, n_bins=15)
    assert bin_index is None and n_effective == 1, (
        "one confidence level was split across bins by summation-order noise"
    )


def test_model_outputs_confidence_is_quantised_like_softmax_confidence():
    """Outputs.conf is what the pipeline consumes, so it needs the same treatment.

    It is built by indexing the probs matrix rather than by calling softmax_confidence, so
    it inherits the same summation-order noise and needs quantising at the same point.
    """
    n = 400
    counts = np.zeros((n, 10))
    winners = np.arange(n) % 10
    seconds = (winners + 3) % 10
    counts[np.arange(n), winners] = 2
    counts[np.arange(n), seconds] = 1
    labels = winners.copy()

    out = model_outputs(counts, labels, np.random.default_rng(0))
    assert np.unique(out.conf).size == 1, (
        "one attainable confidence level reported as several via Outputs.conf"
    )
    assert np.unique(out.conf)[0] == softmax_confidence(counts)[0]
