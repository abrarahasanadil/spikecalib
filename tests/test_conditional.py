"""Tests for the conditional ceiling.

The load-bearing property is that the conditional ceiling is a **refinement** of the
theoretical one, not a separate quantity: setting every competitor to zero must recover
``e^T/(e^T + C - 1)`` exactly. If that stopped holding, the two numbers reported side by
side would not be comparable and the whole comparison would be meaningless.

The second property is monotonicity. Louder competitors can only lower the attainable
confidence, and more classes can only lower it further at fixed activity. Those are the
directions the C-axis argument depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikecalib.calibration import softmax_confidence
from spikecalib.conditional import (
    competitor_profile,
    conditional_ceiling,
    residual_share_explained,
    theoretical_ceiling,
)


def vec(winner_count, competitors, n_classes):
    """A count vector: the winner, then the given competitor counts, then zeros."""
    v = np.zeros(n_classes)
    v[0] = winner_count
    v[1:1 + len(competitors)] = competitors
    return v[None, :]


# ---------------------------------------------------------------------------------------
# The conditional ceiling contains the theoretical one
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C", [(2, 10), (6, 10), (2, 101), (6, 101), (10, 101)])
def test_all_competitors_silent_recovers_the_theoretical_ceiling_exactly(T, C):
    """The identity the whole comparison rests on."""
    got = conditional_ceiling(vec(T, [], C), T)[0]
    assert got == pytest.approx(theoretical_ceiling(T, C), rel=0, abs=1e-12)


def test_theoretical_ceiling_matches_the_closed_form():
    for T, C in ((2, 10), (6, 101)):
        assert theoretical_ceiling(T, C) == pytest.approx(
            np.exp(T) / (np.exp(T) + C - 1), abs=1e-12)


def test_conditional_ceiling_never_exceeds_the_theoretical_one():
    """Competitors can only add to the denominator, so the bound can only tighten."""
    rng = np.random.default_rng(0)
    for T, C in ((2, 10), (6, 101)):
        counts = rng.integers(0, T + 1, size=(200, C)).astype(float)
        counts[:, 0] = T  # make class 0 the winner
        assert (conditional_ceiling(counts, T) <= theoretical_ceiling(T, C) + 1e-12).all()


# ---------------------------------------------------------------------------------------
# Monotonicity: the directions the C-axis argument depends on
# ---------------------------------------------------------------------------------------


def test_louder_competitors_lower_the_attainable_confidence():
    T, C = 6, 101
    ceilings = [conditional_ceiling(vec(T, [c] * 100, C), T)[0] for c in (0, 1, 2, 3)]
    assert ceilings == sorted(ceilings, reverse=True)
    assert ceilings[0] > ceilings[-1]


def test_more_classes_cost_more_at_the_same_competitor_activity():
    """Why the simple bound sufficed at C = 10 and does not at C = 101.

    The denominator's sensitivity scales with the number of competitors, so identical
    per-competitor activity costs an order of magnitude more confidence at 101 classes.
    """
    T = 6
    loss = {}
    for C in (10, 101):
        silent = conditional_ceiling(vec(T, [0] * (C - 1), C), T)[0]
        active = conditional_ceiling(vec(T, [1] * (C - 1), C), T)[0]
        loss[C] = silent - active
    # The DENOMINATOR's sensitivity scales with C - 1, so 11x more at 101 classes. The
    # confidence cost does not scale with it, because the ceiling is a saturating function
    # of the denominator -- measured, the loss is about 5.8x larger, not 11x. Asserting the
    # denominator ratio on the confidence was the first version of this test and it failed.
    assert loss[101] > 5 * loss[10], "the gap must widen sharply with C"
    assert loss[10] < 0.05, "at ten classes the simple bound is nearly tight"
    assert loss[101] > 0.15, "at 101 classes the same activity is expensive"


def test_the_winner_is_replaced_by_T_not_kept_at_its_observed_count():
    """The question is what the sample COULD have reached, not what it did."""
    T, C = 6, 101
    low = vec(2, [0] * 100, C)   # winner fired only twice
    assert conditional_ceiling(low, T)[0] == pytest.approx(theoretical_ceiling(T, C), abs=1e-12)
    # and it is strictly above what the sample actually expressed
    assert conditional_ceiling(low, T)[0] > softmax_confidence(low)[0]


def test_conditional_ceiling_bounds_the_observed_confidence_of_the_same_sample():
    """The bound must actually bound, sample by sample."""
    rng = np.random.default_rng(1)
    T, C = 6, 101
    counts = rng.integers(0, T + 1, size=(500, C)).astype(float)
    assert (conditional_ceiling(counts, T) >= softmax_confidence(counts) - 1e-12).all()


# ---------------------------------------------------------------------------------------
# The competitor profile
# ---------------------------------------------------------------------------------------


def test_competitor_profile_excludes_the_winner_and_counts_the_rest():
    C = 101
    counts = vec(6, [3] * 10, C)  # winner + 10 competitors at 3, 90 silent
    prof = competitor_profile(counts, C)
    assert prof["n_competitors"] == 100
    assert prof["mean_n_active"] == pytest.approx(10.0)
    assert prof["mean_max_competitor"] == pytest.approx(3.0)
    assert prof["mean_count"] == pytest.approx(10 * 3 / 100)
    assert prof["count_histogram"][3.0] == pytest.approx(0.10)
    assert prof["count_histogram"][0.0] == pytest.approx(0.90)


def test_profile_reports_concentration_because_a_mean_count_hides_it():
    """The denominator is a sum of exponentials, so a few loud competitors dominate it.

    Two configurations with the SAME mean competitor count give very different ceilings,
    which is why the distribution is reported rather than the mean alone.
    """
    C, T = 101, 6
    spread = vec(T, [1] * 100, C)          # mean 1.00, spread thin
    concentrated = vec(T, [5] * 20, C)     # mean 1.00, concentrated
    assert competitor_profile(spread, C)["mean_count"] == pytest.approx(
        competitor_profile(concentrated, C)["mean_count"])
    assert conditional_ceiling(concentrated, T)[0] < conditional_ceiling(spread, T)[0]
    assert competitor_profile(concentrated, C)["mean_top1_share_of_sum"] > \
           competitor_profile(spread, C)["mean_top1_share_of_sum"]


# ---------------------------------------------------------------------------------------
# The share of the residual a ceiling forces
# ---------------------------------------------------------------------------------------


def test_share_explained_is_zero_when_the_ceiling_is_slack():
    assert residual_share_explained(0.64, 0.49, 0.80) == 0.0


def test_share_explained_is_one_when_the_ceiling_sits_at_or_below_mean_confidence():
    assert residual_share_explained(0.64, 0.49, 0.49) == pytest.approx(1.0)
    assert residual_share_explained(0.64, 0.49, 0.30) == 1.0, "clipped, never above 1"


def test_share_explained_is_proportional_in_between():
    # accuracy 0.64, mean conf 0.49, residual 0.15; a ceiling at 0.565 forces half of it.
    assert residual_share_explained(0.64, 0.49, 0.565) == pytest.approx(0.5, abs=1e-9)


def test_share_explained_is_zero_for_an_overconfident_model():
    """With mean confidence above accuracy there is no underconfidence to explain."""
    assert residual_share_explained(0.64, 0.70, 0.50) == 0.0
