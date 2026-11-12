"""The cap as a floor on ECE (A1) and tie-respecting bins (A2); added after review."""
from __future__ import annotations

import numpy as np
import pytest

from spikecalib.calibration import ece_equal_mass, ece_equal_width, softmax_confidence
from spikecalib.floor import (
    bin_assignment, ece_distinct, ece_from_gaps, floor_ratio, overconfidence_summary,
    signed_bin_gaps,
)
from spikecalib.theory import murphy_exact


def _random_case(rng, n=2000):
    conf = rng.uniform(0.1, 1.0, n)
    correct = (rng.random(n) < conf ** 0.7).astype(float)
    return conf, correct


@pytest.mark.parametrize("seed", range(5))
def test_gap_ece_matches_the_package_estimators(seed):
    conf, correct = _random_case(np.random.default_rng(seed))
    assert ece_from_gaps(signed_bin_gaps(conf, correct, "equal_width")) == pytest.approx(
        ece_equal_width(conf, correct), abs=1e-12)
    assert ece_from_gaps(signed_bin_gaps(conf, correct, "equal_mass")) == pytest.approx(
        ece_equal_mass(conf, correct), abs=1e-12)


def test_distinct_value_ece_matches_the_exact_decomposition():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 3, size=(5000, 10))
    conf = softmax_confidence(counts)
    correct = (rng.random(5000) < 0.8).astype(float)
    assert ece_distinct(conf, correct) == pytest.approx(murphy_exact(conf, correct)["ece"], abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_ece_is_at_least_the_aggregate_gap_with_equality_iff_one_sign(seed):
    rng = np.random.default_rng(100 + seed)
    conf, correct = _random_case(rng, n=int(rng.integers(50, 500)))
    gap = abs(correct.mean() - conf.mean())
    for scheme in ("equal_width", "equal_mass", "distinct"):
        g = signed_bin_gaps(conf, correct, scheme)
        ece = ece_from_gaps(g)
        assert ece >= gap - 1e-12
        one_sign = (g["gap"] >= -1e-15).all() or (g["gap"] <= 1e-15).all()
        if one_sign:
            assert ece == pytest.approx(gap, abs=1e-12)
        else:
            assert ece > gap + 1e-15


def test_overconfidence_is_half_the_excess_over_the_aggregate_gap():
    rng = np.random.default_rng(7)
    conf, correct = _random_case(rng)
    g = signed_bin_gaps(conf, correct, "equal_width")
    s = overconfidence_summary(g)
    a, p = correct.mean(), conf.mean()
    assert a > p
    assert ece_from_gaps(g) - (a - p) == pytest.approx(2 * s["weighted_overconfidence"], abs=1e-12)


def test_a_capped_readout_never_beats_its_floor():
    # Count readout at T = 2, C = 10: every confidence is at most the cap.
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 3, size=(4000, 10))
    counts[:3000, 0] = 2
    counts[:3000, 1:] = 0
    conf = softmax_confidence(counts)
    cap = np.exp(2) / (np.exp(2) + 9)
    assert conf.max() <= cap + 1e-12
    correct = np.zeros(4000)
    correct[:3900] = 1.0  # accuracy 0.975, far above the cap
    a = correct.mean()
    for scheme in ("equal_width", "equal_mass", "distinct"):
        ece = ece_from_gaps(signed_bin_gaps(conf, correct, scheme))
        assert ece >= a - cap
        assert floor_ratio(a, cap, ece) <= 1.0


def test_floor_ratio_is_undefined_where_the_cap_is_slack():
    assert floor_ratio(0.4, 0.45, 0.05) is None
    with pytest.raises(ValueError):
        floor_ratio(0.9, 0.45, 0.0)


def test_equal_mass_bins_are_not_equal_in_mass_when_one_value_dominates():
    conf = np.concatenate([np.full(9500, 0.45), np.linspace(0.2, 0.4, 500)])
    index, n_eff = bin_assignment(conf, "equal_mass", 15)
    occupancy = np.bincount(index)
    assert n_eff == 1 and occupancy.tolist() == [10000]
    with pytest.raises(ValueError):
        bin_assignment(conf, "quantile")
