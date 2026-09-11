"""Tests for the quantisation analysis.

The load-bearing property is the **identity** the whole experiment is read through:

    AUROC(quantised) - AUROC(full) = pair_fraction_merged * (0.5 - auroc_within)

If that ever stopped holding, the reported "coarsening helps / coarsening hurts" reading
would be unsupported, because the two factors would no longer account for the movement.
It is asserted on three regimes with opposite expected signs -- fine structure that is
signal, fine structure that is noise, and fine structure that is anti-correlated with
correctness -- so a bug that happened to be invisible in one of them is still caught.

The second property is that quantisation is **monotone and tie-preserving**: it may merge
samples but never reorder them, and never splits samples that share a confidence. A split
would manufacture an operating point the model never expressed, which is exactly what
:mod:`spikecalib.selective` refuses to do.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from spikecalib.quantisation import (
    LEVEL_GRID,
    SCHEMES,
    quantisation_step,
    quantisation_sweep,
    quantise_to_levels,
    within_bin_auroc,
)

SCHEME_PARAMS = pytest.mark.parametrize("scheme", SCHEMES)


def informative(n=3000, seed=0):
    """Confidence whose fine structure is real signal: a finer reading ranks better."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n)
    return conf, rng.random(n) < conf


def noisy(n=3000, seed=0, jitter=0.2):
    """A coarse signal plus independent noise: the fine structure carries nothing."""
    rng = np.random.default_rng(seed)
    base = rng.choice(np.linspace(0.5, 0.95, 8), size=n)
    correct = rng.random(n) < base
    conf = np.clip(base + rng.normal(0, jitter, n), 1e-6, 1.0)
    return conf, correct


def anti_correlated(n=3000, seed=0):
    """Fine structure that actively misleads: within each coarse level it ranks backwards.

    This is the regime the hypothesis proposes for the membrane readout, constructed
    deliberately so the tests can show that coarsening *does* raise AUROC when it holds --
    a flat result on real data then means the regime is absent, not that the code is inert.
    """
    rng = np.random.default_rng(seed)
    base = rng.choice(np.linspace(0.5, 0.95, 8), size=n)
    correct = rng.random(n) < base
    # Offset inside the level is pushed the wrong way: wrong answers get the higher offset.
    offset = np.where(correct, -1.0, 1.0) * rng.uniform(0.0, 0.028, n)
    return np.clip(base + offset, 1e-6, 1.0), correct


# ---------------------------------------------------------------------------------------
# Quantisation is monotone, tie-preserving and honest about its resolution
# ---------------------------------------------------------------------------------------


@SCHEME_PARAMS
def test_quantisation_never_reorders_a_pair(scheme):
    conf, _ = informative()
    bins = quantise_to_levels(conf, 20, scheme)
    order = np.argsort(conf, kind="stable")
    assert np.all(np.diff(bins[order]) >= 0), "a lower confidence received a higher bin"


@SCHEME_PARAMS
def test_equal_confidences_always_share_a_bin(scheme):
    """Splitting a tie would invent an ordering the model never expressed."""
    rng = np.random.default_rng(0)
    conf = rng.choice(np.array([0.31, 0.45, 0.62, 0.88]), size=4000)
    bins = quantise_to_levels(conf, 50, scheme)
    for value in np.unique(conf):
        assert np.unique(bins[conf == value]).size == 1


def test_equal_mass_spreads_samples_evenly_and_equal_width_does_not():
    """The reason equal-mass is the primary scheme: its N axis means what it says.

    Both schemes populate most of their bins on skewed data. What separates them is how
    the samples are spread: equal-width piles them into the few bins nearest the mode, so
    the resolution that is actually applied to the bulk of the data is far below ``N``.
    """
    rng = np.random.default_rng(0)
    conf = 1.0 - rng.exponential(0.01, 20000)  # piled up near 1.0, like a real readout

    mass = quantise_to_levels(conf, 100, "equal-mass")
    width = quantise_to_levels(conf, 100, "equal-width")
    assert np.unique(mass).size == 100

    mass_counts = np.bincount(mass.astype(int))
    width_counts = np.bincount(width.astype(int))
    assert mass_counts.max() / mass_counts.min() < 1.1, "equal-mass bins hold equal counts"
    assert width_counts.max() > 5 * mass_counts.max(), "equal-width piles up at the mode"
    # Half the samples land in this many bins: the resolution the bulk of the data sees.
    def half_mass_bins(counts):
        return int(np.searchsorted(np.cumsum(np.sort(counts)[::-1]), counts.sum() / 2) + 1)
    assert half_mass_bins(mass_counts) == 50
    assert half_mass_bins(width_counts) <= 12


def test_equal_mass_does_not_collapse_a_heavy_tie_into_the_level_above_it():
    """Why bins follow the midpoint of a value's mass rather than its upper edge.

    A value holding half the samples spans half the bin grid. Charged to the top of that
    span it lands on its neighbour and a genuinely distinct confidence level disappears --
    which would understate the resolution the readout actually has.
    """
    conf = np.concatenate([np.full(250, 0.3), np.full(500, 0.6), np.full(250, 0.9)])
    bins = quantise_to_levels(conf, 4, "equal-mass")
    assert np.unique(bins).size == 3, "three distinct confidences must keep three bins"
    assert bins[conf == 0.6][0] != bins[conf == 0.9][0]


def test_equal_width_keeps_the_largest_value_inside_the_grid():
    """The maximum sits exactly on the top edge, which must not open an N+1-th bin."""
    conf = np.linspace(0.1, 0.9, 500)
    bins = quantise_to_levels(conf, 10, "equal-width")
    assert bins.max() == 9
    assert np.unique(bins).size == 10


def test_full_precision_is_a_no_op():
    conf, correct = informative()
    assert quantisation_step(conf, correct, None)["auroc"] == pytest.approx(
        roc_auc_score(correct, conf)
    )


def test_single_level_cannot_order_anything():
    conf, correct = informative()
    step = quantisation_step(conf, correct, 1)
    assert step["n_effective"] == 1
    assert step["auroc"] == 0.5
    assert step["pair_fraction_merged"] == pytest.approx(1.0)


def test_rejects_bad_arguments():
    conf, _ = informative(n=10)
    with pytest.raises(ValueError, match="n_levels"):
        quantise_to_levels(conf, 0)
    with pytest.raises(ValueError, match="unknown scheme"):
        quantise_to_levels(conf, 10, "equal-nonsense")
    with pytest.raises(ValueError, match="empty"):
        quantise_to_levels(np.array([]), 10)


# ---------------------------------------------------------------------------------------
# The identity the experiment is read through
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("maker", [informative, noisy, anti_correlated])
@SCHEME_PARAMS
def test_auroc_shift_equals_merged_fraction_times_within_bin_deficit(maker, scheme):
    """AUROC(quantised) - AUROC(full) == pair_fraction_merged * (0.5 - auroc_within).

    Exact, not approximate: quantisation cannot reorder, so every pair it touches moves
    from its full-precision contribution to exactly 0.5.
    """
    conf, correct = maker()
    full = roc_auc_score(correct, conf)
    for n_levels in (5, 20, 100):
        step = quantisation_step(conf, correct, n_levels, scheme)
        predicted = step["pair_fraction_merged"] * (0.5 - step["auroc_within"])
        assert step["auroc"] - full == pytest.approx(predicted, abs=1e-12)


def test_signal_in_the_fine_structure_means_coarsening_costs():
    conf, correct = informative()
    full = roc_auc_score(correct, conf)
    step = quantisation_step(conf, correct, 10)
    assert step["auroc_within"] > 0.5
    assert step["auroc"] < full


def test_anti_correlated_fine_structure_means_coarsening_pays():
    """The positive control: when the hypothesis holds, AUROC rises as levels are removed.

    Without this, a flat curve on the real data would be indistinguishable from a
    quantiser that does nothing.
    """
    conf, correct = anti_correlated()
    full = roc_auc_score(correct, conf)
    step = quantisation_step(conf, correct, 8)
    assert step["auroc_within"] < 0.5
    assert step["auroc"] > full + 0.01, "coarsening should recover a visible amount"


def test_pure_noise_in_the_fine_structure_leaves_auroc_nearly_flat():
    conf, correct = noisy()
    full = roc_auc_score(correct, conf)
    step = quantisation_step(conf, correct, 100)
    assert step["auroc_within"] == pytest.approx(0.5, abs=0.05)
    assert step["auroc"] == pytest.approx(full, abs=0.01)


# ---------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------


def test_sweep_covers_the_grid_and_ends_at_full_precision():
    conf, correct = informative()
    rows = quantisation_sweep(conf, correct)
    assert [r["n_levels_requested"] for r in rows] == list(LEVEL_GRID)
    assert rows[-1]["n_levels_requested"] is None
    assert rows[-1]["n_effective"] == np.unique(np.round(conf, 12)).size
    assert all(r["n_effective"] <= (r["n_levels_requested"] or np.inf) for r in rows)


def test_merged_pair_fraction_falls_as_resolution_rises():
    conf, correct = informative()
    rows = quantisation_sweep(conf, correct)
    merged = [r["pair_fraction_merged"] for r in rows[:-1]]
    assert merged == sorted(merged, reverse=True)
    assert rows[-1]["pair_fraction_merged"] == pytest.approx(0.0, abs=1e-12)


def test_within_bin_auroc_is_undefined_when_no_bin_holds_both_classes():
    """One sample per bin merges nothing, so there is no fine structure to score."""
    conf, correct = informative(n=200)
    out = within_bin_auroc(conf, correct, np.arange(conf.size, dtype=float))
    assert np.isnan(out["auroc_within"])
    assert out["n_pairs_within"] == 0
