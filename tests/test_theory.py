"""Tests for the derived results in :mod:`spikecalib.theory`.

Each result is checked against something that does not share its derivation: brute-force
enumeration of every count vector, random search over an interval, or an algebraic identity
evaluated on data. A bound that is only compared with its own formula has not been tested.

Nothing here touches a trained model. Whether the models respect these bounds is an
experiment, and is recorded in ``NOTES.md`` with its prediction, not asserted in a unit test.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from spikecalib.calibration import (
    IsotonicOneVsRest,
    IsotonicTopLabel,
    TemperatureScaling,
    ece_equal_mass,
    ece_equal_width,
    model_outputs,
    quantise_confidence,
    softmax_confidence,
    softmax_probs,
)
from spikecalib.conditional import theoretical_ceiling
from spikecalib.models import min_timesteps, readout_ceiling
from spikecalib.theory import (
    bounded_logit_ceiling,
    distinct_confidence_bound_unreduced,
    loss_floor,
    max_distinct_confidences,
    min_logit_width,
    min_scale,
    murphy_binned,
    murphy_exact,
    rate_ceiling,
    scaled_ceiling,
)


def all_count_vectors(T: int, C: int) -> np.ndarray:
    """Every count vector up to the order of its classes, which softmax confidence ignores."""
    return np.array(list(itertools.combinations_with_replacement(range(T + 1), C)), dtype=float)


def n_distinct(conf: np.ndarray) -> int:
    return int(np.unique(quantise_confidence(conf)).size)


# ---------------------------------------------------------------------------------------
# 1a. The cap for bounded logits
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C", [(2, 10), (4, 10), (6, 10), (10, 10), (2, 101), (6, 101), (10, 101)])
def test_scale_one_is_the_ceiling_the_study_already_uses(T, C):
    assert scaled_ceiling(T, C, 1.0) == pytest.approx(readout_ceiling("count", T, C), abs=1e-15)
    assert scaled_ceiling(T, C, 1.0) == pytest.approx(theoretical_ceiling(T, C), abs=1e-15)


@pytest.mark.parametrize("T,C,s", [(1, 3, 1.0), (2, 4, 0.5), (2, 4, 2.5), (3, 3, 1.7), (2, 10, 1.0), (2, 10, 3.2)])
def test_no_count_vector_beats_the_scaled_cap_and_the_extreme_one_reaches_it(T, C, s):
    """Enumerate every count vector; the largest confidence must be the cap, not just below it."""
    conf = softmax_probs(s * all_count_vectors(T, C)).max(axis=1)
    cap = scaled_ceiling(T, C, s)
    assert conf.max() <= cap + 1e-12
    assert conf.max() == pytest.approx(cap, abs=1e-12)
    extreme = np.zeros((1, C))
    extreme[0, 0] = T
    assert softmax_probs(s * extreme).max() == pytest.approx(cap, abs=1e-12)


@pytest.mark.parametrize("W,C", [(0.0, 5), (0.7, 3), (1.0, 10), (2.0, 10), (5.0, 101), (12.0, 1000)])
def test_real_valued_logits_in_any_interval_of_width_w_obey_the_cap(W, C):
    """Nothing about the bound is specific to integers, to spikes, or to where the interval sits."""
    rng = np.random.default_rng(0)
    cap = bounded_logit_ceiling(W, C)
    for offset in (-40.0, 0.0, 3.5, 900.0):
        z = offset + W * rng.random((4000, C))
        # push some rows to the corners, where the maximum lives
        z[:200] = offset + W * (rng.random((200, C)) > 0.5)
        assert softmax_probs(z).max() <= cap + 1e-12
        corner = np.full((1, C), offset)
        corner[0, 0] = offset + W
        assert softmax_probs(corner).max() == pytest.approx(cap, abs=1e-12)


def test_clipping_or_quantising_logits_imposes_the_same_cap():
    """A clipped logit and a 3-bit quantised one both live in an interval; the cap follows."""
    rng = np.random.default_rng(1)
    z = rng.normal(0, 8, (5000, 10))
    clipped = np.clip(z, -1.5, 1.5)
    assert softmax_probs(clipped).max() <= bounded_logit_ceiling(3.0, 10) + 1e-12
    step = 0.4
    quantised = np.clip(np.round(z / step), 0, 7) * step          # 8 levels, width 7 * step
    assert softmax_probs(quantised).max() <= bounded_logit_ceiling(7 * step, 10) + 1e-12


@pytest.mark.parametrize("C,expected", [(10, 0.2320), (11, 0.2137), (101, 0.0265)])
def test_rate_readout_cap_is_e_over_e_plus_c_minus_one(C, expected):
    assert rate_ceiling(C) == pytest.approx(math.e / (math.e + C - 1), abs=1e-15)
    assert round(rate_ceiling(C), 4) == expected


@pytest.mark.parametrize("T", [1, 2, 4, 6, 10, 80, 1000])
def test_rate_readout_cap_does_not_depend_on_t(T):
    """z = c / T lies in [0, 1] for every T, so adding timesteps adds no confidence."""
    assert scaled_ceiling(T, 10, 1.0 / T) == pytest.approx(rate_ceiling(10), abs=1e-14)
    rates = all_count_vectors(min(T, 3), 4) / min(T, 3)
    assert softmax_probs(rates).max() == pytest.approx(rate_ceiling(4), abs=1e-12)


def test_the_cap_grows_with_width_and_falls_with_classes():
    widths = [0.0, 0.5, 1.0, 2.0, 6.0, 20.0]
    caps = [bounded_logit_ceiling(w, 10) for w in widths]
    assert caps == sorted(caps) and caps[0] == pytest.approx(0.1) and caps[-1] < 1.0
    assert bounded_logit_ceiling(2.0, 101) < bounded_logit_ceiling(2.0, 10)
    assert bounded_logit_ceiling(800.0, 10) == 1.0          # no overflow where e^W would


@pytest.mark.parametrize("T,C,floor", [(2, 10, 0.7966), (6, 10, 0.0221), (10, 10, 0.0004), (2, 101, 2.6765)])
def test_loss_floor_is_minus_log_of_the_cap_and_matches_the_paper(T, C, floor):
    assert loss_floor(T, C) == pytest.approx(-math.log(scaled_ceiling(T, C)), abs=1e-12)
    assert round(loss_floor(T, C), 4) == floor


def test_no_bounded_logit_vector_gets_under_the_loss_floor():
    rng = np.random.default_rng(2)
    W, C = 1.3, 7
    z = W * rng.random((5000, C))
    labels = rng.integers(0, C, 5000)
    nll = -np.log(softmax_probs(z)[np.arange(5000), labels])
    assert nll.min() >= loss_floor(W, C) - 1e-12


# ---------------------------------------------------------------------------------------
# 1b. The minimum scale
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C,a", [(2, 10, 0.985), (2, 101, 0.4543), (6, 101, 0.6427), (10, 10, 0.99)])
def test_the_cap_at_the_minimum_scale_is_exactly_the_target(T, C, a):
    s = min_scale(T, C, a)
    assert scaled_ceiling(T, C, s) == pytest.approx(a, abs=1e-12)
    assert scaled_ceiling(T, C, 0.99 * s) < a < scaled_ceiling(T, C, 1.01 * s)


@pytest.mark.parametrize("C,a", [(10, 0.95), (101, 0.95), (10, 0.5), (2, 0.9)])
def test_t_min_is_the_minimum_width_read_at_scale_one(C, a):
    """T_min, s_min * T and W_min are one number. The constraint is on the product sT."""
    assert min_logit_width(C, a) == pytest.approx(min_timesteps(C, a), abs=1e-15)
    for T in (1, 2, 5, 40):
        assert min_scale(T, C, a) * T == pytest.approx(min_logit_width(C, a), abs=1e-12)


def test_halving_t_and_doubling_the_scale_leaves_the_cap_where_it_was():
    for T, s in ((10, 0.3), (4, 1.0), (2, 3.6)):
        assert scaled_ceiling(T, 10, s) == pytest.approx(scaled_ceiling(T / 2, 10, 2 * s), abs=1e-15)


def test_minimum_scale_values_for_the_cells_the_predictions_name():
    """The numbers the registered predictions compare a learned scale against."""
    assert min_scale(2, 10, 0.9850) == pytest.approx(3.191, abs=5e-3)
    assert min_scale(2, 101, 0.4543) == pytest.approx(2.211, abs=5e-3)
    assert min_scale(6, 101, 0.6427) == pytest.approx(0.865, abs=5e-3)


# ---------------------------------------------------------------------------------------
# 1c. How many distinct confidences a count readout can emit
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C", [(1, 2), (1, 5), (2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (2, 10)])
@pytest.mark.parametrize("s", [1.0, 0.37, 2.5])
def test_enumeration_gives_exactly_the_bound_at_generic_scales(T, C, s):
    """Count the distinct confidences of every count vector. The bound is attained, not just kept."""
    conf = softmax_probs(s * all_count_vectors(T, C)).max(axis=1)
    assert n_distinct(conf) == max_distinct_confidences(T, C) == math.comb(T + C - 1, C - 1)


def test_the_count_is_55_at_t2_c10_and_the_unreduced_count_is_66():
    assert max_distinct_confidences(2, 10) == 55
    assert distinct_confidence_bound_unreduced(2, 10) == 66
    assert n_distinct(softmax_confidence(all_count_vectors(2, 10))) == 55


@pytest.mark.parametrize("T,C", [(1, 2), (2, 3), (2, 10), (6, 10), (2, 101), (10, 101)])
def test_the_unreduced_count_is_a_valid_but_loose_bound(T, C):
    """Counting by winning count double-counts shifted vectors; removing them telescopes."""
    loose, tight = distinct_confidence_bound_unreduced(T, C), max_distinct_confidences(T, C)
    assert loose == math.comb(T + C, C)                       # hockey-stick identity
    assert loose - tight == math.comb(T + C - 1, C)           # the shifted duplicates
    assert tight < loose


def test_shifting_every_count_by_a_constant_does_not_change_the_confidence():
    """The reason the unreduced count is loose."""
    rng = np.random.default_rng(3)
    c = rng.integers(0, 4, (500, 6)).astype(float)
    assert np.array_equal(softmax_confidence(c), softmax_confidence(c + 3.0))


def test_the_scale_changes_the_values_and_not_how_many_there_are():
    vectors = all_count_vectors(2, 10)
    sets = [np.unique(quantise_confidence(softmax_probs(s * vectors).max(axis=1))) for s in (0.5, 1.0, 3.6)]
    assert [v.size for v in sets] == [55, 55, 55]
    assert sets[0].max() < sets[1].max() < sets[2].max()


def test_a_special_scale_can_merge_values_but_never_add_them():
    """At e^s = 2 three competitors one step behind equal one tied and two two steps behind."""
    vectors = all_count_vectors(2, 4)
    merged = n_distinct(softmax_probs(math.log(2.0) * vectors).max(axis=1))
    assert merged < max_distinct_confidences(2, 4) == 10


def test_the_bound_is_vacuous_once_it_passes_the_number_of_test_samples():
    """Where the bound is informative. Stated as arithmetic so the paper cannot overclaim it."""
    assert max_distinct_confidences(2, 10) == 55
    assert max_distinct_confidences(4, 10) == 715
    assert max_distinct_confidences(6, 10) == 5005 < 10_000 < max_distinct_confidences(8, 10)
    assert max_distinct_confidences(2, 101) == 5151 > 1742


# ---------------------------------------------------------------------------------------
# 1d. The Murphy decomposition, and what each calibrator can do to resolution
# ---------------------------------------------------------------------------------------


def synthetic_counts(n=6000, C=10, T=3, seed=0):
    """Counts where the true class tends to fire more, so confidence carries information."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, n)
    p = np.full((n, C), 0.15)
    p[np.arange(n), labels] = 0.6
    counts = rng.binomial(T, p).astype(float)
    return model_outputs(counts, labels, np.random.default_rng(seed + 100))


def partition(values: np.ndarray) -> np.ndarray:
    return np.unique(quantise_confidence(values), return_inverse=True)[1]


def is_same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = set(zip(partition(a).tolist(), partition(b).tolist()))
    return len(pairs) == n_distinct(a) == n_distinct(b)


def only_merges(before: np.ndarray, after: np.ndarray) -> bool:
    """Every group of ``before`` lands inside a single group of ``after``."""
    pairs = set(zip(partition(before).tolist(), partition(after).tolist()))
    return len(pairs) == n_distinct(before)


def test_exact_decomposition_is_an_identity_with_no_within_group_terms():
    out = synthetic_counts()
    d = murphy_exact(out.conf, out.correct)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-12)
    assert d["within_bin_variance"] == pytest.approx(0.0, abs=1e-20)
    assert d["within_bin_covariance"] == pytest.approx(0.0, abs=1e-20)
    assert d["n_groups"] == n_distinct(out.conf) <= max_distinct_confidences(3, 10)
    assert 0.0 <= d["resolution"] <= d["uncertainty"]


@pytest.mark.parametrize("scheme", ["equal_width", "equal_mass"])
def test_binned_decomposition_needs_the_two_within_bin_terms(scheme):
    rng = np.random.default_rng(4)
    conf = rng.beta(5, 2, 8000)
    correct = rng.random(8000) < conf ** 1.3
    d = murphy_binned(conf, correct, 15, scheme)
    three = d["reliability"] - d["resolution"] + d["uncertainty"]
    five = three + d["within_bin_variance"] - d["within_bin_covariance"]
    assert d["brier"] == pytest.approx(five, abs=1e-12)
    assert abs(d["brier"] - three) > 1e-5       # the three-term form is not an identity here


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binned_decomposition_uses_the_same_bins_as_the_ece_estimators(seed):
    out = synthetic_counts(seed=seed)
    assert murphy_binned(out.conf, out.correct, 15, "equal_width")["ece"] == pytest.approx(
        ece_equal_width(out.conf, out.correct, 15), abs=1e-12)
    assert murphy_binned(out.conf, out.correct, 15, "equal_mass")["ece"] == pytest.approx(
        ece_equal_mass(out.conf, out.correct, 15), abs=1e-12)


def test_exact_ece_is_never_below_either_binned_estimate():
    """Both binnings keep equal confidences together, so each bin is a union of exact groups."""
    for seed in range(4):
        out = synthetic_counts(seed=seed)
        exact = murphy_exact(out.conf, out.correct)
        for scheme in ("equal_width", "equal_mass"):
            binned = murphy_binned(out.conf, out.correct, 15, scheme)
            assert exact["ece"] >= binned["ece"] - 1e-12
            assert exact["resolution"] >= binned["resolution"] - 1e-12


def test_exact_decomposition_says_nothing_with_one_sample_per_value():
    """Why an unbounded readout has to be binned, and why the two must not be compared."""
    rng = np.random.default_rng(5)
    conf = rng.random(3000)
    correct = rng.random(3000) < conf
    d = murphy_exact(conf, correct)
    assert d["samples_per_value"] == pytest.approx(1.0)
    assert d["resolution"] == pytest.approx(d["uncertainty"], abs=1e-12)
    assert d["reliability"] == pytest.approx(d["brier"], abs=1e-12)


def test_temperature_scaling_keeps_the_partition_and_so_cannot_change_resolution():
    cal, test = synthetic_counts(seed=10), synthetic_counts(seed=11)
    ts = TemperatureScaling().fit(cal.conf, cal.correct, outputs=cal)
    after = ts.transform(test.conf, outputs=test)
    assert ts.temperature_ != pytest.approx(1.0, abs=1e-3)       # it did something
    assert not np.allclose(after, test.conf)
    assert is_same_partition(test.conf, after)
    before_d, after_d = murphy_exact(test.conf, test.correct), murphy_exact(after, test.correct)
    assert after_d["resolution"] == pytest.approx(before_d["resolution"], abs=1e-12)
    assert after_d["n_groups"] == before_d["n_groups"]


def test_temperature_scaling_can_merge_groups_only_at_special_temperatures():
    """t = 1/ln 2 makes e^(1/t) = 2, the one kind of value at which distinct multisets collide."""
    vectors = np.repeat(all_count_vectors(2, 4), 3, axis=0)
    before = softmax_probs(vectors).max(axis=1)
    special = softmax_probs(vectors * math.log(2.0)).max(axis=1)
    generic = softmax_probs(vectors / 0.2733).max(axis=1)
    assert is_same_partition(before, generic)
    assert only_merges(before, special) and n_distinct(special) < n_distinct(before)


def test_top_label_isotonic_only_merges_groups_and_so_cannot_raise_resolution():
    for seed in range(4):
        cal, test = synthetic_counts(seed=20 + seed), synthetic_counts(seed=30 + seed)
        iso = IsotonicTopLabel().fit(cal.conf, cal.correct, outputs=cal)
        after = iso.transform(test.conf, outputs=test)
        assert only_merges(test.conf, after)
        assert n_distinct(after) <= n_distinct(test.conf)
        before_d, after_d = murphy_exact(test.conf, test.correct), murphy_exact(after, test.correct)
        assert after_d["resolution"] <= before_d["resolution"] + 1e-12


def test_merging_groups_never_raises_resolution():
    """The lemma under the isotonic claim: resolution is a between-group variance."""
    rng = np.random.default_rng(6)
    conf = rng.integers(0, 12, 5000) / 12.0
    correct = rng.random(5000) < 0.3 + 0.5 * conf
    fine = murphy_exact(conf, correct)["resolution"]
    for width in (2, 3, 4, 6, 12):
        coarse = murphy_exact(np.floor(conf * 12 / width) * width / 12.0, correct)["resolution"]
        assert coarse <= fine + 1e-12


def test_one_vs_rest_isotonic_is_not_covered_by_the_merge_argument():
    """It maps each class through its own function, so one input group can come apart."""
    rng = np.random.default_rng(7)
    n, C, T = 8000, 4, 2
    labels = rng.integers(0, C, n)
    p = np.full((n, C), 0.2)
    p[np.arange(n), labels] = 0.7
    p[labels == 0, 0] = 0.35                                    # class 0 is much harder
    cal = model_outputs(rng.binomial(T, p).astype(float), labels, np.random.default_rng(8))
    ovr = IsotonicOneVsRest().fit(cal.conf, cal.correct, outputs=cal)
    after = ovr.transform(cal.conf, outputs=cal)
    assert not only_merges(cal.conf, after)
