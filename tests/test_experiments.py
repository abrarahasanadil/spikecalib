"""Tests for the plumbing the scale and resolution experiments share.

What is tested here is what would silently corrupt a result rather than crash: a model
looked up under the wrong filename, a calibration subsample that is not stratified or not
reproducible, and the tie rule, which decides the outcome of two registered predictions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from spikecalib.calibration import model_outputs, paired_verdict
from spikecalib.experiments import (
    BATCH_SIZE, DEFAULT_CHANNELS, _integer_counts, stem, stratified_subsample, subset,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("args,expected", [
    (("nmnist", 2, 0), "nmnist_T2_seed0"),
    (("nmnist", 6, 1, "accumulator"), "nmnist_T6_seed1_accumulator"),
    (("nmnist", 2, 2, "count", (32, 64, 128, 128)), "nmnist_T2_seed2_arch32-64-128-128"),
    (("nmnist", 2, 0, "accumulator", (32, 64, 128, 128)), "nmnist_T2_seed0_accumulator_arch32-64-128-128"),
    (("ncaltech101", 10, 0), "ncaltech101_T10_seed0"),
    (("ncaltech101", 2, 1, "count", (32, 64)), "ncaltech101_T2_seed1_arch32-64"),
    (("ncaltech101", 6, 2, "scaled-count"), "ncaltech101_T6_seed2_scaled-count"),
])
def test_stems_match_the_names_the_earlier_experiments_wrote(args, expected):
    assert stem(*args) == expected


def test_every_stem_for_an_existing_cell_has_a_tracked_record():
    """The naming rule is only right if it finds the records already in the repository."""
    cells = ([("nmnist", T, r, None) for T in (2, 6, 10) for r in ("count", "accumulator")]
             + [("ncaltech101", T, r, None) for T in (2, 6, 10) for r in ("count", "accumulator")]
             + [("nmnist", T, r, (32, 64, 128, 128)) for T in (2, 6) for r in ("count", "accumulator")]
             + [("ncaltech101", T, r, (32, 64)) for T in (2, 6) for r in ("count", "accumulator")])
    for key, T, readout, channels in cells:
        for seed in (0, 1, 2):
            path = os.path.join(ROOT, "results", stem(key, T, seed, readout, channels) + "_record.json")
            assert os.path.exists(path), path


def test_defaults_are_the_settings_the_existing_runs_recorded():
    import json
    for key in ("nmnist", "ncaltech101"):
        with open(os.path.join(ROOT, "results", f"{key}_T2_seed0_record.json")) as fh:
            rec = json.load(fh)
        assert rec["batch_size"] == BATCH_SIZE[key]
        assert tuple(rec["model_config"]["channels"]) == DEFAULT_CHANNELS[key]


# ---------------------------------------------------------------------------------------
# Scaled logits back to integer counts
# ---------------------------------------------------------------------------------------


def test_integer_counts_are_recovered_from_scaled_logits_and_anything_else_is_refused():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 7, (200, 10))
    scale = 3.6591
    logits = (counts * np.float32(scale)).astype(np.float32).astype(np.float64)
    assert np.array_equal(_integer_counts(logits, float(np.float32(scale)), 6, "x"), counts)
    with pytest.raises(RuntimeError):
        _integer_counts(logits + 0.3, scale, 6, "x")          # not on the lattice
    with pytest.raises(RuntimeError):
        _integer_counts(logits, scale, 5, "x")                # a count above T


# ---------------------------------------------------------------------------------------
# Calibration subsamples
# ---------------------------------------------------------------------------------------


def test_subsample_is_stratified_without_replacement_and_reproducible():
    rng = np.random.default_rng(1)
    labels = rng.permutation(np.repeat(np.arange(10), 500))
    for size in (100, 1000, 5000):
        idx = stratified_subsample(labels, size, np.random.default_rng([2, size, 7]))
        assert idx.size == size == np.unique(idx).size
        assert np.array_equal(np.bincount(labels[idx], minlength=10), np.full(10, size // 10))
        again = stratified_subsample(labels, size, np.random.default_rng([2, size, 7]))
        assert np.array_equal(idx, again)
    a = stratified_subsample(labels, 100, np.random.default_rng([2, 100, 0]))
    b = stratified_subsample(labels, 100, np.random.default_rng([2, 100, 1]))
    assert not np.array_equal(a, b)
    assert np.array_equal(stratified_subsample(labels, 5000, np.random.default_rng(0)), np.arange(5000))


def test_subsample_refuses_a_size_it_cannot_stratify():
    labels = np.repeat(np.arange(10), 20)
    with pytest.raises(ValueError):
        stratified_subsample(labels, 105, np.random.default_rng(0))
    with pytest.raises(ValueError):
        stratified_subsample(labels, 300, np.random.default_rng(0))


def test_subset_keeps_every_field_aligned():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 3, (300, 10)).astype(float)
    labels = rng.integers(0, 10, 300)
    out = model_outputs(counts, labels, np.random.default_rng(4))
    idx = np.array([5, 17, 201, 299])
    sub = subset(out, idx)
    assert sub.n == 4
    assert np.array_equal(sub.counts, out.counts[idx]) and np.array_equal(sub.labels, out.labels[idx])
    assert np.array_equal(sub.predictions, out.predictions[idx])      # the tie-break is not re-rolled
    assert np.array_equal(sub.conf, out.conf[idx]) and np.array_equal(sub.correct, out.correct[idx])


# ---------------------------------------------------------------------------------------
# The tie rule
# ---------------------------------------------------------------------------------------


def _calibrated(n, seed):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n)
    return conf, rng.random(n) < conf


def test_a_calibrator_that_is_clearly_worse_differs_under_both_estimators():
    good, correct = _calibrated(6000, 0)
    bad = np.clip(good - 0.15, 0, 1)
    v = paired_verdict(good, bad, correct, n_bootstrap=200, seed=0)
    assert v["differ"] and not v["tie"] and v["favours"] == "a"
    assert v["equal_width"]["ci"][1] < 0 and v["equal_mass"]["ci"][1] < 0
    swapped = paired_verdict(bad, good, correct, n_bootstrap=200, seed=0)
    assert swapped["differ"] and swapped["favours"] == "b"


def test_identical_calibrators_tie():
    conf, correct = _calibrated(4000, 1)
    v = paired_verdict(conf, conf.copy(), correct, n_bootstrap=200, seed=0)
    assert v["tie"] and not v["differ"] and v["favours"] is None


def test_excluding_zero_under_one_estimator_only_is_a_tie(monkeypatch):
    """The rule's whole point, and the case the N-MNIST sweep produced five times."""
    import spikecalib.calibration as cal

    def fake(conf_a, conf_b, correct, n_bins=15, n_bootstrap=1000, seed=0, estimator="equal_width"):
        ci = (0.0006, 0.0044) if estimator == "equal_width" else (-0.0013, 0.0011)
        return {"estimator": estimator, "difference": 0.0, "ci": ci,
                "resolvable": ci[0] > 0 or ci[1] < 0}

    monkeypatch.setattr(cal, "bootstrap_ece_difference", fake)
    v = cal.paired_verdict(np.array([0.5]), np.array([0.5]), np.array([1]))
    assert v["equal_width"]["resolvable"] and not v["equal_mass"]["resolvable"]
    assert v["tie"] and v["favours"] is None


def test_excluding_zero_in_opposite_directions_is_a_tie(monkeypatch):
    import spikecalib.calibration as cal

    def fake(conf_a, conf_b, correct, n_bins=15, n_bootstrap=1000, seed=0, estimator="equal_width"):
        ci = (0.001, 0.004) if estimator == "equal_width" else (-0.004, -0.001)
        return {"estimator": estimator, "difference": 0.0, "ci": ci, "resolvable": True}

    monkeypatch.setattr(cal, "bootstrap_ece_difference", fake)
    assert cal.paired_verdict(np.array([0.5]), np.array([0.5]), np.array([1]))["tie"]


def test_the_tie_rule_reproduces_the_fifteen_ties_already_on_record():
    """Applied to the N-MNIST sweep's own intervals, the rule makes every run a tie."""
    import csv
    with open(os.path.join(ROOT, "docs", "sweep.csv")) as fh:
        runs = {(r["timesteps"], r["seed"]): r for r in csv.DictReader(fh)}

    def side(lo, hi):
        return -1 if float(hi) < 0 else (1 if float(lo) > 0 else 0)

    differ = 0
    for r in runs.values():
        ew = side(r["iso_minus_temp_ci_low"], r["iso_minus_temp_ci_high"])
        em = side(r["iso_minus_temp_em_ci_low"], r["iso_minus_temp_em_ci_high"])
        differ += ew != 0 and ew == em
    assert len(runs) == 15 and differ == 0
