"""The class-prior reweighting of the N-Caltech101 calibrators (S1 in NOTES.md)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from reweighted_calibration import N_CLASSES, class_weights, weighted_temperature  # noqa: E402
from spikecalib.calibration import TemperatureScaling, model_outputs  # noqa: E402


def test_weights_are_test_share_over_calibration_share_with_mean_one():
    cal = np.repeat(np.arange(N_CLASSES), 10)
    test = np.concatenate([np.full(30, 0), np.repeat(np.arange(1, N_CLASSES), 6)])
    w = class_weights(cal, test)
    assert abs(w.mean() - 1.0) < 1e-12
    pt = np.bincount(test, minlength=N_CLASSES) / test.size
    expected = pt[cal] * N_CLASSES
    assert np.allclose(w, expected / expected.mean())
    assert w[cal == 0][0] / w[cal == 1][0] == 5.0


def test_equal_shares_give_equal_weights():
    cal = np.repeat(np.arange(N_CLASSES), 10)
    test = np.repeat(np.arange(N_CLASSES), 4)
    assert np.allclose(class_weights(cal, test), 1.0)


def test_unit_weights_give_the_unweighted_temperature():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 3, size=(400, 10)).astype(np.float64)
    labels = rng.integers(0, 10, size=400)
    out = model_outputs(counts, labels, np.random.default_rng(0))
    t = TemperatureScaling().fit(out.conf, out.correct, outputs=out).temperature_
    assert abs(weighted_temperature(counts, labels, np.ones(400)) - t) < 1e-6 * t


def test_integer_weights_equal_duplicated_samples():
    rng = np.random.default_rng(4)
    counts = rng.integers(0, 3, size=(300, 10)).astype(np.float64)
    labels = rng.integers(0, 10, size=300)
    w = rng.integers(1, 4, size=300).astype(np.float64)
    dup_c, dup_l = np.repeat(counts, w.astype(int), axis=0), np.repeat(labels, w.astype(int))
    assert abs(weighted_temperature(counts, labels, w) - weighted_temperature(dup_c, dup_l, np.ones(dup_l.size))) < 1e-5
