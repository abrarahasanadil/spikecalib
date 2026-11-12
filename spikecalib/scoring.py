"""Proper scoring rules, equivalence tests and interval helpers.

Added after review, not pre-registered (NOTES.md, "Post-review analyses, thresholds fixed
2026-11-12 before computation", A3 and A4).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .calibration import _validate, ece_equal_mass, ece_equal_width

__all__ = [
    "CLIP",
    "top_label_scores",
    "multiclass_scores",
    "paired_difference_draws",
    "basic_interval",
    "tost_equivalent",
    "wilson_interval",
]

#: Probabilities are clipped to ``[CLIP, 1 - CLIP]`` before any logarithm (NOTES.md, A4).
CLIP = 1e-12


def top_label_scores(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Top-label Brier score and log loss of a confidence for "this prediction is right".

    The same quantities as ``CalibrationMetrics.brier`` and ``.nll``, with the number of
    confidences that the clip at ``CLIP`` changed.
    """
    conf, correct = _validate(conf, correct)
    clipped = np.clip(conf, CLIP, 1.0 - CLIP)
    n_clipped = int(np.sum(clipped != conf))
    n_clipped_exact = int(np.sum((clipped != conf) & ((conf == 0.0) | (conf == 1.0))))
    # Scores the clip turns into a penalty of -ln(CLIP): certain and wrong (A8).
    n_penalised = int(np.sum(((conf == 1.0) & (correct == 0.0)) | ((conf == 0.0) & (correct == 1.0))))
    log_loss = -float(np.mean(correct * np.log(clipped) + (1.0 - correct) * np.log(1.0 - clipped)))
    brier = float(np.mean((conf - correct) ** 2))
    return {"brier_top_label": brier, "log_loss_top_label": log_loss, "n_clipped_top_label": n_clipped,
            "n_clipped_top_label_exact_zero_or_one": n_clipped_exact,
            "n_penalised_top_label": n_penalised}


def multiclass_scores(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Multiclass negative log-likelihood and Brier score of a full predictive distribution.

    ``nll = -mean log p_y`` with ``p_y`` clipped below at ``CLIP``; ``brier = mean over samples
    of sum_k (p_k - 1[y = k])^2``, which lies in ``[0, 2]``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.size:
        raise ValueError(f"probs {probs.shape} does not match labels {labels.shape}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("rows of probs must sum to 1")
    p_y = probs[np.arange(labels.size), labels]
    n_clipped = int(np.sum(p_y < CLIP))
    nll = -float(np.mean(np.log(np.maximum(p_y, CLIP))))
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.size), labels] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return {"nll_multiclass": nll, "brier_multiclass": brier, "n_clipped_multiclass": n_clipped}


def paired_difference_draws(conf_a: np.ndarray, conf_b: np.ndarray, correct: np.ndarray,
                            seed: int, n_bins: int = 15, n_bootstrap: int = 1000,
                            estimator: str = "equal_width") -> tuple[float, np.ndarray]:
    """The resampled ECE differences behind :func:`spikecalib.calibration.bootstrap_ece_difference`.

    Draws exactly the same index vectors, in the same order, from
    ``numpy.random.default_rng(seed)``, so percentiles 2.5 and 97.5 of the returned draws
    reproduce that function's interval bit for bit, and percentiles 5 and 95 give the 90%
    interval of a two one-sided test on the same resamples. Returns ``(point, draws)``.
    """
    conf_a, y = _validate(conf_a, correct)
    conf_b, _ = _validate(conf_b, correct)
    if conf_a.size != conf_b.size:
        raise ValueError("arms have different sizes")
    ece: Callable = {"equal_width": ece_equal_width, "equal_mass": ece_equal_mass}[estimator]
    point = ece(conf_a, y, n_bins) - ece(conf_b, y, n_bins)
    rng = np.random.default_rng(seed)
    n = conf_a.size
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        draws[i] = ece(conf_a[idx], yy, n_bins) - ece(conf_b[idx], yy, n_bins)
    return float(point), draws


def basic_interval(point: float, draws: np.ndarray, level: float) -> tuple[float, float]:
    """Basic (reverse-percentile) bootstrap interval ``[2d - q(1 - a/2), 2d - q(a/2)]`` (A8)."""
    a = 100.0 * (1.0 - level) / 2.0
    lo_q, hi_q = (float(v) for v in np.percentile(draws, [a, 100.0 - a]))
    return 2.0 * point - hi_q, 2.0 * point - lo_q


def tost_equivalent(draws: np.ndarray, delta: float) -> tuple[bool, tuple[float, float]]:
    """Two one-sided tests at alpha 0.05 by the percentile method.

    Equivalence within ``[-delta, delta]`` is declared when the 90% percentile interval of
    the resampled difference (5th to 95th percentile) lies inside that closed interval.
    Returns ``(equivalent, (low, high))``.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    low, high = (float(v) for v in np.percentile(draws, [5.0, 95.0]))
    return bool(-delta <= low and high <= delta), (low, high)


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at the default ``z``)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
