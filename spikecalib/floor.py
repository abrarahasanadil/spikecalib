"""The confidence cap as a floor on ECE, and what binning does to the estimate.

Added after review, not pre-registered (NOTES.md, "Post-review analyses, thresholds fixed
2026-11-12 before computation", A1 and A2).

For any partition of a test set into bins ``b`` with weights ``w_b = n_b / n``, signed gaps
``g_b = acc_b - conf_b`` and expected calibration error ``ECE = sum_b w_b |g_b|``::

    ECE >= | sum_b w_b g_b | = | a - p_bar |

by the triangle inequality, ``a`` being accuracy and ``p_bar`` mean confidence, because
``sum_b w_b acc_b = a`` and ``sum_b w_b conf_b = p_bar``. Equality holds exactly when every
non-zero ``g_b`` has the same sign. A count readout never emits a confidence above
``p* = e^T / (e^T + C - 1)``, so ``p_bar <= p*``, and wherever ``a > p*``::

    ECE >= a - p_bar >= a - p*

under any binning, including one group per distinct confidence value. This module computes
the per-bin signed gaps that show how close each estimate is to that bound, and the
occupancy of the tie-respecting equal-mass bins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .calibration import _equal_mass_bins, _validate, quantise_confidence

__all__ = [
    "SCHEMES",
    "bin_assignment",
    "signed_bin_gaps",
    "ece_from_gaps",
    "ece_distinct",
    "overconfidence_summary",
    "floor_ratio",
]

SCHEMES = ("equal_width", "equal_mass", "distinct")


def bin_assignment(conf: np.ndarray, scheme: str, n_bins: int = 15) -> tuple[np.ndarray, int]:
    """Bin index per sample under one scheme, and the number of bins that can be occupied.

    ``equal_width`` and ``equal_mass`` reproduce the assignments of
    :func:`spikecalib.calibration.ece_equal_width` and
    :func:`spikecalib.calibration.ece_equal_mass` exactly. ``distinct`` puts every distinct
    confidence value (to 12 decimals) in its own group, the grouping of
    :func:`spikecalib.theory.murphy_exact`.
    """
    conf = np.asarray(conf, dtype=np.float64).ravel()
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1), n_bins
    if scheme == "equal_mass":
        index, n_effective = _equal_mass_bins(conf, n_bins)
        if index is None:
            return np.zeros(conf.size, dtype=np.int64), 1
        return index, n_effective
    if scheme == "distinct":
        _, inverse = np.unique(quantise_confidence(conf), return_inverse=True)
        inverse = inverse.ravel()
        return inverse, int(inverse.max()) + 1
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def signed_bin_gaps(conf: np.ndarray, correct: np.ndarray, scheme: str,
                    n_bins: int = 15) -> dict:
    """Per occupied bin: its size, mean confidence, accuracy and signed gap ``acc - conf``.

    Returns arrays over occupied bins only, in bin order, as a dict with keys ``n``,
    ``weight``, ``mean_confidence``, ``accuracy``, ``gap``.
    """
    conf, correct = _validate(conf, correct)
    index, n_slots = bin_assignment(conf, scheme, n_bins)
    counts = np.bincount(index, minlength=n_slots).astype(np.float64)
    sum_correct = np.bincount(index, weights=correct, minlength=n_slots)
    sum_conf = np.bincount(index, weights=conf, minlength=n_slots)
    occupied = counts > 0
    n = counts[occupied]
    acc = sum_correct[occupied] / n
    mc = sum_conf[occupied] / n
    return {"n": n.astype(np.int64), "weight": n / conf.size, "mean_confidence": mc,
            "accuracy": acc, "gap": acc - mc}


def ece_from_gaps(gaps: dict) -> float:
    """``sum_b w_b |g_b|``: the ECE of the binning that produced ``gaps``."""
    return float(np.sum(gaps["weight"] * np.abs(gaps["gap"])))


def ece_distinct(conf: np.ndarray, correct: np.ndarray) -> float:
    """ECE with one group per distinct confidence value: no binning choice at all."""
    return ece_from_gaps(signed_bin_gaps(conf, correct, "distinct"))


def overconfidence_summary(gaps: dict, tol: float = 0.0) -> dict:
    """How much of a binning is overconfident (signed gap below ``-tol``).

    Returns the number of overconfident bins, the share of samples in them, the largest
    overconfident gap as a positive number (0 when there is none), and the total weighted
    overconfidence ``sum_b w_b max(0, -g_b)``, which is exactly half of
    ``ECE - |a - p_bar|`` when ``a > p_bar``.
    """
    over = gaps["gap"] < -tol
    return {
        "n_bins": int(gaps["gap"].size),
        "n_overconfident_bins": int(over.sum()),
        "share_in_overconfident_bins": float(gaps["weight"][over].sum()),
        "largest_overconfident_gap": float(-gaps["gap"][over].min()) if over.any() else 0.0,
        "weighted_overconfidence": float(np.sum(gaps["weight"] * np.maximum(0.0, -gaps["gap"]))),
    }


def floor_ratio(accuracy: float, cap: float, ece: float) -> Optional[float]:
    """``(a - p*) / ECE`` where the cap binds (``a > p*``), else ``None``.

    By the inequality in the module docstring it never exceeds 1 for a count readout.
    """
    if accuracy <= cap:
        return None
    if ece <= 0.0:
        raise ValueError("ECE is zero although accuracy exceeds the cap; the inputs are inconsistent")
    return (accuracy - cap) / ece
