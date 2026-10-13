"""Derived results about bounded logits: the cap, the scale, and the number of operating points.

Everything in this module is ``[DERIVED]``. Nothing here is measured and nothing is taken
from a paper; each function states the argument it rests on, and ``tests/test_theory.py``
checks every one of them against brute-force enumeration or an identity.

The thesis these results separate is:

**The confidence cap is a scale problem.** A count readout's cap ``e^T/(e^T + C - 1)`` is the
``s = 1`` case of ``e^(sT)/(e^(sT) + C - 1)``. What the cap depends on is the *width* of the
interval the logits live in, ``W = sT``, and not on ``T`` alone. One scalar moves it.

**Resolution is a timestep problem.** The number of distinct confidence values a count
readout can emit is at most ``binomial(T + C - 1, C - 1)`` for *every* ``s > 0``. No scale
changes it, because a scale changes the values and not how many there are.

The second half of this module is the measurement that goes with the second claim: a Murphy
decomposition of the top-label Brier score that needs no binning choice when the confidence
takes few distinct values, and the binned form with its two within-bin terms when it does
not.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .calibration import _equal_mass_bins, _validate, quantise_confidence

__all__ = [
    "bounded_logit_ceiling",
    "scaled_ceiling",
    "rate_ceiling",
    "loss_floor",
    "min_logit_width",
    "min_scale",
    "max_distinct_confidences",
    "distinct_confidence_bound_unreduced",
    "murphy_exact",
    "murphy_binned",
]


# ---------------------------------------------------------------------------------------
# 1a. The cap, for any logits bounded in an interval
# ---------------------------------------------------------------------------------------


def bounded_logit_ceiling(width: float, n_classes: int) -> float:
    """Largest softmax probability when every logit lies in an interval of width ``W``.

    [DERIVED] Write the winner's probability as ``1 / (1 + sum_{j != w} e^(z_j - z_w))``.
    Every logit lies in ``[a, a + W]``, so ``z_j - z_w >= -W`` for each of the ``C - 1``
    competitors, the sum is at least ``(C - 1) e^(-W)``, and::

        p_max <= 1 / (1 + (C - 1) e^(-W)) = e^W / (e^W + C - 1)

    with equality exactly when the winner sits at the top of the interval and every
    competitor at the bottom. The interval's position ``a`` does not appear: softmax is
    invariant to adding a constant to every logit.

    Nothing in the argument is specific to spikes. It covers clipped logits, quantised
    logits with a finite code range, and logits squashed by a bounded activation, in any
    network.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; got {n_classes}")
    if width < 0:
        raise ValueError(f"width must be non-negative; got {width}")
    # 1 / (1 + (C-1) e^-W) rather than e^W / (e^W + C-1): the second overflows near W = 710.
    return float(1.0 / (1.0 + (n_classes - 1) * math.exp(-width)))


def scaled_ceiling(timesteps: float, n_classes: int, scale: float = 1.0) -> float:
    """Cap of a count readout whose logits are ``z_j = s * c_j`` with ``c_j`` in ``[0, T]``.

    [DERIVED] The logits lie in ``[0, sT]``, so this is :func:`bounded_logit_ceiling` at
    ``W = sT``. ``scale = 1`` is the sum convention and recovers ``e^T/(e^T + C - 1)``;
    ``scale = 1/T`` is the rate convention and gives :func:`rate_ceiling`.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive; got {scale}")
    return bounded_logit_ceiling(scale * timesteps, n_classes)


def rate_ceiling(n_classes: int) -> float:
    """Cap of a softmax over firing rates, ``z_j = c_j / T``. It does not depend on ``T``.

    [DERIVED] A rate lies in ``[0, 1]`` whatever ``T`` is, so ``W = 1`` and the cap is
    ``e / (e + C - 1)``: 0.2320 at ten classes, 0.0265 at a hundred and one. Adding
    timesteps to a rate readout adds resolution and no confidence.
    """
    return bounded_logit_ceiling(1.0, n_classes)


def loss_floor(width: float, n_classes: int) -> float:
    """Smallest cross-entropy reachable with logits in an interval of width ``W``.

    [DERIVED] ``-ln`` of :func:`bounded_logit_ceiling`, since the probability of the true
    class cannot exceed the cap: ``ln(1 + (C - 1) e^(-W))``.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; got {n_classes}")
    return float(math.log1p((n_classes - 1) * math.exp(-width)))


# ---------------------------------------------------------------------------------------
# 1b. The minimum width, and the scale that buys it
# ---------------------------------------------------------------------------------------


def min_logit_width(n_classes: int, target_accuracy: float) -> float:
    """Smallest interval width at which confidence ``a`` is reachable.

    [DERIVED] Solve ``e^W / (e^W + C - 1) >= a`` for ``W``::

        W >= ln( (C - 1) a / (1 - a) )

    This is the quantity the paper calls ``T_min``. It is a requirement on the width
    ``W = sT``, and reads as a number of timesteps only under the convention ``s = 1``.
    """
    if not (0.0 < target_accuracy < 1.0):
        raise ValueError(f"target_accuracy must lie strictly in (0, 1); got {target_accuracy}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; got {n_classes}")
    return float(math.log((n_classes - 1) * target_accuracy / (1.0 - target_accuracy)))


def min_scale(timesteps: float, n_classes: int, target_accuracy: float) -> float:
    """Smallest logit scale at which a ``T``-step count readout can express accuracy ``a``.

    [DERIVED] ``s_min = ln((C - 1) a / (1 - a)) / T``, from ``sT >= W_min``. The constraint
    is on the product: halving ``T`` and doubling ``s`` leaves the cap where it was. Can be
    non-positive when the target is below ``1/C``, meaning any positive scale suffices.
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive; got {timesteps}")
    return min_logit_width(n_classes, target_accuracy) / float(timesteps)


# ---------------------------------------------------------------------------------------
# 1c. How many distinct confidences a count readout can emit
# ---------------------------------------------------------------------------------------


def max_distinct_confidences(timesteps: int, n_classes: int) -> int:
    """Upper bound on the distinct top-label confidences of a count readout, for any scale.

    [DERIVED] The top-label confidence of a count vector ``c`` under scale ``s`` is
    ``1 / (1 + sum_{j != w} e^(s (c_j - m)))`` with ``m = max_j c_j``. It depends only on
    the multiset of the ``C - 1`` gaps ``m - c_j``, each an integer in ``[0, T]``. The
    number of multisets of size ``C - 1`` drawn from ``T + 1`` values is::

        binomial(T + C - 1, C - 1)

    55 at ``T = 2, C = 10``. The scale does not enter: it changes each value and not how
    many there are. The bound is attained for every ``s`` at which ``e^s`` is transcendental,
    which includes ``s = 1`` and every other non-zero algebraic ``s``, because distinct
    multisets then give distinct integer polynomials in ``e^(-s)`` (checked by enumeration
    in the tests, not relied on).

    Counting by the winning count ``m`` and the multiset of the other counts, without
    reducing to gaps, gives the larger ``binomial(T + C, C)``; see
    :func:`distinct_confidence_bound_unreduced`. That is still an upper bound, but it counts
    vectors that differ by a constant shift separately, and softmax cannot tell them apart.
    """
    if timesteps < 0 or n_classes < 2:
        raise ValueError(f"need timesteps >= 0 and n_classes >= 2; got {timesteps}, {n_classes}")
    return math.comb(timesteps + n_classes - 1, n_classes - 1)


def distinct_confidence_bound_unreduced(timesteps: int, n_classes: int) -> int:
    """The looser count: winning count ``m`` times multisets of the others in ``[0, m]``.

    [DERIVED] ``sum_{m=0}^{T} binomial(m + C - 1, C - 1) = binomial(T + C, C)`` by the
    hockey-stick identity; 66 at ``T = 2, C = 10``. Valid as an upper bound and never tight
    for ``T >= 1``: the vectors ``c`` and ``c + k`` are counted separately and have the same
    softmax. Removing those duplicates telescopes the sum to
    :func:`max_distinct_confidences`. Kept so the relation between the two is tested rather
    than asserted.
    """
    if timesteps < 0 or n_classes < 2:
        raise ValueError(f"need timesteps >= 0 and n_classes >= 2; got {timesteps}, {n_classes}")
    return sum(math.comb(m + n_classes - 1, n_classes - 1) for m in range(timesteps + 1))


# ---------------------------------------------------------------------------------------
# 1d. The Murphy decomposition of the top-label Brier score
# ---------------------------------------------------------------------------------------


def _decompose(conf: np.ndarray, correct: np.ndarray, group: np.ndarray) -> dict:
    """Brier = REL - RES + UNC + WBV - WBC over an arbitrary grouping of the samples.

    [DERIVED] Inside one group, write each term of ``(f - o)^2`` as
    ``((f - fbar) - (o - obar) + (fbar - obar))^2``. The two cross terms against the
    constant ``fbar - obar`` vanish on summing over the group, leaving

        sum (f - o)^2 = n_k (fbar - obar)^2 + sum (o - obar)^2
                        + sum (f - fbar)^2 - 2 sum (f - fbar)(o - obar)

    and ``sum_k sum (o - obar_k)^2 / n = UNC - RES`` is the usual within/between split of the
    outcome variance. So with

        REL = sum_k (n_k/n) (fbar_k - obar_k)^2     reliability, lower is better
        RES = sum_k (n_k/n) (obar_k - obar)^2       resolution, higher is better
        UNC = obar (1 - obar)                       uncertainty, a property of the task
        WBV = (1/n) sum_k sum (f - fbar_k)^2        within-group forecast variance
        WBC = (2/n) sum_k sum (f - fbar_k)(o - obar_k)   within-group covariance

    the identity ``Brier = REL - RES + UNC + WBV - WBC`` holds for any grouping. When every
    group holds one forecast value, WBV and WBC are zero and it is Murphy's three-term
    decomposition with no approximation.
    """
    n = conf.size
    obar = float(correct.mean())
    rel = res = wbv = wbc = ece = 0.0
    labels = np.unique(group)
    for g in labels:
        m = group == g
        w = m.sum() / n
        f, o = conf[m], correct[m]
        fb, ob = float(f.mean()), float(o.mean())
        rel += w * (fb - ob) ** 2
        res += w * (ob - obar) ** 2
        ece += w * abs(fb - ob)
        wbv += float(((f - fb) ** 2).sum()) / n
        wbc += 2.0 * float(((f - fb) * (o - ob)).sum()) / n
    return {
        "brier": float(((conf - correct) ** 2).mean()),
        "reliability": rel,
        "resolution": res,
        "uncertainty": obar * (1.0 - obar),
        "within_bin_variance": wbv,
        "within_bin_covariance": wbc,
        "ece": ece,
        "n_groups": int(labels.size),
        "n": int(n),
    }


def murphy_exact(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Decomposition with one group per distinct confidence value. No binning choice.

    Exact in the sense that ``brier == reliability - resolution + uncertainty`` to float
    precision, and ``ece`` is the calibration error with every sample compared against the
    accuracy of the samples that share its confidence exactly.

    **Meaningful only when distinct values are few relative to the sample size.** With one
    sample per value every group's accuracy is 0 or 1, resolution equals uncertainty, and
    reliability is the Brier score: the decomposition is still an identity and says nothing.
    ``samples_per_value`` is returned so a caller can see which regime it is in. A count
    readout at low ``T`` is in the first; an unbounded readout with ten thousand distinct
    confidences on ten thousand samples is in the second and must be binned.
    """
    conf, correct = _validate(conf, correct)
    _, group = np.unique(quantise_confidence(conf), return_inverse=True)
    out = _decompose(conf, correct, group)
    out["estimator"] = "exact"
    out["samples_per_value"] = out["n"] / out["n_groups"]
    return out


def murphy_binned(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15,
                  scheme: str = "equal_width") -> dict:
    """Decomposition over the same bins the two ECE estimators use.

    ``scheme`` is ``"equal_width"`` or ``"equal_mass"``, with the bin assignment taken from
    the code path of :func:`spikecalib.calibration.ece_equal_width` and
    :func:`spikecalib.calibration.ece_equal_mass`, so ``ece`` here equals those functions'
    output and a resolution reported beside a binned ECE refers to the same bins.

    The three-term identity does not hold under binning; the five-term one does. Resolution
    measured this way is a property of the bins as well as of the forecasts, so **two
    readouts' resolutions are comparable only under the same scheme and bin count**.
    """
    conf, correct = _validate(conf, correct)
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        group = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    elif scheme == "equal_mass":
        idx: Optional[np.ndarray]
        idx, _ = _equal_mass_bins(conf, n_bins)
        group = np.zeros(conf.size, dtype=np.int64) if idx is None else idx
    else:
        raise ValueError(f"scheme must be 'equal_width' or 'equal_mass'; got {scheme!r}")
    out = _decompose(conf, correct, group)
    out["estimator"] = scheme
    out["n_bins_requested"] = int(n_bins)
    return out
