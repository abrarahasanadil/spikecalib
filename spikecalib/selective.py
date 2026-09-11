"""Selective prediction: what can be acted on, and at what risk.

A calibration number says how well confidence is scaled. A selective-prediction number
says something an engineer can use directly: *if I only act when the model is confident,
how much can I act on and how often will I be wrong?*

**The discreteness of the readout bites hardest here, and it is a result in its own
right.** Risk-coverage asks for accuracy on the most-confident ``c`` fraction of samples.
That requires ordering samples by confidence -- but a spike-count readout produces very few
distinct confidence values, and samples sharing a value cannot be partially ordered. There
is no principled way to take "half of the samples at confidence 0.4509": they are
indistinguishable to the model. So **coverage does not vary continuously. It steps, once
per distinct confidence level.** With 9 distinct levels at ``T = 2`` there are 9 achievable
operating points across 10,000 test samples, not 10,000. A practitioner who wants to
deploy at 60% coverage may find that only 47% and 82% exist.

Everything here follows from that. The functions report the achievable set explicitly
rather than interpolating a smooth curve through points that cannot be reached.

**Calibration mostly cannot change any of it, and the exception matters.** The natural
claim is that every calibrator is a monotone map from confidence to confidence, so none can
reorder samples and all of risk, coverage and AURC are invariant. That is true of two of
the four here and **false of a third**:

- ``Uncalibrated`` and ``IsotonicTopLabel`` are genuinely functions of the scalar top-label
  confidence, ``conf' = f(conf)`` with ``f`` monotone. They cannot reorder. Isotonic can
  *merge* levels -- it maps distinct confidences onto equal outputs -- so the achievable set
  can only get coarser, never finer, and AURC can only rise slightly as operating points
  disappear.
- ``TemperatureScaling`` is a function of the whole logit vector, not of ``conf``, so it is
  not guaranteed to preserve the order. Measured on this study's data it effectively does:
  Kendall tau 0.996-0.9997 against the raw ranking and AURC moving by under 3e-6.
- ``IsotonicOneVsRest`` is **not a function of the top-label confidence at all**. It
  recalibrates every class probability and renormalises, so two samples sharing a top-label
  confidence can come out in either order. It reorders heavily and it costs real
  performance: ``[MEASURED]`` at ``T = 2``, Kendall tau **0.28** against the raw ranking and
  AURC **0.0117 against 0.0023**, five times worse. It also inflates the number of
  achievable operating points from 10 to 185, which looks like a gain and is not -- the
  extra levels come from class probabilities unrelated to the decision being made.

So "calibration cannot change selective prediction" holds for **top-label** calibrators
only, and the one-vs-rest formulation should not be used when the downstream task is
selective prediction. The tests assert each of these separately rather than the blanket
claim.

What calibration *does* change, for the top-label calibrators, is the ability to pick an
operating point by confidence **value** ("act when confidence exceeds 0.9") rather than by
**rank** ("act on the top 40%"). Before calibration the value is not interpretable as a
probability; after it, it is. :func:`threshold_for_coverage` reports both.

One consequence runs the other way and is worth stating: calibration can only **coarsen**
the achievable set. Isotonic regression maps distinct raw confidences onto equal calibrated
ones, which removes operating points; it can never create one, because creating a boundary
would require reordering. So a practitioner gains an interpretable threshold and loses
operating points, and at low ``T`` there were few to begin with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .calibration import quantise_confidence

__all__ = [
    "RiskCoverage",
    "risk_coverage",
    "aurc",
    "optimal_aurc",
    "normalised_excess_aurc",
    "coverage_at_risk",
    "n_achievable_coverage_levels",
    "achievable_coverage",
    "threshold_for_coverage",
    "selective_summary",
]


@dataclass(frozen=True)
class RiskCoverage:
    """The achievable operating points of a selective classifier.

    ``coverage`` and ``risk`` have one entry per achievable point, which is one per
    distinct confidence level, not one per sample.
    """

    coverage: np.ndarray
    risk: np.ndarray
    n_selected: np.ndarray
    threshold: np.ndarray
    n_levels: int
    n_samples: int

    @property
    def accuracy(self) -> np.ndarray:
        return 1.0 - self.risk

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage.tolist(),
            "risk": self.risk.tolist(),
            "n_selected": self.n_selected.tolist(),
            "threshold": self.threshold.tolist(),
            "n_levels": self.n_levels,
            "n_samples": self.n_samples,
        }


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> RiskCoverage:
    """Risk against coverage, evaluated only at achievable operating points.

    Samples are ordered by confidence descending and the error rate is computed on the
    most-confident prefix. **A prefix may only end at a boundary between distinct
    confidence values**, because samples sharing a confidence are indistinguishable to the
    model and splitting them would report an operating point nobody can deploy.

    The returned arrays therefore have one entry per distinct confidence level, ascending
    in coverage, and the final entry is always full coverage with risk equal to the overall
    error rate.
    """
    conf = quantise_confidence(np.asarray(conf, dtype=np.float64).ravel())
    correct = np.asarray(correct).ravel().astype(np.float64)
    if conf.shape != correct.shape:
        raise ValueError(f"conf shape {conf.shape} != correct shape {correct.shape}")
    if conf.size == 0:
        raise ValueError("cannot compute risk-coverage on an empty array")

    n = conf.size
    order = np.argsort(-conf, kind="stable")
    c_sorted = conf[order]
    ok_sorted = correct[order]
    cum_correct = np.cumsum(ok_sorted)

    # A boundary exists where the next sample has a strictly lower confidence, plus the end.
    boundary = np.flatnonzero(np.diff(c_sorted) < 0)
    idx = np.append(boundary, n - 1)

    n_selected = idx + 1
    coverage = n_selected / n
    risk = 1.0 - cum_correct[idx] / n_selected
    return RiskCoverage(
        coverage=coverage, risk=risk, n_selected=n_selected,
        threshold=c_sorted[idx], n_levels=int(np.unique(conf).size), n_samples=int(n),
    )


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve, integrated over the achievable points.

    Trapezoidal between achievable points and anchored at the first one, because coverage
    below the smallest achievable value is not reachable. Lower is better.
    """
    rc = risk_coverage(conf, correct)
    x, y = rc.coverage, rc.risk
    if x.size == 1:
        return float(y[0])
    area = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
    return (area + float(y[0]) * float(x[0])) / float(x[-1])


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of an oracle that ranks every correct sample above every incorrect one.

    The floor for a given accuracy. Reported alongside the measured AURC because the raw
    number is otherwise uninterpretable: an AURC of 0.01 is excellent at 90% accuracy and
    poor at 99.5%.
    """
    correct = np.asarray(correct).ravel().astype(np.float64)
    n = correct.size
    if n == 0:
        raise ValueError("cannot compute optimal AURC on an empty array")
    # Oracle ordering: all correct first. Confidence is a perfect ranker, so every sample
    # is its own achievable level.
    oracle = np.concatenate([np.ones(int(correct.sum())), np.zeros(n - int(correct.sum()))])
    conf = np.linspace(1.0, 0.0, n)
    return aurc(conf, oracle)


def normalised_excess_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """How far above the oracle floor the measured AURC sits, as a fraction of the gap.

    0 means the ranking is as good as possible at this accuracy; 1 means it is no better
    than the worst ordering. Scale-free, so it is comparable across ``T`` where raw AURC
    is not.
    """
    measured = aurc(conf, correct)
    best = optimal_aurc(correct)
    correct = np.asarray(correct).ravel().astype(np.float64)
    n = correct.size
    # The anti-oracle ranks every incorrect sample above every correct one: the ceiling of
    # the gap the normalisation divides by.
    anti = np.concatenate([np.zeros(n - int(correct.sum())), np.ones(int(correct.sum()))])
    worst = aurc(np.linspace(1.0, 0.0, n), anti)
    if worst - best <= 0:
        return 0.0
    return float((measured - best) / (worst - best))


def coverage_at_risk(
    conf: np.ndarray, correct: np.ndarray, targets: Sequence[float] = (0.01, 0.005, 0.001)
) -> dict:
    """Largest achievable coverage whose risk stays at or below each target.

    The practitioner's question: *if I need 99% accuracy on what I act on, what fraction
    can I act on?* Returns ``None`` for a target no achievable operating point meets --
    which is the honest answer, not zero, because zero would suggest a reachable point with
    zero coverage.
    """
    rc = risk_coverage(conf, correct)
    out = {}
    for target in targets:
        ok = rc.risk <= target
        if not ok.any():
            out[float(target)] = None
        else:
            i = int(np.flatnonzero(ok)[np.argmax(rc.coverage[ok])])
            out[float(target)] = {
                "coverage": float(rc.coverage[i]),
                "risk": float(rc.risk[i]),
                "n_selected": int(rc.n_selected[i]),
                "threshold": float(rc.threshold[i]),
            }
    return out


def n_achievable_coverage_levels(conf: np.ndarray) -> int:
    """How many distinct operating points exist at all.

    Equal to the number of distinct confidence values. At ``T = 2`` this is single digits
    across ten thousand samples, so a practitioner cannot choose an arbitrary coverage.
    """
    return int(np.unique(quantise_confidence(np.asarray(conf, dtype=np.float64))).size)


def achievable_coverage(conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """The achievable coverage values themselves, ascending."""
    return risk_coverage(conf, correct).coverage


def threshold_for_coverage(conf: np.ndarray, correct: np.ndarray, target: float) -> dict:
    """Pick an operating point by rank, and report the confidence value it corresponds to.

    This is where calibration earns its keep. The *rank* selection is invariant under any
    monotone calibrator; the *threshold value* is not, and only after calibration is that
    value interpretable as a probability. Both are returned so the difference is visible.
    """
    rc = risk_coverage(conf, correct)
    i = int(np.argmin(np.abs(rc.coverage - target)))
    return {
        "requested_coverage": float(target),
        "achieved_coverage": float(rc.coverage[i]),
        "coverage_shortfall": float(rc.coverage[i] - target),
        "risk": float(rc.risk[i]),
        "threshold_confidence": float(rc.threshold[i]),
        "n_selected": int(rc.n_selected[i]),
        "n_achievable_levels": rc.n_levels,
    }


def selective_summary(
    conf: np.ndarray, correct: np.ndarray, targets: Sequence[float] = (0.01, 0.005, 0.001)
) -> dict:
    """Every selective-prediction number for one (T, seed, arm), in one pass."""
    rc = risk_coverage(conf, correct)
    car = coverage_at_risk(conf, correct, targets)
    out = {
        "aurc": aurc(conf, correct),
        "aurc_optimal": optimal_aurc(correct),
        "aurc_excess_normalised": normalised_excess_aurc(conf, correct),
        "n_achievable_coverage_levels": rc.n_levels,
        "achievable_coverage": rc.coverage.tolist(),
        "achievable_risk": rc.risk.tolist(),
    }
    for target, hit in car.items():
        key = f"coverage_at_risk_{target:g}".replace(".", "p")
        out[key] = None if hit is None else hit["coverage"]
    return out
