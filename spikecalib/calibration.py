"""Calibrators, the two ECE estimators, and seeded tie-breaking.

This module is the contribution. Everything else in the repository is scaffolding that
gets data into it and results out of it.

Owns correctness rule 2: **both equal-width (Guo et al., ICML 2017; ``[LIT]`` for the
method, ``[UNVERIFIED]`` for the 15-bin default attributed to it -- SOURCES.md item 2)
and equal-mass ECE are computed
and both are reported.** :func:`compute_metrics` returns them together in one object, so
there is no code path that produces one without the other. With discrete spike counts the
confidence distribution is lumpy -- only finitely many softmax outputs are reachable from
integer counts in ``[0, T]`` -- and equal-width binning then averages over wide intervals
in which positive and negative calibration gaps cancel, biasing the estimate
**optimistically**. Equal-mass binning adapts its edges to where the data actually is.
Disagreement between the two is itself a finding, and :attr:`CalibrationMetrics.ece_disagreement`
exists so it can be reported rather than quietly resolved.

Owns the tie-breaking decision: **argmax ties are broken uniformly at random under the
experiment seed, never by lowest index.** ``np.argmax`` silently returns the lowest index,
which biases every tie toward class 0; that bias grows as ``T`` falls, because fewer
attainable count values means more collisions -- so it would contaminate precisely the
trend under test. :func:`predict_with_ties` therefore does not use ``np.argmax`` on the
counts. The tie rate is a **reported quantity**, but it is a descriptive measure of how
discrete the output is, *not* the mechanism by which isotonic beats temperature scaling --
neither calibrator breaks ties. See ``NOTES.md``, "What tie rate is, and what it is not".

The mechanism the study argues for is **capacity**: temperature scaling has one parameter
and can only rescale the whole confidence curve, while isotonic has as many effective
degrees of freedom as the calibration set has distinct confidence levels (fewer, after
monotonicity pooling). At low ``T`` there are few levels and many samples supporting each
one, so the usual variance objection to isotonic weakens exactly where SNNs are
energy-competitive.

Owns the calibrator-fitting half of correctness rule 4: **fitting runs on CPU with
deterministic libraries only** -- scikit-learn for the isotonic fits, ``scipy.optimize``
for the one-dimensional temperature search. No torch, no MPS, no global RNG. Fitted
calibrators are therefore reproducible bit-for-bit even though trained weights are not.

Confidence is the softmax over the integer per-class spike counts, which ``NOTES.md``
fixes as the logit. Every calibrator operates on that same quantity, so the comparison
isolates the calibrator rather than the readout.

**A note on the calibrator interface.** The common contract is
``fit(cal_conf, cal_correct)`` / ``transform(conf)``, which is all the top-label
calibrators need. Two of the four need more than that and say so rather than guessing:
temperature scaling is defined on logits and true labels, since a scalar cannot be
recovered from a top-label confidence alone, and one-vs-rest needs the full probability
matrix. They take it through the keyword-only ``outputs`` argument and raise if it is
absent.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

__all__ = [
    "softmax_confidence",
    "quantise_confidence",
    "CONFIDENCE_DECIMALS",
    "softmax_probs",
    "predict_with_ties",
    "tie_rate",
    "discrimination",
    "Outputs",
    "model_outputs",
    "Calibrator",
    "Uncalibrated",
    "TemperatureScaling",
    "IsotonicTopLabel",
    "IsotonicOneVsRest",
    "CALIBRATORS",
    "ece_equal_width",
    "ece_equal_mass",
    "compute_metrics",
    "bootstrap_ece_difference",
    "CalibrationMetrics",
    "DegenerateBinningError",
]

# Guarding log(0) in NLL. Small enough not to move a reported figure, large enough that a
# confidently wrong sample gives a finite loss instead of inf.
_EPS = 1e-12

# Decimal places confidences are rounded to before they are compared for equality.
#
# This is not cosmetic. Two samples whose spike counts are the same multiset -- say
# {2, 1, 0, ..., 0} -- have mathematically identical confidence, but if the winning class
# sits in a different position the softmax sums its terms in a different order and float64
# returns values differing by ~5e-17. Left alone that inflates the distinct-confidence
# count, which is the statistic the capacity hypothesis is stated in, and it lets
# _equal_mass_bins split a genuine tie across two bins -- the one thing that function is
# documented never to do.
#
# 12 decimals sits far above the noise (~1e-16 relative) and far below the smallest gap
# between genuinely different attainable confidences. The closest two attainable values can
# be is set by changing one class's count by one, which moves the softmax denominator by at
# least e/(e^T + C - 1); for C=10 that is ~8e-5 even at T=10. Twelve decimals is seven
# orders of magnitude of headroom.
CONFIDENCE_DECIMALS = 12


def quantise_confidence(conf: np.ndarray) -> np.ndarray:
    """Round confidences so that mathematically equal values compare equal.

    See :data:`CONFIDENCE_DECIMALS` for why this is necessary rather than fastidious.
    """
    return np.round(np.asarray(conf, dtype=np.float64), CONFIDENCE_DECIMALS)


class DegenerateBinningError(ValueError):
    """Raised when a binned ECE cannot form more than one distinct bin.

    At low ``T`` the attainable confidence set is small, and it can collapse to a single
    value entirely. That is a real property of the regime under study, not a numerical
    accident, so it must be visible. What it must never do is silently produce ECE = 0:
    a single bin still has a genuine gap ``|accuracy - confidence|``, and reporting zero
    there would claim perfect calibration for the most degenerate case in the sweep.
    """


# ---------------------------------------------------------------------------------------
# Readout: counts -> probabilities, predictions, ties
# ---------------------------------------------------------------------------------------


def softmax_probs(counts: np.ndarray) -> np.ndarray:
    """Full softmax distribution over integer per-class spike counts.

    The per-class spike count over ``T`` timesteps is the logit (``NOTES.md``, resolved
    design decisions), so this is an ordinary softmax -- the point of the study is that
    the *input* lives on an integer lattice, not that the link function is unusual.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D [n_samples, n_classes], got shape {counts.shape}")
    shifted = counts - counts.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_confidence(counts: np.ndarray) -> np.ndarray:
    """Top-label confidence: the largest softmax probability for each sample.

    Note this is the max over the *distribution*, which is well defined even when the
    argmax is tied -- two classes tied on counts share the same probability, and the
    confidence is that shared value. Which of them is predicted is
    :func:`predict_with_ties`' problem, not this function's.

    The result is quantised (see :data:`CONFIDENCE_DECIMALS`) so that two samples with the
    same multiset of counts get byte-identical confidence regardless of which class held
    the maximum. Without that, summation order alone splits one attainable confidence level
    into several, and the number of levels is exactly what the study measures.
    """
    return quantise_confidence(softmax_probs(counts).max(axis=1))


def _max_mask(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D [n_samples, n_classes], got shape {counts.shape}")
    return counts == counts.max(axis=1, keepdims=True)


def predict_with_ties(
    counts: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax with ties broken **uniformly at random** under the experiment seed.

    Returns ``(predictions, tie_mask)`` where ``tie_mask[i]`` is True when sample ``i``
    had two or more classes sharing the maximum count.

    This deliberately does not call ``np.argmax`` on the counts. ``np.argmax`` resolves
    ties by lowest index, which sends every tie to the lowest-numbered class involved --
    a systematic bias toward class 0 that strengthens as ``T`` falls and ties multiply.
    Since the trend against ``T`` is exactly what the study measures, that bias would
    manufacture part of the effect being tested.

    The implementation draws one uniform per (sample, class), masks out non-maximal
    entries, and takes the argmax of the draws. Among the tied classes the largest draw
    wins, which is uniform over the tied set; the masking guarantees a non-maximal class
    can never be selected.
    """
    counts = np.asarray(counts)
    is_max = _max_mask(counts)
    keys = rng.random(counts.shape)
    # -1 is below every value default_rng().random() can return, so a masked-out class
    # cannot win even if every tied draw is 0.0.
    keys = np.where(is_max, keys, -1.0)
    predictions = keys.argmax(axis=1).astype(np.int64)
    tie_mask = is_max.sum(axis=1) > 1
    return predictions, tie_mask


def tie_rate(counts: np.ndarray) -> float:
    """Fraction of samples whose argmax is tied.

    A reported quantity that belongs in the results table. It rises as ``T`` falls, because
    fewer attainable count values means more collisions, and that rise is direct evidence
    that the output really is discrete -- it makes the premise of the capacity argument
    visible and measurable.

    It is **not** the reason isotonic outperforms temperature scaling, and it cannot be:
    neither calibrator breaks ties. Temperature is order-preserving, so a tie survives any
    temperature; isotonic top-label is a monotone function of the scalar confidence, so two
    samples with identical counts receive identical output from it too. Tie-breaking happens
    once, in :func:`predict_with_ties`, before either calibrator runs.
    """
    return float((_max_mask(counts).sum(axis=1) > 1).mean())


def discrimination(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Can confidence separate correct predictions from incorrect ones at all?

    AUROC of confidence as a score for the binary label "this prediction is right". 0.5
    means confidence carries no information about correctness; 1.0 means it orders them
    perfectly.

    This is prior to calibration, not part of it. A calibrator is a monotone map from
    confidence to a probability, and a monotone map cannot change the ordering of samples
    by confidence -- so it cannot change AUROC at all. Whatever separation exists here is
    the ceiling on what any calibrated confidence can express. If it is 0.5, the best
    possible calibrator outputs the base accuracy rate for every sample, which is a correct
    but useless model, and the interesting question stops being "which calibrator" and
    becomes "why is the readout uninformative".

    Also returns the mean confidence conditioned on correctness, whose gap is the same
    question in units a reader can interpret directly, and the coefficient of variation of
    confidence, which measures how much room a calibrator has to act differently on
    different samples.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)
    n_correct = int(correct.sum())
    n_wrong = int((~correct).sum())

    if n_correct == 0 or n_wrong == 0:
        auroc = float("nan")
        note = f"AUROC undefined: {n_correct:,} correct and {n_wrong:,} incorrect"
    elif np.unique(conf).size == 1:
        auroc = 0.5
        note = "confidence takes a single value, so it cannot order anything: AUROC is 0.5 by construction"
    else:
        auroc = float(roc_auc_score(correct, conf))
        note = ""

    mean_correct = float(conf[correct].mean()) if n_correct else float("nan")
    mean_wrong = float(conf[~correct].mean()) if n_wrong else float("nan")
    mean = float(conf.mean())
    std = float(conf.std(ddof=0))
    return {
        "auroc": auroc,
        "note": note,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "mean_conf_correct": mean_correct,
        "mean_conf_incorrect": mean_wrong,
        "gap": mean_correct - mean_wrong,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else float("nan"),
        "n_distinct": int(np.unique(conf).size),
    }


@dataclass(frozen=True)
class Outputs:
    """Everything one model run produces, on one split.

    Bundled because the calibrators need different subsets of it and passing four loose
    arrays around invites mismatched orderings.
    """

    counts: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    tie_mask: np.ndarray
    probs: np.ndarray
    conf: np.ndarray
    correct: np.ndarray

    @property
    def n(self) -> int:
        return int(self.labels.size)

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    @property
    def tie_rate(self) -> float:
        return float(self.tie_mask.mean())


def model_outputs(counts: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> Outputs:
    """Derive predictions, confidences and correctness from raw spike counts."""
    counts = np.asarray(counts)
    labels = np.asarray(labels, dtype=np.int64)
    if counts.shape[0] != labels.shape[0]:
        raise ValueError(f"counts has {counts.shape[0]} rows but labels has {labels.size}")
    probs = softmax_probs(counts)
    predictions, tie_mask = predict_with_ties(counts, rng)
    return Outputs(
        counts=counts,
        labels=labels,
        predictions=predictions,
        tie_mask=tie_mask,
        probs=probs,
        # Quantised for the same reason softmax_confidence is: indexing the probs matrix
        # inherits the summation-order noise, and Outputs.conf is what every calibrator and
        # every distinct-level count downstream consumes.
        conf=quantise_confidence(probs[np.arange(probs.shape[0]), predictions]),
        correct=(predictions == labels),
    )


# ---------------------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------------------


class Calibrator(ABC):
    """Common contract: ``fit(cal_conf, cal_correct)`` then ``transform(conf)``.

    ``outputs`` is keyword-only and optional at this level because the top-label
    calibrators genuinely do not need it. The ones that do raise a specific error when it
    is missing rather than falling back to something that would silently be a different
    method.
    """

    name: str = "calibrator"
    needs_outputs: bool = False

    @abstractmethod
    def fit(
        self,
        cal_conf: np.ndarray,
        cal_correct: np.ndarray,
        *,
        outputs: Optional[Outputs] = None,
    ) -> "Calibrator":
        ...

    @abstractmethod
    def transform(
        self, conf: np.ndarray, *, outputs: Optional[Outputs] = None
    ) -> np.ndarray:
        ...

    def _require(self, outputs: Optional[Outputs]) -> Outputs:
        if outputs is None:
            raise ValueError(
                f"{type(self).__name__} needs the full model outputs (counts, labels, "
                f"predictions), not just top-label confidence: pass outputs=... . "
                f"A scalar temperature cannot be recovered from top-label confidence "
                f"alone, and one-vs-rest needs the full probability matrix."
            )
        return outputs


class Uncalibrated(Calibrator):
    """Identity baseline: report the model's own confidence unchanged."""

    name = "uncalibrated"

    def fit(self, cal_conf, cal_correct, *, outputs=None) -> "Uncalibrated":
        return self

    def transform(self, conf, *, outputs=None) -> np.ndarray:
        return np.asarray(conf, dtype=np.float64)


class TemperatureScaling(Calibrator):
    """Single scalar temperature, NLL-minimised on the calibration split.

    Fitted on the **raw counts before softmax**, which is where temperature scaling is
    defined: it divides the logits by a scalar and re-softmaxes. Guo et al.'s method, and
    the baseline used for SNN uncertainty by Sun & Bohte (AOT-SNN).

    The claim under test lives here, and it is about **capacity**. One parameter and a
    rigid functional form means temperature can only rescale the entire confidence curve.
    Where miscalibration is a uniform offset that is sufficient; where miscalibration
    varies with confidence -- overconfident at the top of the range, underconfident at the
    bottom -- no single ``t`` corrects both ends, because the only available move applies
    everywhere at once.

    Two properties are worth stating precisely, because both are easy to overclaim.
    Dividing by a positive scalar is order-preserving, so temperature cannot change the
    predicted class or separate a tie -- but neither can isotonic top-label, so this does
    not distinguish the methods. And temperature is *not* trapped by the attainable-
    confidence ceiling of the raw readout: as ``t`` approaches 0 the softmax approaches
    one-hot, so sharpening reaches confidences the readout itself could never emit.

    Optimised with ``scipy.optimize.minimize_scalar`` over ``log(t)``, which keeps the
    temperature positive and makes the search scale-free. Bounded, deterministic, CPU.
    """

    name = "temperature"
    needs_outputs = True

    def __init__(self, bounds: tuple[float, float] = (1e-3, 1e3)) -> None:
        self.bounds = bounds
        self.temperature_: Optional[float] = None
        self.nll_: Optional[float] = None
        self.converged_: Optional[bool] = None

    def _nll(self, log_t: float, counts: np.ndarray, labels: np.ndarray) -> float:
        z = counts / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(z).sum(axis=1))
        return float(-(z[np.arange(labels.size), labels] - log_partition).mean())

    def fit(self, cal_conf, cal_correct, *, outputs=None) -> "TemperatureScaling":
        out = self._require(outputs)
        counts = np.asarray(out.counts, dtype=np.float64)
        labels = np.asarray(out.labels, dtype=np.int64)
        result = minimize_scalar(
            self._nll,
            args=(counts, labels),
            bounds=(np.log(self.bounds[0]), np.log(self.bounds[1])),
            method="bounded",
        )
        self.temperature_ = float(np.exp(result.x))
        self.nll_ = float(result.fun)
        self.converged_ = bool(result.success)
        if not self.converged_:
            warnings.warn(
                f"temperature search did not converge: {getattr(result, 'message', '')}",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def transform(self, conf, *, outputs=None) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaling.transform called before fit")
        out = self._require(outputs)
        scaled = softmax_probs(np.asarray(out.counts, dtype=np.float64) / self.temperature_)
        # The prediction is carried over from the seeded tie-break, not recomputed.
        # Temperature cannot change the argmax anyway; recomputing it here would silently
        # re-roll the tie-break and make the arms differ by more than the calibrator.
        return scaled[np.arange(scaled.shape[0]), out.predictions]


class IsotonicTopLabel(Calibrator):
    """PRIMARY formulation: isotonic regression from max-confidence to correctness.

    Fitted on the calibration split, mapping the model's top-label confidence to binary
    correctness of that top label. Primary because it matches what ECE measures: ECE
    compares top-label confidence against top-label accuracy, so fitting exactly that
    pairing is the like-for-like comparison against temperature scaling.

    Non-parametric and monotone, so a lumpy, discrete input distribution is native to the
    estimator rather than a violated assumption. ``out_of_bounds='clip'`` because test
    confidences can fall outside the calibration range, and clipping is the monotone
    extension -- extrapolating a step function would invent structure.

    This is where the capacity argument cashes out. The fit has at most one free value per
    distinct calibration confidence, and :attr:`n_levels_` reports how many survived the
    pool-adjacent-violators step -- the model's *realised* degrees of freedom. Comparing
    that against the calibration-set size gives the samples-per-level ratio the hypothesis
    is really about.
    """

    name = "isotonic-top-label"

    def __init__(self) -> None:
        self.model_: Optional[IsotonicRegression] = None

    def fit(self, cal_conf, cal_correct, *, outputs=None) -> "IsotonicTopLabel":
        cal_conf = np.asarray(cal_conf, dtype=np.float64)
        cal_correct = np.asarray(cal_correct, dtype=np.float64)
        if cal_conf.shape != cal_correct.shape:
            raise ValueError(
                f"cal_conf shape {cal_conf.shape} != cal_correct shape {cal_correct.shape}"
            )
        self.model_ = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        ).fit(cal_conf, cal_correct)
        return self

    def transform(self, conf, *, outputs=None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("IsotonicTopLabel.transform called before fit")
        return np.asarray(self.model_.transform(np.asarray(conf, dtype=np.float64)))

    @property
    def n_knots_(self) -> int:
        """Distinct calibration confidences the fit was built on: the capacity ceiling."""
        if self.model_ is None:
            raise RuntimeError("IsotonicTopLabel.n_knots_ accessed before fit")
        return int(np.asarray(self.model_.X_thresholds_).size)

    @property
    def n_levels_(self) -> int:
        """Distinct output values after monotonicity pooling: the realised capacity.

        Pool-adjacent-violators merges neighbouring knots that would violate monotonicity,
        so this is at most :attr:`n_knots_` and is usually far smaller. It is the honest
        parameter count for the fitted step function.
        """
        if self.model_ is None:
            raise RuntimeError("IsotonicTopLabel.n_levels_ accessed before fit")
        return int(np.unique(np.asarray(self.model_.y_thresholds_)).size)


class IsotonicOneVsRest(Calibrator):
    """SECONDARY formulation: per-class isotonic, then renormalised.

    Included for comparability with Guo et al., who report this formulation, so a reader
    cross-referencing that paper finds the number they expect. One isotonic fit per class
    maps that class's probability to the indicator of it being the true class; the
    calibrated columns are then renormalised to sum to one.

    The reported confidence is the calibrated probability of the **originally predicted**
    class. Renormalised one-vs-rest can move the argmax, and letting it do so would change
    accuracy between arms, so the arms would no longer differ only by the calibrator.
    Holding the prediction fixed keeps the comparison about calibration.
    """

    name = "isotonic-one-vs-rest"
    needs_outputs = True

    def __init__(self) -> None:
        self.models_: list[IsotonicRegression] = []
        self.n_classes_: Optional[int] = None

    def fit(self, cal_conf, cal_correct, *, outputs=None) -> "IsotonicOneVsRest":
        out = self._require(outputs)
        probs = np.asarray(out.probs, dtype=np.float64)
        labels = np.asarray(out.labels, dtype=np.int64)
        self.n_classes_ = probs.shape[1]
        self.models_ = []
        for k in range(self.n_classes_):
            target = (labels == k).astype(np.float64)
            self.models_.append(
                IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
                ).fit(probs[:, k], target)
            )
        return self

    def transform(self, conf, *, outputs=None) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("IsotonicOneVsRest.transform called before fit")
        out = self._require(outputs)
        probs = np.asarray(out.probs, dtype=np.float64)
        if probs.shape[1] != self.n_classes_:
            raise ValueError(
                f"fitted for {self.n_classes_} classes but given {probs.shape[1]}"
            )
        calibrated = np.column_stack(
            [self.models_[k].transform(probs[:, k]) for k in range(self.n_classes_)]
        )
        totals = calibrated.sum(axis=1, keepdims=True)
        # Isotonic can map every class of a sample to 0, leaving nothing to renormalise.
        # Falling back to uniform is the honest answer for "the calibrator has no opinion",
        # and it keeps the row a probability distribution.
        degenerate_rows = totals[:, 0] <= 0
        if degenerate_rows.any():
            calibrated[degenerate_rows] = 1.0 / self.n_classes_
            totals = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / totals
        return calibrated[np.arange(calibrated.shape[0]), out.predictions]


CALIBRATORS: dict[str, type[Calibrator]] = {
    "uncalibrated": Uncalibrated,
    "temperature": TemperatureScaling,
    "isotonic-top-label": IsotonicTopLabel,
    "isotonic-one-vs-rest": IsotonicOneVsRest,
}


# ---------------------------------------------------------------------------------------
# Calibration error: two estimators, always together
# ---------------------------------------------------------------------------------------


def _bin_ece(conf: np.ndarray, correct: np.ndarray, bin_index: np.ndarray, n_bins: int) -> tuple[float, int]:
    """Weighted mean absolute gap between confidence and accuracy, over occupied bins."""
    total = conf.size
    ece = 0.0
    occupied = 0
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        occupied += 1
        ece += (count / total) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    return ece, occupied


def _validate(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct).ravel().astype(np.float64)
    if conf.shape != correct.shape:
        raise ValueError(f"conf shape {conf.shape} != correct shape {correct.shape}")
    if conf.size == 0:
        raise ValueError("cannot compute calibration error on an empty array")
    if not np.all((conf >= 0.0) & (conf <= 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    return conf, correct


def ece_equal_width(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (Guo et al., ICML 2017).

    ``[LIT]`` for the binning scheme. The ``n_bins = 15`` default is attributed to that
    paper and is ``[UNVERIFIED]``; see SOURCES.md "To verify" item 2.

    Bins partition ``[0, 1]`` into ``n_bins`` intervals of equal width regardless of where
    the data is. With discrete spike counts most intervals are empty and the occupied ones
    are wide, so opposing calibration gaps inside a bin cancel and the estimate is biased
    **optimistically** -- worst at low ``T``, which is the regime the hypothesis is about.
    Reported anyway, because it is the standard and the comparison must be like-for-like.
    """
    conf, correct = _validate(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True so a confidence of exactly 1.0 lands in the last bin rather than past it.
    bin_index = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece, _ = _bin_ece(conf, correct, bin_index, n_bins)
    return ece


def _equal_mass_bins(conf: np.ndarray, n_bins: int) -> tuple[Optional[np.ndarray], int]:
    """Assign each sample an equal-mass bin, keeping equal confidences together.

    Returns ``(bin_index, n_effective)``, or ``(None, 1)`` when the confidence takes a
    single distinct value and no binning is possible.

    Quantile edges are deliberately *not* used. ``np.quantile`` interpolates, so with two
    observed confidences of 0.81 and 0.85 it produces edges like 0.83 -- a value no sample
    has and no model could emit. For continuous scores that is harmless; for spike-count
    confidences, where the attainable set is small and every value is a lattice point,
    inventing edges between them mis-assigns the very structure the study is measuring.

    Instead the unique confidence values are walked in order and greedily packed into bins
    of roughly ``n / n_bins`` samples. Every sample sharing a confidence lands in the same
    bin, so no bin boundary ever falls inside a tie -- splitting a tie by rank would
    manufacture per-bin accuracy differences out of sampling noise, and with discrete
    counts ties are the common case rather than the edge case. The cost is that the
    effective bin count can fall well below ``n_bins``; that is reported rather than
    hidden.
    """
    values, inverse, counts = np.unique(
        quantise_confidence(conf), return_inverse=True, return_counts=True
    )
    if values.size < 2:
        return None, 1

    target = conf.size / n_bins
    bin_of_value = np.empty(values.size, dtype=np.int64)
    current = 0
    cumulative = 0
    for i, count in enumerate(counts):
        bin_of_value[i] = current
        cumulative += int(count)
        if current < n_bins - 1 and cumulative >= target * (current + 1):
            current += 1
    n_effective = int(bin_of_value.max()) + 1
    return bin_of_value[inverse], n_effective


def ece_equal_mass(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, *, strict: bool = False
) -> float:
    """Expected calibration error with equal-mass bins that respect ties.

    Bins carry roughly the same number of samples, so resolution follows the data instead
    of the interval. See :func:`_equal_mass_bins` for why the edges are built by packing
    unique values rather than by quantile interpolation.

    The degenerate limit is real at low ``T``: the attainable confidence set can collapse
    to a single value, leaving exactly one bin. That single bin still has a genuine gap
    ``|accuracy - confidence|``, and this function returns it. What it must never do is
    return 0, which would report perfect calibration for the most degenerate point in the
    sweep. Pass ``strict=True`` to raise :class:`DegenerateBinningError` instead.
    """
    conf, correct = _validate(conf, correct)
    bin_index, n_effective = _equal_mass_bins(conf, n_bins)

    if bin_index is None:
        gap = abs(float(correct.mean()) - float(conf.mean()))
        if strict:
            raise DegenerateBinningError(
                f"confidence collapsed to {np.unique(quantise_confidence(conf)).size} distinct value(s) across "
                f"{conf.size:,} samples, so equal-mass binning cannot form more than one "
                f"bin. The single-bin calibration gap is {gap:.6f}; it is not zero. "
                f"This is expected at low T and must be reported, not discarded."
            )
        return gap

    ece, _ = _bin_ece(conf, correct, bin_index, n_effective)
    return ece


def _equal_mass_effective_bins(conf: np.ndarray, n_bins: int) -> int:
    return _equal_mass_bins(conf, n_bins)[1]


def _ece_from_bins(bin_index: np.ndarray, correct: np.ndarray, conf: np.ndarray, n_bins: int) -> float:
    """Weighted mean |accuracy - confidence| over occupied bins, via bincount.

    Same quantity as :func:`_bin_ece`, computed without a Python loop so the bootstrap can
    afford a thousand of them.
    """
    total = conf.size
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.float64)
    sum_correct = np.bincount(bin_index, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(bin_index, weights=conf, minlength=n_bins)
    occupied = counts > 0
    if not occupied.any():
        return 0.0
    gaps = np.abs(sum_correct[occupied] / counts[occupied] - sum_conf[occupied] / counts[occupied])
    return float(np.sum(counts[occupied] / total * gaps))


def _bootstrap_ci(
    conf: np.ndarray,
    correct: np.ndarray,
    ew_index: np.ndarray,
    n_ew: int,
    em_index: Optional[np.ndarray],
    n_em: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Percentile bootstrap 95% CIs for both ECE estimators.

    Without an interval, no difference between two ECEs can be called real -- and with
    discrete confidences the per-bin accuracies are estimated from few samples, so the
    intervals are wide enough to matter.

    **Bin edges are held fixed** at those computed on the full sample rather than
    recomputed per resample. This conditions on the observed confidence support and
    isolates sampling variability in the per-bin accuracies, which is the quantity of
    interest. It is also well justified in this specific setting: the attainable
    confidence set is determined by ``T`` and the class count, so the support is close to
    fixed by construction rather than by chance. Recomputing bins per resample would add
    binning variability and is the defensible alternative; it is not done here, and this
    choice is recorded so a reader can discount it.

    **Binned ECE is positively biased, and the interval shows it.** Averaging
    ``|accuracy - confidence|`` turns sampling noise in the per-bin accuracies into
    positive error, so the statistic overstates calibration error, and the overstatement
    grows as bins get thinner. A bootstrap resample is noisier than the full sample, so
    the resampled ECEs are more inflated still: **the percentile interval can lie entirely
    above the point estimate.** That is expected for a biased statistic and is not
    corrected here, because the interval's job is to describe how much this estimator
    would move on a different test draw -- which is exactly the question "is this ECE
    difference real?" A test pins the behaviour so it cannot be silently "fixed".

    Because the bias depends on how many samples fall in each bin, it is **not constant
    across the sweep**: at low ``T`` the effective bin count collapses and the bias
    shrinks with it. Comparing raw ECE across ``T`` therefore compares differently-biased
    numbers. Comparing two calibrators *at the same* ``T`` on the same test split does not
    have that problem, which is another reason the headline quantity is a difference; use
    :func:`bootstrap_ece_difference` for it.

    Resampling uses ``numpy.random.default_rng(seed)`` only -- CPU, reproducible
    bit-for-bit (correctness rule 4).
    """
    rng = np.random.default_rng(seed)
    n = conf.size
    ew = np.empty(n_bootstrap, dtype=np.float64)
    em = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        c = correct[idx]
        f = conf[idx]
        ew[i] = _ece_from_bins(ew_index[idx], c, f, n_ew)
        if em_index is None:
            # Degenerate single-bin case: the gap is |accuracy - confidence| outright.
            em[i] = abs(float(c.mean()) - float(f.mean()))
        else:
            em[i] = _ece_from_bins(em_index[idx], c, f, n_em)
    return (
        (float(np.percentile(ew, 2.5)), float(np.percentile(ew, 97.5))),
        (float(np.percentile(em, 2.5)), float(np.percentile(em, 97.5))),
    )


@dataclass(frozen=True)
class CalibrationMetrics:
    """Both ECE estimators plus the scores and the diagnostics needed to read them.

    Returned as one object so that no caller can report equal-width ECE without also
    holding equal-mass ECE. Correctness rule 2 is a property of the interface here, not a
    convention someone has to remember.
    """

    n: int
    accuracy: float
    ece_equal_width: float
    ece_equal_mass: float
    nll: float
    brier: float
    n_bins: int
    n_distinct_confidences: int
    equal_width_bins_occupied: int
    equal_mass_bins_effective: int
    degenerate: bool
    ece_equal_width_ci: Optional[tuple[float, float]] = None
    ece_equal_mass_ci: Optional[tuple[float, float]] = None
    n_bootstrap: int = 0
    bootstrap_seed: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def _fmt_ci(ci: Optional[tuple[float, float]]) -> str:
        return "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    @property
    def ci_overlap_note(self) -> str:
        """Human-readable reminder that overlapping intervals are not a difference."""
        if self.ece_equal_width_ci is None:
            return "no bootstrap requested; no ECE difference can be called real"
        return (
            f"equal-width 95% CI {self._fmt_ci(self.ece_equal_width_ci)}, "
            f"equal-mass 95% CI {self._fmt_ci(self.ece_equal_mass_ci)}"
        )

    @property
    def ece_disagreement(self) -> float:
        """How far the two estimators differ. Disagreement is a finding, so it is a field."""
        return abs(self.ece_equal_mass - self.ece_equal_width)

    def as_dict(self) -> dict:
        d = {
            "n": self.n,
            "accuracy": self.accuracy,
            "ece_equal_width": self.ece_equal_width,
            "ece_equal_mass": self.ece_equal_mass,
            "ece_disagreement": self.ece_disagreement,
            "nll": self.nll,
            "brier": self.brier,
            "n_bins": self.n_bins,
            "n_distinct_confidences": self.n_distinct_confidences,
            "equal_width_bins_occupied": self.equal_width_bins_occupied,
            "equal_mass_bins_effective": self.equal_mass_bins_effective,
            "degenerate": self.degenerate,
            "ece_equal_width_ci_low": None if self.ece_equal_width_ci is None else self.ece_equal_width_ci[0],
            "ece_equal_width_ci_high": None if self.ece_equal_width_ci is None else self.ece_equal_width_ci[1],
            "ece_equal_mass_ci_low": None if self.ece_equal_mass_ci is None else self.ece_equal_mass_ci[0],
            "ece_equal_mass_ci_high": None if self.ece_equal_mass_ci is None else self.ece_equal_mass_ci[1],
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }
        return d


def bootstrap_ece_difference(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    seed: int = 0,
    estimator: str = "equal_width",
) -> dict:
    """Paired bootstrap on the ECE *difference* between two calibrators.

    This is the statistic the study actually needs. Two separate confidence intervals
    cannot answer "is isotonic better than temperature": overlapping intervals do not
    imply no difference, and the two arms are evaluated on the **same** test samples, so
    their errors are correlated. Resampling the sample indices once and computing both
    arms on that same resample keeps the pairing and cancels the shared test-set variation
    -- which also cancels most of the shared binning bias.

    ``conf_a`` and ``conf_b`` are the calibrated confidences of the two arms for the same
    samples, in the same order; ``correct`` is their shared correctness vector, which is
    identical because neither calibrator changes the prediction.

    Returns the point difference ``a - b``, its percentile interval, and the fraction of
    resamples in which ``a`` had the lower ECE. A ``ci`` that straddles zero means the
    difference is not resolvable at this test-set size, whatever the point estimate says.
    """
    conf_a, correct_a = _validate(conf_a, correct)
    conf_b, _ = _validate(conf_b, correct)
    if conf_a.size != conf_b.size:
        raise ValueError(f"arms have different sizes: {conf_a.size} vs {conf_b.size}")
    if estimator not in ("equal_width", "equal_mass"):
        raise ValueError(f"unknown estimator {estimator!r}")

    def ece(c, y):
        if estimator == "equal_width":
            return ece_equal_width(c, y, n_bins)
        return ece_equal_mass(c, y, n_bins)

    point = ece(conf_a, correct_a) - ece(conf_b, correct_a)
    rng = np.random.default_rng(seed)
    n = conf_a.size
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # ONE index set, used for both arms
        y = correct_a[idx]
        diffs[i] = ece(conf_a[idx], y) - ece(conf_b[idx], y)

    low, high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "estimator": estimator,
        "difference": point,
        "ci": (low, high),
        "a_better_fraction": float((diffs < 0).mean()),
        "resolvable": bool(low > 0.0 or high < 0.0),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def compute_metrics(
    conf: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
) -> CalibrationMetrics:
    """Both ECEs, NLL, Brier and accuracy, computed together.

    NLL and Brier are the **top-label** scores: the binary problem of "is this prediction
    right, at this confidence". That is deliberate and it is what makes them comparable
    across all four calibrators -- the top-label calibrators emit a scalar confidence and
    no full distribution, so a multiclass NLL is not defined for them. It also matches
    what ECE measures, keeping every number in a row about the same quantity.
    """
    conf, correct = _validate(conf, correct)
    n_distinct = int(np.unique(quantise_confidence(conf)).size)
    effective = _equal_mass_effective_bins(conf, n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ew_index = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece_ew, occupied = _bin_ece(conf, correct, ew_index, n_bins)

    em_index, n_em = _equal_mass_bins(conf, n_bins)
    ece_em = ece_equal_mass(conf, correct, n_bins)

    ew_ci = em_ci = None
    if n_bootstrap > 0:
        ew_ci, em_ci = _bootstrap_ci(
            conf, correct, ew_index, n_bins, em_index, n_em, n_bootstrap, bootstrap_seed
        )

    clipped = np.clip(conf, _EPS, 1.0 - _EPS)
    nll = float(-(correct * np.log(clipped) + (1.0 - correct) * np.log(1.0 - clipped)).mean())
    brier = float(np.mean((conf - correct) ** 2))

    degenerate = effective < 2
    notes: list[str] = []
    if degenerate:
        notes.append(
            f"confidence collapsed to {n_distinct} distinct value(s); equal-mass ECE is a "
            f"single-bin gap, not a binned estimate"
        )
    if n_distinct < n_bins:
        notes.append(
            f"only {n_distinct} distinct confidences for {n_bins} requested bins; binned "
            f"ECE is coarse here by construction, which is the low-T regime the study is about"
        )

    return CalibrationMetrics(
        n=int(conf.size),
        accuracy=float(correct.mean()),
        ece_equal_width=ece_ew,
        ece_equal_mass=ece_em,
        nll=nll,
        brier=brier,
        n_bins=n_bins,
        n_distinct_confidences=n_distinct,
        equal_width_bins_occupied=occupied,
        equal_mass_bins_effective=effective,
        degenerate=degenerate,
        ece_equal_width_ci=ew_ci,
        ece_equal_mass_ci=em_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        notes=tuple(notes),
    )
