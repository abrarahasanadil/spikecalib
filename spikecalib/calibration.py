"""Calibrators, the two ECE estimators, and seeded tie-breaking.

This module is the contribution. Everything else in the repository is scaffolding that
gets data into it and results out of it.

Owns correctness rule 2: **both equal-width (Guo et al.) and equal-mass ECE are computed
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
counts. The tie rate is a **reported quantity, not a diagnostic**: the hypothesis predicts
it rises as ``T`` falls, and that rise is direct evidence for the discreteness mechanism
independent of any ECE number.

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

__all__ = [
    "softmax_confidence",
    "softmax_probs",
    "predict_with_ties",
    "tie_rate",
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
    "CalibrationMetrics",
    "DegenerateBinningError",
]

# Guarding log(0) in NLL. Small enough not to move a reported figure, large enough that a
# confidently wrong sample gives a finite loss instead of inf.
_EPS = 1e-12


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
    """
    return softmax_probs(counts).max(axis=1)


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

    A reported quantity, not a diagnostic. The hypothesis predicts this rises as ``T``
    falls, because fewer attainable count values means more collisions, and that rise is
    direct evidence for the mechanism the paper argues for -- visible without reference to
    any ECE number. It belongs in the results table.
    """
    return float((_max_mask(counts).sum(axis=1) > 1).mean())


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
        conf=probs[np.arange(probs.shape[0]), predictions],
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

    The claim under test lives here. Dividing every logit by the same positive scalar is
    strictly order-preserving, so it cannot change which class is predicted and cannot
    separate a tie -- two classes on equal counts stay equal at every temperature. On an
    integer lattice that limitation binds hard at low ``T``.

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
    """Expected calibration error with equal-width bins (Guo et al.).

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
    values, inverse, counts = np.unique(conf, return_inverse=True, return_counts=True)
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
                f"confidence collapsed to {np.unique(conf).size} distinct value(s) across "
                f"{conf.size:,} samples, so equal-mass binning cannot form more than one "
                f"bin. The single-bin calibration gap is {gap:.6f}; it is not zero. "
                f"This is expected at low T and must be reported, not discarded."
            )
        return gap

    ece, _ = _bin_ece(conf, correct, bin_index, n_effective)
    return ece


def _equal_mass_effective_bins(conf: np.ndarray, n_bins: int) -> int:
    return _equal_mass_bins(conf, n_bins)[1]


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
    notes: tuple[str, ...] = field(default_factory=tuple)

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
        }
        return d


def compute_metrics(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> CalibrationMetrics:
    """Both ECEs, NLL, Brier and accuracy, computed together.

    NLL and Brier are the **top-label** scores: the binary problem of "is this prediction
    right, at this confidence". That is deliberate and it is what makes them comparable
    across all four calibrators -- the top-label calibrators emit a scalar confidence and
    no full distribution, so a multiclass NLL is not defined for them. It also matches
    what ECE measures, keeping every number in a row about the same quantity.
    """
    conf, correct = _validate(conf, correct)
    n_distinct = int(np.unique(conf).size)
    effective = _equal_mass_effective_bins(conf, n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ew_index = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece_ew, occupied = _bin_ece(conf, correct, ew_index, n_bins)

    ece_em = ece_equal_mass(conf, correct, n_bins)

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
        notes=tuple(notes),
    )
