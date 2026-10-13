"""Decision fusion of two readouts, and what a saturated count readout does to it.

Everything derived here is a corollary of known decision-fusion theory applied to
spike-count readouts, and is tagged ``[DERIVED]``; ``SOURCES.md`` records the theory it
follows from. Nothing in this module is a new theorem.

The setting is two modalities, each read out by a network, each reporting a probability
vector over the same ``C`` classes. Three rules are implemented, because the proposition is
a statement about how they relate:

- the **product rule** on the reported probabilities, which is the Bayes rule for
  conditionally independent modalities under equal priors when the probabilities are the
  true posteriors;
- the product rule after a **per-modality scale** on the logits, which is the same rule
  applied to recalibrated probabilities;
- **reliability-weighted voting**, in which each modality casts one vote for its predicted
  class with weight ``log((C - 1) a / (1 - a))``, the Bayes rule when all that is known of a
  modality is its accuracy ``a`` and its errors are spread evenly over the wrong classes.

**The proposition.** Call a count readout *saturated* on an input when its winning class
fired at every one of the ``T`` steps and no other class fired at all. Its probability vector
is then ``(p*, q, ..., q)`` up to the position of the winner, the same vector for every
saturated input and for every network. So on the event ``S`` that both modalities saturate:

(a) the two reports are identical up to a permutation of the classes, and any rule that uses
    only the two reports, treats the two modalities alike and treats the classes alike
    gives the two winners the same fused score: exchanging the two classes exchanges the two
    reports, and the rule cannot tell which report came from which modality. If the rule also
    rates a class strictly higher when a modality rates it higher, the tie is between exactly
    those two classes, and with uniform tie-breaking it is an unweighted vote: a coin flip
    whenever the two disagree. The product, sum and max rules are of this kind. The min rule
    is not strict and ties every class instead.
(b) Against a rule that sides with modality 1 whenever the two disagree, which is what
    reliability-weighted voting does when modality 1 is the more accurate, it therefore loses

        0.5 * P(S) * (a1_S - a2_S)

    in accuracy, where ``a_k,S`` is modality ``k``'s accuracy on ``S``. This is an identity,
    not an approximation, and needs no model of the errors: when the modalities agree both
    are right or both are wrong, so the difference in their accuracies on ``S`` is carried
    entirely by the inputs on which they disagree, and the coin flip gives up half of it.
(c) A scale ``s_k`` on modality ``k``'s counts with ``s_k * T = log((C - 1) a_k / (1 - a_k))``
    makes the product rule on ``S`` exactly reliability-weighted voting.

Tie-breaking is uniform at random under a seeded generator throughout, as everywhere else in
this repository, because breaking ties by lowest index would hand every tie to class 0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .calibration import predict_with_ties, softmax_probs
from .theory import min_logit_width

__all__ = [
    "saturated", "reliability_weight", "product_rule", "weighted_vote", "unweighted_vote",
    "excess_error_on_saturation", "excess_error_as_first_drafted",
]


def saturated(counts: np.ndarray, timesteps: int) -> np.ndarray:
    """True where a count vector is ``T`` at exactly one class and 0 at every other.

    That vector attains the readout's confidence ceiling ``e^T / (e^T + C - 1)``, and is
    the only one that does. ``counts`` must be integer-valued spike counts; a scaled or
    real-valued readout has to be reduced to its counts first.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D [n_samples, n_classes], got shape {counts.shape}")
    if not np.array_equal(counts, np.rint(counts)):
        raise ValueError("saturation is defined on integer spike counts")
    return (counts.max(axis=1) == timesteps) & (counts.sum(axis=1) == timesteps)


def reliability_weight(n_classes: int, accuracy: float) -> float:
    """Vote weight of a modality known only by its accuracy: ``log((C - 1) a / (1 - a))``.

    [DERIVED] If a modality names the true class with probability ``a`` and each wrong
    class with probability ``(1 - a) / (C - 1)``, the log-likelihood of class ``j`` given
    its vote ``v`` is ``log a`` when ``v = j`` and ``log((1 - a)/(C - 1))`` otherwise. Up to a
    constant that is the same for every class, a vote for ``j`` is therefore worth
    ``log(a (C - 1) / (1 - a))`` to class ``j``. It is the same expression as the minimum
    logit width of :func:`spikecalib.theory.min_logit_width`, which is why a scale that
    lifts the cap to a modality's accuracy is also the scale that weights its vote correctly.
    """
    return min_logit_width(n_classes, accuracy)


def product_rule(logits_a: np.ndarray, logits_b: np.ndarray, rng: np.random.Generator,
                 scale_a: float = 1.0, scale_b: float = 1.0):
    """Product of the two softmax distributions, with optional per-modality logit scales.

    The product of two softmaxes is the softmax of the summed logits, so the fused
    distribution is ``softmax(scale_a * logits_a + scale_b * logits_b)``. Returns
    ``(predictions, probabilities, tie_mask)``; ties in the fused argmax are broken
    uniformly at random under ``rng``.
    """
    fused = scale_a * np.asarray(logits_a, dtype=np.float64) + scale_b * np.asarray(logits_b, dtype=np.float64)
    predictions, tie_mask = predict_with_ties(fused, rng)
    return predictions, softmax_probs(fused), tie_mask


def weighted_vote(votes_a: np.ndarray, votes_b: np.ndarray, weight_a: float, weight_b: float,
                  n_classes: int, rng: np.random.Generator):
    """Each modality votes for one class with its weight; the heavier total wins.

    Returns ``(predictions, probabilities, tie_mask)``. The probabilities are the softmax of
    the vote totals, which is the posterior the weights imply under the model in
    :func:`reliability_weight`.
    """
    votes_a, votes_b = np.asarray(votes_a, dtype=np.int64), np.asarray(votes_b, dtype=np.int64)
    scores = np.zeros((votes_a.size, n_classes), dtype=np.float64)
    rows = np.arange(votes_a.size)
    np.add.at(scores, (rows, votes_a), weight_a)
    np.add.at(scores, (rows, votes_b), weight_b)
    predictions, tie_mask = predict_with_ties(scores, rng)
    return predictions, softmax_probs(scores), tie_mask


def unweighted_vote(votes_a: np.ndarray, votes_b: np.ndarray, n_classes: int,
                    rng: np.random.Generator):
    """:func:`weighted_vote` with both weights equal: a coin flip whenever the two disagree."""
    return weighted_vote(votes_a, votes_b, 1.0, 1.0, n_classes, rng)


def excess_error_on_saturation(p_s: float, acc_a_on_s: float, acc_b_on_s: float) -> float:
    """What an unweighted vote loses on ``S`` to a rule that sides with modality A: part (b).

    ``0.5 * P(S) * (a_A,S - a_B,S)``, with both accuracies conditioned on ``S``. Positive
    when A is the more accurate on ``S``. Exact in expectation over the tie-breaks.
    """
    return 0.5 * float(p_s) * (float(acc_a_on_s) - float(acc_b_on_s))


def excess_error_as_first_drafted(p_s_and_disagree: float, acc_a_on_s: float,
                                  acc_b_on_s: float) -> float:
    """The expression in the first draft of part (b), kept so its error is on record.

    ``0.5 * P(S and disagree) * (a_A,S - a_B,S)``. It multiplies a difference of accuracies
    conditioned on ``S`` by the probability of ``S`` **and** disagreement, which counts the
    probability of disagreement twice: the difference ``a_A,S - a_B,S`` already equals
    ``P(disagree | S)`` times the difference conditioned on disagreement. It is smaller than
    :func:`excess_error_on_saturation` by exactly the factor ``P(disagree | S)``. The
    experiment reports it beside the corrected value; a test shows which one a simulation
    reproduces.
    """
    return 0.5 * float(p_s_and_disagree) * (float(acc_a_on_s) - float(acc_b_on_s))
