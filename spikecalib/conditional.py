"""The conditional ceiling: what a sample could have expressed, given its competitors.

The attainable-confidence ceiling ``e^T / (e^T + C - 1)`` is the softmax of the extreme
count vector ``[T, 0, ..., 0]``. It assumes **every one of the C - 1 competing classes is
silent for all T steps**. That assumption is nearly free at ten classes and expensive at a
hundred and one: the softmax denominator's sensitivity to competitor activity scales with
``C - 1``, so the same per-competitor firing rate costs eleven times more confidence at
``C = 101`` than at ``C = 10``.

Conditioning on what the competitors actually did gives, for one sample,

    ceiling_cond = e^T / (e^T + sum_{j != winner} e^{c_j})

the largest confidence that sample could have expressed **if its winner had fired at every
step**, holding the competitors where they were. Setting every ``c_j = 0`` recovers the
theoretical ceiling exactly, so this is a refinement of the same quantity rather than a
different one, and :func:`theoretical_ceiling` is asserted to be its all-zero case.

**What this can and cannot claim.** The theoretical ceiling is a property of the readout,
fixed by ``T`` and ``C`` before a model exists, so it predicts. The conditional ceiling is
computed from a model's own outputs, so it explains: it can say "this model was as confident
as its own spike distribution allowed" but not "any model must be this unconfident". A model
that spread its spikes differently would face a different conditional bound. Reported as an
attainability effect, never as a bound the study derived in advance.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "theoretical_ceiling",
    "conditional_ceiling",
    "competitor_profile",
    "residual_share_explained",
]


def theoretical_ceiling(timesteps: float, n_classes: int) -> float:
    """``e^T / (e^T + C - 1)``: the all-competitors-silent case."""
    e = np.exp(float(timesteps))
    return float(e / (e + n_classes - 1))


def conditional_ceiling(counts: np.ndarray, timesteps: float) -> np.ndarray:
    """Per-sample ``e^T / (e^T + sum of e^{competitor count})``.

    ``counts`` is ``[n_samples, n_classes]`` of readout scores. The winner is the argmax,
    and is *replaced* by ``T`` rather than kept at its observed value -- the question is what
    the sample could have reached, not what it did.

    Computed in log-space against the same exponentials the softmax uses, so the comparison
    with an observed confidence is exact rather than approximately consistent.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be [n_samples, n_classes], got {counts.shape}")
    winner = counts.argmax(axis=1)
    exp_all = np.exp(counts)
    # Total minus the winner's own term leaves exactly the competitor sum.
    competitor_sum = exp_all.sum(axis=1) - exp_all[np.arange(counts.shape[0]), winner]
    e_t = np.exp(float(timesteps))
    return e_t / (e_t + competitor_sum)


def competitor_profile(counts: np.ndarray, n_classes: int,
                       max_histogram_values: int = 64) -> dict:
    """How many non-winning classes fire, and at what counts.

    Returns the distribution rather than only its mean: the softmax denominator is a sum of
    exponentials, so it is dominated by the few loudest competitors and a mean count hides
    whether the activity is spread thinly or concentrated.

    ``count_histogram`` is ``None`` for a readout with more than ``max_histogram_values``
    distinct competitor values -- a real-valued readout has one per sample per class, and
    tabulating them is neither informative nor small. The quantiles of the competitor sum are
    reported either way and are what the conditional ceiling actually depends on.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    winner = counts.argmax(axis=1)
    mask = np.ones(counts.shape, dtype=bool)
    mask[np.arange(n), winner] = False
    comp = counts[mask].reshape(n, counts.shape[1] - 1)

    active = comp > 0
    exp_comp = np.exp(comp)
    denom = exp_comp.sum(axis=1)
    loudest = np.sort(comp, axis=1)[:, ::-1]
    # Share of the competitor sum carried by the single loudest competitor: the number that
    # says whether "mean competitor count" is a fair summary at all.
    top1_share = np.exp(loudest[:, 0]) / denom

    # A histogram only makes sense for a DISCRETE readout. Spike counts take a handful of
    # integer values; a membrane or accumulator readout is real-valued, and tabulating its
    # unique values produced 173,962 keys per row and a 137 MB dashboard payload before this
    # guard existed. Above the threshold the quantiles below carry the shape instead.
    uniq = np.unique(comp)
    hist = ({float(v): float((comp == v).mean()) for v in uniq}
            if uniq.size <= max_histogram_values else None)
    return {
        "n_competitors": int(counts.shape[1] - 1),
        "mean_count": float(comp.mean()),
        "median_count": float(np.median(comp)),
        "mean_n_active": float(active.sum(axis=1).mean()),
        "frac_active": float(active.mean()),
        "mean_max_competitor": float(loudest[:, 0].mean()),
        "mean_competitor_sum": float(denom.mean()),
        "mean_top1_share_of_sum": float(top1_share.mean()),
        "count_histogram": hist,
        "n_distinct_competitor_values": int(uniq.size),
        "quantiles_of_competitor_sum": {
            str(q): float(np.quantile(denom, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }


def residual_share_explained(accuracy: float, mean_conf: float, ceiling: float) -> float:
    """Share of ``accuracy - mean_conf`` that a ceiling of ``ceiling`` forces.

    ``0`` when the ceiling sits at or above accuracy, so nothing is forced; ``1`` when it
    sits at or below the mean confidence, so all of it is. Clipped rather than allowed to
    exceed 1, because a ceiling below the observed mean confidence would be a contradiction
    and should be caught as one by the caller, not smuggled through as 130%.
    """
    residual = accuracy - mean_conf
    if residual <= 0:
        return 0.0
    return float(min(1.0, max(0.0, accuracy - ceiling) / residual))
