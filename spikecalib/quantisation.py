"""Does coarsening a confidence signal improve the ranking it induces?

The membrane readout ranks *worse* than the spike-count readout from ``T = 6`` upward --
better ECE, worse AUROC and worse AURC -- and nothing in the ceiling argument predicts
that. One candidate explanation is that a spike count is a thresholded membrane signal,
and that the thresholding throws away noise: once the count has enough levels to express
the ordering, the discarding is pure gain, while membrane keeps the variance it discarded.

This module tests that by coarsening the membrane confidences directly. No retraining is
involved; quantisation is applied to the saved confidences.

**The arithmetic makes the test sharp, and it is worth stating up front because it says
what a positive result would have to look like.** Quantisation to ordered bins is a
monotone, non-injective map. It cannot reorder any pair -- it can only merge pairs into
ties. AUROC scores a tie as 0.5. So, writing ``W_within`` for the number of
correct/incorrect pairs that land in the same bin, ``W_total`` for all such pairs, and
``A_within`` for the pooled AUROC *within* bins at full precision:

    AUROC(quantised) - AUROC(full) = (W_within / W_total) * (0.5 - A_within)

This is an identity, not an approximation, and :func:`quantisation_step` returns both
factors so the result can be read mechanistically rather than as a curve that happens to
go up or down. It has three consequences:

1. Quantisation can only help if the fine structure it destroys was **worse than chance**
   at ordering correct from incorrect. ``A_within < 0.5`` is the hypothesis, stated
   exactly.
2. If ``A_within`` is around 0.5, the fine structure is uninformative rather than
   misleading, AUROC is flat in the number of levels, and quantisation is not the
   mechanism behind the membrane deficit.
3. If ``A_within > 0.5``, the fine structure is real signal and coarsening costs.

Two binning schemes are provided because "N levels" is ambiguous on a distribution as
skewed as a confidence distribution:

``equal-mass``
    Bin boundaries at mass quantiles, so each of the ``N`` levels holds roughly ``n / N``
    samples and every level is actually populated. This is the primary scheme: it makes
    the ``N`` axis mean what it says.
``equal-width``
    Bin boundaries evenly spaced across the observed confidence range. Closer to a physical
    threshold ladder, but on a distribution piled up near 1.0 the samples crowd into the
    few bins nearest the mode, so the resolution the bulk of the data actually receives is
    far below ``N`` even when most bins are non-empty. Reported as a robustness check, with
    ``n_effective`` alongside it.

Under both schemes **samples sharing a confidence always share a bin**. Splitting them
would manufacture an ordering the model never expressed, which is the thing
:mod:`spikecalib.selective` exists to refuse.

One caveat on AURC, which does not apply to AUROC. :func:`spikecalib.selective.aurc`
anchors the curve at the smallest achievable coverage, and with ``N`` levels that coverage
is about ``1 / N``. Coarsening therefore moves AURC for a structural reason -- fewer
operating points, a longer anchored segment -- on top of any change in ranking quality.
AURC is reported because it is the deployment-facing number, but **AUROC is the measure
that isolates ranking**, and the hypothesis is a hypothesis about ranking.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .calibration import quantise_confidence
from .selective import aurc

__all__ = [
    "LEVEL_GRID",
    "SCHEMES",
    "quantise_to_levels",
    "within_bin_auroc",
    "quantisation_step",
    "quantisation_sweep",
]

#: Resolutions to test. ``None`` is full precision: the ungrouped signal.
LEVEL_GRID: tuple[Optional[int], ...] = (5, 10, 20, 50, 100, 200, 500, 1000, None)

SCHEMES = ("equal-mass", "equal-width")


def quantise_to_levels(
    conf: np.ndarray, n_levels: Optional[int], scheme: str = "equal-mass"
) -> np.ndarray:
    """Coarsen ``conf`` to at most ``n_levels`` ordered levels, returning bin indices.

    The return value is a float array of bin indices, which is all that AUROC and
    risk-coverage need: both read only the ordering and the tie structure, never the
    magnitudes. ``n_levels=None`` returns the confidences unchanged, quantised to
    :data:`spikecalib.calibration.CONFIDENCE_DECIMALS` so that mathematically equal values
    compare equal.

    Monotone by construction, and equal confidences always receive equal indices.
    """
    conf = quantise_confidence(np.asarray(conf, dtype=np.float64).ravel())
    if n_levels is None:
        return conf
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1 or None, got {n_levels}")
    if conf.size == 0:
        raise ValueError("cannot quantise an empty array")

    # Work on unique values, then scatter back. This is what guarantees that samples
    # sharing a confidence share a bin, under either scheme.
    uniq, inverse, counts = np.unique(conf, return_inverse=True, return_counts=True)
    if uniq.size == 1:
        return np.zeros(conf.size, dtype=np.float64)

    if scheme == "equal-mass":
        # Each unique value sits at the MIDPOINT of the mass it occupies, not at its upper
        # edge. The difference only shows up under heavy ties, and there it is the whole
        # point: a value holding a large share of the mass would otherwise be pushed to the
        # top of its span and collapse into the value above it, losing a level that exists.
        # The midpoint also makes the top bin unreachable by construction -- position is
        # strictly below 1 -- so no clipping is needed here.
        cum = np.cumsum(counts)
        position = (cum - counts / 2.0) / conf.size
        bin_of_unique = (position * n_levels).astype(np.int64)
    elif scheme == "equal-width":
        # Here the largest value lands exactly on the upper edge and must be pulled back
        # into the last bin rather than opening an N+1-th one.
        lo, hi = float(uniq[0]), float(uniq[-1])
        bin_of_unique = np.minimum(((uniq - lo) / (hi - lo) * n_levels).astype(np.int64),
                                   n_levels - 1)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected one of {SCHEMES}")

    return bin_of_unique[inverse].astype(np.float64)


def within_bin_auroc(conf: np.ndarray, correct: np.ndarray, bins: np.ndarray) -> dict:
    """Pooled AUROC of the fine structure that ``bins`` merges away.

    Considers only correct/incorrect pairs falling in the same bin, and asks how well the
    full-precision confidence orders them. Below 0.5 the fine structure is anti-correlated
    with correctness and merging it is a gain; at 0.5 it is uninformative; above 0.5 it is
    signal and merging it is a loss.

    Returns the pooled value, the number of pairs it is pooled over, and that number as a
    fraction of all correct/incorrect pairs -- the weight the identity in the module
    docstring applies to it.
    """
    conf = quantise_confidence(np.asarray(conf, dtype=np.float64).ravel())
    correct = np.asarray(correct).ravel().astype(bool)
    bins = np.asarray(bins, dtype=np.float64).ravel()
    if not (conf.shape == correct.shape == bins.shape):
        raise ValueError(f"shapes differ: {conf.shape}, {correct.shape}, {bins.shape}")

    n_pos_all, n_neg_all = int(correct.sum()), int((~correct).sum())
    total_pairs = n_pos_all * n_neg_all

    concordant = 0.0
    within_pairs = 0
    for b in np.unique(bins):
        mask = bins == b
        y = correct[mask]
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        # Mann-Whitney U with mid-ranks for ties; U / (n_pos * n_neg) is the AUROC of this
        # bin, so summing U over bins and dividing by the pooled pair count pools them.
        ranks = rankdata(conf[mask])
        concordant += float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0
        within_pairs += n_pos * n_neg

    return {
        "auroc_within": float(concordant / within_pairs) if within_pairs else float("nan"),
        "n_pairs_within": int(within_pairs),
        "n_pairs_total": int(total_pairs),
        "pair_fraction_merged": float(within_pairs / total_pairs) if total_pairs else float("nan"),
    }


def quantisation_step(
    conf: np.ndarray, correct: np.ndarray, n_levels: Optional[int], scheme: str = "equal-mass"
) -> dict:
    """AUROC, AURC and the merge diagnostics at one resolution."""
    conf = quantise_confidence(np.asarray(conf, dtype=np.float64).ravel())
    correct = np.asarray(correct).ravel().astype(bool)
    bins = quantise_to_levels(conf, n_levels, scheme)

    within = within_bin_auroc(conf, correct, bins)
    auroc = 0.5 if np.unique(bins).size == 1 else float(roc_auc_score(correct, bins))
    return {
        "n_levels_requested": n_levels,
        "n_effective": int(np.unique(bins).size),
        "scheme": scheme,
        "auroc": auroc,
        "aurc": aurc(bins, correct),
        **within,
    }


def quantisation_sweep(
    conf: np.ndarray,
    correct: np.ndarray,
    levels: Sequence[Optional[int]] = LEVEL_GRID,
    scheme: str = "equal-mass",
) -> list[dict]:
    """:func:`quantisation_step` across a grid of resolutions, coarsest first."""
    return [quantisation_step(conf, correct, n, scheme) for n in levels]
