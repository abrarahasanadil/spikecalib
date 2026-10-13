"""Tests for the fusion proposition: what a saturated count readout does to decision fusion.

Each part is checked two ways that do not share its derivation: by brute force over every
pair of saturated reports, and by simulation. Part (b) was first drafted with a different
expression; a test here shows that a simulation reproduces the corrected one and not the
draft, so the correction is on record as something checked and not just asserted.

Nothing here touches a trained model.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from spikecalib.calibration import softmax_probs
from spikecalib.fusion import (
    class_matched_partners, event_dropout, excess_error_as_first_drafted, excess_error_on_saturation, product_rule, reliability_weight,
    saturated, unweighted_vote, weighted_vote,
)
from spikecalib.theory import (
    max_distinct_confidences, min_scale, n_distinct_probability_vectors, scaled_ceiling,
)


def sat(winner: int, T: int, C: int) -> np.ndarray:
    v = np.zeros(C)
    v[winner] = T
    return v


# ---------------------------------------------------------------------------------------
# The saturation event
# ---------------------------------------------------------------------------------------


def test_saturated_is_exactly_the_vector_that_attains_the_ceiling():
    T, C = 2, 4
    vectors = np.array(list(itertools.product(range(T + 1), repeat=C)))
    flag = saturated(vectors, T)
    assert flag.sum() == C                                    # one per winning class
    conf = softmax_probs(vectors.astype(float)).max(axis=1)
    assert np.allclose(conf[flag], scaled_ceiling(T, C))
    assert conf[~flag].max() < scaled_ceiling(T, C) - 1e-9    # and no other vector reaches it
    with pytest.raises(ValueError):
        saturated(vectors * 1.5, T)


def test_every_saturated_report_is_the_same_vector_up_to_the_winner():
    T, C = 3, 6
    reports = np.sort(softmax_probs(np.array([sat(w, T, C) for w in range(C)])), axis=1)
    assert np.allclose(reports, reports[0])


# ---------------------------------------------------------------------------------------
# (a) On S, a symmetric rule is an unweighted vote
# ---------------------------------------------------------------------------------------

# Rules that use only the two reported probability vectors, treat the modalities alike and
# the classes alike, and rate a class no lower when a modality rates it higher. All but
# "min" rate it strictly higher; "min" is here because it is the one of Kittler et al.'s
# rules that is not strict, and it shows what the strictness is for.
STRICT = {"product", "sum", "max", "geometric mean", "sum of squares"}
SYMMETRIC_RULES = {
    "product": lambda p, q: np.log(p) + np.log(q),
    "sum": lambda p, q: p + q,
    "max": lambda p, q: np.maximum(p, q),
    "min": lambda p, q: np.minimum(p, q),
    "geometric mean": lambda p, q: np.sqrt(p * q),
    "sum of squares": lambda p, q: p ** 2 + q ** 2,
}


@pytest.mark.parametrize("name", sorted(SYMMETRIC_RULES))
@pytest.mark.parametrize("T,C", [(1, 3), (2, 4), (2, 10), (6, 10)])
def test_no_symmetric_rule_can_prefer_one_saturated_modality_to_the_other(name, T, C):
    """Brute force over every pair of winners.

    Whatever the rule, the two winners get the same fused score and nothing scores above
    them. For a rule that is strictly increasing in each report the tie is between exactly
    those two, which is an unweighted vote. The min rule is not strict: on disagreement it
    ties every class, which is worse than a vote and still cannot tell the modalities apart.
    """
    rule = SYMMETRIC_RULES[name]
    for w1, w2 in itertools.product(range(C), repeat=2):
        p, q = softmax_probs(sat(w1, T, C)[None, :])[0], softmax_probs(sat(w2, T, C)[None, :])[0]
        score = rule(p, q)
        assert score[w1] == pytest.approx(score[w2], abs=1e-12)
        assert score.max() == pytest.approx(score[w1], abs=1e-12)
        top = set(np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-12)))
        if name in STRICT or w1 == w2:
            assert top == {w1, w2}, (name, w1, w2, score)
        else:
            assert top == set(range(C))


def test_on_s_the_product_rule_and_an_unweighted_vote_are_the_same_rule():
    """Same decisions where the modalities agree, the same coin where they do not."""
    rng = np.random.default_rng(0)
    T, C, n = 2, 10, 40_000
    w1, w2 = rng.integers(0, C, n), rng.integers(0, C, n)
    a = np.zeros((n, C)); a[np.arange(n), w1] = T
    b = np.zeros((n, C)); b[np.arange(n), w2] = T
    assert saturated(a, T).all() and saturated(b, T).all()
    pred, _, tie = product_rule(a, b, np.random.default_rng(1))
    vote, _, vtie = unweighted_vote(w1, w2, C, np.random.default_rng(2))
    agree = w1 == w2
    assert np.array_equal(pred[agree], w1[agree]) and np.array_equal(vote[agree], w1[agree])
    assert np.array_equal(tie, ~agree) and np.array_equal(vtie, ~agree)
    for chosen in (pred, vote):
        assert np.all((chosen == w1) | (chosen == w2))
        share = (chosen[~agree] == w1[~agree]).mean()
        assert abs(share - 0.5) < 0.012                         # 36,000 coin flips: sd 0.0026


def test_off_s_the_product_rule_does_use_the_reports():
    """The reduction is about S. A modality that is not saturated can be outvoted."""
    T, C = 2, 4
    a = np.array([[2.0, 0, 0, 0]])                             # saturated, for class 0
    b = np.array([[0.0, 1, 0, 0]])                             # one spike, for class 1
    pred, _, tie = product_rule(a, b, np.random.default_rng(0))
    assert pred[0] == 0 and not tie[0]


# ---------------------------------------------------------------------------------------
# (b) What the coin flip costs
# ---------------------------------------------------------------------------------------


def simulate(n, C, a1, a2, sat1, sat2, seed):
    """Two conditionally independent modalities with evenly spread errors.

    ``sat_k`` is ``(P(saturate | correct), P(saturate | wrong))``, so saturation may depend
    on correctness, which is what makes the conditioning on S matter.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)

    def modality(acc, sat_rates):
        correct = rng.random(n) < acc
        wrong = (y + rng.integers(1, C, n)) % C
        vote = np.where(correct, y, wrong)
        saturates = rng.random(n) < np.where(correct, sat_rates[0], sat_rates[1])
        return vote, saturates

    v1, s1 = modality(a1, sat1)
    v2, s2 = modality(a2, sat2)
    return y, v1, v2, s1 & s2


@pytest.mark.parametrize("C,a1,a2,sat1,sat2", [
    (10, 0.985, 0.885, (0.97, 0.40), (0.90, 0.30)),
    (10, 0.95, 0.80, (0.6, 0.6), (0.6, 0.6)),
    (4, 0.90, 0.60, (0.8, 0.5), (0.7, 0.2)),
    (101, 0.60, 0.45, (0.3, 0.05), (0.2, 0.02)),
])
def test_simulation_reproduces_the_corrected_expression_and_not_the_first_draft(C, a1, a2, sat1, sat2):
    n = 400_000
    y, v1, v2, S = simulate(n, C, a1, a2, sat1, sat2, seed=0)
    coin, _, _ = unweighted_vote(v1, v2, C, np.random.default_rng(1))
    # the rule that sides with modality 1 on disagreement: weighted voting with w1 > w2
    weighted, _, _ = weighted_vote(v1, v2, reliability_weight(C, a1), reliability_weight(C, a2),
                                   C, np.random.default_rng(2))
    assert np.array_equal(weighted, v1)                        # two voters, heavier one decides
    lost = (((weighted == y).astype(float) - (coin == y)) * S).mean()   # excess error on S

    p_s, disagree = S.mean(), (v1 != v2)
    a1_s, a2_s = (v1 == y)[S].mean(), (v2 == y)[S].mean()
    corrected = excess_error_on_saturation(p_s, a1_s, a2_s)
    drafted = excess_error_as_first_drafted((S & disagree).mean(), a1_s, a2_s)
    se = 0.5 * math.sqrt((S & disagree).mean() / n)            # sd of the coin-flip average
    assert abs(lost - corrected) < 5 * se + 1e-4
    # the draft is short by exactly the factor P(disagree | S)
    assert drafted == pytest.approx(corrected * disagree[S].mean(), rel=1e-9)
    if abs(corrected - drafted) > 40 * se:
        # ... and wherever that gap is large against sampling error, which includes the
        # regime of the experiment (the first case), the simulation sides with the correction
        assert abs(lost - drafted) > 20 * se
    else:
        assert C == 101, "only the rare-saturation case should lack the power to tell them apart"


def test_the_expression_is_an_identity_that_needs_no_model_of_the_errors():
    """Arbitrary, dependent, lopsided errors: the expected loss is still 0.5 P(S)(a1_S - a2_S).

    With the coin replaced by its expectation the two sides are equal to float precision,
    because where the modalities agree they are right or wrong together.
    """
    rng = np.random.default_rng(3)
    n, C = 50_000, 7
    y = rng.integers(0, C, n)
    v1 = np.where(rng.random(n) < 0.8, y, rng.integers(0, C, n))
    v2 = np.where(rng.random(n) < 0.5, v1, (y + rng.integers(0, 3, n)) % C)   # correlated with v1
    S = rng.random(n) < np.where(v1 == y, 0.9, 0.2)
    c1, c2 = (v1 == y).astype(float), (v2 == y).astype(float)
    expected_coin = np.where(v1 == v2, c1, 0.5 * (c1 + c2))
    lost = ((c1 - expected_coin) * S).mean()
    assert lost == pytest.approx(excess_error_on_saturation(S.mean(), c1[S].mean(), c2[S].mean()), abs=1e-12)


def test_reliability_weights_are_the_log_likelihood_ratio_of_a_vote():
    """Brute force: weighted voting with these weights is the MAP rule under the error model."""
    C = 5
    accs = (0.9, 0.6, 0.55)
    w = [reliability_weight(C, a) for a in accs]
    for votes in itertools.product(range(C), repeat=len(accs)):
        log_post = np.zeros(C)
        for a, v in zip(accs, votes):
            log_post += np.where(np.arange(C) == v, math.log(a), math.log((1 - a) / (C - 1)))
        scores = np.zeros(C)
        for wk, v in zip(w, votes):
            scores[v] += wk
        assert np.allclose(scores - scores.max(), log_post - log_post.max(), atol=1e-12)


# ---------------------------------------------------------------------------------------
# (c) Per-modality scales restore the weighting
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C,a1,a2", [(2, 10, 0.985, 0.885), (2, 101, 0.58, 0.45), (6, 10, 0.99, 0.7)])
def test_scales_at_s_min_turn_the_product_rule_into_weighted_voting_on_s(T, C, a1, a2):
    s1, s2 = min_scale(T, C, a1), min_scale(T, C, a2)
    assert s1 * T == pytest.approx(reliability_weight(C, a1)) and s2 * T == pytest.approx(reliability_weight(C, a2))
    pairs = list(itertools.product(range(C), repeat=2))
    a = np.array([sat(w1, T, C) for w1, _ in pairs])
    b = np.array([sat(w2, T, C) for _, w2 in pairs])
    w1 = np.array([p[0] for p in pairs]); w2 = np.array([p[1] for p in pairs])
    fused, probs, tie = product_rule(a, b, np.random.default_rng(0), scale_a=s1, scale_b=s2)
    voted, vprobs, _ = weighted_vote(w1, w2, reliability_weight(C, a1), reliability_weight(C, a2),
                                     C, np.random.default_rng(1))
    assert not tie.any()
    assert np.array_equal(fused, voted) and np.array_equal(fused, w1)     # the more accurate one decides
    assert np.allclose(probs, vprobs, atol=1e-12)                          # the same posterior, not just the same vote


def test_equal_scales_restore_nothing():
    """Any common scale leaves the coin flip in place: it is the ratio of scales that matters."""
    T, C = 2, 10
    a, b = sat(0, T, C)[None, :], sat(1, T, C)[None, :]
    for s in (0.5, 1.0, 3.66, 20.0):
        _, _, tie = product_rule(a, b, np.random.default_rng(0), scale_a=s, scale_b=s)
        assert tie[0]


# ---------------------------------------------------------------------------------------
# (d) How much a count readout can say
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,C", [(1, 2), (1, 4), (2, 3), (2, 5), (3, 4)])
@pytest.mark.parametrize("s", [1.0, 0.37, math.log(2.0)])
def test_enumeration_gives_exactly_the_number_of_probability_vectors(T, C, s):
    """Every scale, including the one at which top-label confidences collide."""
    vectors = np.array(list(itertools.product(range(T + 1), repeat=C)), dtype=float)
    probs = np.round(softmax_probs(s * vectors), 12)
    assert np.unique(probs, axis=0).shape[0] == n_distinct_probability_vectors(T, C) == (T + 1) ** C - T ** C


def test_almost_everything_a_count_readout_can_say_is_outside_its_confidence():
    assert n_distinct_probability_vectors(2, 10) == 58_025
    assert max_distinct_confidences(2, 10) == 55
    assert n_distinct_probability_vectors(2, 10) // max_distinct_confidences(2, 10) == 1055


# ---------------------------------------------------------------------------------------
# The second modality: event dropout and class-matched partners
# ---------------------------------------------------------------------------------------


def test_event_dropout_thins_counts_binomially_and_never_adds_an_event():
    rng = np.random.default_rng(0)
    frames = rng.poisson(0.4, (300, 2, 2, 34, 34)).astype(np.float32)
    for rate in (0.0, 0.3, 0.9):
        out = event_dropout(frames, rate, np.random.default_rng(1))
        assert out.shape == frames.shape and out.dtype == frames.dtype
        assert np.array_equal(out, np.rint(out)) and out.min() >= 0
        assert np.all(out <= frames)                              # an event can only be lost
        assert np.all(out[frames == 0] == 0)
        assert out.sum() / frames.sum() == pytest.approx(1.0 - rate, abs=0.004)
    assert np.array_equal(event_dropout(frames, 0.0, np.random.default_rng(2)), frames)
    again = event_dropout(frames, 0.3, np.random.default_rng(1))
    assert np.array_equal(again, event_dropout(frames, 0.3, np.random.default_rng(1)))


def test_event_dropout_refuses_what_is_not_an_event_count():
    with pytest.raises(ValueError):
        event_dropout(np.array([0.5, 1.0]), 0.2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        event_dropout(np.array([1.0, 2.0]), 1.0, np.random.default_rng(0))


def test_partners_share_the_class_are_never_the_sample_itself_and_are_used_once():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 10, 5000)
    partner = class_matched_partners(labels, np.random.default_rng(4))
    assert np.array_equal(labels[partner], labels)
    assert np.all(partner != np.arange(labels.size))
    assert np.array_equal(np.sort(partner), np.arange(labels.size))   # a permutation: each used once
    other = class_matched_partners(labels, np.random.default_rng(5))
    assert not np.array_equal(partner, other)
    with pytest.raises(ValueError):
        class_matched_partners(np.array([0, 0, 1]), np.random.default_rng(0))
