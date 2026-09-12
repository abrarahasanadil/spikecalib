# SpikeCalib — Research Design

Post-hoc confidence calibration for event-driven spiking neural networks.

This document is the authoritative design record for the study. It states what is
being claimed, what would falsify it, which datasets are in scope, how the work is
positioned against prior art, and which rules must hold for the reported numbers to
mean anything. Decisions recorded here are settled; where a decision is still open it
is marked as such.

Every numeric constant and factual claim below carries a provenance tag: `[LIT]` from
published literature, `[DERIVED]` from mathematics stated in this repository,
`[MEASURED]` from a run here, and `[UNVERIFIED]` for something believed to come from the
literature whose source has not been confirmed. `SOURCES.md` holds the full references
and the list of what still needs checking. Nothing marked `[UNVERIFIED]` may be cited in
the paper until verified. Code in this repository is expected to follow this document, and
where the two disagree, this document is what needs updating first.

---

## Claim under test

Temperature scaling assumes a continuous logit space. It takes a real-valued logit
vector, divides it by a single learned scalar, and relies on the softmax being able to
move smoothly in response. In an event-driven spiking neural network read out by spike
counts, that assumption does not hold. Each class output is an integer in `[0, T]`,
where `T` is the number of simulation timesteps. The logit vector is a point on an
integer lattice, not a point in a continuous space.

Discreteness has two distinct consequences, and it matters which one does the work.

### 1. The attainable confidence set is small, coarse, and bounded

With `C` classes and counts bounded by `T`, only finitely many softmax outputs are
reachable. At `T = 2` that set is small enough to enumerate. The set is also bounded
above: the most extreme count vector puts `T` on one class and `0` on the rest, so the
largest confidence the raw readout can emit is `e^T / (e^T + C - 1)`. For `C = 10` that
ceiling is **0.451 at `T = 2`**, 0.859 at `T = 4`, and only reaches 0.978 by `T = 6`.
`[DERIVED]` — softmax of the extreme count vector `[T, 0, ..., 0]`; not from any paper.

The scope of that ceiling must be stated precisely, because it is easy to overclaim.
**It bounds the raw readout, not what a calibrator can output.** Temperature scaling with
`t < 1` sharpens, and as `t` approaches 0 the softmax approaches one-hot, so confidence
approaches 1 — sharpening escapes the ceiling entirely. Isotonic can map any input level
to any value in `[0, 1]`. Neither calibrator is trapped by it.

What the ceiling does explain is the shape of the *uncalibrated* error. A model whose
accuracy exceeds 0.451 at `T = 2` cannot emit a correctly confident output at all, so it
is structurally underconfident before any calibration. At high `T` the ceiling is
effectively absent and the familiar overconfidence returns. **The direction of
miscalibration therefore flips across the sweep**, which is why the study reports the
signed calibration gap and not only `|ECE|`: averaging the two regimes would cancel them.
This is observed, not merely argued — on synthetic counts the fitted temperature was
0.556 at `T = 2` (sharpening) and 1.696 at `T = 10` (softening). `[MEASURED]` —
`scripts/synthetic_check.py`, seed 0, uniform mode, 2026-09-09; synthetic data.

#### The ceiling is also a floor on the training loss

A consequence worth stating because it changes how convergence is judged. Cross-entropy on
spike-count logits cannot fall below `-ln(ceiling)`: the best reachable logit vector is
`[T, 0, ..., 0]`, whose softmax gives the true class exactly the ceiling probability. For
`C = 10` that floor is **0.797 at `T = 2`** and **0.153 at `T = 4`**.

Measured: the `T = 2` model plateaued at a training loss of 0.806 and the `T = 4` model at
0.166 — each within 0.014 of its own floor. `[MEASURED]` `scripts/train_one.py`, seed 0,
2026-09-09.

So **training loss is not comparable across `T`**, and a high loss at low `T` is not
evidence of underfitting. Convergence in the sweep is judged by validation accuracy, never
by loss.

### 2. The number of distinct confidence levels is small at low `T` — the mechanism

This is the consequence that does the work, and the argument is about **capacity**.

**Temperature scaling has one parameter and a rigid functional form.** It can rescale the
whole confidence curve and nothing else. Where the miscalibration is uniform — the same
direction and roughly the same magnitude across the confidence range — one scalar is
genuinely sufficient, and isotonic's extra flexibility buys nothing. Where the
miscalibration *varies with confidence* — overconfident at the top of the range,
underconfident at the bottom — no single `t` can correct both ends at once, because
raising confidence everywhere is the only move available.

**Isotonic regression is non-parametric.** Its effective degrees of freedom are bounded
above by the number of distinct confidence values in the calibration set, and reduced
below that by the pool-adjacent-violators step that enforces monotonicity. It can set
each distinct confidence level to its own empirical accuracy.

The inversion is the point. The standard objection to isotonic regression is variance:
with many distinct levels and few samples supporting each one, it fits noise, and that
is exactly why it is usually considered the riskier choice on small calibration sets.
Discreteness removes that objection at low `T`. Few distinct levels means each level is
estimated from many samples.

On a trained N-MNIST model, `T = 2` gave **8** distinct confidence levels on a
5,000-sample calibration split — about **625 samples per level**, so each per-level
accuracy is well conditioned. `T = 4` gave **29** levels, roughly **172 samples per
level**. `[MEASURED]` — `scripts/readout_report.py`, seed 0, 2026-09-10.

Earlier figures quoted here (52 and 3,035 levels, from the synthetic check) were inflated:
softmax summation order split one attainable confidence into several float64 values that
differed at ~1e-16. Confidences are now quantised at source; see `LOG.md` 2026-09-10.

So discreteness, which normally penalises isotonic, protects it — and it does so in
precisely the low-timestep regime where SNNs are energy-competitive. The usual objection
to the method evaporates exactly where the method is needed.

### What tie rate is, and what it is not

Argmax ties are frequent at low `T` and the tie rate is reported, but **it is a
descriptive measure of how discrete the output is, not the reason isotonic wins.**

Neither calibrator breaks ties, so the tie rate cannot distinguish them. Temperature
scaling divides every logit by the same positive scalar, which is order-preserving, so
two classes on equal counts stay equal at every temperature. Isotonic top-label computes
`conf' = f(conf)` for a monotone `f`, so two samples with identical counts have identical
confidence and receive identical output from it as well. Both calibrators hold the
prediction fixed by design; tie-breaking happens once, before either of them runs, under
the experiment seed (see "Tie-breaking" under resolved design decisions).

An earlier version of this document offered "ties cannot be broken by a scalar" as the
mechanism separating the two methods. That was wrong for the reason just given, and the
correction is recorded in `LOG.md`. Tie rate stays in the results table as evidence that
the output really is discrete and as a measure of how coarse it is — it is what makes the
capacity argument's premise visible — but the mechanism is capacity.

## Hypothesis

**The isotonic advantage over temperature scaling is governed by the number of
calibration samples per distinct confidence level, `n_cal / n_distinct`.** Lower `T`
means fewer distinct levels, hence more samples supporting each level, hence a
better-conditioned isotonic fit. So the advantage is expected to grow as `T` falls — but
`T` is the mechanism's *proxy*, and the ratio is the mechanism.

If that holds, isotonic regression is the correct calibration method precisely in the
low-timestep regime, which is the regime where SNNs are energy-competitive in the first
place and therefore the regime anyone deploying one actually cares about. The argument is
then not merely that isotonic calibrates better on average, but that it is best where it
matters most, for a reason that can be stated and measured.

The hypothesis is falsifiable in three distinct ways, and the study reports whichever
occurs:

1. **Flat in `T`.** If isotonic wins by roughly the same margin at `T = 10` as at
   `T = 2`, or wins nowhere, the discreteness account is not driving the difference and
   the hypothesis is false.
2. **Tracks `T` but not the ratio.** This is the sharper test, and it is the one that
   distinguishes the capacity explanation from any other story about low `T`. The ratio
   can be moved *independently of `T`* by changing the calibration-set size: at fixed `T`,
   shrinking `n_cal` lowers `n_cal / n_distinct` and should shrink the isotonic advantage.
   If the advantage tracks `T` but is indifferent to `n_cal` at fixed `T`, then something
   other than capacity is producing the trend and the stated mechanism is wrong even
   though its prediction appeared to hold.
3. **Uniform miscalibration.** The advantage requires miscalibration whose *shape* varies
   with confidence. Where the error is a uniform offset, one scalar is sufficient by
   construction and no amount of extra capacity helps. So the study reports the shape of
   the miscalibration, not only its magnitude; an isotonic advantage that vanishes under
   uniform miscalibration confirms the mechanism rather than undermining it.

A flat trend, or a trend that fails the ratio test, is a publishable result about a
plausible mechanism that turns out not to matter. It is not a reason to reframe the
question after the fact.

### What the synthetic pre-test already shows

The capacity mechanism was tested on synthetic counts before any model existed
(`scripts/synthetic_check.py`), and **it did not survive that test cleanly**. This is
recorded here rather than in the log alone, because it changes what the real experiment
has to measure.

Three observations, all from synthetic data with a smooth monotone true calibration curve:

1. **Isotonic's realised capacity is small and roughly constant.** The pool-adjacent-
   violators step collapses the fit far below the `n_distinct` upper bound. Across runs
   where `n_distinct` spanned 34 to 4,197 — a factor of 123 — the number of distinct
   fitted output values stayed between 4 and 22, with no trend. Isotonic self-regularises:
   the parameters it actually uses are set by the complexity of the true calibration
   curve, not by how many confidence levels exist. **`n_distinct` is therefore a very
   loose bound, and `n_cal / n_distinct` may be the wrong statistic.** The realised block
   count is reported alongside it so both can be checked.

2. **The advantage was indifferent to the ratio at fixed `T = 10`.** Moving `n_cal` from
   250 to 20,000 took samples-per-level from 1.1 to 4.8 and left the isotonic advantage
   flat at 0.137–0.159. At 1.1 samples per level — essentially one sample per confidence
   value — isotonic did not overfit and did not lose. At `T = 2` there was a weak positive
   trend (0.008 to 0.027 as the ratio went 7.4 to 350.9), so the ratio is not irrelevant,
   but it is not the driver the hypothesis names.

3. **`T` is confounded with miscalibration severity.** In the conditional generator the
   uncalibrated ECE rose from 0.084 at `T = 2` to 0.276 at `T = 10` — 3.3x more error to
   correct — because a wider count range gives the injected miscalibration more room to
   vary. The isotonic advantage rose with `T` in step with that severity, which is a
   different explanation from capacity and fits the data better here.

None of this refutes the hypothesis for real SNNs: a synthetic generator with a smooth
monotone calibration curve is the best possible case for isotonic's self-regularisation,
and real event data may not be so kind. But it does impose two requirements on the real
experiment. **Realised isotonic block count is reported, not just `n_distinct`.** And the
isotonic advantage is compared **at matched uncalibrated ECE** wherever possible, because
otherwise a trend in the advantage across `T` cannot be separated from a trend in how much
miscalibration exists to remove.

## Pre-registered prediction, recorded 2026-09-10

**This is written before any calibrator has been fitted to a trained model, and it must
not be revised afterwards.** If the sweep contradicts it, the sweep is right and this
section stands as a record of what was expected and why. The whole point of dating it is
that a prediction which can be edited after the result is not a prediction.

### What was measured first

`scripts/readout_report.py`, N-MNIST seed 0, calibration split of 5,000 samples.
`[MEASURED]` 2026-09-10.

| | `T = 2` | `T = 4` |
|---|---|---|
| distinct confidence levels (cal) | 8 | 29 |
| share of calibration mass at the single top level | **95.0%** (4,750 / 5,000) | **89.0%** (4,449 / 5,000) |
| accuracy at that level | 0.9973 | 1.0000 |
| calibration gap at that level (accuracy − confidence) | +0.5464 | +0.1415 |
| mass-weighted mean gap over all levels | +0.5412 | +0.1449 |
| confidence std / mean (test) | 0.0609 | 0.0733 |
| AUROC of confidence for correctness (test) | 0.9106 | 0.9728 |

Two facts matter and they pull in opposite directions. Confidence **does** carry
information: AUROC is 0.91 and 0.97, so confidence orders correct predictions above
incorrect ones, and calibration is a meaningful question rather than a vacuous one. But
confidence is **extremely concentrated**: nine tenths of the calibration mass sits at a
single value, and the mass-weighted mean gap (+0.5412) is within 0.006 of the gap at that
one level (+0.5464).

### The prediction

Expected calibration error is mass-weighted. When ~90% of the mass sits at one confidence
level, ECE is very nearly the gap at that level, and **correcting that single level
correctly is almost the entire job**. A single scalar can do that: temperature scaling
sharpens `softmax(z/t)` and can map the dominant level anywhere in `(0, 1)`, including onto
its empirical accuracy. Isotonic's extra capacity buys it only the remaining ~10% of mass,
spread across 7 or 28 sparsely-populated levels where its per-level estimates are noisiest.

So, on N-MNIST:

> **Temperature scaling will match or beat top-label isotonic at every `T`, and the
> isotonic advantage will not be statistically resolvable at any `T`** — that is, the 95%
> paired-bootstrap interval on the equal-width ECE difference will straddle zero at
> `T` in {2, 4, 6, 8, 10}.

**This contradicts the study's own hypothesis**, which predicts an isotonic advantage that
grows as `T` falls. That is deliberate. The hypothesis was formed from the structure of the
readout; this prediction is formed from the measured distribution of the readout's outputs
on a trained model, and the measurement postdates the hypothesis. Recording the conflict
in advance is the only way the eventual result can settle it.

### Falsification

The prediction is wrong if, at any `T`, isotonic beats temperature by a margin whose 95%
paired-bootstrap CI excludes zero. Should that happen, the concentration argument above is
incorrect — most likely because the sparsely-populated tail of low-confidence samples
carries more ECE weight than its mass share suggests, or because temperature cannot reach
the dominant level's empirical accuracy without badly mis-mapping the tail.

One caveat stated in advance rather than discovered later: the per-level gap is **not**
constant. At `T = 2` it ranges from +0.154 to +0.698 across the eight levels. So this is
not a uniform-offset miscalibration in the strict sense, and a scalar is not guaranteed to
be sufficient. The claim rests specifically on the *mass weighting*, not on the shape of
the gap curve. If isotonic wins, this is the most likely reason.

### Outcome, recorded 2026-09-10 after the sweep

The prediction above is left exactly as written. This section records what happened.

The sweep ran 15 runs (5 timesteps x 3 seeds) at a 50-epoch cap with patience 5.
**All 15 stopped on the patience rule**, so no result below is confounded by one model
being less converged than another, and no seed trailed the best at its `T` by more than
0.0034. `[MEASURED]` `scripts/run_sweep.py --epochs 50 --patience 5`, 2026-09-10.

Mean equal-width ECE over 3 seeds:

| `T` | uncalibrated | temperature | isotonic top-label | isotonic − temperature |
|---|---|---|---|---|
| 2 | 0.5386 | **0.0018** | 0.0042 | **+0.0024** (temperature better) |
| 4 | 0.1447 | 0.0031 | 0.0029 | −0.0002 |
| 6 | 0.0273 | 0.0026 | 0.0027 | +0.0001 |
| 8 | 0.0072 | 0.0028 | 0.0031 | +0.0003 |
| 10 | 0.0029 | 0.0026 | 0.0028 | +0.0002 |

**The letter of the prediction is falsified; its substance holds.** **Thirteen** of the
fifteen paired-bootstrap intervals straddle zero. Two do not: one favours temperature
(`T = 2`, seed 1, CI [+0.0006, +0.0044]) and one favours isotonic (`T = 6`, seed 0, CI
[−0.0024, −0.0003]). *(Corrected 2026-09-11: this paragraph previously said "fourteen of
the fifteen" in the same sentence as "the two that do not", which cannot both be true. The
artifact has always said two are resolvable.)* The prediction said the interval would straddle zero at *every* `T`,
and at `T = 6` it did not, so the prediction as stated is wrong. But one resolvable result
in fifteen at a 95% level is what multiple comparisons produce on their own — the expected
count under no true difference is about 0.75 — and it does not survive as evidence of an
isotonic advantage.

**The study's own hypothesis fares worse.** It predicted an isotonic advantage that grows
as `T` falls. At `T = 2`, the most discrete regime, with 583 calibration samples per
distinct confidence level, **temperature scaling beat isotonic** by more than at any other
`T`. The advantage does not grow as `T` falls; it is zero from `T = 4` upward and negative
at `T = 2`.

**The capacity mechanism gets no support either.** Samples per distinct level ran 583, 162,
70, 30, 18 across the sweep — a 32-fold range, exactly the manipulation the refined
hypothesis said the advantage should track. The advantage was flat and near zero across all
of it.

Two things did hold, and both matter to the paper:

- **The ceiling result is dramatic.** Uncalibrated ECE at `T = 2` is **0.5386**, almost
  entirely structural underconfidence forced by the 0.4509 attainable-confidence ceiling.
  It falls to 0.0029 by `T = 10` as the ceiling recedes. Post-hoc calibration removes
  essentially all of it — both methods reach ~0.002–0.004 at every `T`. The finding is not
  "which calibrator", it is "**calibration matters enormously at low `T`, and which method
  you use does not**".
- **The two ECE estimators disagree exactly where predicted.** At `T = 2` equal-width gives
  temperature 0.0018 and isotonic 0.0042, while equal-mass gives 0.0008 and 0.0015 — the
  estimators differ by a factor of two on the same data, in the regime where confidence is
  most concentrated. Correctness rule 2 earns its place.

The honest summary for the paper: on N-MNIST the discreteness argument does not produce a
calibrator preference. Whether it does on DVS-Gesture, where the calibration split is
roughly 330 samples rather than 5,000 and isotonic's per-level estimates are far noisier,
is untested and is now the more interesting question.

## Pre-registered prediction 2, recorded 2026-09-11

**Written before the readout experiment was run, and before any alternative-readout model
existed.** Not to be revised afterwards.

### The question

The ceiling `e^T / (e^T + C - 1)` follows from applying a softmax to integer spike counts
bounded by `T`. But **softmax-over-spike-counts is a choice of readout, not a property of
spiking networks.** If the cap is an artifact of that choice, the study changes character:
from "here is a limitation of low-`T` SNNs" to "here is a limitation of one readout, and
here is the fix". That is the most consequential experiment in the study, so the
prediction is recorded first.

### Three readouts, identical network

Only the final aggregation differs. Every layer up to and including the last
`Linear → BatchNorm1d` is the same; changing anything else would confound the comparison.

| Readout | Aggregation | Attainable maximum | Derived ceiling | Granularity |
|---|---|---|---|---|
| **count** (current) | sum of output spikes over `T` | `T` | `e^T / (e^T + C - 1)` = **0.4509** at `T=2, C=10` | `T + 1` levels per class |
| **weighted-count** | `Σ_t w_t s_t`, `w_t = 2(t+1)/(T+1)` | `T` (weights sum to `T`) | **identical**, 0.4509 at `T=2` | up to `2^T` levels per class |
| **membrane** | final potential of a non-spiking LIF integrator | unbounded | **none**; supremum is 1 | continuous |

`[DERIVED]` The weighted-count weights are chosen so `Σ_t w_t = 2·(1+…+T)/(T+1) = T`,
giving it the **same cap as count but a far finer attainable set**. That separates two
confounded properties: is the problem that confidence is *capped*, or that it is *coarse*?
Count is capped and coarse; weighted-count is capped and fine; membrane is neither.

### The prediction

1. **Membrane readout removes the cap and most of the uncalibrated error.** At `T = 2`,
   membrane should reach confidences far above 0.4509, and its uncalibrated ECE should be
   **dramatically lower** than count's 0.5386 at comparable test accuracy. If so, the
   ceiling is a readout artifact and the study reports a fix, not just a limitation.
2. **Weighted-count stays near count.** It shares the cap, so if the cap is what matters
   its uncalibrated ECE at `T = 2` should remain near 0.5, despite being far
   finer-grained. If instead it lands near membrane, the problem was coarseness rather
   than the bound.
3. **Failure mode that would be more interesting.** If membrane shows the *same* severe
   underconfidence at `T = 2`, the effect is not a readout artifact at all — it is
   something deeper about directly-trained SNNs at low `T`, and that is a stronger result
   than the fix.

### Falsification

The prediction is wrong if membrane's uncalibrated ECE at `T = 2` is within a factor of
two of count's, at comparable accuracy. It is also wrong, in the opposite direction, if
weighted-count matches membrane rather than count — that would mean granularity, not the
bound, was doing the work, and the ceiling framing would need rewriting.

### Connection to the literature, stated in advance

The SNN literature reports **over**confidence as the characteristic failure. This study
measures the **opposite** at low `T`. If membrane readout removes the cap *and* restores
overconfidence, then both observations are the same phenomenon seen through different
readouts, and the apparent contradiction dissolves. That connection is to be **tested**,
not asserted, and it is written here before the data exists so that a positive result
cannot be constructed after the fact.

### Outcome of prediction 2, recorded 2026-09-11 after the run

The prediction above is left exactly as written. This records what happened.
`[MEASURED]` `scripts/readout_experiment.py`, seed 0, 2026-09-11, N-MNIST, 15 runs
(3 readouts x 5 timesteps), identical architecture throughout.

| `T` | readout | test acc | cap | max conf | conf − acc | ECE (ew) | levels |
|---|---|---|---|---|---|---|---|
| 2 | count | 0.9852 | 0.4509 | 0.4509 | **−0.5387** | **0.5387** | 10 |
| 2 | weighted-count | 0.9854 | 0.4509 | 0.4509 | **−0.5381** | **0.5381** | 17 |
| 2 | membrane | 0.9851 | none | **0.9976** | **−0.0333** | **0.0333** | 9,994 |
| 4 | count | 0.9886 | 0.8585 | 0.8585 | −0.1461 | 0.1463 | 37 |
| 4 | weighted-count | 0.9880 | 0.8585 | 0.8585 | −0.1438 | 0.1440 | 164 |
| 4 | membrane | 0.9873 | none | 1.0000 | −0.0024 | 0.0048 | 10,000 |
| 10 | count | 0.9899 | 0.9996 | 0.9996 | −0.0007 | 0.0028 | 339 |
| 10 | weighted-count | 0.9881 | 0.9996 | 0.9996 | +0.0004 | 0.0031 | 3,254 |
| 10 | membrane | 0.9909 | none | 1.0000 | −0.0017 | 0.0025 | 10,000 |

**Prediction 1 held.** Membrane readout removes the cap — observed maximum confidence
0.9976 against the count readout's 0.4509 — and cuts uncalibrated ECE at `T = 2` by
**16.2x**, from 0.5387 to 0.0333, at indistinguishable accuracy (0.9851 against 0.9852).
The falsification threshold was "within a factor of two"; the result is eight times clear
of it.

**Prediction 2 held, and it is the sharper of the two.** Weighted-count has the *same cap*
as count and 1.7x more attainable levels at `T = 2`, and its ECE is 0.5381 against count's
0.5387 — a difference of 0.0006. At `T = 6` it has 7x more levels and still tracks count.
**Granularity buys nothing; the bound is doing all the work.** That distinction was the
reason for constructing the weighted readout, and it is what licenses calling the effect a
*ceiling* rather than a *quantisation* effect.

**The readouts converge once the ceiling stops binding.** At `T = 10` all three land
between 0.0025 and 0.0031, because the cap (0.9996) is no longer below the accuracy. The
effect is present only where the bound binds, which is the strongest available evidence
that the bound is the mechanism.

**So the ceiling is a property of the readout, not of spiking networks.** The study now
reports a limitation *and* a fix: at low `T`, use a membrane readout. The cost is that the
output is no longer an integer spike count, which matters for neuromorphic hardware that
transmits spikes rather than potentials — that trade is real and is not evaluated here.

### The connection to prior work: tested, and it does not hold

The prediction above said that if membrane removes the cap **and restores overconfidence**,
then the literature's overconfidence reports and this study's underconfidence measurement
would be the same phenomenon seen through different readouts, and the contradiction would
dissolve. That was to be tested rather than asserted.

**It was tested. It does not hold** — though at three seeds the result is more qualified
than at one, and the qualification is recorded here rather than smoothed over.

`[MEASURED]` The membrane readout's **mean** confidence stays below its accuracy at every
timestep setting: −0.0289, −0.0010, −0.0013, −0.0036 and −0.0008 at `T` = 2, 4, 6, 8 and
10, averaged over three seeds. But **only at `T = 2` is that gap larger than the spread
between seeds.** From `T = 4` up the individual seeds straddle zero, and six of the 45
cells in the experiment are positive:

| readout | `T` | seed | `conf − acc` |
|---|---|---|---|
| membrane | 4 | 2 | +0.0020 |
| membrane | 8 | 2 | +0.0009 |
| membrane | 10 | 2 | +0.0020 |
| weighted-count | 10 | 0 | +0.0004 |
| weighted-count | 10 | 1 | +0.0003 |
| count | 10 | 2 | +0.0002 |

At `T = 2`, where the cap binds hardest, all three membrane seeds are underconfident and
removing the cap shrinks the gap by 11.9× to 63.7×. That is the one regime where the
comparison with the literature is unambiguous.

*(Corrected 2026-09-11. This paragraph previously read "underconfident at all five timestep
settings and overconfident at none", gave seed 0's five values as though they were the
experiment's, said the shrinkage was "a factor of sixteen" — seed 0's 16.2× — and asserted
that "the single positive value anywhere in the experiment" was weighted-count at `T = 10`.
Three seeds had been run when that was written; only one had been read. The pre-registered
prediction above is untouched.)*

So the disagreement with the literature on the *sign* of SNN miscalibration is **not**
explained by the readout choice, and the tidy resolution is unavailable. What can be said
is narrower and still useful: the *magnitude* of the underconfidence this study reports is
almost entirely a readout artifact, and once the readout is changed the residual
miscalibration is small and **not consistently signed** — at `T ≥ 4` this study cannot
resolve its direction at all. Why the prior reports find overconfidence where
this finds near-zero remains unexplained, and the most likely candidates — different
timestep regimes, converted rather than directly-trained networks, harder tasks where
accuracy is far from the ceiling — are untested here.

## Readout comparison at three seeds, and what it changes

`[MEASURED]` `scripts/readout_experiment.py --seeds 0 1 2`, 2026-09-11, N-MNIST, 45 runs
(3 readouts x 5 timesteps x 3 seeds). Mean +- sd across seeds.

| `T` | readout | test acc | conf − acc | ECE (ew) | AUROC | AURC | operating points |
|---|---|---|---|---|---|---|---|
| 2 | count | 0.9850±0.0005 | −0.5386±0.0004 | 0.5386±0.0004 | 0.9204±0.0042 | 0.00236±0.00005 | 9±1 |
| 2 | weighted-count | 0.9851±0.0006 | −0.5385±0.0008 | 0.5385±0.0008 | 0.9030±0.0110 | 0.00289±0.00046 | 17±0 |
| 2 | **membrane** | 0.9851±0.0006 | **−0.0289±0.0153** | **0.0290±0.0153** | **0.9659±0.0045** | **0.00072±0.00016** | **9998±3** |
| 4 | count | 0.9887±0.0009 | −0.1446±0.0011 | 0.1447±0.0012 | 0.9735±0.0025 | 0.00048±0.00001 | 37±2 |
| 4 | **membrane** | 0.9872±0.0002 | **−0.0010±0.0021** | **0.0038±0.0009** | **0.9772±0.0008** | **0.00040±0.00001** | 10000±0 |
| 6 | count | 0.9902±0.0004 | −0.0272±0.0011 | 0.0273±0.0011 | **0.9842±0.0010** | **0.00021±0.00002** | 98±2 |
| 6 | membrane | 0.9891±0.0008 | −0.0013±0.0007 | **0.0031±0.0002** | 0.9710±0.0034 | 0.00046±0.00007 | 10000±0 |
| 8 | count | 0.9898±0.0010 | −0.0065±0.0011 | 0.0072±0.0004 | **0.9837±0.0014** | **0.00022±0.00001** | 221±23 |
| 8 | membrane | 0.9882±0.0005 | −0.0036±0.0048 | **0.0054±0.0035** | 0.9772±0.0009 | 0.00035±0.00002 | 10000±0 |
| 10 | count | 0.9908±0.0006 | −0.0008±0.0009 | **0.0029±0.0005** | **0.9863±0.0012** | **0.00017±0.00001** | 372±27 |
| 10 | membrane | 0.9890±0.0014 | −0.0008±0.0021 | 0.0039±0.0010 | 0.9773±0.0043 | 0.00033±0.00009 | 10000±0 |

### The membrane advantage at `T = 2` survives, with a wide spread

Per-seed ratios of count ECE to membrane ECE at `T = 2`: **16.2x, 11.9x, 63.5x** — minimum
11.9x, mean 30.5x. Every seed clears the factor-of-two falsification threshold by a wide
margin, so the finding is not an artifact of seed 0.

But the **spread is large and must be reported**: membrane ECE at `T = 2` is
0.0290 ± 0.0153 across seeds — 0.0085, 0.0333 and 0.0451, a fivefold range. Count's is
0.5386 ± 0.0004, which is almost perfectly stable. Removing the cap removes a *deterministic*
error and exposes an ordinary, *variable* one. The headline "16.2x" from the seed-0 run is
therefore the middle of a wide distribution, and the paper should quote the range.

### Three of fifteen cells contradict the pattern, and they are all at high `T`

Membrane is **worse** than count at `(T=8, seed 1)`, `(T=10, seed 1)` and `(T=10, seed 2)`.
Reported rather than averaged away. All three are at timestep settings where the ceiling no
longer binds — at `T = 10` the cap is 0.9996 against an accuracy of 0.9908 — so there is no
structural error left for the membrane readout to remove, and it is simply a different model
with ordinary variance. At `T = 10` the **mean** also favours count, 0.0029 against 0.0039.

### Membrane is NOT strictly dominant, and the recommendation narrows accordingly

This is the result that changes the advice, and it only appeared with the extra seeds and
the per-readout selective metrics.

| `T` | better ECE | better AUROC | better AURC |
|---|---|---|---|
| 2 | membrane | **membrane** (0.9659 vs 0.9204) | **membrane** (0.00072 vs 0.00236) |
| 4 | membrane | **membrane** (0.9772 vs 0.9735) | **membrane** (0.00040 vs 0.00048) |
| 6 | membrane | **count** (0.9842 vs 0.9710) | **count** (0.00021 vs 0.00046) |
| 8 | membrane | **count** (0.9837 vs 0.9772) | **count** (0.00022 vs 0.00035) |
| 10 | **count** | **count** (0.9863 vs 0.9773) | **count** (0.00017 vs 0.00033) |

**Membrane calibrates better wherever the ceiling binds, and ranks better only at
`T` in {2, 4}.** From `T = 6` upward the count readout separates correct from incorrect
predictions *better* — AUROC 0.9842 against 0.9710 at `T = 6`, and AURC roughly half.
Calibration and ranking are different properties, and the readout that wins one does not
automatically win the other. Why the ranking reverses is a separate question, taken up
below in two sections. The first candidate explanation — that the count readout's
coarseness is denoising its confidence — was tested and falsified. The second — that the
membrane integrator's **leak** shortens its evidence window — was tested and accounts for
**about half** of the deficit at `T = 6` and **62%** at `T = 10`. What survives the removal
of the leak is attributable to the output threshold itself, and is not yet explained.

So the claim is **not** "membrane readout is better". It is:

> Where the ceiling binds — accuracy above `e^T/(e^T + C − 1)`, which for 10 classes means
> `T ≤ 4` — a membrane readout removes a large structural calibration error and also ranks
> at least as well. Above that point the cap is inert, the count readout ranks better, and
> there is no reason to change it.

`T_min` is exactly the test for which regime you are in.

### The second deployment consequence: operating points

Count gives **9 ± 1** achievable selective-prediction operating points at `T = 2` across
10,000 test samples. Membrane gives **9,998 ± 3** — effectively a continuum, since the
readout is real-valued and almost every sample has its own confidence. A practitioner who
needs to deploy at a specific coverage can do so with a membrane readout and cannot with a
count readout at low `T`. That is a separate consequence of the same choice from the
calibration one, and it points the same way at low `T`.

## Why membrane ranks worse at high `T`: it is not quantisation

Membrane calibrates better wherever the ceiling binds and *ranks* worse from `T = 6`
upward. Nothing in the ceiling argument predicts that, and it was left open. The first
candidate explanation is attractive because it needs no new mechanism: a spike count is a
thresholded membrane signal, thresholding discards information, and if what it discards is
noise then the count readout is getting a free denoiser. Once `T` is large enough for the
count to have the levels it needs to express the ordering, discarding would be pure gain,
while membrane keeps the variance the count threw away.

### What the test had to show

The test coarsens the *saved* membrane confidences — no retraining — to `N` ordered levels
for `N` in {5, 10, 20, 50, 100, 200, 500, 1000, full}, and recomputes AUROC and AURC at
each. Equal-mass binning is primary, so that every one of the `N` levels is actually
populated; equal-width is reported as a check. Under both, samples sharing a confidence
always share a level, because splitting them would invent an ordering the model never
expressed.

The arithmetic makes the prediction exact rather than impressionistic. Coarsening is a
monotone, non-injective map: it cannot reorder a pair, only merge pairs into ties, and
AUROC scores a tie as 0.5. So, with `W` the fraction of correct/incorrect pairs merged and
`A` the pooled AUROC *within* levels at full precision:

    AUROC(coarsened) − AUROC(full) = W × (0.5 − A)

That is an identity, not an approximation, and it is asserted in the tests on three
constructed regimes. It says coarsening can only help where `A < 0.5` — where the fine
structure is actively **worse than chance** at ordering correct from incorrect. The
hypothesis, stated exactly, is `A < 0.5`. If instead `A ≈ 0.5` the fine detail is merely
uninformative and AUROC is flat in `N`; if `A > 0.5` it is signal and coarsening costs.

**Predicted signature if the hypothesis holds:** AUROC rises as `N` falls from full, peaks
somewhere in the tens or low hundreds, then falls once the binning is too coarse to carry
the ordering — with the peak near count's own level count at that `T`.

### The signature does not appear at any `T` or any seed

`[MEASURED]` `scripts/quantisation_test.py`, 2026-09-11, N-MNIST, 5 `T` × 3 seeds,
membrane AUROC by level count, mean over seeds:

| `T` | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N=1000 | full | count |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.8970 | 0.9413 | 0.9587 | 0.9648 | 0.9654 | 0.9658 | 0.9659 | 0.9659 | **0.9659** | 0.9205 |
| 4 | 0.8999 | 0.9476 | 0.9675 | 0.9759 | 0.9768 | 0.9771 | 0.9772 | 0.9772 | **0.9772** | 0.9735 |
| 6 | 0.8939 | 0.9403 | 0.9604 | 0.9691 | 0.9705 | 0.9710 | 0.9710 | 0.9710 | **0.9710** | 0.9842 |
| 8 | 0.9008 | 0.9497 | 0.9687 | 0.9759 | 0.9768 | 0.9770 | 0.9772 | 0.9772 | **0.9772** | 0.9837 |
| 10 | 0.9020 | 0.9483 | 0.9683 | 0.9754 | 0.9768 | 0.9772 | 0.9773 | 0.9773 | **0.9773** | 0.9863 |

AUROC **rises monotonically with `N` and saturates at full precision**, at every `T` and
every seed. There is no interior peak anywhere in the grid. The curve is not flat in `N`
either, which is a stronger refutation than the flat case anticipated: coarsening does not
fail to help, it actively **costs**, and the cost grows as the binning gets coarser.

Coarsening did nudge AUROC up in 10 of 15 cells, and the number is the point: **the
largest gain anywhere is +0.000073** (`T = 6`, seed 1, `N = 200`). The deficit it would
have to close at `T = 6` is 0.0132. It is **180 times too small**, and the sign flutter
across cells is what floating-point noise around zero looks like. Equal-width binning
agrees: its best gain anywhere is +0.0013, still an order of magnitude short, and it is
worse than equal-mass everywhere the two can be compared.

### Why, from the identity

`[MEASURED]` the within-level AUROC, mean over seeds and `T`, against the fraction of
correct/incorrect pairs each resolution merges:

| N | pairs merged | within-level AUROC |
|---|---|---|
| 5 | 19.06% | 0.8935 |
| 10 | 9.01% | 0.8137 |
| 20 | 4.16% | 0.7158 |
| 50 | 1.53% | 0.5996 |
| 100 | 0.75% | 0.5633 |
| 200 | 0.37% | 0.5269 |
| 500 | 0.14% | 0.5230 |
| 1000 | 0.07% | 0.5049 |

**The fine structure of the membrane confidence is signal, not noise, at every resolution
that merges a meaningful number of pairs.** At `N = 5`, where a fifth of all pairs are
merged, the detail being destroyed orders correct from incorrect at AUROC 0.89. Of 266
measured points, 24 fall below 0.5 — all at `N ≥ 200`, where under 1% of pairs are merged,
so the largest AUROC shift any of them could produce is 0.0013. There is no resolution at
which membrane's extra precision is misleading.

### What this rules in, not just out

Coarsening membrane to **count's own level count** is the sharpest form of the comparison,
and it separates "how finely" from "what":

| `T` | membrane at count's N | membrane, full | count | gap to count |
|---|---|---|---|---|
| 2 | 0.9367 | 0.9659 | 0.9204 | **+0.0164** |
| 4 | 0.9745 | 0.9772 | 0.9735 | +0.0009 |
| 6 | 0.9705 | 0.9710 | 0.9842 | **−0.0138** |
| 8 | 0.9771 | 0.9772 | 0.9837 | −0.0067 |
| 10 | 0.9773 | 0.9773 | 0.9863 | −0.0091 |

At `T = 2` membrane restricted to count's nine or ten levels **still beats count** by
0.0164. At `T = 6` membrane given count's ninety-odd levels is **still 0.0138 behind**.
Held at identical resolution the two readouts rank differently in both directions, so the
difference cannot be resolution.

**Conclusion: quantisation is not the mechanism.** The remaining explanation is that the
two readouts rank *different quantities*. A count is a sum of thresholded events, so it is
invariant to how far above threshold each crossing was; the membrane potential is not.
Whether that invariance is what helps at high `T` — and why it would only help once the cap
stops binding — is untested. It is the next question, and this result does not answer it.

*(The C axis adds one constraint on any answer: at `C = 101` the accumulator loses to count
on accuracy by 4 points at `T` in {6, 10}, the same direction as its ranking deficit at
`C = 10`. Whatever the threshold is doing, it is not specific to ten classes.)*

## Pre-registered prediction 3, recorded 2026-09-11

Written before the non-leaky readout was trained. Two explanations for the high-`T`
ranking deficit were confounded until now, and this separates them.

### The measurement that motivates it

`[MEASURED]` The membrane readout uses the same charge equation as every other neuron,
`v += (z - v)/tau`, with `tau = 2.0`. Expanding the recurrence, the input `k` steps before
the end enters the final potential with weight `(1/tau)(1 - 1/tau)^k = 0.5 × 0.5^k`.

| property | value |
|---|---|
| per-step retention `1 - 1/tau` | 0.5 |
| half-life | **exactly 1 step** |
| exponential time constant | 1.443 steps |

So the readout **leaks, and leaks hard**. The share of the final potential contributed by
the last two steps, against the count readout, which weights every step equally at `1/T`:

| `T` | leaky membrane, last 2 steps | count, last 2 steps |
|---|---|---|
| 2 | **100.0%** | **100.0%** |
| 6 | 76.2% | 33.3% |
| 10 | 75.1% | 20.0% |

**At `T = 2` the two readouts integrate over exactly the same window.** By `T = 6` they
have come apart, and membrane is reading essentially the last two steps whatever `T` is.
That coincides with the observed crossover, which is what makes leak a live hypothesis
rather than a guess.

### The hypothesis

The membrane readout's effective evidence window is shorter than `T`. At low `T` there is
little history to discard, so the leak costs nothing and membrane's freedom from the cap
wins. At high `T` the count integrates all `T` steps equally while the potential is
dominated by recent input, so count ranks better.

### The control

A fourth readout, `accumulator`: the same non-spiking integrator with the leak removed,
`v += z`. Every step carries equal weight, as in count. Everything else — every layer up to
and including the final `BatchNorm1d`, the training settings, the splits — is identical.

This is the control that separates the two confounded explanations, because:

- **accumulator vs count** differ *only* by the output threshold (both weight steps equally);
- **accumulator vs membrane** differ *only* by the leak (both are unthresholded).

### The prediction

1. If leak is the mechanism, the accumulator should **match or beat count at `T = 6` and
   `T = 10` on AUROC and AURC**, while keeping membrane's calibration advantage at `T = 2`.
2. The accumulator should beat the leaky membrane at `T = 6` and `T = 10`. If the leak is
   what costs membrane its ranking, removing it must recover something.
3. At `T = 2` accumulator and membrane should be close, since the windows nearly coincide
   there and the leak has almost nothing to discard.

### Falsification

Wrong if the accumulator still ranks below count at `T = 6` and `T = 10` — specifically if
its AUROC deficit against count remains more than half the leaky membrane's deficit
(membrane is 0.0132 behind at `T = 6` and 0.0090 at `T = 10`, so "more than half" means a
remaining deficit above 0.0066 and 0.0045 respectively). In that case the leak is not the
explanation and the difference is something about *what the two quantities measure* — a
count is invariant to how far above threshold each crossing was, and a potential is not.

Also wrong, in the opposite direction, if the accumulator loses membrane's `T = 2`
calibration advantage, which would mean the leak was doing load-bearing work at low `T`
that the ceiling argument did not account for.

## Outcome of prediction 3, recorded 2026-09-11 after the run

`[MEASURED]` `scripts/readout_experiment.py --readouts accumulator --timesteps 2 6 10
--seeds 0 1 2`, 9 runs, 2026-09-11. The prediction above is unedited.

| `T` | readout | test acc | `conf − acc` | ECE (ew) | AUROC | AURC |
|---|---|---|---|---|---|---|
| 2 | count | 0.9850 | −0.5386 | 0.5386 | 0.9204 | 0.00236 |
| 2 | membrane (leaky) | 0.9851 | −0.0289 | 0.0290 ± 0.0153 | 0.9659 | 0.00072 |
| 2 | **accumulator** | 0.9861 | **+0.0000** | **0.0031 ± 0.0002** | 0.9698 | 0.00059 |
| 6 | count | 0.9902 | −0.0272 | 0.0273 | **0.9842** | **0.00021** |
| 6 | membrane (leaky) | 0.9891 | −0.0013 | 0.0031 | 0.9710 | 0.00046 |
| 6 | **accumulator** | 0.9888 | **+0.0032** | 0.0042 | 0.9781 | 0.00032 |
| 10 | count | 0.9908 | −0.0008 | 0.0029 | **0.9863** | **0.00017** |
| 10 | membrane (leaky) | 0.9890 | −0.0008 | 0.0039 | 0.9773 | 0.00033 |
| 10 | **accumulator** | 0.9879 | **+0.0047** | 0.0050 | 0.9828 | 0.00029 |

### The prediction passes its own test and fails its own headline

These two statements are both true and they were both in the prediction, so both are
reported:

- **The stated falsification condition is not met, so the prediction stands.** It said the
  prediction was wrong if the accumulator's AUROC deficit against count remained *more than
  half* the leaky membrane's, and named the thresholds 0.0066 at `T = 6` and 0.0045 at
  `T = 10`. Measured: **0.0061 at `T = 6`** and **0.0035 at `T = 10`**. Removing the leak
  closes **53%** of the deficit at `T = 6` and **62%** at `T = 10`. Both clear the bar, the
  `T = 6` one narrowly.
- **The headline wording is not satisfied.** It said the accumulator "should match or beat
  count at `T = 6` and `T = 10`". It does not. Count ranks better in **every seed at both
  timestep settings** — 0/6 wins. The gap narrows and does not close.

So the leak is **a** mechanism, and the largest single one identified, but it is not the
whole mechanism. Roughly 40% of the deficit survives with the leak removed, and that
residue is what the remaining explanation has to account for: an accumulator and a count
differ *only* by the output threshold, both weighting every step equally, so whatever is
left is attributable to the thresholding itself.

Part 2 of the prediction held cleanly: the accumulator beats the leaky membrane on AUROC at
`T = 6` and `T = 10` in **3 of 3 seeds each**.

### Part 3 was wrong, and being wrong is the most useful thing here

Part 3 said accumulator and membrane should be *close* at `T = 2`, because their windows
nearly coincide there. On ranking they are: AUROC 0.9698 against 0.9659. **On calibration
they are not close at all, and the difference answers a question left open by the previous
session.**

| `T = 2` | ECE per seed | mean | sd |
|---|---|---|---|
| count | 0.5387, 0.5380, 0.5390 | 0.5386 | 0.0004 |
| membrane (leaky) | 0.0333, 0.0451, 0.0085 | 0.0290 | **0.0153** |
| accumulator | 0.0032, 0.0028, 0.0033 | **0.0031** | **0.0002** |

The open question was *why membrane's `T = 2` ECE is so seed-variable* — a fivefold range —
when count's is stable to 0.0004. **The leak is the answer.** Removing it drops the mean
ECE ninefold and the standard deviation **77-fold**, to a spread narrower than count's. A
leaky integrator at `T = 2` is reading essentially the final step, so its calibration rides
on whatever that one step happened to produce; an accumulator averages the evidence and the
seed dependence disappears with it.

### An unpredicted result that bears on the literature disagreement

`[MEASURED]` The accumulator is the **first readout in this study that is consistently
overconfident**:

| `T` | `conf − acc` per seed | positive in |
|---|---|---|
| 2 | −0.0015, +0.0001, +0.0014 | 2 of 3 |
| 6 | +0.0022, +0.0028, +0.0046 | **3 of 3** |
| 10 | +0.0047, +0.0042, +0.0051 | **3 of 3** |

Every other readout measured here is underconfident or unsigned. This does not resolve the
disagreement with the literature, but it narrows it usefully: **an uncapped, non-leaky
readout at `T ≥ 6` is mildly overconfident**, which is the direction the SNN calibration
literature reports. The magnitude is small — under half a percentage point — and one
architecture on one dataset cannot carry the claim. It is recorded as a lead, not a finding.

### What this costs

The accumulator is not free. Test accuracy is 0.9879 ± 0.0009 at `T = 10` against count's
0.9908 ± 0.0006, and 0.9888 against 0.9902 at `T = 6` — a consistent shortfall of about
0.2 points at high `T`, though it is *higher* than count at `T = 2` (0.9861 against 0.9850).
Nothing here recommends replacing the count readout at high `T`: count still ranks best,
still costs least, and is what the hardware exposes.

## The C axis: N-Caltech101 at 101 classes

The ceiling `e^T/(e^T + C − 1)` has two variables and until now only `T` had been varied.
Every measurement in the study was at `C = 10`, which left the formula half tested and made
the central claim rest on a single value of its second argument. N-Caltech101 has **101
classes**, where the `T = 2` ceiling is **0.068806** against 0.450853 at `C = 10`.

`[MEASURED]` `scripts/caltech_experiment.py`, 2026-09-11, 18 runs: 2 readouts × 3 `T` × 3
seeds. All 18 stopped on the patience rule between 14 and 40 epochs against a 50 cap, so
nothing below is confounded by unequal convergence.

### The ceiling holds, and the distinction between *valid* and *saturated* matters

| `T` | derived ceiling | observed max (mean) | max abs deviation | saturated? |
|---|---|---|---|---|
| 2 | 0.068806 | 0.068806 | 2.36e-14 | yes, to float precision |
| 6 | 0.801362 | 0.801362 | 3.83e-13 | yes, to float precision |
| 10 | 0.995481 | 0.993992 | 2.35e-03 | **no** |

**The bound was respected in 9 of 9 count runs**, and at `T` in {2, 6} the models saturate
it to the same precision they did at `C = 10` — one part in `10^13`. The formula is not
wrong at 101 classes.

At `T = 10` the observed maximum sits 1.5e-3 *below* the ceiling rather than on it. That is
not a failure and must not be reported as one: the ceiling is an **upper bound**, and
reaching it requires some test sample to produce the extreme count vector `[T, 0, ..., 0]`
— one class firing at every step while **all 100 others stay silent for all 10 steps**. At
`C = 10` that is easy and every run achieved it. At `C = 101` it is not, so the bound stops
being tight. Validity and tightness are different properties and only the first is a claim
this study makes.

### Where it binds, and the discovery that the residual is not small

A ceiling forces underconfidence only while accuracy exceeds it.

| `T` | ceiling | test accuracy | binds? | `conf − acc` | ECE (ew) |
|---|---|---|---|---|---|
| 2 | 0.068806 | 0.4543 | **3/3 seeds** | −0.3969 | 0.3969 |
| 6 | 0.801362 | 0.6427 | 0/3 | −0.1534 | 0.1539 |
| 10 | 0.995481 | 0.6435 | 0/3 | **+0.0344** | 0.0442 |

At `T = 2` the ceiling binds hard and accounts for **97.1%** of uncalibrated ECE — the same
story as `C = 10`, where it accounted for 99.2%.

**At `T = 6` it does not bind, and the study's `C = 10` intuition fails.** On N-MNIST, once
accuracy fell below the ceiling the miscalibration collapsed to 0.0029. Here accuracy is
0.6427 against a ceiling of 0.8014 — comfortably below it, so nothing is forced — and the
readout is **still underconfident by 0.1534**, an ECE **4.4× to 6.0×** N-MNIST's at the same
`T` depending on architecture. *(Corrected 2026-09-12: this read "fifty times N-MNIST's at
the same `T`". Fifty is the ratio against N-MNIST at `T = 10`, a different setting. See
"Outcome of prediction 5".)*
So:

> **The ceiling explains the miscalibration where it binds and explains none of it where it
> does not.** At `C = 10` that was invisible, because the regime where it stopped binding was
> also a regime where almost no error remained. At `C = 101` a large error remains and the
> ceiling has nothing to say about it.

This is a limitation of the study's central claim that only the C axis could expose, and it
is the most important thing on this page.

**Tested and still unexplained.** The obvious rescue is that the *theoretical* ceiling is too
generous — it assumes all 100 competitors silent — and that a ceiling conditioned on real
competitor activity would bind after all. That was pre-registered as prediction 4 and
**falsified**: the conditional ceiling at `T = 6` is 0.6804, *above* the accuracy of 0.6427,
so these models had 0.19 of unused headroom. The residual is genuine miscalibration. See
"Outcome of prediction 4".

### The cap costs accuracy, not only calibration

`[MEASURED]` Accuracy, accumulator minus count, per seed:

| `T` | count | accumulator | gap |
|---|---|---|---|
| 2 | 0.4543 | **0.5825** | **+0.1282** (+0.1515, +0.0959, +0.1372) |
| 6 | 0.6427 | 0.5976 | −0.0452 |
| 10 | 0.6435 | 0.6072 | −0.0364 |

At `T = 2`, removing the cap is worth **12.8 accuracy points**, in every seed, *with this
architecture*. At `C = 10` the same comparison was worth 0.1 points — the readouts were
indistinguishable. *(Qualified 2026-09-12: at the other architecture tested the gain is
**+3.1 points**, still positive in 3 of 3 seeds. The direction replicates and the magnitude
does not; see "Outcome of prediction 5".)* The
difference is the severity of the bound: the cross-entropy floor at `C = 101, T = 2` is
**2.6765 nats** against 0.7966 at `C = 10`, so the objective cannot be reduced below a value
that dominates the gradient signal, and the model is optimising against a target it cannot
reach.

**So the ceiling is not only a calibration artifact. Where it is severe it is a capacity
limit on learning itself.** Nothing at `C = 10` showed this, because 0.4509 was never severe
enough to matter for accuracy. Above `T = 2` the effect reverses and count wins by 4 points,
so this is a property of the binding regime and not of the readout.

### The first genuinely uncertain regime, and the first overconfident count readout

Accuracy across all 18 runs spans **0.4323 to 0.6561, mean 0.5880**, against N-MNIST's
0.9842–0.9914. The standing limitation that "no genuinely uncertain regime was sampled" is
now addressed.

Two sign results follow from it:

- **The count readout is overconfident at `T = 10`**, `conf − acc = +0.0344`, positive in
  3 of 3 seeds. This is the first time in the study that a count readout is overconfident,
  and it is the direction the SNN literature reports.
- **The accumulator is overconfident everywhere**: +0.0816, +0.1843, +0.2254 at
  `T` = 2, 6, 10, positive in 9 of 9 runs. That confirms at `C = 101` what was only a lead
  at `C = 10`.

Together these sharpen the disagreement with the literature considerably. Underconfidence is
what a *capped* readout does in a regime where the cap binds. Take away either condition —
raise `T` until the cap is slack, or remove the cap — and this study measures overconfidence
too. The disagreement was a regime difference, and the C axis is what made it visible.

### What the split is

No official train/test split exists, so the study defines one: **proportional per class at
20% with a floor of 6**, seeded, and written to `results/ncaltech101_T*_seed*_split.npz`.
Sizes: **train 5,654 / val 303 / cal 1,010 / test 1,742**, partitioning all 8,709 samples.

Those `.npz` files are not committed — `results/*.npz` is ignored, as it is for N-MNIST —
and they do not need to be. `make_dataset_splits` regenerates them byte-identically from
the seed alone, which is verified: seeds 0, 1 and 2 reproduce fingerprints `17e6b35480e6d410`,
`0087190fd7fba8ef` and `65947f632bdb55cc`, and **every row in
`results/caltech_experiment.json` carries the fingerprint of the split that produced it**.
A result can therefore prove which split it came from rather than assert it. Fixed per-class holdouts are used for val (3) and
cal (10) so every class reaches the calibration set; the test split is proportional because
the dataset runs from 31 samples in its smallest class to 800 in its largest and no fixed
count serves both.

Frames are average-pooled 4× to 45×60 before the model sees them, which is recorded in the
registry rather than hidden: it makes the 18-run grid tractable. Pooling cannot affect the
ceiling, which contains neither resolution nor anything downstream of it.

**Correction, 2026-09-11.** This paragraph originally also claimed that pooling let
N-MNIST's own two-block `(32, 64)` architecture fit unchanged, so that the `C = 101`
comparison differed from the `C = 10` one "in the class count and the data alone". **That is
false.** The runs used **four** blocks, `(32, 64, 128, 128)` with 464,074 parameters, against
N-MNIST's two blocks and 1,070,868 — `scripts/caltech_experiment.py` set `CHANNELS` before
the pooling decision was made and was never updated to match the argument built on it. The
comparison is therefore confounded with depth and parameter count.

What this does and does not touch:

- **The ceiling verification is unaffected.** `e^T/(e^T + C − 1)` contains no architectural
  term, and the bound was respected in 9 of 9 runs and saturated to 1e-13 where reachable.
  That result holds whatever the trunk was.
- **The T = 6 residual is now less cleanly attributable.** 0.1534 of underconfidence at
  `C = 101` against 0.0272 at `C = 10` was read as a C effect; part of it could be a depth or
  capacity effect. Establishing it as a C effect requires re-running `C = 101` with `(32, 64)`
  — about 1.5 hours — and that has not been done.
- **The accuracy comparison between readouts is unaffected**, since count and accumulator
  share the same trunk within N-Caltech101.

## Pre-registered prediction 4, recorded 2026-09-11

Written before any competitor count was measured. The target is the one quantity the C
axis left unexplained: at `C = 101, T = 6` the theoretical ceiling is 0.8014, accuracy is
0.6427 so the ceiling is **slack**, and yet `conf − acc` is **−0.1534**.

### The conditional ceiling

The theoretical ceiling is the softmax of the extreme count vector `[T, 0, ..., 0]`, which
assumes **every one of the `C − 1` competitors is silent for all `T` steps**. That is a
worst case for the denominator and it is what makes the bound loose: with 100 competitors,
assuming all of them silent is a strong assumption.

Conditioning on what the competitors actually do gives, for one sample whose non-winning
classes hold counts `c_j`:

    ceiling_cond = e^T / (e^T + Σ_{j ≠ winner} e^{c_j})

which is the largest confidence that sample could have expressed **if its winner had fired
at every step**, holding its competitors where they actually were. Setting every `c_j = 0`
recovers `e^T/(e^T + C − 1)` exactly, so the theoretical ceiling is the special case, not a
different quantity. In the grouped form the task states it, with `k` competitors at count
`c` and the rest at zero, it is `e^T / (e^T + k·e^c + (C − 1 − k))`.

### Why this is a live hypothesis and not a guess

`[DERIVED]` The denominator's sensitivity to competitor activity scales with `C − 1`, so the
same per-competitor activity costs eleven times more confidence at `C = 101` than at
`C = 10`. Evaluated a priori, before measuring anything:

| mean competitor count | ceiling at `C = 101`, `T = 6` | ceiling at `C = 10`, `T = 6` |
|---|---|---|
| 0 (the theoretical assumption) | 0.8014 | 0.9782 |
| 0.5 | 0.7099 | 0.9645 |
| 1.0 | 0.5974 | 0.9428 |
| 2.0 | 0.3532 | 0.8585 |

So a mean competitor count around 1 would push the `C = 101` ceiling **below** the measured
accuracy of 0.6427 and account for the entire residual, while the same activity at `C = 10`
would move the ceiling by only 0.035.

*(Note added 2026-09-11, after the run. The sentence above saying the same per-competitor
activity "costs eleven times more confidence" at `C = 101` is imprecise: 11× is the ratio of
**denominator** sensitivities, `(C − 1)` against `(C − 1)`, but the ceiling saturates as the
denominator grows, so the confidence cost ratio is about **5.8×** at these values. The table
in this section was computed correctly and none of the predictions or thresholds depended on
the misstated ratio, so the prediction stands as written. Recorded rather than edited.)*

### The prediction

1. **The conditional ceiling accounts for most of the `C = 101, T = 6` residual** — more
   than half of the 0.1534 gap between accuracy and mean confidence.
2. **The conditional and theoretical ceilings nearly coincide at `C = 10`** — within 0.05 of
   each other at `T = 6` — which is why the simple bound sufficed there and does not here.
3. The gap between them at `C = 101, T = 6` will exceed 0.15.

### Falsification

Define the share of the residual that attainability forces as
`min(1, max(0, accuracy − ceiling_cond) / (accuracy − mean_conf))`.

- **Wrong if that share is below 0.5** at `C = 101, T = 6`, which means a mean conditional
  ceiling above **0.5660**. In that case the residual is not an attainability effect and is
  genuine miscalibration — the more interesting negative result, because it would leave the
  study's largest unexplained quantity still unexplained and rule out the tidiest
  explanation for it.
- **Also wrong, in the opposite direction, if the conditional ceiling at `C = 10, T = 6`
  sits more than 0.05 below the theoretical one.** That would mean the simple bound was
  always the wrong object and the `C = 10` results were reported against a bound that
  happened to be loose there too, rather than one that was nearly tight.

### What it would not establish

Even if the prediction holds entirely, the conditional ceiling is **not** a bound the study
can claim in the way the theoretical one is. The theoretical ceiling is a property of the
readout, fixed by `T` and `C` before any model exists. The conditional ceiling is measured
*from the model's own outputs*, so "the model is as confident as it could be given how it
distributed its spikes" is close to circular: a model that spread its spikes differently
would face a different bound. It explains the residual as an attainability effect rather
than a calibration failure; it does not predict the residual from first principles.

## Outcome of prediction 4, recorded 2026-09-11 after the run

`[MEASURED]` `scripts/conditional_ceiling.py`, 24 cells, no training. The prediction above
is unedited.

### The competitor distribution

| dataset | `T` | competitors | mean count | mean # firing | % firing | mean loudest | Σe^c |
|---|---|---|---|---|---|---|---|
| N-Caltech101 | 2 | 100 | 0.097 | 8.6 | 8.6% | 1.43 | 119.7 |
| N-Caltech101 | 6 | 100 | 0.237 | 16.4 | 16.4% | 3.01 | 198.3 |
| N-Caltech101 | 10 | 100 | 0.408 | 24.9 | 24.9% | 4.27 | 423.2 |
| N-MNIST | 2 | 9 | 0.006 | 0.1 | 0.6% | 0.05 | 9.1 |
| N-MNIST | 6 | 9 | 0.064 | 0.4 | 4.6% | 0.45 | 11.1 |
| N-MNIST | 10 | 9 | 0.245 | 1.5 | 17.0% | 1.37 | 21.6 |

The distribution is heavily skewed, which is why it is reported rather than its mean: at
`C = 101, T = 6`, **81.6% of competitor slots hold zero**, 12.1% hold one, 4.1% hold two,
and the tail reaches six. The single loudest competitor carries 14.7% of the whole
denominator sum at `T = 6` and 25.6% at `T = 10`, so "mean competitor count" understates
what the softmax actually sees.

### Part 2 held, and it explains why the simple bound sufficed at `C = 10`

`[MEASURED]` Theoretical minus conditional ceiling:

| `T` | `C = 101` | `C = 10` | ratio |
|---|---|---|---|
| 2 | 0.0100 | 0.0026 | 3.8× |
| 6 | **0.1210** | **0.0047** | **25.9×** |
| 10 | 0.0140 | 0.0006 | 24.7× |

At `C = 10` the two ceilings differ by **at most 0.0067 at any `T`**, far inside the 0.05
the prediction allowed. With nine competitors that are silent 95–99% of the time, the
extreme vector is nearly reachable and the simple bound is nearly tight. At `C = 101` the
gap is up to 26 times larger. **The prediction's account of why `C = 10` never needed the
refinement is correct.**

The refinement also improves the accounting wherever the ceiling does bind: at
`C = 101, T = 2` it explains 99.6% of the residual against the theoretical bound's 97.1%,
and at `C = 10, T = 6` it explains 61.7% against 44.5%.

### Part 1 is falsified, and that is the result

The prediction said the conditional ceiling would account for **more than half** of the
`C = 101, T = 6` residual, and named its own falsification threshold: wrong if the mean
conditional ceiling exceeds **0.5660**.

| seed | accuracy | mean conf | residual | conditional ceiling | share explained |
|---|---|---|---|---|---|
| 0 | 0.6372 | 0.4775 | +0.1597 | 0.6571 | **0%** |
| 1 | 0.6349 | 0.4858 | +0.1491 | 0.6812 | **0%** |
| 2 | 0.6561 | 0.5048 | +0.1513 | 0.7029 | **0%** |

**The conditional ceiling is 0.6804, which sits 0.0377 *above* the accuracy of 0.6427.**
Every seed exceeds the 0.5660 threshold. Nothing forced these models to be underconfident:
given exactly the competitor configurations they produced, they could have expressed a mean
confidence of 0.68 and expressed 0.49 instead. **They left 0.1910 of headroom unused.**

> **The `C = 101, T = 6` residual is genuine miscalibration, not an attainability effect.**
> The tidiest available explanation is ruled out, and the study's largest unexplained
> quantity stays unexplained.

Part 3 also missed, in the same direction: the predicted gap of more than 0.15 at
`C = 101, T = 6` measured **0.1210**.

### Why this conclusion survives the architecture confound

The `C = 101` runs used four conv blocks against N-MNIST's two (see the correction under
"The C axis"), so whether the *size* of the residual is a `C` effect or a depth effect is
open. **This result does not depend on which.** The conditional ceiling is computed from
each model's own output distribution, so it asks only whether *that* model could have been
more confident than it was. It could. Whatever produced the residual, attainability did not.

### What the conditional ceiling is good for

It is a strictly better descriptive bound than the theoretical one — never looser, up to
0.12 tighter at 101 classes, and it raises the explained share wherever the ceiling binds.
It is **not** a claim of the same kind. The theoretical ceiling is fixed by `T` and `C`
before a model exists and therefore predicts; the conditional ceiling is measured from a
model's own spikes and therefore only explains. A model that distributed its spikes
differently would face a different conditional bound. It belongs in the paper as an
analysis tool, not as a second derived result.

## Pre-registered prediction 5, recorded 2026-09-11

Written before any matched-architecture model was trained. This is the 2×2 that removes the
depth confound introduced when N-Caltech101 was run with four conv blocks against N-MNIST's
two, and it doubles as the alternative-architecture check the study has never had.

### What is actually confounded, and what is not

The distinction matters, and one claim has been described as confounded when it is not:

**Genuinely confounded (cross-dataset, so depth varies with `C`):**

- "The `T = 6` residual is 0.1534 at `C = 101` against 0.0272 at `C = 10`, fifty times
  larger." Depth differs between those two numbers.
- "The ceiling explains the error where it binds and none of it where it does not." The
  `C = 101` half of that comparison is a four-block result and the `C = 10` half a two-block
  result.
- The competitor-activity comparison between datasets, and therefore the claim that the
  conditional ceiling's usefulness grows with `C`.

**Not confounded, but single-architecture:**

- **The +12.8 accuracy gain from removing the cap at `C = 101, T = 2` is not confounded with
  depth.** Count and accumulator share the same trunk within N-Caltech101, so that comparison
  was always internally matched. What it lacks is replication at a second depth, which is a
  weaker charge and is what this run supplies.
- The falsification of prediction 4. The conditional ceiling is computed from each model's
  own outputs, so "this model had headroom it did not use" holds for the models measured
  whatever their depth. Whether it holds for a *different* depth is untested.

**Untouched by any of this:** the ceiling verification itself. `e^T/(e^T + C − 1)` contains
no architectural term.

### The design

Both architectures on both datasets, `T` in {2, 6}, seeds 0–2, count and accumulator in
every cell, so the readout comparison is matched as well as the dataset comparison. Two of
the four quadrants already exist; 24 runs are new.

| | two blocks (32, 64) | four blocks (32, 64, 128, 128) |
|---|---|---|
| N-MNIST | exists | **new** |
| N-Caltech101 | **new** | exists |

### The prediction

1. **The ceiling still holds at `C = 101` with two blocks.** The bound is respected in every
   run. Saturation is likely but is not predicted: reaching it needs some sample with all 100
   competitors silent, which is a property of the trained model, not of the formula.
2. **The accuracy gain from removing the cap at `C = 101, T = 2` survives at both depths**,
   remaining above +0.05 in each. The mechanism claimed for it — a 2.68-nat cross-entropy
   floor dominating the gradient — has no architectural term either.
3. **The `C = 101, T = 6` residual persists at both depths, staying above 0.08** — that is,
   more than half of the 0.1534 survives a 5.8× change in parameter count. If it instead
   falls below 0.08 with the larger two-block model, it is substantially a capacity effect
   and the "C effect" reading of it is wrong.
4. **The conditional ceiling's explanatory share at `C = 101, T = 6` stays at 0%** under both
   architectures — the conditional ceiling remains above accuracy, so the residual remains
   genuine miscalibration rather than an attainability effect.

### Falsification

- **Prediction 1 is wrong** if any run's observed maximum confidence exceeds its derived
  ceiling by more than float tolerance. This would be the most important negative result in
  the study and is reported first if it happens.
- **Prediction 2 is wrong** if the accuracy gain at `C = 101, T = 2` falls below +0.05 at
  either depth, or reverses sign.
- **Prediction 3 is wrong** if the `C = 101, T = 6` residual falls below 0.08 with two
  blocks. That is the outcome that would reattribute the study's largest unexplained quantity
  from `C` to capacity.
- **Prediction 4 is wrong** if the conditional ceiling drops below accuracy at
  `C = 101, T = 6` under either architecture, giving a non-zero share.

### What is suspended until this completes

Every cross-dataset quantitative comparison in "The C axis" and in "Outcome of prediction 4"
is provisional. They are left in place, unedited, with this section as the pointer: a reader
should treat the `C = 10` against `C = 101` numbers as unmatched until the table below is
filled in.

## Outcome of prediction 5, recorded 2026-09-12 after the run

`[MEASURED]` `scripts/arch_matched.py`, 48 cells — 2 datasets × 2 architectures × 2 `T` ×
3 seeds × 2 readouts, of which 24 were reused and 24 newly trained. The prediction above is
unedited.

### First, what the two architectures actually are

They differ in depth **and** in size, in opposite directions, because more pooling shrinks
the flatten:

| | blocks | N-MNIST params | N-Caltech101 params |
|---|---|---|---|
| `(32, 64)` | 2 | 1,070,868 | 2,749,130 |
| `(32, 64, 128, 128)` | 4 | 375,060 | 464,074 |

**The shallower model is the larger one, by about 6×.** So this design separates "one
network" from "another network", which is what the alternative-architecture check needed,
but it cannot separate depth from capacity — they are anti-correlated here. Any statement
below of the form "it changed with architecture" must not be read as "it changed with depth".

### 1. The ceiling holds — prediction HELD, and more strongly than predicted

`[MEASURED]` Signed excess (observed max − derived), worst case over 3 seeds:

| dataset | blocks | `T` | derived | worst signed excess | saturated |
|---|---|---|---|---|---|
| N-Caltech101 | 2 | 2 | 0.068806 | +2.4e-14 | yes |
| N-Caltech101 | 2 | 6 | 0.801362 | +3.8e-13 | yes |
| N-Caltech101 | 4 | 2 | 0.068806 | +2.4e-14 | yes |
| N-Caltech101 | 4 | 6 | 0.801362 | +3.8e-13 | yes |
| N-MNIST | 2 | 2 | 0.450853 | −2.8e-13 | yes |
| N-MNIST | 4 | 6 | 0.978178 | +3.8e-14 | yes |

**Zero violations across all 24 count cells**, and the bound is saturated to float precision
in every one. The prediction declined to claim saturation, since reaching the bound needs a
sample with all competitors silent and that is a property of the trained model; it happened
anyway at both depths on both datasets. The ceiling result is architecture-independent, as
the formula says it must be.

### 2. The accuracy gain does not survive — prediction FALSIFIED

The prediction said the gain from removing the cap at `C = 101, T = 2` would stay above
**+0.05 at both depths**.

| blocks | count | accumulator | gain | per seed |
|---|---|---|---|---|
| 4 | 0.4543 | 0.5825 | **+0.1282** | +0.1515, +0.0959, +0.1372 |
| 2 | 0.4891 | 0.5199 | **+0.0308** | +0.0482, +0.0138, +0.0304 |

**+0.0308 is below the threshold, so the prediction is falsified.** The gain shrinks
fourfold with the other architecture. It stays positive in 3 of 3 seeds, so the *direction*
replicates, but the magnitude does not.

> **The headline "+12.8 accuracy points" is an architecture-specific number and must be
> reported as one.** What replicates is "removing the cap helps accuracy at `C = 101, T = 2`,
> by between 3 and 13 points depending on the network". The claim that the cross-entropy
> floor dominates the gradient — offered as a mechanism with no architectural term — does not
> survive this: if it were the whole story the gain would not depend on the network.

At `T = 6` the sign reverses at both depths (−0.0224 and −0.0452), so "the cap costs accuracy
only where it binds hard" does replicate.

### 3. The `C = 101, T = 6` residual persists — prediction HELD

| blocks | params | accuracy | mean conf | residual | per seed |
|---|---|---|---|---|---|
| 4 | 464,074 | 0.6427 | 0.4894 | **+0.1534** | 0.1597, 0.1491, 0.1513 |
| 2 | 2,749,130 | 0.5618 | 0.4459 | **+0.1159** | 0.1054, 0.1194, 0.1229 |

The threshold was 0.08 and the residual is 0.1159 at the other architecture, so it holds.
**About a quarter of it is attributable to the network and three quarters are not.** For
contrast, N-MNIST's residuals barely move: 0.5386 against 0.5398 at `T = 2`, and 0.0272
against 0.0253 at `T = 6`. The architecture sensitivity is specific to `C = 101`.

### 4. The conditional ceiling still explains none of it — prediction HELD

| blocks | residual | theoretical | share | conditional | share |
|---|---|---|---|---|---|
| 4 | +0.1534 | 0.8014 | 0% | 0.6804 | **0%** |
| 2 | +0.1159 | 0.8014 | 0% | 0.7146 | **0%** |

The conditional ceiling sits above accuracy at both depths (0.7146 against 0.5618; 0.6804
against 0.6427), so it forces none of the residual either way. **The `C = 101, T = 6`
residual is genuine miscalibration at both architectures**, and prediction 4's falsification
survives the matched design.

### The cross-dataset comparison, with the confound removed

`[MEASURED]` Residual at `C = 101` against `C = 10`, count readout, **matched depth**:

| blocks | `T` | `C = 101` | `C = 10` | ratio |
|---|---|---|---|---|
| 2 | 2 | +0.4343 | +0.5386 | 0.8× |
| 4 | 2 | +0.3969 | +0.5398 | 0.7× |
| 2 | 6 | +0.1159 | +0.0272 | **4.3×** |
| 4 | 6 | +0.1534 | +0.0253 | **6.1×** |

**The `T = 6` result survives the matched comparison at both depths.** More classes leave a
much larger residual where the ceiling is slack, by a factor of 4 to 6.

Two corrections follow from this table:

- ***Corrected 2026-09-12.*** The C-axis section previously called the `C = 101, T = 6` ECE
  "fifty times N-MNIST's at the same `T`". It is **4.4× to 6.0×** at the same `T`. The
  fifty-fold figure is the ratio against N-MNIST at `T = 10` (0.1539 / 0.0029 = 53), which is
  a different timestep setting. The comparison was mislabelled, not miscomputed.
- At `T = 2` the residual is *smaller* at `C = 101` than at `C = 10`, which is not a
  contradiction: at `C = 10` accuracy is 0.985 against a ceiling of 0.4509, and at `C = 101`
  accuracy is only 0.45–0.49 against a ceiling of 0.0688. The ceiling is far more binding and
  the absolute gap is still smaller, because the model is much worse.

### What is no longer suspended

Everything listed under prediction 5. The cross-dataset `T = 6` comparison holds at matched
depth with a corrected ratio; the ceiling verification holds at both depths; the conditional
ceiling result holds at both depths. The one claim that did not survive is the magnitude of
the `T = 2` accuracy gain, which is now reported as a range across architectures.

## Does the ceiling bind in published work? Resolved: no, and the reason is informative

The verification pass left one consequential question open. Li, Jones & Furber report
**94.11% accuracy at 4 timesteps on CIFAR-10**, and at `C = 10, T = 4` the
attainable-confidence ceiling is **0.8585**. If their confidence were a softmax over integer
spike counts, the bound would constrain a published ICCV result. Their method section was
read to settle it, rather than inferring anything from the accuracy number.

### Outcome (b): a real-valued readout. The cap does not apply.

`[LIT]` Li, Jones & Furber, arXiv:2303.10276v3, experimental setup, **verbatim**:

> "In our experiments, we duplicate QCFS and QFFS by quantizing the activation in ANNs to
> 2-bit by LSQ in all hidden layers and **keeping the output layer at full precision**. All
> weights are in full precision. We adopt the soft-reset IF neuron and its variant in SNNs
> as described in, and **the neurons in the output layer only accumulate input currents**."

The output neurons **integrate but do not fire**. Their logits are real-valued accumulated
currents at full precision, not integers in `[0, T]`, so `e^T/(e^T + C − 1)` does not apply
to them and **their published numbers are not constrained by it**. Their confidence is

    P̂_t = max_i ( e^{z_t^i / α} / Σ_j e^{z_t^j / α} )

with `z_t` those accumulated currents.

**This is the same construction as this study's `accumulator` readout** — a non-spiking
integrator with no leak — arrived at independently and for a different reason. Theirs comes
from the standard conversion recipe, which keeps the output layer at full precision
*because conversion needs the output precision*. So conversion-based work sidesteps the cap
as a side effect of a choice made for accuracy, not calibration.

### The mirror-image failure mode, documented independently

The same paragraph that rules out the cap corroborates this study's account of what happens
when the cap is removed. `[LIT]` verbatim:

> "The outputs of a rate-coded SNN are accumulated spikes or, in some cases that need higher
> output precision, the membrane potential of integrate-but-not-fire neurons. In both
> situations, the output logits of SNNs increase with `t` continuously, which causes the
> **saturated confidence** `P̂_t` after applying Softmax to the SNN outputs."

and their Figure 2 caption:

> "a. `P̂_t` **saturates to 100% quickly with time**. b. After introducing a scalar
> regulation parameter `α`, `P̂_t` is softened."

**An unbounded accumulating readout saturates towards 1 and is overconfident.** That is
precisely what this study measures for its own accumulator: `conf − acc` of **+0.0032 at
`T = 6` and +0.0047 at `T = 10`** on N-MNIST, positive in 3 of 3 seeds at both, and **+0.08
to +0.23** at `C = 101`. Two independent lines of work, two different datasets, same
direction.

So the two readout families have **opposite** failure modes, and the literature has hit both:

| readout | bound | failure mode | who reports it |
|---|---|---|---|
| softmax over spike counts | capped at `e^T/(e^T+C−1)` | **under**confidence where the cap binds | this study, `T ≤ 6` |
| non-spiking accumulator | unbounded, grows with `t` | **over**confidence, saturating to 1 | Li et al.; this study's accumulator |

### And their fix is temperature scaling

`α` divides the logits before the softmax, which is exactly temperature scaling with
temperature `α`; the paper says so directly, applying "temperature scaling, a post-training
calibration algorithm". This study's null result — that calibrator choice does not matter
and temperature is entirely adequate — is consistent with a published method choosing
temperature for the uncapped case and finding it sufficient.

### What the paper may now claim, and what it may not

**May claim.** That no verified published result in this study's reference set is
constrained by the ceiling, and that this is *because of a readout choice* rather than
because the bound is weak:

- **Li, Jones & Furber** use a full-precision non-spiking accumulator at `T ≈ 2.5–14`. Cap
  does not apply. `[LIT]` verified in the method section.
- **Chen, Park & Simeone, "Agreeing to Stop"** use softmax over integer spike counts, so the
  cap *does* apply in principle — but at `T = 80` it equals 1.0 to double precision and is
  inert. `[LIT]` verified in the full text.

The ceiling therefore binds in a regime the literature has not published in: a **spike-count
readout at low `T`**. That is a narrower and more defensible statement than "the bound
constrains published work", and it is the one the evidence supports.

**May not claim.** That the ceiling constrains any published number. It does not, in either
verified case. Nor may it claim anything about **Wu et al., arXiv:2405.00699**, whose
abstract mentions a "cutoff based on softmax output" on directly-trained SNNs but whose
readout and timestep counts could not be verified — the arXiv HTML is not served for that
paper and it was not read. It stays in "Still to verify".

## Why the count readout exists

The readout experiment shows the cap is avoidable, which makes it easy to read this study
as saying the field has been doing something obviously wrong. It has not, and a paper that
implies otherwise will read as naive. There are real reasons to sum spikes, and they are
not all dissolved by the result.

### 1. Spikes are the communication primitive; potentials are not

In event-driven neuromorphic hardware the thing that crosses a core or chip boundary is a
spike event. A per-class count is then obtained by an **accumulator at the destination**:
one integer per class, incremented when an event arrives, costing no traffic beyond the
events already being sent. Membrane potential is internal compartment state. Exporting it
means moving a real value per neuron per timestep, dense and continuous, which is precisely
the traffic that event-driven architectures exist to avoid. `[DERIVED]` from the
architecture of event-driven systems in general; the specific per-chip claims below are
`[UNVERIFIED]`.

### 2. Counts are monotone; potentials are not

This is the sharpest argument and it survives the experiment. A spike count is
**non-decreasing in `T`**: once class `c` has reached `k` spikes it can never fall below
`k`. A stopping rule built on it is therefore stable — a threshold crossing is final.

A leaky membrane potential is **not monotone**. It decays between inputs and can fall back
below a threshold it previously crossed, so a stopping rule on it can un-fire. `[DERIVED]`
from the LIF update `v ← v + (z − v)/tau`, which is a contraction toward `z` and decreases
whenever `z < v`.

Anytime inference and early exit need a quantity that updates each timestep and behaves
predictably as evidence accumulates, and the Dynamic Confidence and SpikeCP line of work
depends on exactly that. So even where a membrane readout is available, a system that must
stop early may still want counts — and then the cap applies and `T_min` is the right tool.

### 3. What the hardware actually exposes — `[UNVERIFIED]`

**A verification pass on 2026-09-12 could not confirm these claims from primary
documentation, and they remain `[UNVERIFIED]`.** Intel's NxSDK API reference is not publicly
reachable and the sPyNNaker limitations page does not discuss recording bandwidth. What
follows is therefore weaker than it was: the strong form of the argument is withdrawn and
only what could be established is stated.

- **Intel Loihi.** `[UNVERIFIED]` Loihi's neuron model is a CUBA leaky integrate-and-fire
  with membrane potential as an internal state variable, and Rueckauer et al.'s NxTF paper
  (arXiv:2101.04261) states that during the management phase the embedded CPUs may read out
  the current SNN state. That is **consistent with** membrane state being readable and says
  **nothing about the per-timestep cost** of reading it, which is the only part this argument
  needed. The claim that it is probe-only, or that reading requires a host round-trip, is
  **not established** and is no longer asserted.
- **SpiNNaker.** `[UNVERIFIED]` Membrane potential is exposed through PyNN's `record('v')`,
  cores carry at most 255 neurons, and host transfers are bounded by on-board Ethernet
  bandwidth. Whether that bound binds for an output layer of ten to a thousand neurons —
  the only thing a membrane readout needs — is **not established**.

The structural observation that motivated this section still stands on its own, without
needing either hardware claim: a membrane readout needs the potential of the **output layer
only**, `C` values per timestep rather than the whole network. Whether that is cheap on any
particular device is exactly what could not be verified, so the argument for the count
readout should rest on the two points above it — spikes as the communication primitive, and
count monotonicity — and not on an unverified cost model.

### 4. The recommendation, with its scope

- **Where a membrane readout is available** — simulation, GPU or CPU inference, or hardware
  that exposes output-layer compartment state — the cap is avoidable and **should be
  avoided**. It costs 16x in uncalibrated ECE at `T = 2` for nothing.
- **Where only spike events leave the chip**, the cap is a hard constraint, not a design
  error. `T_min = ln((C − 1)a/(1 − a))` then gives the timestep budget: 10 classes at 95%
  accuracy needs `T ≥ 6`, and no amount of calibration changes that.
- **Where early exit is required**, counts may be preferable regardless, because they are
  monotone and a stopping rule on them is stable. The cap is then the price of that
  property, and it should be paid knowingly.

The study's contribution here is not "use membrane" but **"the cap is a property of a
choice, the choice has real reasons behind it, and if you make it you can compute what it
costs you in timesteps."**

## Scope

A multi-dataset benchmark. One dataset is an anecdote; the claim is about a mechanism
(output discreteness), and a mechanism should not be sensitive to which event camera
recorded the data.

Datasets, in priority order:

| Order | Dataset     | Resolution | Classes | Samples (train/test) | Status                |
|-------|-------------|------------|---------|----------------------|-----------------------|
| 1     | N-MNIST     | 34 x 34    | 10      | 60,000 / 10,000      | Primary, done first   |
| 2     | DVS-Gesture | 128 x 128  | 11      | 1,176 / 288          | Second                |
| 3     | CIFAR10-DVS | 128 x 128  | 10      | 10,000 total         | Third, if time allows |

The sweep is `T` in `{2, 4, 6, 8, 10}` with 3 seeds, on every dataset that makes it into
the study. That is 15 runs per dataset per calibration arm.

The paper reports whether the isotonic-vs-`T` trend holds **across all of them**. A
trend that holds on N-MNIST and fails on DVS-Gesture is a finding about where the
mechanism applies — most likely interacting with calibration-set size or task
difficulty — and it goes in the paper as such. It is not a failure to be hidden by
dropping the inconvenient dataset, and any dataset that is run is reported.

## Prior work

The study claims novelty over none of the following. Each is cited, and the difference
is stated rather than implied.

**Verified against primary documents on 2026-09-12.** Every reference here was checked;
`SOURCES.md`, "Verified 2026-09-12", carries the evidence. Two did not survive:

- **"Sun & Bohte, AOT-SNN, arXiv:2405.00699" was a wrong citation** and is withdrawn. That
  identifier is **Wu, Qi, Cai, Jin, Yi & Huang**, "Direct Training Needs Regularisation:
  Anytime Optimal Inference Spiking Neural Network" — method AOI-SNN, not AOT-SNN. A real
  AOT-SNN line exists and is Sun & Bohté's, but the paper the brief meant was never
  identified, so no claim is made about it.
- **"Skatchkovsky & Simeone on SNN overconfidence" could not be confirmed** and is
  withdrawn. The overconfidence result this study compares itself against is **Chen, Park &
  Simeone's**. No Skatchkovsky paper on SNN overconfidence was identified, and guessing at
  one would be worse than omitting it.

**Li, Jones & Furber, "Unleashing the Potential of Spiking Neural Networks by Dynamic
Confidence", ICCV 2023, pp. 13350–13360** (arXiv:2303.10276). `[LIT]` *Dynamic Confidence is
the method; the title above is the paper, which this document previously conflated.*
Constructs a runtime confidence signal for early exit in SNNs, so that inference can stop as
soon as the network is confident enough. `[LIT]` **verified:** the setting is ANN-to-SNN
**conversion** on static image data (CIFAR-10, ImageNet), and temperature scaling is applied
as post-training calibration. It runs at **2.5–14 timesteps**, which is inside this study's
own regime — the closest overlap of any reference here. This study differs on three axes: the networks
are **directly trained** as SNNs rather than converted; the data is **event data** rather
than static frames; and the object of study is the **calibration method itself** rather
than a stopping rule built on top of a confidence signal.

**Wu, Qi, Cai, Jin, Yi & Huang, "Direct Training Needs Regularisation: Anytime Optimal
Inference Spiking Neural Network", arXiv:2405.00699.** `[LIT]` Introduces a
Spatial-Temporal Regulariser so that accuracy degrades gracefully under an early cutoff on
the softmax output, on event data with directly-trained SNNs. The setting overlaps this
study closely, but the object is a **stopping rule and a training regulariser**, not the
calibration of the confidence itself. *(This entry previously misattributed the paper to
Sun & Bohte; corrected 2026-09-12.)*

**Chen, Park & Simeone, "Knowing When to Stop: Delay-Adaptive Spiking Neural Network
Classifiers with Reliability Guarantees", arXiv:2305.11322.** `[LIT]` Applies conformal prediction to the SNN stopping problem,
producing coverage guarantees for early exit. Conformal methods target set-valued
coverage; this study targets scalar confidence calibration measured by ECE. Related
problem, different quantity.

## The sign of the miscalibration: a disagreement with the literature

**The SNN literature reports overconfidence as the characteristic calibration failure.
This study measures the opposite.**

`[MEASURED]` `scripts/run_sweep.py`, seeds 0–2, 2026-09-10: mean confidence minus
accuracy is **−0.5386 at `T = 2`** and **−0.1446 at `T = 4`** on N-MNIST, averaged over the
three seeds. *(Corrected 2026-09-11: the figures previously given here, −0.5390 and −0.1437,
were seed 2's, labelled as though they covered seeds 0–2.)* That is severe
*under*confidence, and it is not a training artifact — it is forced by the
attainable-confidence ceiling, which at `T = 2` with 10 classes is 0.4509 against a test
accuracy of 0.9850. No reachable output could have been correctly confident.

Chen, Park & Simeone's "Agreeing to Stop" (*Entropy* 26(2):126, 2024) uses the **same
readout** — `[LIT]` verified in its full text as `f_c = e^{r_c} / Σ e^{r_c'}` over
accumulated spike counts `r_c` — and states verbatim that SNNs "tend to be poorly
calibrated, producing **overconfident** decisions". It too is directly trained with
surrogate gradients. On the face of it that is a direct disagreement on the sign, using the
same construction.

**It is not a disagreement. It is a regime difference, and the numbers now say so.**
`[LIT]` **verified 2026-09-12:** their experiments run at **`T = 80`**, with checkpoints at
`{20, 40, 60, 80}`. `[DERIVED]` At `T = 80` with `C = 10` the attainable-confidence ceiling
is **1.0 to double precision**; even at their earliest checkpoint, `T = 20`, it is
0.99999998. **The cap is completely inert in their regime.** Nothing forces underconfidence
there, so what they measure is ordinary overconfidence, and what this study measures at
`T = 2` — where the ceiling is 0.4509 — is a different phenomenon in a different place. The
two results are compatible and this document previously asserted that without knowing their
timestep count.

The remaining regime differences are secondary and are listed for completeness:

- **Timestep count.** The ceiling only binds while accuracy exceeds it. At `T = 8` and
  `T = 10` this study's own models sit *below* the ceiling, and the structural
  underconfidence duly disappears — but it does **not** turn into overconfidence. Measured
  `conf − acc` for the count readout is −0.0065 at `T = 8` (all three seeds negative) and
  −0.0008 at `T = 10` (two negative, one +0.0002). What the ceiling stops forcing is simply
  not replaced by anything this study can resolve. Work at higher `T`, or on tasks where
  accuracy is lower, may well see overconfidence; this study does not on N-MNIST, and
  cannot claim it does there.

  **Updated 2026-09-11 by the C axis, which settles this.** On N-Caltech101, where accuracy
  is 0.64 rather than 0.99, the *count* readout is overconfident at `T = 10` — `conf − acc`
  = +0.0344, positive in 3 of 3 seeds — and the accumulator is overconfident at every `T`
  measured, 9 of 9 runs. So the prediction in this bullet was right: a harder task does show
  overconfidence with the same readout. The disagreement with the literature is a **regime**
  difference, not a contradiction, and underconfidence is specifically what a capped readout
  does where the cap binds. *(Corrected 2026-09-11: this bullet previously asserted that the residual at `T = 8`
  and `T = 10` "is ordinary overconfidence", which the artifact contradicts.)*
- **Readout.** The bound is a property of softmax-over-spike-counts. Work using a membrane
  potential, a voting layer, or an ANN-to-SNN converted network with real-valued outputs
  has no such cap. Whether this is the whole explanation is tested directly in the readout
  experiment.
- **Training regime.** These are directly trained SNNs; much of the calibration literature
  concerns converted networks.

Both positions are stated so a reader can judge. What this study can defend is narrow and
specific: **at low `T`, with a spike-count readout, on a task where accuracy exceeds the
ceiling, the miscalibration is underconfidence and it is structural.** It does not claim
that SNNs are underconfident in general, and it does not claim the overconfidence reports
are wrong.

## Correctness rules

These are invariants, not preferences. **Breaking any one of them invalidates every
number in the paper**, so each is asserted in code rather than checked by eye, and each
stub in this repository restates the rule it is responsible for.

1. **Split disjointness.** The calibration split is disjoint from both the test split
   and the validation split. This is asserted on index sets, not assumed from the
   construction that produced them. A post-hoc calibrator fitted on data that overlaps
   the evaluation set reports its own training error as a result, which is the single
   most damaging thing that can silently go wrong in this study.

2. **Two ECE estimators, both reported.** Equal-width binning (Guo et al., ICML 2017;
   `[LIT]` for the method, `[UNVERIFIED]` for the 15-bin default attributed to it) and
   equal-mass binning are both computed and both reported. With discrete spike counts
   the confidence distribution is lumpy: equal-width binning leaves bins empty and
   concentrates mass in a few others, which biases the estimate **optimistically** at low
   `T` — precisely the regime the hypothesis is about. Reporting only equal-width would
   flatter every method in the regime where the comparison matters. **Disagreement
   between the two estimators is itself a finding** and is reported when it occurs,
   rather than being resolved by picking the more convenient one.

3. **Energy is a proxy, stated as such.** Energy is reported as the SynOps proxy using
   45 nm figures attributed to Horowitz, "Computing's Energy Problem (and what we can do
   about it)", ISSCC 2014: `E_AC = 0.9 pJ` for an accumulate and `E_MAC = 4.6 pJ` for a
   multiply-accumulate. `[LIT]` for the paper; **`[UNVERIFIED]` for the values** — the
   figure or table number, the exact 0.9 pJ (believed to be the 32-bit floating-point ADD
   at 45 nm), and whether 4.6 pJ is stated in the paper or is the community convention of
   FP multiply plus FP add, are all unconfirmed. See `SOURCES.md`, "To verify" item 1.
   These constants must not be cited with a figure number until checked. Every appearance
   of an energy number in
   text, table, or figure describes it as a **design-comparison proxy**. It is never
   described as measured power, and no claim about a physical device is derived from it.

4. **Seeded and reproducible.** Every experiment is seeded and every seed is recorded
   alongside its result. One caveat is stated explicitly rather than discovered by a
   reader: **MPS floating-point operations are not bitwise deterministic**, so training
   on Apple silicon varies slightly at a fixed seed. Consequently, everything that can be
   made exactly reproducible is kept off the accelerator — **data splits and calibrator
   fitting run on CPU with deterministic libraries only**: NumPy's `default_rng` for
   splits, scikit-learn for the isotonic fits, and `scipy.optimize` for the
   one-dimensional temperature search, which has no scikit-learn equivalent. No torch and
   no MPS anywhere in this path. Split index sets and fitted calibrators
   are therefore reproducible bit-for-bit; trained weights are reproducible only up to
   MPS float variation, which is why results are reported over 3 seeds rather than one.

## Resolved design decisions

These questions were settled before implementation, and are recorded here so that the
answers are not silently re-litigated once numbers start appearing.

**Confidence.** The per-class spike count over `T` timesteps is the logit. Confidence is
the softmax over these integer counts. Both calibrators operate on that same quantity, so
the comparison isolates the calibrator and not the readout. This also keeps the study
aligned with the standard practice it is testing: temperature scaling is defined on
logits, and the spike count is the natural logit of a rate-coded SNN.

**Isotonic formulation.** Two formulations are fitted and both are reported.

- *Primary — top-label isotonic.* Maps the maximum confidence to binary correctness,
  fitted on the calibration split. This is chosen as primary because it matches what ECE
  actually measures: ECE compares top-label confidence against top-label accuracy, so a
  calibrator fitted on exactly that pairing is the like-for-like comparison. This
  formulation appears in the headline table.
- *Secondary — one-vs-rest with renormalisation.* Included for comparability with Guo
  et al., who report this formulation. It is reported alongside so that readers
  cross-referencing that paper can find the number they expect.

**Tie-breaking.** Argmax ties are broken **uniformly at random under the experiment
seed**. They are never broken by lowest index, which is the library default and which
biases every tie toward class 0 — an artifact that would grow as `T` falls and would
contaminate the exact trend under test.

The tie rate is a **reported quantity, not a diagnostic**, and it appears in the results
table next to the calibration metrics. Ties become more frequent as `T` falls, because
fewer available count values means more collisions, and that rise is direct evidence that
the output is genuinely discrete — it makes the premise of the capacity argument visible
and measurable.

It is *not* the reason isotonic outperforms temperature scaling. Neither calibrator breaks
ties, so the tie rate cannot separate them; see "What tie rate is, and what it is not"
under the claim. Tie-breaking happens once, here, before any calibrator runs, and both
arms inherit the same predictions — which is what keeps the comparison about calibration
alone.

## Per-dataset hazards

Each of these is a known property of the dataset that is designed around here, in
advance, rather than discovered mid-analysis.

**DVS-Gesture — sample count.** Two different totals for this dataset circulate, and the
study needs to commit to one.

Amir et al. (CVPR 2017) describe "1,342 instances of a set of 11 hand and arm gestures".
`[LIT]`, quoted verbatim in the installed `spikingjelly/datasets/dvs128_gesture.py`
docstring, which is where this repository read it. SpikingJelly produces **1,464**, as
1,176 train and 288 test — `[LIT]` from that same docstring, and **not yet reproduced by
building the dataset here** (`SOURCES.md`, "To verify" item 10). The gap is 122, which is
also the number of recording trials, so it is one extra sample per trial on average.

The mechanism is not a twelfth category. Both counts cover the same 11 classes, and
"other gestures" is already one of them — it is class 11 in the per-trial label CSVs,
which SpikingJelly maps to label index 10. What differs is segmentation. SpikingJelly
emits one sample per *row* of each trial's `_labels.csv`, and some trials label a gesture
more than once: the `user01_fluorescent` CSV reproduced in SpikingJelly's own
documentation has twelve rows for eleven classes, because class 8 appears twice, and
SpikingJelly's source carries a note that `user26_fluorescent_labels.csv` has several
samples sharing a label. So the extra instances are repeated gesture segments, not a new
category. SpikingJelly attributes the difference to "different pre-processing methods",
and snnTorch's loader independently arrives at the same 1,176 / 288.

**This study uses SpikingJelly's convention: 11 classes, 1,176 train / 288 test, 1,464
total.** The reason is that the pipeline is built on SpikingJelly, so those are the
samples that will actually exist on disk, and every count in the paper has to match the
artifacts that produced it rather than a figure inherited from a citation. The paper will
report both numbers and this explanation, so a reader comparing against Amir et al. is
not left to guess whether the difference is a bug.

One consequence worth recording: SpikingJelly's train/test division is by recording
trial, listed in `trials_to_train.txt` and `trials_to_test.txt`, so no subject appears on
both sides. Splits carved out of the training set do **not** automatically inherit that
property — validation and calibration samples can come from subjects also seen in
training. Whether the calibration split should be subject-disjoint too is open; it is
recorded in `LOG.md` rather than silently decided here.

Either way the dataset is small — about 107 training samples per class. A 500-per-class
calibration split is arithmetically impossible, and more importantly isotonic regression
**overfits on small calibration sets**: a step function with enough steps can fit
calibration-set noise exactly. Two consequences for the design: split sizes are
**per-dataset configuration** rather than a global constant, and **calibration-set size
is reported alongside every result** so a reader can judge whether an isotonic win is
real or is small-sample overfitting. **Cross-validated isotonic is planned for the small
datasets**, so that the reported isotonic performance is not an in-sample artifact.

**CIFAR10-DVS — no official split.** The dataset ships as 10,000 samples with no
canonical train/test division, which means every paper using it has invented one and few
have published it. The split used here is **defined, seeded, saved to disk, and
documented in this file** when the dataset is brought into the study, so the numbers are
reproducible by someone who has only the raw dataset and this repository.

**Heterogeneous input geometry.** Spatial size and class count differ across the three
datasets (34x34 / 10, 128x128 / 11, 128x128 / 10). Model sizing is therefore
**configuration, never hardcoded** — input dimensions, downsampling, and the classifier
width all derive from the dataset registry. A hardcoded 34x34 assumption would surface as
a shape error at best and a silently wrong model at worst when the second dataset is
added.

---

## AI disclosure, for the paper

The paper is not yet written. When it is, this goes in the methods or an acknowledgements
note, once, in these words:

> AI assistance was used for implementation and drafting. All research design, hypotheses,
> the pre-registered prediction and its outcome, and all verification decisions are the
> author's. Every citation was checked against primary documents; two were withdrawn on
> verification and one energy constant was reclassified from cited to derived.

It is not to be softened and not to be expanded, and it belongs in the paper rather than on
the dashboard. Both of its factual claims are supported here: the two withdrawn references
are recorded in `SOURCES.md`, and the energy constant is E_MAC = 4.6 pJ, reclassified from
`[LIT]` to `[DERIVED]` on 2026-09-12 because it appears nowhere in Horowitz and is the
0.9 + 3.7 multiply-plus-add convention.

---

## Status

Implemented and run on **N-MNIST and N-Caltech101**: data acquisition, splits, model,
training, both ECE estimators, all four calibrators, bootstrap intervals, selective
prediction, and the figures. **96 models have been trained** — 66 on N-MNIST and 30 on
N-Caltech101, counted as weight files in `results/`. Six experiments have completed, and each outcome is recorded above beneath the
prediction that preceded it where one was registered:

| experiment | scale |
|---|---|
| main calibration sweep | 5 `T` × 3 seeds × 4 calibrators, 60 rows |
| readout experiment | 4 readouts × up to 5 `T` × 3 seeds, 54 runs |
| quantisation test | 5 `T` × 3 seeds × 2 schemes × 9 resolutions, 296 rows |
| C axis (N-Caltech101) | 2 readouts × 3 `T` × 3 seeds, 18 runs |
| conditional ceiling | 24 cells, no training |
| matched-architecture 2×2 | 2 datasets × 2 depths × 2 `T` × 3 seeds × 2 readouts, 48 cells |

Five predictions were pre-registered and committed before their runs, and none has been
revised after the fact:

| # | subject | outcome |
|---|---|---|
| 1 | the isotonic advantage grows as `T` falls | **falsified in the letter**; its substance — that the two calibrators tie — held |
| 2 | membrane removes the cap and most of the error | **held**, both halves, at every seed |
| 3 | the leak explains the high-`T` ranking deficit | **passed its own falsification threshold, failed its headline wording** — it closes half the deficit, it does not close it |
| 4 | a conditional ceiling explains the `C = 101` residual | **falsified in its central claim**; its secondary claim, that the two ceilings coincide at `C = 10`, held |
| 5 | the matched-architecture 2×2 | three parts **held**, one **falsified** — the accuracy gain does not replicate in magnitude |

So one prediction held cleanly, two were falsified in what they mainly claimed, and two
split. That ratio is the point of registering them.

Four readouts exist — count, weighted-count, membrane (leaky) and accumulator
(non-leaky) — with 54 readout runs recorded on N-MNIST: all four at `T` in {2, 6, 10} and
the first three at all five timestep settings.

**Two datasets are now implemented and run.** N-Caltech101 adds the C axis at 101 classes,
with 18 runs at `T` in {2, 6, 10} × 3 seeds × {count, accumulator}. It also supplies the
first genuinely uncertain accuracy regime in the study, 0.43–0.66 against N-MNIST's 0.99.

Not implemented: DVS-Gesture and CIFAR10-DVS, which raise `NotImplementedError` at the
loader. DVS-Gesture remains the interesting one — its ~330-sample calibration split is the
regime where the calibrator null result could plausibly break. A preprint is in
preparation.
