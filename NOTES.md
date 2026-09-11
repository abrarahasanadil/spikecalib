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

**The letter of the prediction is falsified; its substance holds.** Fourteen of the
fifteen paired-bootstrap intervals straddle zero, and of the two that do not, one favours
temperature (`T = 2`, seed 1) and one favours isotonic (`T = 6`, seed 0, CI
[−0.0024, −0.0003]). The prediction said the interval would straddle zero at *every* `T`,
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

**It was tested. It does not hold.** `[MEASURED]` The membrane readout remains
**underconfident at all five** timestep settings and is **overconfident at none** of them:
mean confidence minus accuracy is −0.0333, −0.0024, −0.0005, −0.0016 and −0.0017 at
`T` = 2, 4, 6, 8 and 10. Removing the cap shrinks the underconfidence by a factor of
sixteen but **never reverses it**. The single positive value anywhere in the
experiment is weighted-count at `T = 10`, at +0.0004, which is negligible.

So the disagreement with the literature on the *sign* of SNN miscalibration is **not**
explained by the readout choice, and the tidy resolution is unavailable. What can be said
is narrower and still useful: the *magnitude* of the underconfidence this study reports is
almost entirely a readout artifact, and once the readout is changed the residual
miscalibration is small in both directions. Why the prior reports find overconfidence where
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
automatically win the other.

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

**Nothing in this section has been confirmed against primary documentation, and it must be
before the preprint ships.** Recorded as the belief being relied on, so it can be checked.

- **Intel Loihi.** Believed to keep membrane voltage as per-compartment state that the host
  API can probe, while inter-core communication is spike-only. If so, membrane readout is
  available in a host-in-the-loop setting but not as a native on-chip routed quantity.
  *To verify:* whether compartment voltage is readable at inference time without a host
  round-trip, and at what cost.
- **SpiNNaker.** Simulates neurons in software on ARM cores, so membrane potential is a
  variable in memory and is recordable through PyNN. Recording it for many neurons is
  believed to be bandwidth-limited in a way that spike recording is not.
  *To verify:* whether that bandwidth limit binds at the output layer only, which is all
  this readout needs — ten to a thousand neurons, not the whole network.

That last point matters and cuts in this study's favour: the membrane readout needs the
potential of the **output layer only**, which is `C` values per timestep, not the whole
network. A dense-readout objection scaled to an entire network may not apply to ten
neurons.

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

**Every reference in this section is `[UNVERIFIED]`.** The original four were supplied in
the project brief; three more were added on 2026-09-11 against the study's final claims —
Atarashi et al. on learned probability bounds, Sun & Bohte on anytime inference, and
"Agreeing to Stop" on latency-adaptive stopping. None has been checked against a primary
document in this repository. `SOURCES.md` carries what each does, how this study differs,
and what specifically must be confirmed.
Authors, titles, venues and years must be confirmed before the paper cites any of them;
see `SOURCES.md`, "To verify" items 3–6.

**Li, Jones & Furber, "Dynamic Confidence" (ICCV 2023).** Constructs a runtime
confidence signal for early exit in SNNs, so that inference can stop as soon as the
network is confident enough. The setting is ANN-to-SNN conversion on static image data,
and the calibration reference is an ANN. This study differs on three axes: the networks
are **directly trained** as SNNs rather than converted; the data is **event data** rather
than static frames; and the object of study is the **calibration method itself** rather
than a stopping rule built on top of a confidence signal.

**Sun & Bohte, AOT-SNN.** Uses temperature scaling as the calibration baseline for SNN
uncertainty estimation. This is the baseline the present study interrogates: the question
here is whether that choice of calibrator is appropriate given a spike-count readout, not
whether uncertainty estimation for SNNs is worthwhile.

**Skatchkovsky & Simeone.** Documents overconfidence in SNNs and addresses it with a
Bayesian treatment. That is a training-time and model-class intervention. This study is
strictly **post-hoc**: the trained network is fixed, and only the mapping from its
outputs to confidences is changed. The two are complementary rather than competing.

**"Knowing When to Stop."** Applies conformal prediction to the SNN stopping problem,
producing coverage guarantees for early exit. Conformal methods target set-valued
coverage; this study targets scalar confidence calibration measured by ECE. Related
problem, different quantity.

## The sign of the miscalibration: a disagreement with the literature

**The SNN literature reports overconfidence as the characteristic calibration failure.
This study measures the opposite.**

`[MEASURED]` `scripts/run_sweep.py`, seed 0–2, 2026-09-10: mean confidence minus accuracy
is **−0.5390 at `T = 2`** and **−0.1437 at `T = 4`** on N-MNIST. That is severe
*under*confidence, and it is not a training artifact — it is forced by the
attainable-confidence ceiling, which at `T = 2` with 10 classes is 0.4509 against a test
accuracy of 0.9850. No reachable output could have been correctly confident.

"Agreeing to Stop" (Simeone group) uses the **same readout** — softmax over spike counts —
and states that SNNs are poorly calibrated and overconfident. That is a direct disagreement
on the sign, using the same construction.

**This is reported as a contradiction of the received position, carefully.** The difference
is most likely **regime**, not error in the prior reports:

- **Timestep count.** The ceiling only binds while accuracy exceeds it. At `T = 8` and
  `T = 10` this study's own models sit *below* the ceiling and the residual miscalibration
  is ordinary overconfidence. Work at higher `T`, or on tasks where accuracy is lower,
  would see overconfidence and would be right to report it.
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

## Status

Design fixed; implementation in progress. Data acquisition and environment verification
are implemented for N-MNIST at `T = 10`. Model, training, and calibration code are not
yet written. A preprint is in preparation.
