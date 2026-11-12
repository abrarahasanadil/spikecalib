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

*(Pointer added 2026-10-13. "At high `T` the ceiling is effectively absent and the familiar
overconfidence returns" was written before any model was trained and is superseded. On
N-MNIST the count readout does not become overconfident once the ceiling is slack:
`conf − acc` is −0.0065 at `T = 8` and −0.0008 at `T = 10`. The sign does flip where the
task is harder or the readout is unbounded: +0.0344 for the count readout at
`C = 101, T = 10`, and positive for the accumulator at `T ≥ 6`. See "The sign of the
miscalibration: a disagreement with the literature" and "The C axis: N-Caltech101 at 101
classes".)*

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
fifteen paired-bootstrap intervals straddle zero *under equal-width binning — see the
addendum below this paragraph for both estimators.* Two do not: one favours temperature
(`T = 2`, seed 1, CI [+0.0006, +0.0044]) and one favours isotonic (`T = 6`, seed 0, CI
[−0.0024, −0.0003]). *(Corrected 2026-09-11: this paragraph previously said "fourteen of
the fifteen" in the same sentence as "the two that do not", which cannot both be true. The
artifact has always said two are resolvable.)* The prediction said the interval would straddle zero at *every* `T`,
and at `T = 6` it did not, so the prediction as stated is wrong. But one resolvable result
in fifteen at a 95% level is what multiple comparisons produce on their own — the expected
count under no true difference is about 0.75 — and it does not survive as evidence of an
isotonic advantage.

**Addendum 2026-09-12 — the outcome under both ECE estimators.** Across the two
estimators, 13 of 15 intervals straddle zero under equal-width binning and 12 of 15 under
equal-mass.

**The runs that resolve are disjoint.** Equal-width resolves `T = 2` seed 1, favouring
temperature, and `T = 6` seed 0, favouring isotonic. Equal-mass resolves `T = 4` seeds 1
and 2 and `T = 8` seed 0, all three favouring isotonic and none temperature. **No run
resolves under both.**

**That strengthens the null result rather than weakening it.** The two estimators are
computed on identical calibrated confidences and differ only in how they bin them. A run
whose interval excluded zero because the calibrators genuinely differed would be found by
both, in the same place and the same direction. Two estimators on the same data
disagreeing entirely about which runs resolve is evidence that none of the five do — each
is binning noise that one scheme pushes across zero and the other does not. That is a
stronger argument than either count alone, and it is the form in which the calibrator
result should enter the paper.

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

### The paired bootstrap under both estimators, added 2026-09-12

Correctness rule 2 says both ECE estimators are reported wherever a number is claimed, and
this was the one place that never was: `iso_minus_temp_ece_ew` existed with no equal-mass
counterpart, so "thirteen of fifteen" was an equal-width statement standing in for the
finding. `scripts/rerun_paired_bootstrap.py` closes it — no training, weights reused,
calibrators refit on CPU, the same bootstrap seed, and **the equal-width column recomputed
and checked against the CSV before anything was written (worst |delta| 0.00e+00)**, so the
two arms are comparable rather than merely adjacent.

| estimator | straddle zero | favour temperature | favour isotonic |
|---|---|---|---|
| equal-width | **13** of 15 | 1 — `T = 2` seed 1 | 1 — `T = 6` seed 0 |
| equal-mass | **12** of 15 | 0 | 3 — `T = 4` seeds 1 and 2, `T = 8` seed 0 |

**No run resolves under both estimators.** The union of the two resolvable sets is five
runs; their intersection is empty. As the addendum to the outcome above sets out, that is
evidence none of the five reflects a real difference between the calibrators, and it
strengthens the null. That every equal-mass exception favours isotonic does not change
this: they are three runs that the other estimator, on the same confidences, does not
confirm.

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

*(Pointer added 2026-10-13. "At low `T`, use a membrane readout" is superseded as a
recommendation. At three seeds membrane is not dominant: it ranks worse than count from
`T = 6`, and its `T = 2` ECE varies fivefold across seeds, which the non-leaky accumulator
removes. The recommendation as it stands is in "Membrane is NOT strictly dominant, and the
recommendation narrows accordingly", "Outcome of prediction 3" and "Why the count readout
exists", part 4. The prediction above and its outcome table are unchanged.)*

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

*(Pointer added 2026-10-13. The mechanism offered in this subsection, a cross-entropy floor
that dominates the gradient, does not survive the matched-architecture run. The gain is
+0.1282 with four blocks and +0.0308 with two, and a mechanism with no architectural term
cannot produce a fourfold difference between networks. "A capacity limit on learning
itself" should be read as "removing the cap raises accuracy at `C = 101, T = 2` by 3 to 13
points depending on the network", with the cause open. See "Outcome of prediction 5",
part 2.)*

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

**Zero violations across all 24 count cells of the matched 2×2 grid**, and the bound is saturated to float precision
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

## The cap is a scale problem; resolution is a timestep problem

Recorded 2026-10-13, before any run described under predictions 6 to 11. The derivations are
in `spikecalib/theory.py` and each is checked by `tests/test_theory.py` against enumeration
or an identity, not against its own formula.

**None of the inequalities in parts 1 and 2 is new, and this document does not claim them.**
A literature check on 2026-10-13 (`SOURCES.md`, "Literature check, 2026-10-13") found each of
them in print for non-spiking networks. `[LIT]` The cap for logits in `[0, s]` is Eq. (5) of
Zhang et al. (AdaCos); NormFace gives it at width 2 as "0.45 when n = 10", which is this
study's 0.4509; the loss floor is Proposition 3.3 of Wei et al. (LogitNorm), who quote
0.7966 at ten classes, this study's floor at `T = 2`; Wei et al. (LogitClip) bound the loss
by the logit range directly; the minimum scale is Eq. (6) of Wang et al. (CosFace) and
Eq. (9) of Ranjan et al.; and a trainable scale as the remedy is NormFace's scale layer and,
in directly trained spiking networks, the trained threshold of Meng et al. (DSR). What this
study adds is the reading of `T` as the logit range of a spike-count readout, the
measurement of what that costs in calibration, the rate case in part 1, and parts 3 and 4.

The thesis under test is in two halves, and the experiments below are chosen to separate
them.

> **The confidence cap is a scale problem.** One scalar on the logits moves it.
> **Resolution is a timestep problem.** The number of distinct confidence values, and so the
> number of usable operating points, is fixed by `T` and `C` and no scalar changes it.

### 1. The cap depends on the width of the logit interval, not on `T`

`[DERIVED]` Write the winner's softmax probability as
`1 / (1 + Σ_{j≠w} e^{z_j − z_w})`. If every logit lies in an interval of width `W`, each of
the `C − 1` exponents is at least `−W`, so

    p_max ≤ 1 / (1 + (C − 1) e^{−W}) = e^W / (e^W + C − 1)

with equality when the winner is at the top of the interval and every competitor at the
bottom. Three cases of it:

| convention | logits | width `W` | cap | at `C = 10` |
|---|---|---|---|---|
| sum (this study so far) | `c_j`, in `[0, T]` | `T` | `e^T / (e^T + C − 1)` | 0.4509 at `T = 2` |
| scaled | `s·c_j`, `s > 0` | `sT` | `e^{sT} / (e^{sT} + C − 1)` | any value below 1 |
| rate | `c_j / T`, in `[0, 1]` | 1 | `e / (e + C − 1)` | **0.2320 at every `T`** |

The rate convention's cap contains no `T`: adding timesteps to a rate readout buys
resolution and no confidence. At `C = 101` it is 0.0265. The argument uses nothing about
spikes, so it also covers clipped or quantised logits in a non-spiking network. The
cross-entropy floor follows as before, `ln(1 + (C − 1) e^{−W})`.

### 2. `T_min` is a constraint on the product `sT`

`[DERIVED]` Requiring the cap to reach an accuracy `a` gives `W ≥ ln((C − 1) a / (1 − a))`,
so for a count readout

    s_min = ln((C − 1) a / (1 − a)) / T .

What this document has called `T_min` is the minimum *width*, read at `s = 1`. Halving `T`
and doubling `s` leaves the cap where it was. For the cells the predictions name: `s_min` is
3.191 on N-MNIST at `T = 2` (`a = 0.9850`), 2.211 on N-Caltech101 at `T = 2` (`a = 0.4543`),
and 0.865 at `T = 6` (`a = 0.6427`).

`[MEASURED]`, already on record: the temperatures fitted in the N-MNIST sweep imply scales
`1/t` of 3.57 to 3.66 at `T = 2`, 1.98 to 2.08 at `T = 4`, 1.39 to 1.51 at `T = 6`, 1.15 to
1.26 at `T = 8` and 1.01 to 1.10 at `T = 10`, and `1/t` exceeds that run's `s_min` in all
fifteen. So the post-hoc half of "one scale fixes calibration" is not a prediction. It is
the existing calibrator result, read as a scale.

### 3. The number of distinct confidences is at most `binomial(T + C − 1, C − 1)`, for any `s`

`[DERIVED]` Under scale `s` the top-label confidence of a count vector is
`1 / (1 + Σ_{j≠w} e^{−s(m − c_j)})` with `m` the largest count. It depends only on the
multiset of the `C − 1` gaps `m − c_j`, each an integer in `[0, T]`. There are

    binomial(T + C − 1, C − 1)

such multisets: **55 at `T = 2, C = 10`**, 715 at `T = 4`, 5,005 at `T = 6`. The scale
changes each value and not how many there are.

Counting instead by the winning count `m` and the multiset of the other counts in `[0, m]`
gives `Σ_{m=0}^{T} binomial(m + C − 1, C − 1) = binomial(T + C, C)`, which is 66 at
`T = 2, C = 10`. That is a valid upper bound and a loose one: it counts `c` and `c + k`
separately, and softmax cannot tell them apart. Removing the shifted duplicates telescopes
the sum to the figure above, which is the bound this study uses. Enumerating every count
vector at `T = 2, C = 10` gives exactly 55 distinct confidences at `s = 1`, at `s = 0.37`
and at `s = 2.5`.

**Where the bound says something.** It is informative only while it is below the number of
test samples: on N-MNIST (10,000 test samples) at `T ≤ 6`. At `C = 101` it is 5,151 at
`T = 2`, above N-Caltech101's 1,742 test samples, so there the sample count is the binding
limit and the bound cannot be violated.

**A check, not a prediction.** Whether observed distinct values stay within the bound is
already answerable from the result files, so it is not registered: 8 to 10 against 55 on
N-MNIST at `T = 2`, 35 to 39 against 715 at `T = 4`, 95 to 112 against 5,005 at `T = 6`, and
59 to 154 against 5,151 on N-Caltech101 at `T = 2`. It is recomputed over every count cell,
including the learned-scale runs, and reported as a check beside prediction 9.

### 4. A decomposition with no binning choice, where the values are few

`[DERIVED]` Group the test samples by their exact confidence value. With `f_k` the value,
`ō_k` the accuracy and `n_k` the size of group `k`, and `ō` the overall accuracy,

    Brier = REL − RES + UNC
    REL = Σ_k (n_k/n)(f_k − ō_k)²    RES = Σ_k (n_k/n)(ō_k − ō)²    UNC = ō(1 − ō)

exactly, and `Σ_k (n_k/n)|f_k − ō_k|` is a calibration error with no bins. This is
meaningful only while distinct values are few relative to the sample size. With one sample
per value every group's accuracy is 0 or 1, `RES = UNC`, and the identity says nothing. A
count readout at low `T` is in the first regime (9 values on 10,000 samples); the accumulator
(10,000 values on 10,000 samples) is in the second and has to be binned. Under binning the
identity needs two further terms, the within-bin forecast variance and covariance:
`Brier = REL − RES + UNC + WBV − WBC`.

**Rule adopted: two readouts' resolutions are compared only under the same estimator.** Exact
against exact, or equal-width against equal-width and equal-mass against equal-mass at the
same bin count. Both binned estimators are reported wherever a binned resolution is, as rule
2 requires for ECE.

What each calibrator can do to resolution, `[DERIVED]`:

- **Temperature scaling keeps the partition, so it cannot change exact resolution.** Two
  samples share a confidence exactly when they share a gap multiset, and that stays true at
  every temperature except finitely many, those at which two different multisets happen to
  give the same sum. `t = 1/ln 2` is one: three competitors one step behind then equal one
  tied competitor and two two steps behind. At such a temperature groups merge; they never
  split.
- **Top-label isotonic can only merge groups, so it cannot raise exact resolution.** It maps
  confidence through a monotone function, equal inputs stay equal, and merging groups never
  increases a between-group variance.
- **One-versus-rest isotonic is not covered.** It maps each class through its own function,
  so samples that shared a confidence can come apart.

So no post-hoc calibrator in this study can add resolution to a count readout, and the one
that changes nothing about it is the one-parameter method.

## The tie definition for predictions 10 and 11, fixed 2026-10-13

Fixed before any calibrator was fitted on N-Caltech101 and before any calibration-set
subsample was drawn.

> **Two calibrators differ on a run only if the paired 95% bootstrap interval on their ECE
> difference excludes zero under both binned estimators, equal-width and equal-mass, in the
> same direction. Otherwise they tie.**

This is the argument already made under "The paired bootstrap under both estimators": a
difference that is real is found by both estimators, in the same place and with the same
sign. Under this definition all fifteen N-MNIST runs at the full calibration set are ties,
since no run resolves under both. Intervals are the paired bootstrap of
`spikecalib.calibration.bootstrap_ece_difference`, 1,000 resamples of the test set, 15 bins,
top-label isotonic against temperature scaling.

## The learned-scale arm, as registered 2026-10-13

A count readout with a learnable logit scale: logits `z_j = s·c_j`, `s = exp(θ)`, `θ`
initialised at 0 so that `s = 1`, trained jointly with the network by the same optimiser at
the same learning rate. Everything else is the existing count run's: architecture, neuron
settings, batch size, epoch cap and patience, seeds 0 to 2, and the split, which must
reproduce the existing run's fingerprint. Three cells are trained, nine runs: N-MNIST at
`T = 2` with two blocks, and N-Caltech101 at `T = 2` and `T = 6` with four.

The learned `s` is logged for every run and compared with that run's own
`s_min = ln((C − 1) a / (1 − a)) / T`, with `a` its test accuracy. The comparison is
reported. It is not one of the falsification criteria below.

Throughout, the gap is signed: mean confidence minus accuracy on the test split, before any
calibrator, averaged over the three seeds. Negative is underconfident.

## Pre-registered prediction 6a, recorded 2026-10-13

> On N-MNIST at `T = 2` (2 blocks), the learned-scale count readout has
> |mean confidence − accuracy| < 0.05, against 0.5386 at `s = 1`.

*Falsification.* Wrong if the magnitude of the three-seed mean signed gap is 0.05 or more.

*Known before registration.* The accumulator, which is also unbounded and trained by the
same loss, has a gap of +0.0000 in this cell (−0.0015, +0.0001, +0.0014).

## Pre-registered prediction 6b, recorded 2026-10-13

> On N-Caltech101 at `T = 2` (4 blocks), the learned-scale count readout has
> mean confidence − accuracy above −0.05, and above 0: it is overconfident, against −0.3969
> at `s = 1`.

*Falsification.* Wrong if the three-seed mean signed gap is 0 or below.

*Known before registration.* The accumulator's gap in this cell is **+0.0816** (+0.0634,
+0.0802, +0.1012). This prediction was first drafted as |gap| < 0.05 and was changed, before
any learned-scale model was trained, because that result was already on record and pointed
the other way.

## Pre-registered prediction 7, recorded 2026-10-13

> On N-Caltech101 at `T = 2` (4 blocks), the learned-scale count readout recovers at least
> half of the 12.82-point accuracy gap to the accumulator.

*Falsification.* Wrong if its three-seed mean test accuracy is below 0.5184, which is the
count readout's 0.4543 plus half of 0.1282.

*Known before registration.* The gap is 3.08 points with two blocks, so its size depends on
the network. The mechanism first offered for it, a cross-entropy floor that dominates the
gradient, did not survive prediction 5. This prediction is that mechanism's second test: a
learnable scale removes the floor and leaves the threshold in place.

## Pre-registered prediction 8, recorded 2026-10-13

> On N-Caltech101 at `T = 6` (4 blocks), the learned-scale count readout has
> mean confidence − accuracy above −0.05, against −0.1534 at `s = 1`. The sign is reported.

*Falsification.* Wrong if the three-seed mean signed gap is −0.05 or below.

*What it would show.* If it holds, the residual that neither ceiling explains is removed by
freeing the logit range during training, which would make it an effect of the bounded range
on what the network learns and not of anything the readout forbids.

*Known before registration.* The cap is slack in this cell, 0.8014 against an accuracy of
0.6427, and the conditional ceiling is above accuracy too. The accumulator in the same cell
is **+0.1843**, overconfident. A learned scale that behaves as the accumulator does meets
this prediction by overshooting, which is why the sign is reported with it.

## Pre-registered prediction 9, recorded 2026-10-13

> On N-MNIST at `T = 2` (2 blocks), the count readout's resolution term stays below the
> accumulator's under both binned estimators, with or without the learned scale and after
> either calibrator.

*Definitions.* Resolution is `RES` of `spikecalib.theory.murphy_binned` at 15 bins, on the
test split, averaged over the three seeds. There are twelve comparisons: the count readout
at `s = 1` and with the learned scale, each uncalibrated, after temperature scaling and
after top-label isotonic, each against the accumulator in the same state, under equal-width
and under equal-mass binning. Calibrators are fitted on the calibration split, on CPU.

*Falsification.* Wrong if the count readout's mean resolution is at or above the
accumulator's in any of the twelve.

*Scope.* Restricted to N-MNIST because the two readouts' accuracies match there (0.9850 and
0.9861), so their uncertainty terms match and a difference in resolution is not a difference
in accuracy. On N-Caltech101 the accuracies differ by 13 points; resolution there is
reported descriptively as `RES/UNC` and no prediction is made about it.

*Known before registration.* The AUROC of confidence for correctness at `T = 2` on N-MNIST is
0.9204 for count and 0.9698 for the accumulator, which points the same way. No resolution
had been computed for any model when this was recorded.

## Pre-registered prediction 10, recorded 2026-10-13

> On N-Caltech101, temperature scaling and top-label isotonic tie in at least 7 of 9
> count-readout runs (`T` in {2, 6, 10}, 3 seeds, 4 blocks).

*Falsification.* Wrong if the two calibrators differ, under the tie definition above, in
three or more of the nine runs.

*Known before registration.* No calibrator had been fitted on N-Caltech101. The calibration
set is 1,010 samples, ten per class. On N-MNIST, with 5,000, fifteen of fifteen runs tie
under this definition.

## Pre-registered prediction 11, recorded 2026-10-13

> On N-MNIST at `T = 2`, subsampling the calibration set to 100 samples makes temperature
> scaling beat isotonic in at least 2 of 3 seeds; at 1,000 and 5,000 samples they tie.

*Definitions.* For each seed and each size, 20 subsamples of the 5,000-sample calibration
split are drawn without replacement, stratified so that each class supplies a tenth of the
size, with NumPy's `default_rng` seeded from the seed, the size and the draw index. Both
calibrators are fitted on each subsample and compared on the full test split under the tie
definition above. At 5,000 the subsample is the whole split, so the 20 draws coincide.

*Falsification.* At 100 samples, a seed counts for the prediction if temperature scaling
beats isotonic, that is, the two differ under the tie definition with temperature the lower
ECE, in at least 10 of its 20 draws. Wrong if fewer than 2 of the 3 seeds count. At 1,000
and at 5,000, a seed ties if the two calibrators differ, in either direction, in fewer than
10 of its 20 draws. Wrong if any seed fails to tie at either size. The 100-sample rule is
the author's; the rule for 1,000 and 5,000 is its mirror and was written with it.

*Known before registration.* The 5,000-sample result exists: at `T = 2`, seed 1 resolves in
favour of temperature under equal-width binning and not under equal-mass, which is a tie
under the definition above. Nothing has been fitted on a subsample.

## Outcome of predictions 6a to 11, recorded 2026-10-14 after the runs

The predictions above are unedited. They were committed at 23:29 on 2026-10-13 (`293d73c`);
the first model was trained after that, and every experiment below ran between then and the
first minutes of 2026-10-14. Each script evaluates its prediction against the registered
condition and stores the verdict beside the numbers.

| prediction | outcome | what was measured |
|---|---|---|
| 6a | **held** | N-MNIST, `T = 2`: mean signed gap +0.0044 |
| 6b | **held** | N-Caltech101, `T = 2`: mean signed gap +0.1259, positive in 3 of 3 seeds |
| 7 | **falsified** | N-Caltech101, `T = 2`: accuracy 0.4969 against a threshold of 0.5184; 33% of the gap recovered |
| 8 | **held**, by overshooting | N-Caltech101, `T = 6`: mean signed gap +0.1655, overconfident; its magnitude is not below 0.05 |
| 9 | **falsified** | count resolution below the accumulator's in 4 of 12 comparisons |
| 10 | **falsified** | calibrators tie in 4 of 9 runs; isotonic is better in the other 5 |
| 11 | **falsified** | at 100 samples temperature is better in 9, 9 and 8 of 20 draws; the ties at 1,000 and 5,000 held |

Three held and four were falsified. Two of the falsifications change what the study claims.

### The learned-scale arm (predictions 6a, 6b, 7, 8)

`[MEASURED]` `scripts/scaled_count_experiment.py`, nine runs, each on the split of the
existing count run (fingerprints checked before training).

| cell | seed | learned `s` | `s` at the last epoch | `s_min` | accuracy | count acc. | accum. acc. | `conf − acc` | ECE (ew) | ECE (em) | levels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N-MNIST, `T = 2` | 0 | 4.444 | 5.089 | 3.015 | 0.9788 | 0.9852 | 0.9864 | +0.0026 | 0.0048 | 0.0026 | 9 |
| N-MNIST, `T = 2` | 1 | 4.996 | 5.456 | 3.052 | 0.9803 | 0.9843 | 0.9876 | +0.0060 | 0.0067 | 0.0060 | 9 |
| N-MNIST, `T = 2` | 2 | 5.200 | 5.573 | 3.124 | 0.9829 | 0.9854 | 0.9844 | +0.0046 | 0.0046 | 0.0046 | 10 |
| N-Caltech101, `T = 2` | 0 | 6.002 | 6.251 | 2.307 | 0.5023 | 0.4323 | 0.5838 | +0.1228 | 0.1231 | 0.1228 | 65 |
| N-Caltech101, `T = 2` | 1 | 5.285 | 5.829 | 2.283 | 0.4902 | 0.4816 | 0.5775 | +0.1454 | 0.1474 | 0.1454 | 62 |
| N-Caltech101, `T = 2` | 2 | 4.338 | 5.105 | 2.299 | 0.4983 | 0.4489 | 0.5861 | +0.1097 | 0.1125 | 0.1135 | 62 |
| N-Caltech101, `T = 6` | 0 | 1.840 | 2.505 | 0.794 | 0.5390 | 0.6372 | 0.5614 | +0.0916 | 0.1022 | 0.0930 | 1,114 |
| N-Caltech101, `T = 6` | 1 | 3.054 | 3.195 | 0.825 | 0.5850 | 0.6349 | 0.6142 | +0.2098 | 0.2098 | 0.2098 | 924 |
| N-Caltech101, `T = 6` | 2 | 3.151 | 3.354 | 0.835 | 0.6005 | 0.6561 | 0.6171 | +0.1951 | 0.1951 | 0.1951 | 948 |

**The network uses the scale, in every run.** The learned `s` exceeds that run's `s_min` in 9
of 9, by a factor of 1.5 to 3.8. So the cap, where it bound, was a cap the network would move
if it could. **The scale had not settled.** In all nine runs it was still rising at the last
epoch: early stopping watches validation accuracy and ended training first. The values above
are where training stopped.

**6a held.** On N-MNIST the underconfidence is gone, +0.0044 against −0.5386, with ECE of
0.0054 equal-width and 0.0044 equal-mass. It costs accuracy, 0.9807 against the count
readout's 0.9850. The count of distinct confidences is 9 or 10, as before: the scale moved
the values and added none.

**6b held, and it is the accumulator's result again.** On N-Caltech101 at `T = 2` the readout
is overconfident by +0.1259, more than the accumulator's +0.0816. The first draft of this
prediction, |gap| < 0.05, would have failed.

**7 is falsified.** The learned scale reaches 0.4969, a third of the way from the count
readout's 0.4543 to the accumulator's 0.5825. Removing the cap, with the threshold left in
place, does not recover half of the accuracy. The cross-entropy-floor account of the gain
failed prediction 5 on its dependence on architecture, and it fails here on its size.

**8 held, by overshooting, which was the stated risk.** At `T = 6` the gap goes from −0.1534
to +0.1655. The underconfidence that neither ceiling explained is removed by freeing the
logit range, so it is tied to the bounded range. It is not replaced by calibration. It is
replaced by overconfidence of about the same size, and accuracy falls from 0.6427 to 0.5748.
"Underconfident because capped" and "overconfident because uncapped" are the two states
measured on this dataset; nothing here lands between them without a calibrator.

### Resolution (prediction 9, and the check)

`[MEASURED]` `scripts/resolution_experiment.py`, no training.

**The check passes.** In all 45 count and scaled-count cells the number of distinct
confidences is within `binomial(T + C − 1, C − 1)`. In the 18 cells where the bound is below
the test-set size the models use at most 18% of it.

**9 is falsified, in 8 of 12 comparisons.** N-MNIST, `T = 2`, two blocks, three-seed means:

| count readout | state | estimator | count `RES` | accumulator `RES` | count below? |
|---|---|---|---|---|---|
| `s = 1` | uncalibrated | equal-width | 0.003951 | 0.003833 | **no** |
| `s = 1` | uncalibrated | equal-mass | 0.000000 | 0.002182 | yes |
| `s = 1` | temperature | equal-width | 0.003695 | 0.003715 | yes |
| `s = 1` | temperature | equal-mass | 0.000000 | 0.002182 | yes |
| `s = 1` | isotonic-top-label | equal-width | 0.004544 | 0.003484 | **no** |
| `s = 1` | isotonic-top-label | equal-mass | 0.000000 | 0.001936 | yes |
| learned scale | uncalibrated | equal-width | 0.006836 | 0.003833 | **no** |
| learned scale | uncalibrated | equal-mass | 0.002506 | 0.002182 | **no** |
| learned scale | temperature | equal-width | 0.006949 | 0.003715 | **no** |
| learned scale | temperature | equal-mass | 0.002506 | 0.002182 | **no** |
| learned scale | isotonic-top-label | equal-width | 0.007058 | 0.003484 | **no** |
| learned scale | isotonic-top-label | equal-mass | 0.002506 | 0.001936 | **no** |

- Under equal-width binning the count readout's resolution is **not** below the
  accumulator's: 0.0040 against 0.0038, or 0.27 against 0.28 of their uncertainty terms.
- Under equal-mass binning it is zero, because the count readout's confidences fall into a
  single equal-mass bin in every seed. Three of the four comparisons that go the predicted
  way are this artefact. The fourth is equal-width after temperature scaling, by 0.00002.
- With the learned scale the count readout is above the accumulator in all six. Part of that
  is its lower accuracy, which leaves a larger uncertainty term to resolve (0.0190 against
  0.0137).

**What this does to the thesis.** "Resolution is a timestep problem" is true of the *number*
of confidence values and false of the Murphy resolution term. Eight to ten values separate
correct from incorrect predictions, in the Brier sense, as well as ten thousand do. What the
count readout gives up at `T = 2` is ranking, AUROC 0.9204 against 0.9698, and deterministic
operating points. Those are different quantities from resolution, and this document had been
treating them as one.

**The derived claims about calibrators hold on the data.** Grouped by exact value, the count
readout's resolution is 0.0048 before and after temperature scaling in every seed, with the
same 8 to 10 groups. Isotonic merges them to 4 to 6 and lowers it to 0.0046. Temperature
scaling does change AUROC, in 34 of 36 count cells and by at most 0.0052: it keeps the groups
and reorders them.

### Calibrators on N-Caltech101 (prediction 10)

`[MEASURED]` `scripts/caltech_calibrators.py`, no training. Count readout, four blocks,
calibration set 1,010, test set 1,742.

| `T` | accuracy | uncalibrated (ew / em) | temperature (ew / em) | isotonic top-label (ew / em) | fitted `t` | runs that differ |
|---|---|---|---|---|---|---|
| 2 | 0.4543 | 0.3969 / 0.3969 | 0.1449 / 0.1523 | **0.0320 / 0.0349** | 0.413 | 3 of 3, all favouring isotonic |
| 6 | 0.6427 | 0.1539 / 0.1534 | 0.0579 / 0.0532 | **0.0375 / 0.0379** | 0.837 | 2 of 3, all favouring isotonic |
| 10 | 0.6435 | 0.0442 / 0.0462 | 0.0324 / 0.0261 | **0.0300 / 0.0311** | 1.104 | 0 of 3 |

**10 is falsified, and this is the most consequential result of the set.** The calibrators
tie in 4 of 9 runs. In the other five, all three at `T = 2` and two at `T = 6`, isotonic is
better under both estimators. At `T = 2` temperature scaling leaves an ECE of about 0.15 where
isotonic leaves 0.03. The fitted temperature there, 0.413, is a scale of 2.42, above `s_min`,
so the cap is lifted and the error that remains is not one a single scale removes.

Three things follow, and none may be softened.

- **"Which calibrator you use does not matter" is an N-MNIST result.** It does not hold at 101
  classes where the cap binds.
- **"One scale fixes calibration" is false on N-Caltech101 at `T = 2`**, both post hoc (this
  table) and learned (prediction 6b). The cap is a scale problem. The miscalibration is not
  only the cap.
- **The study's starting hypothesis is what these nine runs show.** It predicted an isotonic
  advantage that grows as `T` falls. On N-MNIST it failed. Here the advantage is 0.11 at
  `T = 2`, 0.02 at `T = 6` and nothing at `T = 10`. The capacity mechanism offered for it was
  not tested here, and nine runs of one architecture do not establish it.

Calibrators were also fitted to the unbounded readouts, for the first time. For the
accumulator the two differ in 6 of 9 runs, always favouring isotonic. For the learned-scale
readout, of 6 runs, 3 tie, 2 favour temperature and 1 favours isotonic.

### Calibration-set size (prediction 11)

`[MEASURED]` `scripts/calibration_size_experiment.py`, no training. N-MNIST, `T = 2`.

| calibration samples | seed | temperature better | isotonic better | tie |
|---|---|---|---|---|
| 100 | 0 | 9 of 20 | 2 of 20 | 9 of 20 |
| 1,000 | 0 | 4 of 20 | 1 of 20 | 15 of 20 |
| 5,000 | 0 | 0 of 20 | 0 of 20 | 20 of 20 |
| 100 | 1 | 9 of 20 | 0 of 20 | 11 of 20 |
| 1,000 | 1 | 7 of 20 | 1 of 20 | 12 of 20 |
| 5,000 | 1 | 0 of 20 | 0 of 20 | 20 of 20 |
| 100 | 2 | 8 of 20 | 1 of 20 | 11 of 20 |
| 1,000 | 2 | 6 of 20 | 0 of 20 | 14 of 20 |
| 5,000 | 2 | 0 of 20 | 0 of 20 | 20 of 20 |

**11 is falsified on its first clause.** At 100 samples temperature scaling is better in 9, 9
and 8 of 20 draws, and the registered bar was 10 in two of three seeds. The ties at 1,000 and
5,000 held. The direction is the predicted one and it is graded: temperature is better in 26
of 60 draws at 100 samples, 17 of 60 at 1,000 and none at 5,000, while isotonic is better in
3, 2 and none. The mean ECE at 100 samples is 0.0079 against 0.0114 equal-width and 0.0061
against 0.0090 equal-mass. Scarce calibration data leans towards the one-parameter method,
less decisively than predicted.

### What changes in what the study says

- The calibrator result is stated per regime: a tie on N-MNIST with ample calibration data,
  a lean to temperature when that data is scarce, and isotonic on N-Caltech101 where the cap
  binds.
- The taxonomy is conditional on where the cap sits relative to accuracy. A learned scale
  moves a count readout from one side to the other.
- The `T = 6` residual on N-Caltech101 is no longer wholly unexplained: it goes when the
  logit range is freed. Why the capped network was underconfident when nothing forced it is
  still open.
- "Resolution" is no longer used for the number of confidence values.

## Consequence for multisensory fusion

Recorded 2026-10-14, before any fusion result was computed. **This is a corollary of known
decision-fusion theory applied to spike-count readouts. It is not a new theorem**, and the
theory it follows from is recorded in `SOURCES.md` as each source is verified. The
derivations are in `spikecalib/fusion.py` and `spikecalib/theory.py`, checked by brute force
and by simulation in `tests/test_fusion.py`.

Two modalities, each read out by a count readout over `T` steps, each reporting a probability
vector over the same `C` classes. A readout is *saturated* on an input when its winner fired
at every step and no other class fired: the count vector `[T, 0, ..., 0]` up to position, the
one vector that attains the ceiling. `S` is the event that both modalities saturate.

`[DERIVED]` **(a) On `S` a symmetric rule cannot weight the modalities.** Every saturated
report is the same vector up to which class wins. So a fusion rule that uses only the two
reports, treats the two modalities alike and treats the classes alike gives the two winners
the same fused score: exchanging the two classes exchanges the two reports, and the rule
cannot tell which came from which modality. If the rule is also strictly increasing in each
report, the tie is between exactly those two classes, and with uniform tie-breaking it is an
unweighted vote, a coin flip whenever the modalities disagree. The product, sum and max
rules are of this kind.

*Two conditions were added to the proposition as first stated*, which asked only for symmetry
between modalities. Symmetry between classes is needed for the exchange argument. Strict
monotonicity is needed for the tie to be two-way: the min rule is symmetric and not strict,
and on disagreement it ties every class, which is worse than a vote and still cannot tell
the modalities apart. Both conditions are checked by brute force over every pair of winners.

`[DERIVED]` **(b) What the coin flip costs.** Against a rule that sides with modality 1
whenever the two disagree, which is what reliability-weighted voting does when modality 1 has
the larger weight `log((C − 1) a_k / (1 − a_k))`, the symmetric rule loses, in accuracy,

    0.5 · P(S) · (a_{1,S} − a_{2,S})

with `a_{k,S}` the accuracy of modality `k` on `S`. This is an identity and needs no model
of the errors: where the modalities agree they are right or wrong together, so the
difference in their accuracies on `S` is carried entirely by the inputs on which they
disagree, and the coin gives up half of it. The symmetric-error model enters only in calling
reliability-weighted voting the optimal rule for two modalities known by their accuracies.

**The expression in the first draft of (b) was `0.5 · P(S and disagree) · (a_{1,S} − a_{2,S})`,
and it is wrong by a factor.** `a_{1,S} − a_{2,S}` already equals `P(disagree | S)` times the
difference in accuracy conditioned on disagreement, so multiplying it by the probability of
`S` *and* disagreement counts the probability of disagreement twice. The drafted expression
is smaller than the correct one by exactly `P(disagree | S)`. This was found by simulation
before any fusion run, and `tests/test_fusion.py` shows the simulation reproducing the
corrected expression and not the draft. Both are computed and reported.

`[DERIVED]` **(c) A per-modality scale restores the weighting.** With logits `s_k · c` and
`s_k · T = log((C − 1) a_k / (1 − a_k))`, the product rule on `S` is exactly
reliability-weighted voting, vote for vote and posterior for posterior. This is the known
remedy, calibrating each source before combining, and the scale is `s_min` of
"The cap is a scale problem": the scale that lifts a modality's cap to its accuracy is the
scale that weights its vote correctly. A scale common to both modalities restores nothing.

`[DERIVED]` **(d) How much a count readout can say.** Its top-label confidence takes
`binomial(T + C − 1, C − 1)` values at a generic scale, 55 at `T = 2, C = 10`. Its full
probability vector takes `(T + 1)^C − T^C` values at every scale, 58,025: the count vectors
with at least one zero, since softmax ignores a constant shift.

### The validation, as fixed before any fusion result

Inference only, on the existing N-MNIST models. No training.

- **Two conditionally independent modalities.** Modality A is a test sample. Modality B is a
  different test sample of the same class, drawn uniformly at random, with event dropout.
  Given the class the two are independent draws.
- **Event dropout** removes each event independently with probability `r`: a frame's event
  count `n` at a pixel becomes a draw from `Binomial(n, 1 − r)`. One independent dropout draw
  is made for each pairing.
- **The rate `r`** is chosen once, on the calibration split, with the seed-0 count model at
  `T = 2`: the value on a grid of 0.00 to 0.99 in steps of 0.01 whose calibration accuracy
  under dropout is closest to the clean calibration accuracy minus 0.10. It is fixed and
  recorded below before any fusion result is computed, and used unchanged for every `T`,
  seed and readout.
- **Readouts.** Count at `T` in {2, 4, 6, 8, 10} and accumulator at `T` in {2, 6, 10}, three
  seeds, two blocks, 20 random pairings each.
- **Rules.** R0, the product rule on the raw softmax outputs, ties broken uniformly at
  random. R1, the product rule after temperature scaling each modality, the temperatures
  fitted on the calibration split (A on clean calibration samples, B on calibration samples
  under one seeded dropout draw). R2, reliability-weighted voting, the weights from each
  modality's accuracy on the same calibration data.
- **`delta = acc(R1) − acc(R0)`**, pooled over the 20 pairings, and split into the part from
  samples in `S` and the part from the rest. For an accumulator, which has no counts, `S` is
  empty and the whole of `delta` is off `S`.
- **Intervals** are 95% percentile bootstrap intervals over the 10,000 test samples taken as
  modality A, each carrying the mean of its 20 pairings, 1,000 resamples, seeded.
- **ECE** of each rule's fused confidence is reported under both estimators.

**The dropout rate, fixed 2026-10-14 before any fusion result: `r = 0.50`.**
`[MEASURED]` `scripts/fusion_experiment.py --select-rate`, seed-0 count model at `T = 2`, on
the 5,000-sample calibration split. Clean calibration accuracy is
0.9838, so the target was 0.8838; the grid
value closest to it is 0.50, at which the accuracy under dropout is
0.8800. Half of all events are removed from modality B. The
grid is in `results/fusion_dropout_rate.json`. That step computed no fused prediction, and
the main run reads the rate from the file and cannot change it.

## Pre-registered prediction 12, recorded 2026-10-14

> For count readouts at `T = 2`, `delta` is positive with its interval excluding zero in
> every seed.

*Falsification.* Wrong if in any of the three seeds the lower end of the interval on `delta`
is at or below zero.

## Pre-registered prediction 13, recorded 2026-10-14 (a check of the proposition)

As drafted:

> The on-`S` part of `delta` agrees with `0.5 · P(S and disagree) · (a_{1,S} − a_{2,S})`
> within its interval in every seed.

**Registered in corrected form, for the reason given under (b):**

> The on-`S` part of `delta` agrees with `0.5 · P(S) · (a_{1,S} − a_{2,S})` within its
> interval in every seed.

*Definitions.* Evaluated for the count readout at `T = 2`, the cell prediction 12 names;
other cells are reported. "Agrees within its interval" means the predicted value lies
inside the 95% bootstrap interval of the on-`S` part of `delta`. `P(S)` and the two
accuracies are measured on the same pooled pairings.

*Falsification.* Wrong if in any seed the corrected expression lies outside that interval.
The drafted expression is evaluated the same way and reported beside it, labelled as the
draft; its outcome does not decide this prediction.

*What it tests.* Part (b) describes a rule that sides with modality A on every disagreement
in `S`. R1 is that rule only if temperature scaling gives A the larger scale in every seed,
which is expected, since A is the more accurate, and is not guaranteed. If B were given the
larger scale the measured value would have the opposite sign.

## Pre-registered prediction 14, recorded 2026-10-14

> `delta` falls as `T` rises, and its interval includes zero at `T = 8` and `T = 10`.

*Definitions.* Count readouts. "Falls as `T` rises": the three-seed mean of `delta` is
strictly decreasing over `T` = 2, 4, 6, and its values at `T = 8` and `T = 10` are both below
its value at `T = 6`.

*Falsification.* Wrong if that ordering fails, or if in any seed the interval on `delta`
excludes zero at `T = 8` or at `T = 10`.

## Pre-registered prediction 15, recorded 2026-10-14

> For accumulator readouts, `delta`'s interval includes zero at every `T`.

*Falsification.* Wrong if in any of the nine runs (`T` in {2, 6, 10}, three seeds) the
interval on `delta` excludes zero.

*Known before registration, for all four.* On N-MNIST at `T = 2` between 94.9% and 95.4% of
the count readout's test predictions sit at the ceiling (`LOG.md`, 2026-09-28), so saturation
is common for the clean modality. No dropout rate had been chosen, no model had been run on a
dropped-out input, and no fused prediction had been computed when these were recorded.

## Outcome of predictions 12 to 15, recorded 2026-10-14 after the run

`[MEASURED]` `scripts/fusion_experiment.py`, written to `results/fusion_experiment.json`,
which carries the verdict block the script computed against the conditions above.
`tests/test_provenance.py` re-derives each verdict from the rows. The predictions above are
as they were registered; nothing in them was edited.

| | outcome | registered condition | measured |
|---|---|---|---|
| 12 | **held** | count, `T = 2`: lower end of the interval on `delta` above zero in every seed | `delta` +0.0064, +0.0033, +0.0032; lower ends +0.0058, +0.0028, +0.0026 |
| 13 | **held** | count, `T = 2`: `0.5 · P(S) · (a_{1,S} − a_{2,S})` inside the interval of the on-`S` part of `delta` in every seed | predicted +0.0041, +0.0018, +0.0020 against measured +0.0043, +0.0019, +0.0020; inside in 3 of 3 |
| 14 | **falsified** | count: mean `delta` strictly decreasing over `T` = 2, 4, 6, lower again at 8 and 10, and no interval excluding zero at 8 or 10 | means +0.00429, +0.00007, +0.00001, −0.00001, +0.00001; `T = 10` is not below `T = 6`; no interval excludes zero at 8 or 10 |
| 15 | **falsified** | accumulator: no interval on `delta` excluding zero in nine runs | one of nine excludes zero: `T = 2`, seed 2, `delta` −0.00016 [−0.00026, −0.00005] |

Two held and two were falsified.

**Prediction 12 held.** Calibrating each modality before the product rule raised fused
accuracy at `T = 2` in every seed: 0.9911 to 0.9954 on the three-seed mean, 0.43 points. Both
modalities saturate on 64% to 77% of pairs. In absolute terms the gain is small, because both
modalities are accurate on N-MNIST and the raw product rule already beats either alone
(0.9850 and 0.9142). In relative terms it removes 48% of the fused errors, 0.89% to 0.46%.

**Prediction 13 held, and the draft would not have.** The corrected expression lies inside
the interval of the measured on-`S` part in all three seeds. The expression as first drafted
gives +0.00007, +0.00002 and +0.00002, outside the interval in all three: it is short by the
factor `P(disagree | S)`, which is 0.012 here. Temperature scaling gave modality A the larger
scale in all three seeds at `T = 2`, which the identity needs. 63% of `delta` comes from
pairs in `S`; the remaining +0.0016 comes from pairs outside it, which the proposition does
not describe.

**Prediction 14 was falsified, on its ordering clause.** The three-seed means at `T = 6` and
`T = 10` are equal: each is a net 6 pairs in 600,000, +0.00001. The prediction required the value at
`T = 10` to be below the value at `T = 6`, and equal is not below. Its other clause held: no
interval excludes zero at `T = 8` or `T = 10`. What the data show is a fall from +0.0043 at
`T = 2` to nothing measurable from `T = 4`, where all twelve intervals include zero. The
prediction asked for a strict ordering among values that cannot be told from zero, and that
is a fault in how it was written, recorded here and not corrected. `P(S)` falls more slowly
than the effect, 0.70, 0.50, 0.20, 0.03, 0.01: at `T = 4` half of all pairs still saturate,
but two saturated modalities disagree on 0.03% of pairs against 0.81% at `T = 2`. The effect
needs saturation and disagreement together, and from `T = 4` saturated readouts are almost
never wrong.

**Prediction 15 was falsified, by one run of nine.** At `T = 2`, seed 2, temperature scaling
made the accumulator fusion worse by 0.00016, a net 32 pairs in 200,000, with an interval that
excludes zero. The other eight intervals include it, and no accumulator run has `delta` above
+0.00004. With nine intervals at the 95% level one exclusion is not surprising, and the
prediction allowed none; it is reported as falsified. The direction is against calibration,
not for it: there is no sign of the count readout's gain in the accumulator.

**Two things the run showed that were not predicted.**

- *Reliability-weighted voting is not the better rule overall.* With two voters the heavier
  vote decides every disagreement, so R2 is modality A alone, 0.9850 at `T = 2`, which is
  0.0061 below the raw product rule. Outside `S` the counts carry graded evidence that a vote
  throws away. Part (b) compares rules on `S` only, and the paper says so.
- *Temperature scaling does not always favour the more accurate modality.* At `T = 2` it gave
  A the larger scale in 3 of 3 seeds. At `T ≥ 6` it gave the event-dropped modality B the
  larger scale in 9 of 9 count runs: dropping events lowers the counts themselves, and the
  fitted temperature compensates for the smaller logits. There the identity has the wrong sign
  for R1, as the note under prediction 13 said it would, and it falls outside the measured
  interval in three runs, each resting on fewer than ten saturated disagreements in 200,000
  pairs. Across all 15 count runs it lies inside in 12.

ECE of the fused confidence at `T = 2`, equal-width / equal-mass, three-seed mean: raw
product rule 0.1801 / 0.1801, after per-modality temperature scaling 0.0027 / 0.0015,
reliability-weighted vote 0.0013 / 0.0013.

**What the paper may say.** That the proposition's identity was confirmed at `T = 2`; that the
cost is real there, about half the fused errors and under half a point of accuracy; that it is
absent from `T = 4` on this dataset; that two of four predictions were falsified. It may not
say that calibration before fusion matters in general for spiking networks: one dataset, a
constructed second modality, and an effect at one timestep setting.

## Post-review analyses, thresholds fixed 2026-11-12 before computation

**Written before any of the analyses below was computed, and not to be revised afterwards.**
They were added in response to a review received on 2026-11-12. None of them is a
pre-registered prediction in the sense of the sections above, and the paper labels each one
"added after review, not pre-registered". Every prediction, date and outcome recorded above
stays exactly as it is. What is fixed here is every threshold, margin, stopping rule, seed and
compute limit, so that none can be chosen after seeing a result.

Common settings. Bootstraps use 1,000 resamples of the test set drawn with
`numpy.random.default_rng(seed)`, `seed` being the run's seed, exactly as
`spikecalib.calibration.bootstrap_ece_difference` draws them. Bins: 15, both estimators, as
in every earlier result. Means over seeds are reported with the sample standard deviation
(`ddof = 1`) and the range. No correction for multiplicity is applied anywhere below; the
number of tests is stated beside each family instead.

### A1. The cap as a floor on ECE

For any partition of the test set into bins, `ECE >= |a - p_bar|` by the triangle
inequality, `a` being accuracy and `p_bar` mean confidence, with equality exactly when every
occupied bin's signed gap (accuracy minus mean confidence) has the same sign. Every
confidence of a count readout is at most `p* = e^T / (e^T + C - 1)`, so `p_bar <= p*`, and
wherever `a > p*` it follows that `ECE >= a - p*` under any binning, including grouping by
distinct confidence value. For every count-readout cell with `a > p*`: the per-bin signed
gaps under equal-width bins, equal-mass bins and grouping by distinct value; the number of
overconfident bins (signed gap below zero), the share of test samples in them and the
largest overconfident gap; the floor ratio `(a - p*) / ECE` under each estimator, as mean,
standard deviation and range over seeds, and a test-set bootstrap 95% percentile interval.
For the interval, each resample redraws every seed's test set with that seed's generator,
recomputes accuracy and ECE (bins rebuilt on the resample, as in the paired bootstrap), and
averages the ratio over seeds. Descriptive; no threshold.

### A2. Equal-mass bins with repeated values

`_equal_mass_bins` packs unique confidence values greedily and never splits a tie, so bins
are not equal in mass. Per cell: the effective bin count, the smallest and largest bin
occupancy, and ECE grouped by distinct value, `sum_k (n_k / n) |acc_k - conf_k|`, beside both
binned estimates. Descriptive; no threshold.

### A3. Equivalence of the two calibrators: two one-sided tests

The registered rule of 2026-10-13 stays as registered. Its unresolved outcome is no longer
to be called a tie or a match in the paper. Added beside it, for every run on which the
paired bootstrap of isotonic minus temperature ECE was computed (N-MNIST: 15 count runs;
N-Caltech101: 9 count, 9 accumulator and 6 scaled-count runs), under both estimators:

- Equivalence at `alpha = 0.05` is declared at margin `delta` when the 90% paired-bootstrap
  percentile interval (5th to 95th percentile of the same 1,000 resampled differences)
  lies inside `[-delta, delta]`.
- The resampled differences are regenerated with the same seed and the same draws as the
  stored 95% intervals, and the regenerated 95% interval must reproduce the stored one to
  within `1e-12` before any 90% interval is used.
- Three margins, all reported, as a sensitivity analysis:
  - `delta = 0.005`. Strict. It is about three binomial standard errors of an accuracy of
    0.97 on the 10,000-sample N-MNIST test set, and of the order of N-MNIST's ECE after
    temperature scaling.
  - `delta = 0.01`. One percentage point of ECE, a conventional practical margin below
    which the choice between two calibrators would not change a deployment decision.
  - `delta = 0.02`. Lenient. It is about 1.7 binomial standard errors of an accuracy near
    0.5 on the 1,742-sample N-Caltech101 test set, so smaller differences there are hard
    to tell from sampling noise.
- A run is reported as equivalent under both estimators only if each estimator's interval
  lies inside the margin. The calibration-size subsamples are not included.

### A4. Run-level metrics, scoring rules, reliability diagrams

Per run, every seed and configuration: accuracy, ECE under both estimators, top-label Brier
score, top-label log loss and the AUROC of confidence for correctness
(`sklearn.metrics.roc_auc_score`). Multiclass negative log-likelihood and Brier score where
the output is a full distribution (uncalibrated and temperature-scaled); isotonic regression
is top-label and gets top-label scores only. Probabilities are clipped at `1e-12` before any
logarithm, and the number clipped is reported. Calibrators refitted for this purpose must
reproduce the published ECE to within `1e-9`, or the script stops. Reliability diagrams:
for count readouts, accuracy against each distinct confidence value, marker area
proportional to mass, Wilson 95% intervals (`z = 1.959964`); for continuous readouts,
equal-mass bins.

### A5. Spiking activity

Spike rasters use the seed-0 weights and test-set index 0 of the seed-0 split, for N-MNIST
and for N-Caltech101, at `T = 2` and `T = 10`, fixed here before any raster is drawn. Hidden
neurons are subsampled with `numpy.random.default_rng(0)`, 32 per hidden layer, and every
output neuron is shown. Firing statistics over the full test set, per layer: mean spikes per
neuron per timestep, and the share of neurons that never fire on any test sample at any
timestep. Output-layer count patterns at `T = 2`: the share of test samples whose top class
fires at every step while every other class is silent (the vector that attains `p*`), the
share with every class silent, and the share with a tie for the top count. Descriptive.

### A6. Same weights, other readouts

Exploratory. The count-trained networks are read without retraining: the output LIF is
bypassed and the final BatchNorm1d output is aggregated as the accumulator (`v += z`) and as
the leaky membrane (`v += (z - v) / tau`, `tau = 2`). Accuracy and ECE under both
estimators are reported before and after temperature scaling fitted on the calibration
split. No threshold; the paper labels it exploratory.

### A7. The learned scale, trained to convergence

The nine scaled-count models (N-MNIST `T = 2`; N-Caltech101 `T = 2` and `T = 6`; seeds 0,
1 and 2) are retrained with every setting of the original runs except the stopping rule:
architecture, neuron settings, optimiser (Adam, learning rate `1e-3`), batch size, split,
seed, MPS device and zero loader workers. Each model is initialised after
`set_seed(seed)`, so its initialisation depends on its seed alone.

- **Convergence.** `theta = ln s` is read after every epoch. A run has converged at the
  first epoch `e >= 5` at which the six end-of-epoch values `theta_(e-5), ..., theta_e`
  span less than 0.01, so that `theta` has moved by less than 0.01 over five consecutive
  epochs.
- **Cap.** 150 epochs, three times the original cap of 50. Training stops at convergence
  or at the cap, whichever comes first.
- **Logged every epoch.** `theta`, `s`, training loss, validation accuracy (the original
  evaluation), and validation ECE under both estimators.
- **Two checkpoints reported.** The checkpoint the original rule would have selected in
  this run (best validation accuracy, stopping after 5 epochs without improvement, within
  50 epochs) and the end point (the weights at convergence or at the cap). For both: test
  accuracy, ECE under both estimators, mean confidence minus accuracy, `s` against
  `s_min`, the cap `e^(sT) / (e^(sT) + C - 1)` and the floor `max(0, a - cap)`.
- **Compute limit.** One epoch per cell is timed first. If the projected total, at 150
  epochs for every run, exceeds five hours, the three N-MNIST runs are trained in full and
  then N-Caltech101 runs in the order `T = 2` seed 0, `T = 6` seed 0, `T = 2` seed 1,
  `T = 6` seed 1, `T = 2` seed 2, `T = 6` seed 2. A run starts only if its projected
  worst-case time fits within five hours of the start of training. What ran is reported
  exactly.
- The original nine runs remain the pre-registered results.

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

*(Pointer added 2026-10-13. This overreaches. No calibrator was fitted to the accumulator or
to any other unbounded readout in this study; the four calibrators were fitted to the count
readout on N-MNIST only. The null result therefore says nothing about whether temperature
scaling is adequate for the uncapped case, and Li, Jones & Furber's use of it is their
finding rather than corroboration of this one. The matching dashboard claim was removed on
2026-09-28, recorded in `LOG.md`, and the paper's Limitations give the scope under
"Calibrators were compared on N-MNIST only".)*

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

*(Pointer added 2026-10-13. Two phrases in the first bullet are superseded. "16x" is seed 0's
ratio; over three seeds it is 11.9× to 63.5×, mean 30.5×, as "The membrane advantage at
`T = 2` survives, with a wide spread" records. And "should be avoided" holds only where the
cap binds, which on N-MNIST is `T ≤ 6`: above that the cap forces no error and the count
readout ranks better, and membrane's ranking advantage holds at `T ≤ 4` only. See "Membrane
is NOT strictly dominant, and the recommendation narrows accordingly".)*

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

*(Pointer added 2026-10-13. This section records the scope as planned on 2026-09-09 and is
superseded by what was run. The second dataset is N-Caltech101, added for the C axis, not
DVS-Gesture; DVS-Gesture and CIFAR10-DVS are registered and not implemented. The five-`T`
calibrator sweep was run on N-MNIST only, so no isotonic-versus-`T` trend is reported
across datasets, and N-Caltech101 was run at `T` in {2, 6, 10} with no calibrator fitted.
See "The C axis: N-Caltech101 at 101 classes" and "Status".)*

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

*(Pointer added 2026-10-14. The first sentence is too strong. Chen, Park & Simeone,
arXiv:2305.11322, report that a spike-count softmax is "first under-confident and then
over-confident" as time passes, so early underconfidence of this readout is a published
observation and not this study's own. What this study adds is the reason, the cap
`e^t / (e^t + C − 1)` on a count read after `t` steps, and the measurement at `T = 2` to 10.
See `SOURCES.md`, "Literature check, 2026-10-13".)*

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

2. **Two ECE estimators, both reported.** Equal-width binning (as described by Guo et al.,
   ICML 2017, who attribute ECE to Naeini et al., 2015, and report it with 15 bins) and
   equal-mass binning are both computed and both reported. With discrete spike counts
   the confidence distribution is lumpy: equal-width binning leaves bins empty and
   concentrates mass in a few others, which biases the estimate **optimistically** at low
   `T` — precisely the regime the hypothesis is about. Reporting only equal-width would
   flatter every method in the regime where the comparison matters. **Disagreement
   between the two estimators is itself a finding** and is reported when it occurs,
   rather than being resolved by picking the more convenient one.

3. **Energy is a proxy, stated as such.** Energy is reported as the SynOps proxy using
   45 nm figures from Horowitz, "Computing's Energy Problem (and what we can do about
   it)", ISSCC 2014. **Verified against the primary document on 2026-09-12**, and the two
   constants do not have the same status:

   - `E_AC = 0.9 pJ` is **`[LIT]`**, confirmed as the 32-bit floating-point add in
     **Figure 1.1.9**, "Rough energy costs for various operations in 45nm 0.9V". A caption
     may cite that figure number for this value.
   - `E_MAC = 4.6 pJ` is **`[DERIVED]`**, not `[LIT]`. **It appears nowhere in that
     paper** — there is no MAC row and no combined figure. It is the community convention
     of FP multiply plus FP add, 0.9 + 3.7, and **must not be attributed to Horowitz**.

   The source describes its own figures as "Rough", so nothing may turn on their
   precision. Every appearance of an energy number in text, table, or figure describes it
   as a **design-comparison proxy**. It is never described as measured power, and no claim
   about a physical device is derived from it.

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
  et al., who describe this formulation and attribute its one-versus-all extension to
  Zadrozny & Elkan (2002). It is reported alongside so that readers
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

The paper, `paper/main.tex`, carries this in its Acknowledgements, once, in these words:

> AI assistance was used for implementation and drafting. All research design, hypotheses,
> the pre-registered prediction and its outcome, and all verification decisions are the
> author's. Every citation was checked against primary documents; two were withdrawn because
> no primary document supported them, and one energy constant was reclassified from cited to
> derived.

(Amended 2026-09-29: "withdrawn on verification" told a reader nothing, since the withdrawn
references do not appear in the paper. The clause now states the reason, which covers both
cases in `SOURCES.md`: one identifier belonged to a different paper, the other paper could not
be found.)

It is not to be softened and not to be expanded, and it belongs in the paper rather than on
the dashboard. Both of its factual claims are supported here: the two withdrawn references
are recorded in `SOURCES.md`, and the energy constant is E_MAC = 4.6 pJ, reclassified from
`[LIT]` to `[DERIVED]` on 2026-09-12 because it appears nowhere in Horowitz and is the
0.9 + 3.7 multiply-plus-add convention.

**Superseded 2026-10-14 by the author.** The disclosure above is no longer the paper's. The
whole Acknowledgements section of `paper/main.tex` is now exactly this, and this text
supersedes the fixed wording above, including the sentence about two withdrawn citations:

> AI assistance was used in preparing this work. The author verified all results and takes
> full responsibility for the content. Every citation was checked against its primary
> document or, where that was inaccessible, against a verified secondary source, as recorded
> in SOURCES.md.

The rules above on where it goes still hold: once, in the paper, not on the dashboard. The
earlier text is kept here as a record. Its "two were withdrawn" had stopped being accurate:
both papers were found on 2026-10-13 and are recorded in `SOURCES.md`. The new last sentence
is supported there too: the citations that could not be opened, Nitzan and Paroush (1982) and
Chair and Varshney (1986), are each recorded with the verified secondary source they are
cited through. `tests/test_provenance.py` checks that the paper carries this text verbatim.

**Superseded 2026-11-12, for three authors.** The paper now has three authors. The whole
Acknowledgements section of `paper/main.tex` is now exactly this, and it supersedes the
2026-10-14 text above, which is kept as a record:

> AI assistance was used in preparing this work. Every citation was checked against its
> primary document or, where that was inaccessible, against a verified secondary source, as
> recorded in SOURCES.md. The first author verified all results; all authors reviewed the
> manuscript and take responsibility for its content.

The disclosure still names no tool. `tests/test_provenance.py` checks that the paper carries
the latest superseding text verbatim.

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
| 1 | temperature scaling matches or beats top-label isotonic at every `T`, and no isotonic advantage is resolvable | **falsified in the letter**; its substance — that the two calibrators tie — held |
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
regime where the calibrator null result could plausibly break.

The paper is written: `paper/main.tex`, ten pages, with every measured value a macro that
`paper/make_numbers.py` regenerates from `results/`. A preprint is in preparation.

**Added 2026-10-14.** Four further experiments have run, on seven further predictions (6a,
6b, 7, 8, 9, 10, 11), of which three held and four were falsified; see "Outcome of predictions
6a to 11". Nine more models were trained, the learned-scale arm, which makes 105: 69 on
N-MNIST and 36 on N-Caltech101. Twelve predictions have now been registered in all.

| experiment | scale |
|---|---|
| learned-scale arm | 3 cells × 3 seeds, 9 runs |
| resolution and the bound check | 45 cells checked, 675 decomposition rows, no training |
| calibrators on N-Caltech101 | 3 readouts × up to 3 `T` × 3 seeds × 4 calibrators, no training |
| calibration-set size | 3 seeds × 3 sizes × 20 draws, no training |

The bound itself is prior work and is cited as such; see "The cap is a scale problem". The
paper was revised around these results on 2026-10-14 and is thirteen pages.

**Added 2026-10-14, later.** A fifth experiment has run, fusion of two readouts, on four
further predictions (12 to 15), of which two held and two were falsified; see "Outcome of
predictions 12 to 15". It trained nothing: 24 runs of inference, 20 pairings each. Sixteen
predictions have now been registered in all. The fusion section is in the paper, which is
sixteen pages with it.

**Added 2026-10-14, after the author's review.** The author accepted prediction 13 in its
corrected form and the overnight decisions on the fusion section. The paper is set in the
original eleven-point type, not ten-point, under the title "Counting Spikes Caps Confidence:
The Calibration Cost of a Known Softmax Bound in Low-Timestep Spiking Networks". Three items
moved to an appendix with nothing deleted: the N-Caltech101 calibrator table, the fusion
table's rows for `T ≥ 4`, and the paragraph on probabilistic population codes. It is 19 pages:
17 of main text, references, and a one-page appendix.

**Added 2026-11-12, after review.** A review received on 2026-11-12 asked for further analyses.
Their thresholds were fixed in "Post-review analyses, thresholds fixed 2026-11-12 before
computation" and committed before any was run. None is a pre-registered prediction, and nothing
above was edited. Outcomes, every one generated into `paper/numbers.tex`:

- A1. Where accuracy exceeds the cap, ECE is at least `a - p*` under any binning (triangle
  inequality). At `T = 2` the floor is 99.17% of the equal-width ECE on N-MNIST and 97.10% on
  N-Caltech101 (seed means; equal-mass gives the same). No binned N-Caltech101 bin is
  overconfident at `T = 2`, but grouped by distinct value 40 to 73 small groups are, so accuracy
  above the cap does not make every group underconfident. `results/ece_floor.json`.
- A2. Equal-mass bins never split a tie, so they are not equal in mass: one bin of 10,000 at
  N-MNIST `T = 2`, two very unequal bins at `T = 4`. The paper's method text said otherwise and
  was corrected.
- A3. Two one-sided tests: on N-MNIST the calibrators are equivalent within 0.01 ECE in all 15
  runs under both estimators (12 of 15 within 0.005); on N-Caltech101 no count run is within
  0.01 and 2 of 9 are within 0.02. `results/calibration_metrics.json`.
- A4. Every published ECE was reproduced from the stored outputs to within 1e-9 before use.
  AUROC at N-MNIST `T = 2`: count 0.9204, accumulator 0.9698, paired difference interval
  -0.0647 to -0.0348.
- A5. The stored count outputs were reproduced bit for bit on MPS for all 36 count networks.
  At `T = 2` the top class fires at both steps with all others silent on 95.1% of N-MNIST test
  samples; on N-Caltech101 58.0% of samples tie for the top count. `results/spiking_activity.json`.
- A6. Read without retraining through the accumulator, the count-trained N-Caltech101 networks
  reach 0.5744 at `T = 2` against 0.4543 through their spikes: 94% of the gap to a separately
  trained accumulator is present with identical weights, so it is lost at the readout and not
  in training. The same-weights accumulator is overconfident on N-Caltech101 in every cell.
  `results/same_weights_readouts.json`.
- A7. The nine scaled-count models were trained again with every setting unchanged except
  stopping (theta settling by less than 0.01 over five epochs, or 150 epochs). The projected
  worst case was 4.3 hours, so all nine ran (2.1 hours). Five settled, all on N-Caltech101; the
  three N-MNIST runs and one N-Caltech101 `T = 6` run reached the cap with theta still rising
  slowly. Every learned scale stays above `s_min`, so the floor is zero throughout. On N-MNIST
  the end points stay calibrated (|conf - acc| at most 0.0068). On N-Caltech101 the
  overconfidence grows with training: at `T = 2` from +0.1177 at the checkpoint the original
  rule would keep to +0.1737 at the end point. Accuracy rises too: at `T = 2` the end points
  recover 72% of the gap from the count readout to the accumulator, against 28% at the
  original rule's checkpoint of the same runs. The pre-registered runs remain the results of
  predictions 6a to 8. `results/scale_convergence.json`.
