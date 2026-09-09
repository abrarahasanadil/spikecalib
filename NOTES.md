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

On synthetic counts with 10 classes and a 10,000-sample calibration split, `T = 2` gave
52 distinct confidence levels — about **190 samples per level**, so each per-level
accuracy is well conditioned. `T = 10` gave 3,035 levels, roughly **3 samples per
level**, which is where isotonic starts fitting noise. `[MEASURED]` —
`scripts/synthetic_check.py`, seed 0, 2026-09-09; **synthetic data, not a result about
SNNs**.

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

**All four references in this section are `[UNVERIFIED]`.** They were supplied in the
project brief and have not been checked against primary documents in this repository.
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
