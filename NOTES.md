# SpikeCalib — Research Design

Post-hoc confidence calibration for event-driven spiking neural networks.

This document is the authoritative design record for the study. It states what is
being claimed, what would falsify it, which datasets are in scope, how the work is
positioned against prior art, and which rules must hold for the reported numbers to
mean anything. Decisions recorded here are settled; where a decision is still open it
is marked as such. Code in this repository is expected to follow this document, and
where the two disagree, this document is what needs updating first.

---

## Claim under test

Temperature scaling assumes a continuous logit space. It takes a real-valued logit
vector, divides it by a single learned scalar, and relies on the softmax being able to
move smoothly in response. In an event-driven spiking neural network read out by spike
counts, that assumption does not hold. Each class output is an integer in `[0, T]`,
where `T` is the number of simulation timesteps. The logit vector is a point on an
integer lattice, not a point in a continuous space.

Two consequences follow, and both sharpen as `T` falls:

1. **The attainable confidence set is small and coarse.** With `C` classes and counts
   bounded by `T`, only finitely many softmax outputs are reachable. At `T = 2` that set
   is small enough to enumerate. Temperature scaling can only slide confidence along the
   curve traced by dividing one of those finitely many integer vectors by a scalar; it
   cannot reach values between the rungs.

   The set is also *bounded*, which turns out to matter more than its coarseness. The
   most extreme count vector puts `T` on one class and `0` on the rest, so the largest
   confidence any spike-count readout can emit is `e^T / (e^T + C - 1)`. For `C = 10`
   that ceiling is **0.451 at `T = 2`**, 0.859 at `T = 4`, and only reaches 0.978 by
   `T = 6`. A model whose accuracy exceeds 0.451 at `T = 2` is therefore *forced* into
   underconfidence: there is no reachable output that would be correctly confident, and no
   calibrator can be blamed for the shortfall. So the direction of miscalibration is
   expected to flip across the sweep — structurally underconfident at low `T`, plausibly
   overconfident at high `T` — and the study must report the signed gap rather than only
   `|ECE|`, or the two regimes will average each other away.
2. **Ties cannot be broken by a scalar.** If two classes finish with the same spike
   count, dividing both by the same temperature leaves them equal. No choice of
   temperature separates them, because temperature is monotone in the logit and applies
   identically to every class.

Isotonic regression does not carry the assumption that fails here. It is
non-parametric and only requires monotonicity, so it fits a step function over the
observed confidence values. Discreteness is native to that estimator rather than a
violated precondition: a coarse, lumpy input distribution is exactly the kind of input a
step function represents well.

## Hypothesis

**The isotonic advantage over temperature scaling grows as `T` falls.** If that holds,
isotonic regression is the correct calibration method precisely in the low-timestep
regime — which is the regime where SNNs are energy-competitive in the first place, and
therefore the regime anyone deploying one actually cares about. The argument would then
be not merely that isotonic calibrates better on average, but that the gap is largest
where it matters most.

The hypothesis is falsifiable and the falsification is cheap to observe. If the
advantage is flat in `T` — isotonic wins by roughly the same margin at `T = 10` as at
`T = 2`, or wins nowhere — then the discreteness mechanism is not what is driving the
difference, the hypothesis is false, and **the study reports that**. A flat trend is a
publishable result about a plausible mechanism that turns out not to matter; it is not a
reason to reframe the question after the fact.

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

2. **Two ECE estimators, both reported.** Equal-width binning (Guo et al.) and
   equal-mass binning are both computed and both reported. With discrete spike counts
   the confidence distribution is lumpy: equal-width binning leaves bins empty and
   concentrates mass in a few others, which biases the estimate **optimistically** at low
   `T` — precisely the regime the hypothesis is about. Reporting only equal-width would
   flatter every method in the regime where the comparison matters. **Disagreement
   between the two estimators is itself a finding** and is reported when it occurs,
   rather than being resolved by picking the more convenient one.

3. **Energy is a proxy, stated as such.** Energy is reported as the SynOps proxy using
   the Horowitz (ISSCC 2014) 45 nm figures: `E_AC = 0.9 pJ` for an accumulate and
   `E_MAC = 4.6 pJ` for a multiply-accumulate. Every appearance of an energy number in
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

The tie rate is a **reported quantity, not a diagnostic**. The hypothesis predicts that
ties become more frequent as `T` falls, because fewer available count values means more
collisions. Observing that rise is direct evidence for the discreteness mechanism the
paper argues for — it is the mechanism made visible, independent of any ECE number.
The tie rate therefore appears **in the results table**, next to the calibration
metrics.

## Per-dataset hazards

Each of these is a known property of the dataset that is designed around here, in
advance, rather than discovered mid-analysis.

**DVS-Gesture — sample count.** Two different totals for this dataset circulate, and the
study needs to commit to one.

Amir et al. (CVPR 2017) describe "1,342 instances of a set of 11 hand and arm gestures".
SpikingJelly produces **1,464**, as 1,176 train and 288 test. The gap is 122, which is
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
