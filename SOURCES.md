# Sources

Every numeric constant and factual claim in this repository carries its provenance, of
exactly one of three kinds:

- **`[LIT]`** — from published literature. Author, year, venue, and the specific value.
- **`[DERIVED]`** — follows from mathematics stated in this repository. A one-line
  derivation, not a citation.
- **`[MEASURED]`** — produced by a run in this repository. Script, seed, and date.

A fourth label exists for honesty: **`[UNVERIFIED]`** marks something believed to come
from the literature whose specific source has not been confirmed against the primary
document. Unverified items are listed under "To verify" below and **must not be cited in
the paper until checked**. Leaving an item unverified is correct; guessing an author,
year, venue, figure number or value is not.

---

## Cited works

### Verified against a primary or locally checkable source

**Orchard, Jayawant, Cohen & Thakor (2015). "Converting Static Image Datasets to Spiking
Neuromorphic Datasets Using Saccades." *Frontiers in Neuroscience*.**
DOI 10.3389/fnins.2015.00437
*Verification:* title and venue read directly from the installed
`spikingjelly/datasets/n_mnist.py` docstring, which links the DOI.
*Taken from it:* the N-MNIST dataset itself. 34x34 spatial resolution, 10 classes.
*Used in:* `spikecalib/datasets.py` (N-MNIST registry entry), `NOTES.md` (Scope).
*Note:* the 60,000 / 10,000 train/test counts are `[MEASURED]`, not from the paper — see
below.

**Amir, Taba, Berg, Melano, McKinstry, Di Nolfo, Nayak, Andreopoulos, Garreau, Mendoza,
Kusnitz, Debole, Esser, Delbruck, Flickner & Modha (2017). "A Low Power, Fully
Event-Based Gesture Recognition System." *CVPR 2017*.**
*Verification:* title and venue read from the installed
`spikingjelly/datasets/dvs128_gesture.py` docstring, which links the CVPR open-access
page. The author list above is `[UNVERIFIED]` beyond "Amir et al." — see To verify.
*Taken from it:* the DVS128 Gesture dataset; the figure of **1,342 instances of a set of
11 hand and arm gestures**, quoted verbatim in SpikingJelly's docstring.
*Used in:* `NOTES.md` ("DVS-Gesture — sample count"), `spikecalib/datasets.py`.

**Li, Liu, Ji, Li & Shi (2017). "CIFAR10-DVS: An Event-Stream Dataset for Object
Classification." *Frontiers in Neuroscience*.**
DOI 10.3389/fnins.2017.00309
*Verification:* title and venue read from the installed
`spikingjelly/datasets/cifar10_dvs.py` docstring, which links the DOI. The author list is
`[UNVERIFIED]` beyond "Li et al." — see To verify.
*Taken from it:* the CIFAR10-DVS dataset; 10,000 samples, 10 classes, 128x128.
*Used in:* `spikecalib/datasets.py`, `NOTES.md` (Scope, per-dataset hazards).

**Fang et al. SpikingJelly (software).** Version 0.0.0.0.14, PKU MLG / PCL and
contributors, https://github.com/fangwei123456/spikingjelly
*Verification:* `pip show spikingjelly` in this environment.
*Taken from it:* dataset loaders, `LIFNode`, the `ATan` surrogate implementation, the
train/test split convention for DVS-Gesture (1,176 / 288), and the MD5 checksums used to
verify N-MNIST archives.
*Used in:* throughout `spikecalib/` and `scripts/`.

### Believed but not yet confirmed against the primary document

These are cited in the repository today and are **flagged**; each also appears under
To verify.

**Horowitz (2014). "Computing's Energy Problem (and what we can do about it)."
*ISSCC 2014*.** `[LIT]` for the paper, `[UNVERIFIED]` for the values.
*Taken from it:* 45 nm energy costs, `E_AC = 0.9 pJ` and `E_MAC = 4.6 pJ`.
*Used in:* `NOTES.md` correctness rule 3, `spikecalib/models.py`, `spikecalib/figures.py`.
*Status:* the paper's existence and topic are not in doubt. What is **not confirmed** is
the specific figure or table number, the exact value 0.9 pJ (believed to be the 32-bit
floating-point ADD at 45 nm), and whether 4.6 pJ appears in the paper at all or is the
community convention of FP multiply + FP add. Do not cite a figure number until checked.

**Guo, Pleiss, Sun & Weinberger (2017). "On Calibration of Modern Neural Networks."
*ICML 2017*.** `[LIT]` for the paper, `[UNVERIFIED]` for the bin count.
*Taken from it:* equal-width-binned ECE; temperature scaling as a post-hoc calibrator;
the one-vs-rest-with-renormalisation isotonic formulation; the default of 15 bins.
*Used in:* `spikecalib/calibration.py` (`ece_equal_width`, `TemperatureScaling`,
`IsotonicOneVsRest`, `n_bins=15` default), `NOTES.md` correctness rule 2 and prior work.
*Status:* author list, year and venue are high-confidence but unconfirmed here; the
**`n_bins = 15` default is attributed to this paper and has not been checked**.

**Four prior-work citations carried over from the project brief.** **Checked 2026-09-12;
two did not survive.** See "Verified 2026-09-12" below for the evidence.

| as supplied in the brief | outcome |
|---|---|
| Li, Jones & Furber, "Dynamic Confidence" (ICCV 2023) | **confirmed**, with the title corrected — *Dynamic Confidence* is the method, not the paper |
| Sun & Bohte, "AOT-SNN" | **withdrawn** — the arXiv identifier attached to it belongs to Wu et al. and a different method |
| Skatchkovsky & Simeone on SNN overconfidence | **withdrawn** — no such paper identified; the overconfidence result is Chen, Park & Simeone's |
| "Knowing When to Stop" on conformal prediction for SNN early exit | **confirmed** — Chen, Park & Simeone, arXiv:2305.11322 |

Two of four supplied references being wrong is the reason this pass was run, and the reason
nothing in the brief is now treated as a source until it has been opened.

### Added 2026-09-11, against the study's final claims

**N-Caltech101 (Orchard, Jayawant, Cohen & Thakor).** The C axis. `[LIT]` for the dataset
itself; the copy used here is the Mendeley Data mirror, `doi:10.17632/cy6cvx3ryv.1`,
version 1. `Caltech101.zip` is 3,989,375,649 bytes with SHA-256
`c70d387facd67f38f04a6f0afc4b1181416f2610cb924afa1f16fed99f7ebb9f`, and its MD5 matches the
hash SpikingJelly publishes for the same file exactly, so the mirror is serving the archive
upstream expects. `[MEASURED]` 8,709 samples in 101 class directories, class sizes 31
(`inline_skate`) to 800 (`airplanes`), median 59 — counted from the extracted tree rather
than taken from a paper.

*Not verified:* the dataset homepage states 101 categories as "100 classes plus one
background class", with the duplicate `Faces_easy` excluded. The class *semantics* have not
been checked against the Caltech101 original, and nothing in this study depends on them —
only on the count, which was measured.

The claims changed during the work. What is claimed now is the attainable-confidence
ceiling, the cross-entropy floor it implies, and a **null result** on calibrator choice —
not the original isotonic hypothesis. These are the references a reviewer of *those*
claims will raise.

**Atarashi, Oyama, Arai & Kashima (2026). "Probability Bounding: Post-Hoc Calibration via
Box-Constrained Softmax." arXiv:2506.10572, February 2026.** `[UNVERIFIED]`
*Status:* taken from the abstract; **not read in full**. Must be read before the preprint
ships, because it is the closest prior work to the ceiling claim and the distinction below
depends on details the abstract may not settle.
*What it does:* post-hoc calibration that **learns** lower and upper bounds on output
probabilities through a box-constrained softmax.
*Distinction, which a reviewer who knows this paper will ask for:* their bounds are
**learned parameters fitted to the outputs of a general ANN** — a modelling choice, tunable,
and imposed to improve calibration. The ceiling in this study is **structural**: it is not
fitted, not adjustable, and not a design decision. It follows arithmetically from applying
a softmax to integer spike counts bounded by `T`, and the trained models saturate it to
within 4e-13. The two are near-opposites in kind — one is a bound put *onto* a model, the
other a bound the readout *already has*. Where they may genuinely meet is that a learned
box constraint could in principle be used to *correct* the structural ceiling, which is a
direction this study does not pursue.

**Wu, Qi, Cai, Jin, Yi & Huang. "Direct Training Needs Regularisation: Anytime Optimal
Inference Spiking Neural Network." arXiv:2405.00699, 15 April 2024.** `[LIT]` verified
against the arXiv record 2026-09-12.

> **Attribution corrected 2026-09-12.** This entry previously read "Sun & Bohte". It is not
> theirs. arXiv:2405.00699 is by **Dengyu Wu, Yi Qi, Kaiwen Cai, Gaojie Jin, Xinping Yi and
> Xiaowei Huang**, and the method is **AOI-SNN** (Anytime Optimal Inference), not AOT-SNN.
> The brief supplied the wrong author pair against this identifier and it was carried
> unchecked. A separate AOT-SNN line does exist and is Sun & Bohté's — see the next entry —
> so two different papers with similar acronyms had been merged into one citation.

*What it does:* introduces a Spatial-Temporal Regulariser that balances spike strength
against membrane potential at each timestep, producing an SNN whose accuracy degrades
gracefully under early cutoff; combined with a softmax-output cutoff it reports 2.14-2.89x
faster inference at 0.50-0.64% accuracy cost on event-based data.
*Distinction:* the object is a **stopping rule and a training regulariser**, not the
calibration of the confidence itself.
*Distinction:* the setting overlaps closely — directly trained, event data, softmax over
the output — but the object is a **stopping rule**, not the calibration of the confidence
itself. This study fixes the network and the timestep count and asks only whether the
confidence is correctly scaled. The overlap makes this the reference most likely to have
already observed the ceiling in passing, which is a reason to read it before claiming the
observation is new.

**Chen, Park & Simeone. "Agreeing to Stop: Reliable Latency-Adaptive Decision Making via
Ensembles of Spiking Neural Networks." *Entropy* 26(2):126, February 2024.
doi:10.3390/e26020126, arXiv:2310.16675.** `[LIT]` verified against the arXiv full text
2026-09-12: authors Jiechen Chen, Sangwoo Park, Osvaldo Simeone (King's College London).
*What it does:* uses the softmax over spike counts as a confidence signal for
latency-adaptive stopping via ensembles, and states verbatim that "SNN models, like their
conventional neural network counterpart, tend to be poorly calibrated, producing
**overconfident** decisions".
*Readout, verified:* spike counts `r_c` accumulated over the readout neuron, then
`f_c = e^{r_c} / sum_c' e^{r_c'}`. **This is exactly the readout this study analyses.**
*Regime, verified and decisive:* **`T = 80`**, with checkpoints at `{20, 40, 60, 80}`, on
MNIST-DVS, DVS128-Gesture and CIFAR-10, **directly trained with surrogate gradients**,
calibration set of 50 examples. `[DERIVED]` At `T = 80, C = 10` the attainable-confidence
ceiling is 1.0 to double precision, and even at their earliest checkpoint `T = 20` it is
0.99999998. **The cap is completely inert in their regime**, which is why they can observe
ordinary overconfidence while this study observes structural underconfidence at `T = 2`.
The two results do not conflict; they are measurements in non-overlapping regimes, and
until 2026-09-12 this study had asserted that without knowing their timestep count.
*Distinction, and a direct disagreement on the sign:* it uses the **same readout this study
uses** — softmax over spike counts — and reports the opposite direction of miscalibration.
See "The sign of the miscalibration" in `NOTES.md`. This study measures mean confidence
minus accuracy of **−0.539 at T = 2**, severe *under*confidence, and attributes it to the
attainable-confidence ceiling. The disagreement is most likely one of regime rather than of
fact, and is reported as such rather than as a correction.

---

## Derivations

Each follows from mathematics stated in the repository. No citation applies.

**Attainable-confidence ceiling.** `ceiling = e^T / (e^T + C - 1)`.
The most extreme reachable count vector is `[T, 0, ..., 0]`; its softmax puts `e^T` over
`e^T + (C-1)e^0`. For `C = 10`: 0.4509 at `T = 2`, 0.8585 at `T = 4`, 0.9782 at `T = 6`,
0.9970 at `T = 8`, 0.9996 at `T = 10`.
*Used in:* `NOTES.md` claim section, `scripts/synthetic_check.py`.

**Temperature scaling cannot change the prediction or separate a tie.**
Dividing every logit by a positive scalar is strictly order-preserving, so the argmax set
is invariant in `t`. *Used in:* `NOTES.md`, `spikecalib/calibration.py`, and asserted by
`test_temperature_scaling_cannot_change_the_prediction_or_break_a_tie`.

**Neither can isotonic top-label.** It computes `conf' = f(conf)` for monotone `f`, so
inputs that are equal remain equal. Therefore tie-breaking does not distinguish the two
calibrators. *Used in:* `NOTES.md` "What tie rate is, and what it is not".

**Temperature is not bounded by the ceiling.** `softmax(z/t) -> one-hot` as `t -> 0`, so
confidence approaches 1 regardless of `T`. *Used in:* `NOTES.md` claim section.

**`E_MAC` as multiply-plus-accumulate.** If `E_MAC = 4.6 pJ` is not stated directly by
Horowitz, it is the sum of a 32-bit FP multiply (believed 3.7 pJ) and a 32-bit FP add
(believed 0.9 pJ). This decomposition is `[DERIVED]` *conditional on* two `[UNVERIFIED]`
inputs, so it inherits their status.

**DVS-Gesture count gap.** `1,464 - 1,342 = 122`, which equals the number of recording
trials (98 train + 24 test). One extra labelled segment per trial on average.
*Used in:* `NOTES.md` "DVS-Gesture — sample count".

**Per-class availability.** N-MNIST: `60,000 / 10 = 6,000` per class, so `500 + 500`
holds out one sixth. DVS-Gesture: `1,176 / 11 ~ 107` per class, so 500 per class is
impossible and `10 + 30` leaves ~67 to train on. *Used in:* `spikecalib/datasets.py`.

**Equal-width bin geometry.** With `n_bins = 15` the bin width is `1/15 = 0.0667`, so
0.81 and 0.85 both fall in `[0.8, 0.8667)` while 0.80 and 0.87 do not.
The two-value example then gives equal-width ECE `|0.5 - 0.83| = 0.33` and equal-mass ECE
`0.5|1.00 - 0.81| + 0.5|0.00 - 0.85| = 0.52`.
*Used in:* `tests/test_calibration.py`.

**LIF check trajectory.** With `tau = 2.0`, `v_threshold = 1.0`, `v_reset = 0` and a
constant drive of 1.5, the membrane potential follows `0.75 -> 1.125` (fire, reset to 0)
`-> 0.75`. Step 1 must be silent and step 2 must fire, which is what makes the check
sensitive to a comparator broken in either direction. *Used in:* `scripts/check_env.py`.

**Ceiling and loss floor at 101 classes.** Same arithmetic, second argument varied.
`e^2/(e^2 + 100) = 0.068806` against `e^2/(e^2 + 9) = 0.450853`, and the cross-entropy floor
`-ln(ceiling)` is `2.6765` nats against `0.7966`. At `T = 6`, 0.801362 and 0.2214 nats; at
`T = 10`, 0.995481 and 0.0045 nats. *Used in:* `NOTES.md` "The C axis",
`scripts/caltech_experiment.py`, and asserted by
`test_ceiling_at_101_classes_is_far_more_binding_than_at_10`.

**The conditional ceiling.** `ceiling_cond = e^T / (e^T + sum_{j != winner} e^{c_j})`, the
largest confidence a sample could express if its winner fired at every step while its
competitors stayed where they were. Setting every `c_j = 0` gives
`e^T / (e^T + (C - 1))`, so the theoretical ceiling is its all-silent special case and not a
separate quantity; this identity is asserted by
`test_all_competitors_silent_recovers_the_theoretical_ceiling_exactly`. **It is a
descriptive bound, not a derived one:** it is computed from a trained model's own outputs,
so it explains what a particular model could have done and predicts nothing about what any
model must do. *Used in:* `NOTES.md` "Pre-registered prediction 4" and its outcome,
`spikecalib/conditional.py`.

**Why the ceiling is saturated at some (T, C) and not others.** Reaching the ceiling needs a
test sample whose count vector is exactly `[T, 0, ..., 0]`: one class firing at every step
and **every one of the other `C - 1` classes silent at every step**. That is easy at `C = 10`
and was achieved in all 15 N-MNIST runs; at `C = 101, T = 10` it was not, so the observed
maximum sits 1.5e-3 below the bound. The bound is an upper bound, so this is slackness, not
violation, and the two must not be reported as the same thing. *Used in:* `NOTES.md`
"The ceiling holds, and the distinction between valid and saturated matters".

**Leaky-integrator weights and the effective evidence window.** The non-spiking readout
uses `v_t = v_(t-1) + (z_t - v_(t-1))/tau`, which rearranges to
`v_t = (1 - 1/tau) v_(t-1) + (1/tau) z_t`. Unrolling it, the input `k` steps before the end
enters the final potential with weight `(1/tau)(1 - 1/tau)^k`. At `tau = 2` that is
`0.5 x 0.5^k`: per-step retention 0.5, half-life exactly 1 step, exponential time constant
`-1/ln(0.5) = 1.443` steps. The share of the final potential contributed by the last two
steps is `(w_0 + w_1) / sum_k w_k` = 100.0% at `T = 2`, 76.2% at `T = 6`, 75.1% at `T = 10`,
against `2/T` for the count readout, which weights every step equally. The `accumulator`
readout sets the leak to zero, `v_t = v_(t-1) + z_t`, giving every step weight 1.
*Used in:* `NOTES.md` "Pre-registered prediction 3", `spikecalib/models.py`
`potential_trace`, and asserted by
`test_membrane_leaks_with_a_one_step_half_life_and_the_accumulator_does_not`.

**Tie-break sampling tolerance.** 4,000 exact two-way ties resolved uniformly gives a
binomial share with standard deviation `sqrt(0.25/4000) = 0.0079`, so the `0.45 < p <
0.55` window in the test is over six standard deviations wide and cannot fail by chance.
*Used in:* `tests/test_calibration.py`.

---

## Measured in this repository

All produced by scripts here; none is a claim about published results. Synthetic figures
are marked as such and say nothing about spiking networks.

| Value | Source |
|---|---|
| torch 2.14.0, spikingjelly 0.0.0.0.14, scikit-learn 1.9.0, Python 3.13.1 | `scripts/check_env.py`, 2026-09-09 |
| LIF spikes per step `[0, 8192, 0]`, final `v = 0.7500` | `scripts/check_env.py`, 2026-09-09, MPS |
| N-MNIST 60,000 train / 10,000 test, frame shape `(10, 2, 34, 34)` | `scripts/fetch_data.py --dataset nmnist --timesteps 10`, 2026-09-09 |
| N-MNIST archive MD5s `20959b8e…` / `69ca8762…` | read from `NMNIST.resource_url_md5()` at runtime; never hardcoded |
| Mirror archive 1,181,573,087 bytes | HTTP `HEAD` on the Dropbox mirror, 2026-09-09 |
| Disk: 3.54 GiB one-time, 224.7 MiB per T | `scripts/fetch_data.py`, 2026-09-09 |
| Build 176.8 s; 8 macOS metadata entries recreated mid-build | `scripts/fetch_data.py`, 2026-09-09 |
| Split fingerprint `e524beccace851af`; train imbalance 4,421–5,742, ratio 1.299 | `scripts/check_splits.py --seed 0`, 2026-09-09 |
| All synthetic tie rates, ECEs, advantages, samples-per-level, isotonic block counts | `scripts/synthetic_check.py`, seed 0, 2026-09-09. **Synthetic data** |
| Isotonic realised blocks 4–22 while `n_distinct` spanned 34–4,197 | `scripts/synthetic_check.py`, seed 0, 2026-09-09. **Synthetic data** |

---

## Design choices that are not sourced at all

Listed so they are not mistaken for findings. These are decisions, and the justification
is engineering judgement recorded in `LOG.md`, not a citation.

- Split sizes: N-MNIST 500/500 per class, DVS-Gesture 10/30, CIFAR10-DVS 100/100.
  `[DERIVED]` from dataset size, but the specific numbers are a choice.
- Synthetic generator parameters: `p_hi = 0.75`, `p_lo = 0.15`, `noise = 0.20`,
  `noise_lo = 0.02`, `noise_hi = 0.45`. Chosen to produce visible miscalibration; not
  tuned and not meaningful outside the synthetic check.
- Model depth per dataset: `(32, 64)` for N-MNIST, `(32, 64, 128, 128)` for N-Caltech101.
  Neither is tuned. **These two differ in depth and in parameter count in opposite
  directions** -- more pooling shrinks the flatten, so the four-block model has roughly a
  sixth the parameters. Every result reported at both is an alternative-architecture check,
  never a depth experiment. `[MEASURED]` parameter counts: N-MNIST 1,070,868 / 375,060;
  N-Caltech101 2,749,130 / 464,074.
- LIF `tau = 2.0`, `v_threshold = 1.0`. Conventional defaults, not tuned. **`tau` became
  load-bearing on 2026-09-11**: it sets the membrane readout's effective evidence window,
  which the leak experiment shows accounts for about half the high-`T` ranking deficit. A
  different `tau` would move that window and probably the crossover. Nothing in this
  repository establishes `tau = 2` as a sensible default; the experiments only measure what
  it does. See To verify.
- Adam learning rate `1e-3`. The Adam default. **Not tuned for this model or dataset**;
  see To verify.
- Bootstrap resamples: 1,000. Conventional; no source.

---

## Verified 2026-09-12 against primary documents

A verification pass was run over every `[UNVERIFIED]` item because these are load-bearing in
a preprint. What follows is what could and could not be confirmed. **Nothing was invented to
close a gap.**

### Confirmed from the primary document

**1. Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014,
Session 1 Plenary 1.1, Digest of Technical Papers pp. 10-14.** The energy table is
**Figure 1.1.9**, captioned *"Rough energy costs for various operations in 45nm 0.9V"*
(Digest p. 14). Read directly from the paper:

| | 8/16 bit | 32 bit |
|---|---|---|
| Integer Add | 0.03 pJ | **0.1 pJ** |
| Integer Mult | 0.2 pJ | **3.1 pJ** |
| FP FAdd | 0.4 pJ (16 bit) | **0.9 pJ** |
| FP FMult | 1.1 pJ (16 bit) | **3.7 pJ** |
| Cache (64 bit) | 8KB 10 pJ, 32KB 20 pJ, 1MB 100 pJ | |
| DRAM | 1.3-2.6 nJ | |

- **`E_AC = 0.9 pJ` is CONFIRMED** as the 45 nm, 0.9 V, **32-bit floating-point add**. Note
  the coincidence that the supply voltage is also 0.9: the caption's "0.9V" is not the
  energy figure.
- **`E_MAC = 4.6 pJ` is NOT stated anywhere in the paper.** Figure 1.1.9 has no MAC row and
  the text gives no combined value. 4.6 = 0.9 + 3.7 is the **community convention** of
  multiply-plus-add, exactly as this repository suspected. It must be described that way and
  never attributed to Horowitz as a measurement.
- The caption says **"Rough"**. The source presents these as order-of-magnitude figures, so
  no result here may turn on their precision.

**Consequence for the paper.** The absolute energy numbers may be reported, with `E_AC`
cited to Figure 1.1.9 and `E_MAC` described as the multiply-plus-add convention rather than
a citation. The safer alternative the task offered — report ratios only — is not necessary
now that the add value is confirmed, but the proxy must keep its existing framing as a
design-comparison quantity and never as measured power.

**2. Atarashi, Oyama, Arai & Kashima. "Probability Bounding: Post-Hoc Calibration via
Box-Constrained Softmax." arXiv:2506.10572, 12 June 2025 (v2, 23 February 2026).**
Title, four authors and date confirmed. Read for the distinction rather than taken from the
abstract:

- It introduces **BCSoftmax**, a generalisation of softmax enforcing lower and upper bounds
  on output probabilities, and fits those bounds **by empirical risk minimisation on a
  validation set**. The bounds are *learned parameters*.
- **It never considers a model whose probabilities are bounded structurally.** Every
  experiment uses a standard unbounded softmax over ResNet-50, DenseNet-121, ViT-B/16 or a
  GloVe-embedding CNN, on TinyImageNet, CIFAR-100, ImageNet-1K and 20NewsGroups.
- **The precise distinction.** Atarashi et al. *impose* a box constraint as a post-hoc remedy
  for miscalibration. This study *finds* a box constraint the readout already imposes, before
  any calibrator is fitted, and shows it *causes* miscalibration. The two are inverses in
  direction: one adds a bound to fix calibration, the other identifies a bound that breaks
  it. They meet only in the untried direction of using a learned bound to correct a
  structural one, which this study does not pursue.

**3a. Li, Jones & Furber.** Title confirmed as **"Unleashing the Potential of Spiking Neural
Networks by Dynamic Confidence"** — *Dynamic Confidence is the method, not the title*, and
this repository had been using the method name as the title. Chen Li, Edward Jones, Steve
Furber; **ICCV 2023, pp. 13350-13360**; arXiv:2303.10276, 17 March 2023. (The ICCV
open-access listing renders the title with "with" in place of "by"; the arXiv record uses
"by".)
*Regime, verified:* **ANN-to-SNN converted**, not directly trained — the paper places itself
in the conversion category explicitly. Static image data: CIFAR-10 at about 4 timesteps
falling to ~2.52 with the method, ImageNet at 14 falling to 4.71. It applies **temperature
scaling** as a post-training calibration step.
*Note for the paper, not a claim:* this operates at `T` of roughly 2.5-14 with `C = 10` on
CIFAR-10, which is inside this study's regime, and reports 94.11% accuracy at 4 timesteps
against a `C = 10, T = 4` ceiling of 0.8585. **If** its readout is a softmax over
accumulated spike counts, the ceiling would bind there too. The readout description
available to this verification was ambiguous between accumulated spikes and accumulated
membrane potential, so this is recorded as a **checkable question**, not a finding.

**3b. Chen, Park & Simeone. "Knowing When to Stop: Delay-Adaptive Spiking Neural Network
Classifiers with Reliability Guarantees."** arXiv:2305.11322, 18 May 2023 (v4, 29 June
2024). Authors **Jiechen Chen, Sangwoo Park, Osvaldo Simeone** confirmed. Introduces
**SpikeCP**, wrapping a pre-trained SNN with **conformal prediction** to give guaranteed
reliability at input-dependent stopping times. The repository's description of it was
correct.

**3c. Chen, Park & Simeone. "Agreeing to Stop."** *Entropy* 26(2):126, February 2024. See
the Cited works entry above: the **readout and the overconfidence statement are confirmed**,
and the regime is **`T = 80`**, which settles the apparent disagreement with this study.

### Could NOT be confirmed — and what follows

**4. "Sun & Bohte, AOT-SNN, arXiv:2405.00699" is a wrong citation and has been removed.**
Verified: arXiv:2405.00699 is **Wu, Qi, Cai, Jin, Yi & Huang**, "Direct Training Needs
Regularisation: Anytime Optimal Inference Spiking Neural Network" — method **AOI-SNN**. A
genuine **AOT-SNN** line exists and is Tao Sun & Sander Bohté's; the one reference confirmed
is **Sun & Bohté, "Average-Over-Time Spiking Neural Networks for Uncertainty Estimation in
Regression", arXiv:2412.00278, 29 November 2024**, which is a **regression** paper and
therefore probably not what the brief intended. A related earlier paper, **Sun, Yin &
Bohté, "Efficient Uncertainty Estimation in Spiking Neural Networks via MC-dropout",
arXiv:2304.10191, 20 April 2023**, is confirmed to exist but was not read.
**Action taken:** the Wu et al. paper is cited correctly under its real authors. The claim
that "AOT-SNN uses temperature scaling as its calibration baseline" is **withdrawn** — it
was attached to the wrong paper and the right paper has not been read.

**5. "Skatchkovsky & Simeone on SNN overconfidence" could not be confirmed and has been
withdrawn.** The overconfidence claim this study compares itself against is in **Chen, Park
& Simeone**, not Skatchkovsky. Nicolas Skatchkovsky has co-authored SNN papers with Simeone,
but **no specific paper of theirs on SNN overconfidence was identified**, and none is cited.
Rather than guess at which was meant, the reference is removed.

**6. The Loihi and SpiNNaker hardware claims could not be confirmed from primary
documentation.** Intel's NxSDK API reference is not publicly reachable, and the sPyNNaker
limitations page does not discuss recording bandwidth. What could be established, from
secondary but peer-reviewed sources:
- Loihi's neuron model is a CUBA leaky integrate-and-fire with membrane potential as an
  internal state variable, and Rueckauer et al., "NxTF: An API and Compiler for Deep Spiking
  Neural Networks on Intel Loihi" (arXiv:2101.04261) states that during the management phase
  the embedded CPUs may read out the current SNN state. That is **consistent with** membrane
  state being readable, and says nothing about the cost of doing so per timestep.
- sPyNNaker exposes membrane potential through PyNN's `record('v')`, has a 255-neuron
  per-core limit, and its host transfers are bounded by on-board Ethernet bandwidth. Whether
  that limit binds for an output layer of ten to a thousand neurons — the only thing a
  membrane readout needs — is **not established**.
**Action taken:** the hardware argument in `NOTES.md` "Why the count readout exists" is
softened to say what is actually known, and both claims stay `[UNVERIFIED]`.

---

## Still to verify

1. **Guo et al. ICML 2017 — full author list and the `n_bins = 15` default.** Not attempted
   in this pass.
2. **Amir et al. CVPR 2017 — full author list** for DVS-Gesture. Only "Amir et al." and the
   title are confirmed, via SpikingJelly's docstring.
3. **Li et al. 2017 — full author list** for CIFAR10-DVS. Same situation.
4. **ATan surrogate gradient — originating paper.** SpikingJelly documents the function but
   attaches no citation. The general method is usually credited to Neftci, Mostafa & Zenke;
   the specific arctan form is unattributed here.
5. **DVS-Gesture 1,176 / 288 split counts.** From SpikingJelly's docstring; not built.
6. **CIFAR10-DVS 10,000 sample count.** Same.
7. **Whether "Dynamic Confidence" uses a spike-count or membrane readout**, which decides
   whether the attainable-confidence ceiling binds in its `T = 4`, `C = 10` setting. The most
   consequential remaining question, because a positive answer would mean the ceiling already
   constrains published work.
8. **The AOT-SNN classification paper**, if one exists, and whether temperature scaling is
   its calibration baseline.
9. **Loihi compartment-state access cost** and **SpiNNaker recording bandwidth for a small
   output layer** — see item 6 above.
10. **Adam `lr = 1e-3`** — an unjustified constant, never tuned.
11. **LIF `tau = 2.0` as a readout parameter.** Adopted as a neuron default, but the leak
    experiment shows it sets the membrane readout's evidence window and therefore drives a
    reported result. No sweep has been run. Every statement about "a membrane readout" is a
    statement about `tau = 2`.
12. **N-Caltech101 class semantics** — the 101 = 100 + background composition is from the
    dataset homepage; only the counts were measured here.
