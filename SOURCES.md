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

**Four prior-work citations carried over from the project brief.** All `[UNVERIFIED]`.
Li, Jones & Furber "Dynamic Confidence" (ICCV 2023); Sun & Bohte "AOT-SNN";
Skatchkovsky & Simeone on SNN overconfidence; "Knowing When to Stop" on conformal
prediction for SNN early exit. These appear in `NOTES.md` "Prior work" exactly as
supplied. None has been checked against a primary source in this repository.

### Added 2026-09-11, against the study's final claims

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

**Sun & Bohte. "Direct Training Needs Regularisation: Anytime Optimal Inference Spiking
Neural Network." arXiv:2405.00699.** `[UNVERIFIED]`
*What it does:* applies a cutoff on the softmax output over event-based datasets, for
anytime inference in directly-trained SNNs.
*Distinction:* the setting overlaps closely — directly trained, event data, softmax over
the output — but the object is a **stopping rule**, not the calibration of the confidence
itself. This study fixes the network and the timestep count and asks only whether the
confidence is correctly scaled. The overlap makes this the reference most likely to have
already observed the ceiling in passing, which is a reason to read it before claiming the
observation is new.

**"Agreeing to Stop: Reliable Latency-Adaptive Decision Making via Ensembles of Spiking
Neural Networks" (Simeone group).** `[UNVERIFIED]`
*What it does:* uses the softmax over spike counts as a confidence signal for
latency-adaptive stopping, and states that SNNs are poorly calibrated and **overconfident**.
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

## To verify

Nothing in this list may be cited in the paper until it has been checked against the
primary document.

1. **Horowitz ISSCC 2014 — figure/table number and exact values.** Confirm `E_AC = 0.9 pJ`
   is the 45 nm 32-bit floating-point ADD, confirm the 32-bit FP MULT value (believed
   3.7 pJ), and establish whether `E_MAC = 4.6 pJ` is stated in the paper or is the
   community convention MULT + ADD. Until then, `spikecalib/models.py` marks both
   constants `[UNVERIFIED]`.
2. **Guo et al. ICML 2017 — full author list, and the `n_bins = 15` default.** Confirm the
   paper uses 15 equal-width bins for ECE, and that the one-vs-rest isotonic formulation
   is reported there as this repository claims.
3. **Li, Jones & Furber, "Dynamic Confidence", ICCV 2023.** Supplied in the project brief;
   not checked. Confirm authors, title, venue, year, and that it concerns runtime
   confidence for early exit in ANN-to-SNN-converted networks on static data.
4. **Sun & Bohte, "AOT-SNN".** Supplied in the brief; not checked. Confirm full reference
   and that temperature scaling is its calibration baseline.
5. **Skatchkovsky & Simeone on SNN overconfidence.** Supplied in the brief; not checked.
   Confirm which specific paper is meant — these authors have several on SNNs.
6. **"Knowing When to Stop."** Supplied in the brief; not checked. Confirm authors, venue,
   year, and that it applies conformal prediction to SNN early exit.
7. **Amir et al. CVPR 2017 — full author list.** Only "Amir et al." and the title are
   confirmed, via SpikingJelly's docstring.
8. **Li et al. 2017 — full author list for CIFAR10-DVS.** Same situation.
9. **ATan surrogate gradient — originating paper.** SpikingJelly documents the function
   `g(x) = (1/pi) arctan((pi/2) alpha x) + 1/2` but attaches no citation in source. The
   general surrogate-gradient method is usually credited to Neftci, Mostafa & Zenke, but
   that attribution is **not confirmed here** and the specific arctan form may be due to
   someone else.
10. **DVS-Gesture 1,176 / 288 split counts.** Read from SpikingJelly's docstring, not yet
    reproduced by building the dataset in this repository. Verify on first build.
11. **CIFAR10-DVS 10,000 sample count.** Same: from SpikingJelly's docstring, not built.
12. **Atarashi et al., arXiv:2506.10572.** Read in full. Confirm the authors, the date,
    and specifically whether the learned box constraint is ever applied to a readout with a
    structural bound. The distinction drawn above rests on the abstract alone.
13. **Sun & Bohte, arXiv:2405.00699.** Read in full. Check whether the attainable-confidence
    ceiling, or any equivalent bound on softmax-over-spike-counts, is already noted there.
    If it is, this study's first claim is a re-derivation and must say so.
14. **"Agreeing to Stop" (Simeone group).** Confirm authors, venue and year. Establish the
    regime in which overconfidence was measured — timestep count, readout, whether directly
    trained or converted — because the comparison with this study's underconfidence result
    is only meaningful once those are known.
15. **Intel Loihi compartment-state access.** Confirm from Intel's documentation whether
    membrane voltage is readable at inference time, whether that requires a host round-trip,
    and what it costs. `NOTES.md` "Why the count readout exists" relies on the belief that
    it is probe-only and that inter-core traffic is spike-only.
16. **SpiNNaker membrane recording bandwidth.** Confirm from the SpiNNaker/PyNN
    documentation whether recording membrane potential is bandwidth-limited, and critically
    whether that limit binds for an output layer of ten to a thousand neurons — which is all
    a membrane readout needs — or only for whole-network recording.
17. **Adam `lr = 1e-3` for this model.** Not a citation issue but an unjustified constant:
    no tuning has been done, and no evidence exists that it is appropriate for a
    surrogate-gradient SNN at low `T`.
18. **LIF `tau = 2.0` as a readout parameter.** Also not a citation issue. It was adopted as
    a conventional neuron default, but the leak experiment of 2026-09-11 shows it sets the
    membrane readout's evidence window — half-life one step — and therefore drives a result
    the study now reports. No sweep over `tau` has been run. Until one is, every statement
    about the membrane readout is a statement about `tau = 2` specifically, and the paper
    must say so rather than describing "a membrane readout" in general.
