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
*Authors in full, for citation:* **Garrick Orchard, Ajinkya Jayawant, Gregory K. Cohen and
Nitish Thakor**; *Frontiers in Neuroscience* volume 9, published 16 November 2015 — from the
Crossref record for the DOI (https://api.crossref.org/works/10.3389/fnins.2015.00437), whose
title matches. Used in `paper/refs.bib`.
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

**Fang, Chen, Ding, Yu, Masquelier, Chen, Huang, Zhou, Li & Tian (2023). "SpikingJelly: An
open-source machine learning infrastructure platform for spike-based intelligence." *Science
Advances* 9(40), eadi1480.** DOI 10.1126/sciadv.adi1480. `[LIT]` verified 2026-09-28 against
the open-access full text, PubMed Central PMC10558124 (PMID 37801497): title, the ten authors in
this order, volume 9, issue 40, article eadi1480 and publication date 6 October 2023 read from
the article record, and the abstract states "We present the SpikingJelly framework". The
Crossref record for the DOI agrees on every field, and the project's own README asks users to
cite this article. The article states no SpikingJelly version and does not name the ATan
surrogate, so neither is taken from it: the version and the surrogate formula come from the
installed software, below. *Used in:* `paper/refs.bib` (`fang2023spikingjelly`), the Networks
paragraph of `paper/main.tex`.

**SpikingJelly (software).** Version 0.0.0.0.14, PKU MLG / PCL and
contributors, https://github.com/fangwei123456/spikingjelly
*Verification:* `pip show spikingjelly` in this environment.
*Taken from it:* dataset loaders, `LIFNode`, the `ATan` surrogate implementation, the
train/test split convention for DVS-Gesture (1,176 / 288), and the MD5 checksums used to
verify N-MNIST archives.
*Used in:* throughout `spikecalib/` and `scripts/`.
*Year and author, for citation:* PyPI records the 0.0.0.0.14 upload on **20 March 2023**
and lists the author as **"PKU MLG, PCL, and other contributors"**
(https://pypi.org/pypi/spikingjelly/0.0.0.0.14/json). The paper states the version in its
text and cites the journal article above, which replaced a software entry in `paper/refs.bib`
on 2026-09-28.

**Guo, Pleiss, Sun & Weinberger (2017). "On Calibration of Modern Neural Networks."
*ICML 2017*, PMLR 70:1321-1330.** `[LIT]` verified 2026-09-12 against the proceedings PDF
and the PMLR record, https://proceedings.mlr.press/v70/guo17a.html: authors **Chuan Guo,
Geoff Pleiss, Yu Sun and Kilian Q. Weinberger**, published 17 July 2017.
*What it says, and what it does not:*
- **Temperature scaling is theirs.** "Temperature scaling, the simplest extension of Platt
  scaling, uses a single scalar parameter T > 0 for all classes", and they write that they
  are "not aware of any prior use" of it for calibration.
- **Equal-width ECE is described, not introduced.** ECE "approximates (2) by partitioning
  predictions into M equally-spaced bins", and the paper attributes Expected Calibration
  Error to **Naeini et al. (2015)**. Crediting Guo et al. for the estimator would be wrong;
  crediting them for describing the equal-width form used here is accurate.
- **15 bins is their reported setting, not a stated default.** Table 1 and the results text
  give ECE "with M = 15 bins". Nothing in the paper recommends 15 as a default, so it is
  described here as following their setting rather than as their default.
- **Isotonic regression and the one-versus-all extension are attributed to Zadrozny & Elkan
  (2002).** Guo et al. describe that extension — K one-versus-all calibrators, confidence
  the maximum of the vector normalised by its sum — and apply it to isotonic regression.
- **Equal-mass binning does not appear in the paper.**
*Used in:* `spikecalib/calibration.py` (`ece_equal_width`, `TemperatureScaling`,
`IsotonicOneVsRest`), `NOTES.md` correctness rule 2, `paper/main.tex`.
*Not read:* Naeini et al. (2015) and Zadrozny & Elkan (2002). Neither is cited.

### Prior-work citations from the project brief, checked 2026-09-12

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
version 1. The DOI record at DataCite gives the creator as **Garrick Orchard**, the publisher
as Mendeley and the issue date as **12 January 2022**
(https://api.datacite.org/dois/10.17632/cy6cvx3ryv.1). *(Corrected 2026-11-12: `paper/refs.bib` now
follows the repository's own citation, which lists four authors; see "References verified
2026-11-12" below.)* `Caltech101.zip` is 3,989,375,649 bytes with SHA-256
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

**Wu, Qi, Cai, Jin, Yi & Huang. "Direct Training Needs Regularisation: Anytime Optimal
Inference Spiking Neural Network." arXiv:2405.00699, 15 April 2024.** `[LIT]` verified
against the arXiv record 2026-09-12.

> **Attribution corrected 2026-09-12.** This entry previously read "Sun & Bohte". It is not
> theirs. arXiv:2405.00699 is by **Dengyu Wu, Yi Qi, Kaiwen Cai, Gaojie Jin, Xinping Yi and
> Xiaowei Huang**, and the method is **AOI-SNN** (Anytime Optimal Inference), not AOT-SNN.
> The brief supplied the wrong author pair against this identifier and it was carried
> unchecked. A separate AOT-SNN line does exist and is Sun & Bohté's — see "Could NOT be
> confirmed", item 4 —
> so two different papers with similar acronyms had been merged into one citation.

*What it does:* introduces a Spatial-Temporal Regulariser that balances spike strength
against membrane potential at each timestep, producing an SNN whose accuracy degrades
gracefully under early cutoff; combined with a softmax-output cutoff it reports 2.14-2.89x
faster inference at 0.50-0.64% accuracy cost on event-based data.
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
attainable-confidence ceiling. At their `T` the cap cannot bind, so the disagreement is one of regime rather than of
fact.

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

**`E_MAC` as multiply-plus-accumulate.** `E_MAC = 4.6 pJ` is not stated by Horowitz. It
is the sum of the 32-bit floating-point multiply, 3.7 pJ, and the 32-bit floating-point add,
0.9 pJ, both read from Figure 1.1.9 (see "Verified 2026-09-12", item 1). The inputs are
`[LIT]`; the sum is `[DERIVED]`, and the convention of adding them is not a citation.

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

**The cap for logits bounded in an interval (added 2026-10-13).** If every logit lies in an
interval of width `W`, the largest softmax probability is `e^W / (e^W + C - 1)`: the winner's
probability is `1 / (1 + sum_{j != w} e^(z_j - z_w))` and each of the `C - 1` exponents is at
least `-W`. The count readout's ceiling is the case `W = T`; a scaled count readout
`z = s c` has `W = sT`; a rate readout `z = c/T` has `W = 1` and so a cap of
`e / (e + C - 1)` that contains no `T`, 0.2320 at ten classes and 0.0265 at a hundred and
one. The loss floor is `ln(1 + (C - 1) e^(-W))`. *Used in:* `spikecalib/theory.py`
(`bounded_logit_ceiling`, `scaled_ceiling`, `rate_ceiling`, `loss_floor`), `NOTES.md` "The
cap is a scale problem". Asserted by enumeration of every count vector and by random search
over intervals in `tests/test_theory.py`. **Whether this bound is already in print for
non-spiking networks is recorded under "Literature check, 2026-10-13" below; the derivation
here is independent of that and is not a claim of priority.**

**The minimum scale (added 2026-10-13).** `s_min = ln((C - 1) a / (1 - a)) / T`, from
requiring the cap at width `sT` to reach `a`. `T_min` is the same number read at `s = 1`, so
the constraint is on the product `sT`. *Used in:* `spikecalib/theory.py` (`min_logit_width`,
`min_scale`), and asserted against `spikecalib.models.min_timesteps`.

**The number of distinct top-label confidences of a count readout (added 2026-10-13).** At
most `binomial(T + C - 1, C - 1)` for every scale `s > 0`: the confidence depends only on
the multiset of the `C - 1` gaps between the winning count and each other count, each an
integer in `[0, T]`. 55 at `T = 2, C = 10`. Counting by winning count and the multiset of
the other counts instead gives `binomial(T + C, C)`, 66, which is valid and loose because
softmax does not distinguish `c` from `c + k`; the difference is `binomial(T + C - 1, C)`.
*Used in:* `spikecalib/theory.py` (`max_distinct_confidences`,
`distinct_confidence_bound_unreduced`). Asserted by enumeration, which gives exactly 55 at
three different scales, and by a case (`e^s = 2`, `T = 2`, `C = 4`) where a special scale
merges two values.

**The Murphy decomposition by exact confidence value, and its binned form (added
2026-10-13).** Grouping samples by exact confidence gives
`Brier = REL - RES + UNC` with no remainder. Over any coarser grouping the identity is
`Brier = REL - RES + UNC + WBV - WBC`, with `WBV` the within-group variance of the forecast
and `WBC` twice the within-group covariance of forecast and outcome; it follows from
expanding `(f - o)^2` about the group means, where the cross terms against the constant
`fbar - obar` sum to zero. Temperature scaling keeps the partition by exact value at all but
finitely many temperatures, so it leaves exact resolution unchanged; top-label isotonic is a
monotone map and can only merge groups, so it cannot raise it; one-versus-rest isotonic can
split a group and is not covered. *Used in:* `spikecalib/theory.py` (`murphy_exact`,
`murphy_binned`). Each statement is asserted in `tests/test_theory.py` against the real
calibrator classes. *Not cited:* the three-term decomposition is usually attributed to
Murphy (1973) and the two extra terms to later work on binned forecasts; neither source has
been opened, so neither is cited and the result is stated here as derived.

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
Session 1 Plenary 1.1, Digest of Technical Papers pp. 10-14.** Author **Mark Horowitz**; DOI
**10.1109/ISSCC.2014.6757323** — from the Crossref record for that DOI, whose title,
proceedings (2014 IEEE ISSCC Digest of Technical Papers) and pages 10-14 match this entry. Used
in `paper/refs.bib`. The energy table is
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
Title, four authors and date confirmed. **Authors in full: Kyohei Atarashi, Satoshi Oyama,
Hiromi Arai and Hisashi Kashima** — from the arXiv record (https://arxiv.org/abs/2506.10572,
`citation_author` metadata), submitted 12 June 2025 and last revised 23 February 2026 (v2).
Used in `paper/refs.bib`. Read for the distinction rather than taken from the
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
*Readout, verified 2026-09-12 in the method section:* **a non-spiking accumulator, not
spike counts.** Verbatim: *"keeping the output layer at full precision... the neurons in the
output layer only accumulate input currents."* Confidence is
`P_t = max_i e^{z_i/alpha} / sum_j e^{z_j/alpha}` over those accumulated currents, where
dividing by `alpha` is temperature scaling and the paper describes it as such. **The
attainable-confidence ceiling therefore does not apply to their 94.11%-at-4-timesteps
result**, and the question raised on 2026-09-12 is closed.
*What it does corroborate:* the same passage states that with such a readout "the output
logits of SNNs increase with `t` continuously, which causes the saturated confidence", and
their Figure 2a shows confidence "saturates to 100% quickly with time". That is the
mirror-image failure of the ceiling and matches this study's own accumulator measurements
(`conf - acc` of +0.0032 and +0.0047 at `T` = 6, 10 on N-MNIST; +0.08 to +0.23 at
`C = 101`).

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

## Literature check, 2026-10-13: is the bound already in print?

Run before any change to the paper's novelty wording. Each entry was read in the arXiv PDF,
and the statement quoted was found in the extracted text of that PDF; titles, authors and
dates are from the arXiv record's `citation_*` metadata. **The answer is yes.** The
inequality, the loss floor and the minimum scale are all published for non-spiking networks,
two of them with the same numbers this study reports at `T = 2, C = 10`. None of these may
be presented as new. What none of them contains is recorded at the end.

### The cap and the loss floor for bounded logits

**Wei, Xie, Cheng, Feng, An & Li. "Mitigating Neural Network Overconfidence with Logit
Normalization." ICML 2022, arXiv:2205.09310 (19 May 2022).** `[LIT]` Hongxin Wei, Renchunzi
Xie, Hao Cheng, Lei Feng, Bo An, Yixuan Li. Logits are normalised to norm `1/tau`, so each
lies in `[-1/tau, 1/tau]`. **Proposition 3.3 (Lower Bound of Loss):** "the per-sample loss
defined in Eq. (4) has a lower bound: L_logit_norm >= log(1 + (k - 1)e^{-2/tau}), where k is
the number of classes." The proof in Appendix C states the probability cap
`1 / (1 + (k - 1) e^{-2/tau})` on the way. The text continues: "when k = 10 and tau = 1 ...
the lower bound is about **0.7966**. The high lower bound can cause optimization difficulty."
*That is this study's cross-entropy floor at `T = 2, C = 10`, to four decimals, because
`2/tau = 2 = T`.* Venue from the PDF's footer, "Proceedings of the 39th International
Conference on Machine Learning".

**Wei, Zhuang, Xie, Feng, Niu, An & Li. "Mitigating Memorization of Noisy Labels by Clipping
the Model Prediction." ICML 2023, arXiv:2212.04055 (8 December 2022).** `[LIT]` Hongxin Wei,
Huiping Zhuang, Renchunzi Xie, Lei Feng, Gang Niu, Bo An, Yixuan Li. Eq. (2) bounds the loss
by the logit range directly: `L_CE <= log(1 + (K - 1) e^{z_max - z_min})`, with the remark
that "if z_max - z_min is upper bounded, L_CE could not reach infinity". **Proposition 2.1:**
"log(1+(K-1)e^{-2tau}) <= L_CE^tau(z, y) <= log(1+(K-1)e^{2tau})" for logits clipped to
`[-tau, tau]`. This is the floor for width `W = 2 tau`.

**Zhang, Zhao, Qiao, Wang & Li. "AdaCos: Adaptively Scaling Cosine Logits for Effectively
Learning Deep Face Representations." CVPR 2019, arXiv:1905.00292 (1 May 2019).** `[LIT]` Xiao
Zhang, Rui Zhao, Yu Qiao, Xiaogang Wang, Hongsheng Li. **Eq. (5):** "Given scale parameter s,
the range of probabilities in all cosine-based softmax losses is
1/(1 + (C - 1) e^s) <= P_{i,j} <= e^s/(e^s + (C - 1))", the upper bound "achieved when
f_{i,j} = s and f_{i,k} = 0 for all k != j", and: "when s is too small ... the maximal value
of P_{i,y_i} could not reach 1." *This is the cap in the form this study writes it, with
logits in `[0, s]` in place of `[0, T]`.* Venue from the arXiv record's comments field,
"CVPR 2019 Oral".

**Wang, Xiang, Cheng & Yuille. "NormFace: L2 Hypersphere Embedding for Face Verification."
ACM Multimedia 2017, arXiv:1704.06369 (21 April 2017).** `[LIT]` Feng Wang, Xiang Xiang, Jian
Cheng, Alan L. Yuille. With features and weights normalised the logits lie in `[-1, 1]`, and
Section 3.2 says: "In the extreme case, e^1 / (e^1 + (n-1)e^{-1}) is very small (**0.45 when
n = 10**; 0.007 when n = 1000)". *That is this study's ceiling at `T = 2, C = 10`, 0.4509.*
**Proposition 2** gives the matching loss bound, `log(1 + (n - 1) e^{-(n/(n-1)) l^2})` for
norm `l`, and the remedy: "we may implement this by directly appending a scale layer after
the cosine layer. The scale layer has only one learnable parameter s = l^2."

### The minimum scale

**Wang, Wang, Zhou, Ji, Gong, Zhou, Li & Liu. "CosFace: Large Margin Cosine Loss for Deep
Face Recognition." CVPR 2018, arXiv:1801.09414 (29 January 2018).** `[LIT]` Hao Wang, Yitong
Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu. **Eq. (6):** with
`P_W` "the expected minimum posterior probability of class center", "the lower bound of s is
given by: s >= ((C - 1)/C) log((C - 1) P_W / (1 - P_W))". *This is `s_min` with the target
posterior in place of the accuracy and `C/(C - 1)` in place of `T`.* Venue from the arXiv
record's comments field, "Accepted by CVPR 2018".

**Ranjan, Castillo & Chellappa. "L2-constrained Softmax Loss for Discriminative Face
Verification." arXiv:1703.09507 (28 March 2017).** `[LIT]` Rajeev Ranjan, Carlos D. Castillo,
Rama Chellappa. **Eq. (9):** "we can obtain the lower bound on alpha to achieve a
probability score of p ... alpha_low = log(p(C - 2)/(1 - p))". The same inversion with
`C - 2` from their geometric assumption. The paper also trains the scale as a parameter and
reports that the network then drives it high.

**He, Zhang, Zhang, Zhang, Xie & Li. "Bag of Tricks for Image Classification with
Convolutional Neural Networks." arXiv:1812.01187 (4 December 2018).** `[LIT]` Tong He, Zhi
Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. **Eq. (5):** under label smoothing the
optimal logit gap is `log((K - 1)(1 - eps)/eps)`. With `a = 1 - eps` this is the expression
`ln((C - 1) a / (1 - a))`, as an optimum instead of a requirement. *Venue not confirmed
here:* it is commonly given as CVPR 2019, which the arXiv record does not state.

### Checked and found not to state the bound

- **Guo et al. (2017)**, arXiv:1706.04599: no bound on confidence from the logit range; only
  the limits of temperature scaling as `T -> 0` and `T -> infinity`.
- **Atarashi et al. (2025)**, arXiv:2506.10572: bounds the output probabilities directly and
  learns them; its Proposition 4.1 runs from probability bounds to logit clip levels, and no
  closed-form cap from a logit range is given. The entry under "Verified 2026-09-12" stands.
- **Müller, Kornblith & Hinton (2019)**, arXiv:1906.02629: says label smoothing makes the
  logit gap "a constant dependent on alpha" and does not write the constant.

### What this leaves, and what it does not

Not found in any of the above, as far as the search went: the statement for a spike-count
readout, where the width is the number of timesteps; the calibration reading, in which the
quantity that must be reachable is the model's own accuracy; the count of distinct
confidences; and any measurement of the calibration error the cap forces. **A failed search
is not proof of absence.** The paper may claim the application and the measurements. It may
not claim the inequality, the loss floor, the minimum scale, or the idea of a learnable
scale as a remedy.

### Spiking networks: who has seen underconfidence, a scale, or a cap

Each entry below was read in the arXiv PDF and the quoted sentence found in its extracted
text, unless marked otherwise.

**Chen, Park & Simeone. "Knowing When to Stop" (SpikeCP), arXiv:2305.11322v4.** Already
listed above for its conformal method. **Read again for calibration on 2026-10-13, and it
changes what this study may claim.** It uses the softmax over spike counts,
`p_c = e^{r_c} / sum e^{r_c'}`, and says in its introduction, of a pre-trained SNN on
MNIST-DVS as a function of time `t`: "It is observed that the SNN's classification decisions
tend to be **first under-confident and then over-confident** with respect to the decision's
ground-truth, unknown, test accuracy." The Figure 1 caption repeats it. Experiments use
`T = 80`. `[LIT]` *Consequence:* underconfidence of a spike-count softmax at early timesteps
is a published observation. The README's sentence that the underconfident direction "is
this study's own observation" is wrong and must be withdrawn, and the paper's Related work,
which describes these authors as reporting overconfidence only, is incomplete. What the paper offers is the reason: read at step `t`, their counts lie in
`[0, t]`, so the cap is `e^t / (e^t + C - 1)`. That reading is `[DERIVED]` here and is not in
their text.

**Meng, Xiao, Yan, Wang, Lin & Luo. "Training High-Performance Low-Latency Spiking Neural
Networks by Differentiation on Spike Representation" (DSR). arXiv:2205.00459v2; CVPR 2022
per the search, not confirmed by me.** `[LIT]` Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen
Wang, Zhouchen Lin, Zhi-Quan Luo. Directly trained; the output is the spike train of the
last layer, its threshold-weighted firing rate is the network output, and "We use
cross-entropy as the loss function". The thresholds are trained: "using small spike
thresholds ... weakens the approximation capacity of the SNNs, since the scaled (weighted)
firing rate will be in a small range then ... we treat the spike threshold of each layer as
parameters to be trained". *Consequence:* a trained multiplicative scale on output firing
rates under cross-entropy, in a directly trained SNN, is published. A learnable logit scale
may not be presented as new. DSR gives no bound on confidence or loss.

**Fan, Zhang, Liu, Li & Lu. "SFOD: Spiking Fusion Object Detector." arXiv:2403.15192v1; CVPR
2024 per the search, not confirmed by me.** `[LIT]` Section 3.3: "SNNs produce limited
discrete output values. This leads to optimization issues in the neurons when using CE loss
... consider two cases with decoded outputs of [0.2, 0.8] and [0.2, 1.0]. Using CE, the
results of the softmax computations are [0.35, 0.65] and [0.31, 0.69], respectively." The
closest published anticipation of the rate cap: a worked example, no formula, no statement
that it is independent of `T`, and the remedy offered is a squared-error loss.

**Wu, Chua, Zhang, Yang, Li & Li. "Deep Spiking Neural Network with Spike Count based
Learning Rule." arXiv:1902.05705v1.** `[LIT]` "the output spike counts are upper bounded by
the maximum time steps r_max := T/dt ... we have not noticed any performance drop due to
r_max". Cross-entropy on spike counts and the bound on the counts in one paper, with the
consequence for confidence not drawn.

**Sun, Yin & Bohté. "Efficient Uncertainty Estimation in Spiking Neural Networks via
MC-dropout." arXiv:2304.10191v1, 20 April 2023.** `[LIT]` Tao Sun, Bojian Yin, Sander Bohté.
**This is the AOT-SNN classification paper that item 4 of "Could NOT be confirmed" left
unread.** AOT is Average-Over-Time. The readout is a membrane potential, not a spike count:
"probabilities of AOT-SNNs are computed based on the membrane potentials of readout
integrator neurons". It reports ECE, Brier score and NLL at one to ten timesteps on
CIFAR-10, and compares against "the temperature scaling method that re-calibrates
probabilities output by the LTS-SNN". So the brief's original claim, that this line uses
temperature scaling as a calibration baseline, was right about the paper and wrong about the
identifier. ICANN 2023 per the search's Crossref check, not confirmed by me. The cap does
not apply to its readout.

**Skatchkovsky, Jang & Simeone. "Bayesian Continual Learning via Spiking Neural Networks."
arXiv:2208.13723v2.** `[LIT]` Nicolas Skatchkovsky, Hyeryung Jang, Osvaldo Simeone. **This is
very probably the "Skatchkovsky & Simeone on SNN overconfidence" reference withdrawn as item
5.** It reports reliability diagrams and ECE and says: "frequentist learning algorithms
produce overconfident decisions, while Bayesian learning outputs decisions whose confidence
levels match well the test accuracies", and "frequentist learning tends to be more confident
in its decisions, especially when a few samples t have been observed." Its confidence is a
time-average of per-step softmax outputs through fixed random weights, not a softmax over
spike counts, and it runs for 50 to 100 timesteps. Frontiers in Computational Neuroscience
2022 per the search, not confirmed by me.

**Katti, Nimbekar, Li, Acharyya, Al-Hashimi & Rajendran. "Bayesian Inference Accelerator for
Spiking Neural Networks." arXiv:2401.15453.** `[LIT]` Converted Bayesian binary networks on
CIFAR-10; "The spike outputs of the networks are accumulated and processed (softmax)"; ECE
against timestep, with "a slow increase of ECE for Bayesian SNNs after 4 timesteps". Nothing
about a cap or about underconfidence below four steps.

**Li, Jones & Furber, title.** The PDF of arXiv:2303.10276v3 is headed "Unleashing the
Potential of Spiking Neural Networks **with** Dynamic Confidence"; the arXiv record says
"by". The entry under "Verified 2026-09-12" noted the same split with the ICCV listing.
`paper/refs.bib` uses "by" and should follow the published title.

**The two withdrawn references, revisited.** Both exist, under identifiers and author lists
the brief did not give. The withdrawals were correct as made, since neither could be tied to
a document at the time, and the test that keeps the old strings out of live files should stay
until the author decides whether to cite the papers under their real records.

**Reported by the search and not opened by me. `[UNVERIFIED]`; not to be cited until
opened.** Sajid, Aburaed & Maskeliūnas, "ConvSNN", *Scientific Reports* 2026 (said to report
an underconfident spiking classifier head with fitted temperature below 1, attributed to
label smoothing); Jang & Simeone, arXiv:2007.11894 (ECE for a probabilistic SNN); Katti et
al., arXiv:2302.01302; Yang et al., "SpikeMatch", arXiv:2509.22581; Abdennadher & Garner,
arXiv:2604.08624; Banteywalu & Leroux, *Engineering Reports* 2026, a Bayesian SNN evaluated
with ECE on N-MNIST, abstract only; and an embargoed University of Porto dissertation,
"Calibration of Spiking Neural Networks for Ordinal Classification" (2026), whose content is
unknown.

**Not found by any search:** a statement of `e^T / (e^T + C - 1)` or an equivalent ceiling
for a spike-count softmax; the `T`-independent cap of a rate readout as a formula; any
comparison of temperature scaling with isotonic regression on a spiking network. Searches
covered arXiv titles and abstracts, OpenAlex, Europe PMC full text, Crossref and general web
search; Google Scholar and Semantic Scholar were not usable, and arXiv does not index full
text.

### References verified 2026-10-13 for the revision

Each was opened in full text and the quoted passage found in it. Bibliographic fields are
from Crossref, the proceedings page or the arXiv record, as stated.

**Murphy (1973). "A New Vector Partition of the Probability Score." *Journal of Applied
Meteorology* 12(4):595-600.** Allan H. Murphy; DOI
10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 (Crossref). `[LIT]` Read as the scanned PDF
from the AMS site. Abstract: "This partition consists of three terms: 1) a measure of the
uncertainty inherent in the events ... 2) a measure of the reliability of the forecasts; and
3) a new measure of the resolution of the forecasts." The partition groups on the distinct
forecast values actually issued, with no binning. The journal's name in 1973 was *Journal of
Applied Meteorology*; the AMS site now labels it with the current name.

**Stephenson, Coelho & Jolliffe (2008). "Two Extra Components in the Brier Score
Decomposition." *Weather and Forecasting* 23(4):752-757.** D. B. Stephenson, C. A. S. Coelho,
I. T. Jolliffe; DOI 10.1175/2007WAF2006116.1 (Crossref). `[LIT]` Read in full. "The Brier
score has five rather than three components: BS = REL - RES + UNC + WBV - WBC", with the
within-bin variance and covariance as defined in `spikecalib/theory.py`, and "Both terms
vanish if only one value of probability is forecast for each bin". The article prints
initials only; full first names are not confirmed and are not used. The five-term identity
was derived here independently and is asserted by a test; the paper is cited as its source.

**Gupta & Ramdas (2022). "Top-label calibration and multiclass-to-binary reductions." ICLR
2022.** Chirag Gupta, Aaditya Ramdas; arXiv:2107.08353; https://openreview.net/forum?id=WqoBaaPHS-.
`[LIT]` Read in the arXiv PDF (v4). Remark 1: histogram binning "produces discrete
predictions whose ECE can be estimated without further binning ... we report unbinned ECE
estimates while assessing HB"; Appendix E.2: "Further binning was not done ... since the
output is already discrete and sufficiently many points take each of the predicted values.
Note that due to Jensen's inequality, any further binning will only decrease the ECE
estimate". *What may be said:* for a discrete-output predictor the ECE is estimated by
grouping on the output value, with no binning choice, provided enough samples share each
value. *What may not:* that it is then computed "exactly". The plug-in estimate still has
finite-sample bias, which Gupta & Ramdas (ICML 2021, arXiv:2105.04656) say in terms. No DOI;
venue from the OpenReview record.

**Dimitriadis, Gneiting & Jordan (2021). "Stable reliability diagrams for probabilistic
classifiers." *PNAS* 118(8), e2016191118.** Timo Dimitriadis, Tilmann Gneiting, Alexander I.
Jordan; DOI 10.1073/pnas.2016191118 (Crossref). `[LIT]` Read as the PubMed Central full text,
PMC7923594. The CORP approach "is based on nonparametric isotonic regression and implemented
via the pool-adjacent-violators (PAV) algorithm", with "no need for tuning parameters or
implementation decisions", and gives a decomposition into miscalibration, discrimination and
uncertainty: "in the special case of discrete forecasts with nondecreasing CEPs, the MCB,
DSC, and UNC terms ... agree with the REL, RES, and UNC components, respectively, in the
classical Murphy decomposition", and "Murphy's decomposition is exact in the discrete case,
but fails to be exact under continuous forecasts". CORP is defined through four headings
(consistency, optimality, reproducibility, PAV algorithm-based), not one expanded phrase.

**Fawcett (2006). "An introduction to ROC analysis." *Pattern Recognition Letters*
27(8):861-874.** Tom Fawcett; DOI 10.1016/j.patrec.2005.10.010 (Crossref). `[LIT]` Read in
the publisher-typeset PDF. Section 10, "Interpolating classifiers": "Sometimes the
performance desired of a classifier is not exactly produced by any available classifier, but
lies between two available classifiers. The desired performance can be obtained by sampling
the decisions of each classifier." Section 5, on tied scores: the curve "should represent the
expected performance of the classifier", obtained "by not emitting an ROC point until all
instances of equal f values have been processed". *What may be said:* an operating point
between two deterministic ones is reachable by randomising between them. *What may not be
attributed to it:* that tied scores are themselves to be broken at random. Applying Section
10 to the two thresholds either side of a tied group is this study's step, `[DERIVED]`.

**Galil, Dabbah & El-Yaniv (2023). "What Can We Learn From The Selective Prediction And
Uncertainty Estimation Performance Of 523 Imagenet Classifiers." ICLR 2023.** Ido Galil,
Mohammed Dabbah, Ran El-Yaniv; arXiv:2302.11874; https://openreview.net/forum?id=p66AzKi6Xim.
`[LIT]` Read in the arXiv PDF (v1). "(a) TS consistently and greatly improves AUROC and
selective performance ... it also improves the partial order of all instances induced by
those improved estimations ... (b) While TS is usually beneficial, it could harm some
models". *What may be said:* temperature scaling changes the ranking a softmax confidence
induces, usually for the better. *What may not:* anything about AURC by name, which the
passage does not mention. Title as on the arXiv record; the OpenReview record ends it with
a question mark and has a lowercase "we".

**Published records for works cited from arXiv above**, from Crossref and the proceedings
pages. Where the open-access proceedings and the IEEE record paginate differently, the
open-access pages are used, which are those printed on the PDF.
- AdaCos: CVPR 2019, pp. 10823-10832; DOI 10.1109/CVPR.2019.01108.
- NormFace: Proceedings of the 25th ACM International Conference on Multimedia, 2017,
  pp. 1041-1049; DOI 10.1145/3123266.3123359.
- LogitNorm: Proceedings of the 39th International Conference on Machine Learning, PMLR
  162:23631-23644, 2022.
- LogitClip: Proceedings of the 40th International Conference on Machine Learning, PMLR
  202:36868-36886, 2023.
- CosFace: CVPR 2018, pp. 5265-5274; DOI 10.1109/CVPR.2018.00552.
- Ranjan, Castillo & Chellappa: arXiv:1703.09507 only; no published version was found.
- DSR: CVPR 2022, pp. 12444-12453; DOI 10.1109/CVPR52688.2022.01212.
- SpikeCP: *IEEE Journal of Selected Topics in Signal Processing* 19(1):88-102, January 2025;
  DOI 10.1109/JSTSP.2024.3431996. The published title capitalises "With".
- Li, Jones & Furber: ICCV 2023, pp. 13350-13360; DOI 10.1109/ICCV51070.2023.01228. The
  proceedings title is "... **with** Dynamic Confidence".
- SFOD: CVPR 2024, pp. 17191-17200; DOI 10.1109/CVPR52733.2024.01627.
- Skatchkovsky, Jang & Simeone: *Frontiers in Computational Neuroscience* 16:1037976, 2022;
  DOI 10.3389/fncom.2022.1037976.

## Literature check, 2026-10-14: fusion

For "Consequence for multisensory fusion" in `NOTES.md`. Each entry was read in full text and
the quoted passage found in the extracted text of the downloaded document, unless marked.
Bibliographic fields are from Crossref, PMLR or the arXiv record as stated.

### Decision fusion: the theory the proposition is a corollary of

**Kittler, Hatef, Duin & Matas (1998). "On Combining Classifiers." *IEEE TPAMI*
20(3):226-239.** Josef Kittler, Mohamad Hatef, Robert P. W. Duin, Jiri Matas; DOI
10.1109/34.667881 (Crossref). `[LIT]` Read in full. "Let us assume that the representations
used are conditionally statistically independent", from which the product rule on the
classifiers' posteriors follows (Eq. 7), and with it the sum, max, min, median and majority
vote rules. On hardening: replacing each posterior by a one-hot vector "results in combining
decision outcomes rather than combining a posteriori probabilities", and the sum rule then
"simply counts the votes".

**Shapley & Grofman (1984). "Optimizing group judgmental accuracy in the presence of
interdependencies." *Public Choice* 43(3):329-343.** Lloyd Shapley, Bernard Grofman; DOI
10.1007/BF00118940 (Crossref). `[LIT]` Read in the scan on the second author's site.
"Theorem II: (The Bayesian Optimal Decision Rule) In a heterogeneous group, in which
individual choices are mutually independent ... the decision rule that maximizes P_N is a
majority weighted voting rule, obtained by assigning weights, w_i proportional to
log(p_i / (1 - p_i))." No proof is given; note 8 says the result "was also independently
discovered by two economists (Nitzan and Paroush, 1980a, 1982)" and was known earlier in
pattern recognition.

**Nitzan & Paroush (1982). "Optimal Decision Rules in Uncertain Dichotomous Choice
Situations." *International Economic Review* 23(2):289-297.** Shmuel Nitzan, Jacob Paroush;
DOI 10.2307/2526438 (Crossref; pages from the RePEc record). **The paper could not be
opened** (JSTOR). `[LIT]` *by a verified secondary source only:* Shapley & Grofman (1984),
read in full, give it as the published proof of their Theorem II and as an independent
discovery of the log-odds weights. That, and no more, is what may be said of it.

**Chair & Varshney (1986). "Optimal Data Fusion in Multiple Sensor Detection Systems." *IEEE
Transactions on Aerospace and Electronic Systems* AES-22(1):98-101.** Z. Chair, P. K.
Varshney; DOI 10.1109/TAES.1986.310699 (Crossref). **The paper could not be opened** (IEEE
Xplore). `[LIT]` *by its abstract and a verified secondary source:* the abstract, as carried
by the OpenAlex record, says "Individual decisions are weighted according to the reliability
of the detector and then a threshold comparison is performed to obtain the global decision."
Mirjalily, Luo & Davidson (*IEEE Trans. AES* 39(1):34-52, 2003), read in an author-hosted
copy, restate the weights as logarithms of each detector's miss and false-alarm
probabilities. Only the abstract's statement is used in the paper.

**Kuncheva & Rodríguez (2014). "A weighted voting framework for classifiers ensembles."
*Knowledge and Information Systems* 38(2):259-275.** Ludmila I. Kuncheva, Juan J. Rodríguez;
DOI 10.1007/s10115-012-0586-6 (Crossref). `[LIT]` Read in full. Under independence given the
class, "When P(s_i = w_k | w_k) = p_i and P(s_i = w_j | w_k) = (1 - p_i)/(c - 1) ... then the
weighted majority vote is the optimal combiner", with weights `w_i = log(p_i/(1 - p_i))` and,
in Eq. (15), a further term of `log(c - 1)` per vote and a class-prior term. Folding the
`log(c - 1)` into the weight gives `log((c - 1) p_i / (1 - p_i))`. There is no numbered
proposition in the paper; the result is a stated condition and a derivation.

**Li & Yu (2014). "Error Rate Bounds and Iterative Weighted Majority Voting for
Crowdsourcing." arXiv:1411.4086.** Hongwei Li, Bin Yu. `[LIT]` Read in full. "Corollary 9 (The
oracle MAP rule under Homogenous Dawid-Skene model) Suppose the task assignment is based on a
constant probability q in (0, 1], and the prevalence of the true labels are balanced. Then the
oracle MAP rule is a weighted majority voting rule ... where nu_i = ln((L - 1) w_i / (1 -
w_i))". *This is the weight in part (b) of the proposition, exactly, and it carries the
condition that classes are balanced.* No journal version was found; cited as a preprint.

**Brümmer, Burget, Černocký, Glembek, Grézl, Karafiát, van Leeuwen, Matějka, Schwarz &
Strasheim (2007). "Fusion of Heterogeneous Speaker Recognition Systems in the STBU Submission
for the NIST Speaker Recognition Evaluation 2006." *IEEE Transactions on Audio, Speech, and
Language Processing* 15(7):2072-2084.** DOI 10.1109/TASL.2007.902870 (Crossref). `[LIT]` Read
in the author-hosted preprint, so cited by section. Sec. IV: scores are fused by a linear
combination whose weights "were obtained by logistic regression", so that the fused score
"functions as a well-calibrated log-likelihood-ratio"; and "each sub-system on its own was
subjected to an affine calibration transformation, also trained via logistic regression". The
remedy in part (c), calibrate each source and then combine, is this.

**Kuncheva (2002). "A theoretical study on six classifier fusion strategies." *IEEE TPAMI*
24(2):281-286.** DOI 10.1109/34.982906 (Crossref). `[LIT]` Read in full. It derives error
formulas for the average, minimum, maximum, median and majority vote rules for two classes,
and leaves the product rule out. It notes, of a simulation, that when one estimate is 1 and
another 0 "both the minimum and the maximum will produce a tie, and the random tie break will
induce bias". It does not state the proposition.

**Genest & Zidek (1986). "Combining Probability Distributions: A Critique and an Annotated
Bibliography." *Statistical Science* 1(1).** DOI 10.1214/ss/1177013825 (Crossref). **Could not
be opened** (Project Euclid). `[UNVERIFIED]` for content, and not cited.

### Prior statements of parts of the proposition

The proposition is not found assembled anywhere. Its parts are.

**Ju, Bibaut & van der Laan (2018). "The relative performance of ensemble methods with deep
convolutional neural networks for image classification." *Journal of Applied Statistics*
45(15):2800-2818.** Cheng Ju, Aurélien Bibaut, Mark van der Laan; DOI
10.1080/02664763.2018.1441383 (Crossref); read as arXiv:1704.01664. `[LIT]` Sec. 4.4.4: "When
all the models are over-confident, the unweighted average is identical to the majority vote."
**The closest prior statement of part (a).** It is asserted for the average of probabilities
that have saturated towards one, without proof and without the link to reliability weights.
Part (a) here concerns a cap below one and is stated for any symmetric rule.

**Kumar, Ma, Liang & Raghunathan (2022). "Calibrated ensembles can mitigate accuracy
tradeoffs under distribution shift." UAI 2022, PMLR 180:1041-1051.** Ananya Kumar, Tengyu Ma,
Percy Liang, Aditi Raghunathan; read as arXiv:2207.08977. `[LIT]` Proposition 4.1: for two
calibrated models whose outputs are conditionally independent given the label, with balanced
classes, the ensemble that adds their logits "is better than any such combination
classifier". **The closest prior statement of part (c)**: calibrate, then apply the product
rule. Author list and pages from the search's reading of the PMLR record; the proposition was
found in the arXiv text.

**Not found by the search:** the exchange argument of part (a), that a rule treating two
classifiers and all classes alike cannot separate two reports that are permutations of each
other. It is elementary, and is stated in this repository as `[DERIVED]`, without a claim that
it is new. May (1952) on simple majority rule may be its classical antecedent and was not
opened.

### Dynamic multimodal fusion

**Zhang, Wu, Zhang, Hu, Fu, Zhou & Peng (2023). "Provable Dynamic Fusion for Low-Quality
Multimodal Data." Proceedings of the 40th International Conference on Machine Learning, PMLR
202:41753-41769.** Qingyang Zhang, Haitao Wu, Changqing Zhang, Qinghua Hu, Huazhu Fu, Joey
Tianyi Zhou, Xi Peng; arXiv:2306.02050. `[LIT]` Theorem 2 compares **upper bounds** on
generalisation error for two-class late fusion: the dynamic bound is no larger than the
static one if the mean dynamic weight equals the static weight and
"r(w^m_dynamic, l(f^m)) <= 0 ... for all input modalities, where r is the Pearson correlation
coefficient which measures the correlation between fusion weights ... and unimodal loss".
The paper adds: "in static multimodal fusion w^m_static is a constant, which means Term-Cov =
0 for any static fusion methods." *What may be said:* weights that are constant across
samples have zero covariance with every loss, so the advantage the theorem establishes for
dynamic fusion is zero. *What may not:* that the theorem "fails" or is contradicted, or that
QMF's own weights are constant. QMF takes its weights from a regularised energy score of the
logits, not from the maximum softmax probability, which appears only as a baseline in its
Table 3.

**Cao, Xia, Ding, Zhang & Hu (2024). "Predictive Dynamic Fusion." Proceedings of the 41st
International Conference on Machine Learning, PMLR 235:5608-5628.** Bing Cao, Yinan Xia, Yi
Ding, Changqing Zhang, Qinghua Hu; arXiv:2406.04802. `[LIT]` Theorem 3.1 bounds the
generalisation error with a term in the covariance of each modality's weight with its own
loss and with the others', and concludes: "Corollary 3.2. A negative correlation should exist
between a modality's weight and its loss. Corollary 3.3. A positive correlation is desirable
between a modality's weight and the losses of the other modalities." The stated requirement
is strict, `Cov(w^m, l^m) < 0`. *What may be said:* constant weights have zero covariance and
do not meet it. *What may not:* that this method's weights are constant under a capped
readout; it predicts them from encoder features with a separately trained predictor of the
true-class probability.

### Probabilistic population codes

**Ma, Beck, Latham & Pouget (2006). "Bayesian inference with probabilistic population codes."
*Nature Neuroscience* 9(11):1432-1438.** Wei Ji Ma, Jeffrey M. Beck, Peter E. Latham,
Alexandre Pouget; DOI 10.1038/nn1790 (Crossref). `[LIT]` Read in the publisher-typeset PDF
hosted by the third author. With Poisson-like variability, "taking the sum of the two
population codes, r1 and r2, is equivalent to optimal Bayesian inference", and the gain of a
population encodes its reliability. On saturation, one paragraph of the Discussion, in full:
"One limitation of our scheme, and of any scheme that reduces Bayesian inference to addition
of activities, is that neural activities are likely to saturate when sequential inferences
are required. To circumvent this problem, a nonlinearity is needed to keep neurons within
their dynamical range. A nonlinearity like divisive normalization would be ideal because it is
near linear for low firing rates, where uncertainty is large and thus there is much to be
gained from performing exact inference, and saturating at high firing rates, where
uncertainty is small and there is little to be gained from exact inference". *What may be
said:* the paper argues, qualitatively and without a number, that saturation at high rates
costs little because uncertainty is small there. *What may not:* that it analyses a hard
ceiling, the relative weighting of two saturated cues, or a categorical readout. Its setting
is a population code for a continuous variable. Nothing about human or animal behaviour is
taken from it here.

**Beck, Ma, Kiani, Hanks, Churchland, Roitman, Shadlen, Latham & Pouget (2008).
"Probabilistic Population Codes for Bayesian Decision Making." *Neuron* 60(6):1142-1152.**
DOI 10.1016/j.neuron.2008.09.021 (Crossref). `[LIT]` Read in the publisher PDF. Networks "can
accumulate evidence without loss of information through linear integration of neural
activity", and: "The problem with this simple summing operation, however, is that LIP
activity would saturate very quickly. Fortunately, it is possible to show that global
inhibition can be used to alleviate this problem while preserving optimality". It proposes a
way to avoid saturation and gives no cost for it.

### Calibration is not enough

**Perez-Lebel, Le Morvan & Varoquaux (2023). "Beyond calibration: estimating the grouping
loss of modern neural networks." ICLR 2023.** Alexandre Perez-Lebel, Marine Le Morvan, Gaël
Varoquaux; arXiv:2210.16315; https://openreview.net/forum?id=6w1k-IixnL8. `[LIT]` "even a
perfectly calibrated classifier with the best possible accuracy can have confidence scores
that are far from the true posterior probabilities. This is due to the grouping loss, created
by samples with the same confidence scores but different true posterior probabilities." The
decomposition into calibration, grouping and irreducible loss is attributed by them to Kull
and Flach (2015), which was not opened and is not cited.

### Read and found not to bear on the proposition

**Tenali, Sidheekh, Mathur, Saravanan, Blasch, Kersting & Natarajan. "Reliable Fusion of
Conflicting Experts." arXiv:2609.27913.** `[LIT]` Dynamic aggregation of large language
models with learned credibility on multiple-choice questions. It does not concern saturated
confidence, bounded logits or spiking networks, and does not state the proposition.

**Not found by the search:** any statement that confidence-weighted or uncertainty-weighted
multimodal fusion degenerates to constant weights when unimodal confidences are saturated,
clipped or quantised. Eight searches were run and every candidate opened. The nearest is the
remark in Zhang et al. quoted above. A failed search is not proof of absence.

## References verified 2026-10-17

**Skatchkovsky, Jang & Simeone (2022). "Bayesian continual learning via spiking neural
networks." *Frontiers in Computational Neuroscience* 16:1037976.** Nicolas Skatchkovsky,
Hyeryung Jang, Osvaldo Simeone; DOI 10.3389/fncom.2022.1037976 (Crossref: title, journal,
volume, article number 1037976, published 2022-11-16, three authors); arXiv:2208.13723. `[LIT]`
Read in the Frontiers PDF and in arXiv v2. The author decided on 2026-10-17 to cite it; it is
the paper the reference withdrawn on 2026-09-12 as "Skatchkovsky & Simeone on SNN
overconfidence" very probably meant (see "Literature check, 2026-10-13").
- *Calibration is measured:* "performance metrics include calibration measures, namely
  reliability diagrams and expected calibration error (ECE)"; "frequentist learning
  algorithms produce overconfident decisions, while Bayesian learning outputs decisions whose
  confidence levels match well the test accuracies".
- *Timesteps:* two-moons "over T = 100 time-steps"; split-MNIST "rate-encoded over T = 50
  time-steps"; MNIST-DVS "restrict the length to T = 80 time-samples"; DVS-Gestures
  "sequences of length 500 ms for training and 1500 ms for testing, with a sampling rate of
  10 ms", that is 50 and 150 steps. Hence "50 to 150 timesteps" in the paper.
- *The readout is not a count readout.* Eq. (37): the predictive probability is
  `(1/T) Σ_t Softmax_k(B^(L) f(w, x_t))`, where `f(w, x_t)` is the output of the read-out
  layer at step `t` and `B^(L)` "are random, fixed weights". It is a time average of per-step
  softmaxes over a fixed random projection of the output spikes. Training uses DECOLLE
  surrogate gradients, so the networks are directly trained.
- *What may be said:* they measure SNN calibration at 50 to 150 timesteps, where a count
  readout's cap would be slack, and find frequentist training overconfident. *What may not:*
  that their readout is subject to the count cap, or that its cap is slack. The cap of a
  per-step softmax over `B^(L) s_t` is set by the range of `B^(L) s_t`, which the paper does
  not report, not by `T`. The brief's phrase "where the cap is slack" is therefore used only
  as a statement about count readouts at those lengths.

**Grillo, Geminiani, Alessandro, D'Angelo, Pedrocchi & Casellato (2022). "Bayesian Integration
in a Spiking Neural System for Sensorimotor Control." *Neural Computation*
34(9):1893-1914.** DOI 10.1162/neco_a_01525 (Crossref: title, journal, volume, issue, pages,
six authors). `[LIT]` Read in the publisher's typeset PDF (CC BY 4.0), fetched through the
Zenodo record 8135935; the MIT Press and Politecnico di Milano links refused automated access.
- *Abstract:* the estimator "receives spiking activity from two neural populations encoding
  the sensory feedback and the cerebellar prediction, and it continuously computes the spike
  variability within each population as a reliability index of the signal these populations
  encode."
- *Delay:* "the trajectory of the point mass was delayed by 100 ms before translating it into
  the neural activity of the sensory population" (Section 2.1).
- *Simulator and inputs:* the populations "were implemented as spiking neural networks (SNN)
  ... in NEST"; the sensory and cerebellar neurons are "Poisson single-point neurons"; "We
  simulated a reaching task" (Section 2.2 and abstract).
- *Reliability:* "the variance in the number of incoming spikes across neurons was computed
  within a 25 ms time window", scaled by the mean count (Eq. 2.3), and the estimate is "the
  weighted sum of X_Fbk and μ_Pred based on their relative variabilities" (Eq. 2.2).
- *Isolated block, variability set by the authors:* "Our model is a proof-of-concept and was
  tested here as an isolated block, receiving spike-encoded input signals"; "we had to make
  assumptions on the statistics of these areas' neural activity"; "we considered two
  scenarios: high variability to simulate inaccurate cerebellar predictions (prelearning),
  and low variability to simulate accurate cerebellar predictions (postlearning)"
  (Sections 2.2 and 4).
- *What may be said:* everything in the paper's one paragraph on it. *What may not:* that it
  uses a readout's confidence, or that it was run in a closed loop. Closed-loop integration is
  stated as future work.

## References verified 2026-11-12

Every entry of `paper/refs.bib` was checked again on 2026-11-12, field by field, against
Crossref (`https://api.crossref.org/works/<DOI>`), DataCite, the DOI landing page, arXiv,
OpenReview, PMLR or the publisher. Entries marked *corrected* were changed in `refs.bib` on
that date; the others were found correct as they stood.

- **zhang2023provable** (correct). authors (Qingyang Zhang, Haitao Wu, Changqing Zhang, Qinghua Hu, Huazhu Fu, Joey Tianyi Zhou, Xi Peng), title, Proceedings of the 40th ICML, PMLR 202:41753-41769, 2023 verified against PMLR page proceedings.mlr.press/v202/zhang23ar.html; PMLR assigns no DOI.
- **cao2024predictive** (correct). authors (Bing Cao, Yinan Xia, Yi Ding, Changqing Zhang, Qinghua Hu), title, Proceedings of the 41st ICML, PMLR 235:5608-5628, 2024 verified against PMLR page proceedings.mlr.press/v235/cao24c.html; PMLR assigns no DOI.
- **ma2006bayesian** (correct). authors (Wei Ji Ma, Jeffrey M. Beck, Peter E. Latham, Alexandre Pouget), title, Nature Neuroscience 9(11):1432-1438, 2006 (print Nov 2006, online 22 Oct 2006), DOI 10.1038/nn1790 verified against Crossref.
- **beck2008probabilistic** (correct). all nine authors in order (Beck, Ma, Kiani, Hanks, Churchland, Roitman, Shadlen, Latham, Pouget) with given names, title, Neuron 60(6):1142-1152, Dec 2008, DOI 10.1016/j.neuron.2008.09.021 verified against Crossref.
- **perezlebel2023beyond** (corrected). authors, title, year verified against the OpenReview ICLR 2023 record (forum 6w1k-IixnL8, poster) and arXiv:2210.16315v3 (27 Apr 2023); booktitle expanded to 'The Eleventh International Conference on Learning Representations' and OpenReview URL added; no DOI exists.
- **skatchkovsky2022bayesian** (correct). authors (Nicolas Skatchkovsky, Hyeryung Jang, Osvaldo Simeone), title, Frontiers in Computational Neuroscience vol. 16, article number 1037976 (Crossref 'article-number', given as pages), published 16 Nov 2022, DOI 10.3389/fncom.2022.1037976 verified against Crossref; arXiv:2208.13723 (latest v2, 1 Nov 2022, cs.NE) is the preprint of this article.
- **grillo2022bayesian** (correct). six authors in order (Massimo Grillo, Alice Geminiani, Cristiano Alessandro, Egidio D'Angelo, Alessandra Pedrocchi, Claudia Casellato), title, Neural Computation 34(9):1893-1914, 2022, MIT Press, DOI 10.1162/neco_a_01525 verified against Crossref (confirmed by Semantic Scholar).
- **fan2024sfod** (correct). authors (Yimeng Fan, Wei Zhang, Changsong Liu, Mingyang Li, Wenrui Lu), title, pages 17191-17200, year 2024 and DOI 10.1109/CVPR52733.2024.01627 match Crossref and the CVF open-access record. The booktitle matches the CVF BibTeX word for word (Crossref container: '2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)').
- **murphy1973partition** (correct). author Allan H. Murphy, title, Journal of Applied Meteorology 12(4):595-600, June 1973, and the DOI all match Crossref (publisher: American Meteorological Society).
- **stephenson2008two** (correct). authors (D. B. Stephenson, C. A. S. Coelho, I. T. Jolliffe, which are initials only in Crossref), title, Weather and Forecasting 23(4):752-757, August 2008, and the DOI all match Crossref. Full given names could not be confirmed from the primary record because the AMS page returned 403, so the initials are kept.
- **gupta2022toplabel** (corrected). authors, title and year checked against the OpenReview ICLR 2022 record (forum WqoBaaPHS-, ICLR 2022 Poster) and arXiv 2107.08353 (latest v4, 2022-09-02). Added the edition to the booktitle and the OpenReview url. No DOI exists.
- **dimitriadis2021stable** (correct). authors (Timo Dimitriadis, Tilmann Gneiting, Alexander I. Jordan), title, PNAS 118(8), article number e2016191118 (Crossref article-number, given here as pages), year 2021 (online 17 Feb, print 23 Feb) and DOI all match Crossref.
- **fawcett2006roc** (correct). author Tom Fawcett, title 'An introduction to ROC analysis', Pattern Recognition Letters 27(8):861-874, June 2006 and DOI all match Crossref (Elsevier). {ROC} is already protected.
- **galil2023selective** (corrected). checked against the OpenReview ICLR 2023 record (forum p66AzKi6Xim, ICLR 2023 poster). The title is changed to the published form, which ends in '?'. Added the edition to the booktitle and the OpenReview url. The arXiv preprint 2302.11874 has only v1 (2023-02-23). No DOI exists for the conference version.
- **kittler1998combining** (correct). title, IEEE TPAMI 20(3):226-239, March 1998, and DOI 10.1109/34.667881 match Crossref. The author order Kittler, Hatef, Duin, Matas matches Crossref, which gives initials only (J., M., R.P.W., J.). The full given names in refs.bib are consistent with those initials and with Semantic Scholar ('Jiri Matas').
- **shapley1984optimizing** (correct). every field checked against Crossref 10.1007/BF00118940 and all match. Authors Lloyd Shapley and Bernard Grofman in that order; title exact; Public Choice 43(3):329-343, January 1984; DOI matches.
- **nitzan1982optimal** (correct). checked against Crossref/JSTOR 10.2307/2526438, which gives Shmuel Nitzan and Jacob Paroush, the exact title, International Economic Review 23(2), June 1982, and first page 289. Crossref records only the first page, so the last page (297) was confirmed from the RePEc/IDEAS record (pages 289-297). All bib fields match.
- **chair1986optimal** (correct). checked against Crossref 10.1109/TAES.1986.310699 and IEEE Xplore 4104179. Both list the authors only as initials, Z. Chair and P.K. Varshney, so the bib keeps the initials. Title exact; IEEE Transactions on Aerospace and Electronic Systems AES-22(1):98-101, January 1986. All fields match.
- **kuncheva2014weighted** (correct). checked against Crossref 10.1007/s10115-012-0586-6. Authors Ludmila I. Kuncheva and Juan J. Rodriguez (with accent); title exact ('A weighted voting framework for classifiers ensembles'); Knowledge and Information Systems 38(2):259-275. It was online 7 Dec 2012 and appeared in the February 2014 issue, so year 2014 is right. All fields match.
- **li2014error** (corrected). checked against arXiv 1411.4086 (API and abstract page) and DataCite DOI 10.48550/arXiv.1411.4086. Authors Hongwei Li and Bin Yu; exact title; v1 of 15 Nov 2014 is the only version; primary category stat.ML. No peer-reviewed version found in Crossref, OpenAlex or a web search; DBLP and Semantic Scholar could not be reached.
- **brummer2007fusion** (correct). checked against Crossref 10.1109/TASL.2007.902870 and IEEE Xplore 4291590. All 10 authors in the right order with matching given names; exact title, with STBU and NIST braced; IEEE Transactions on Audio, Speech, and Language Processing 15(7):2072-2084, September 2007. The diacritics are the authors' own spellings (BUT FIT record); Crossref and IEEE store ASCII forms.
- **ju2018relative** (correct). checked against Crossref 10.1080/02664763.2018.1441383. Authors Cheng Ju, Aurélien Bibaut and Mark van der Laan in that order; exact title; Journal of Applied Statistics 45(15):2800-2818. It was online 26 Feb 2018 and in the November 2018 issue, so year 2018 is right. All fields match.
- **kumar2022calibrated** (corrected). checked against PMLR v180 kumar22a (abstract page and official BibTeX). Authors, title, volume 180, pages 1041-1051 and year 2022 match. The booktitle is changed to PMLR's exact name, 'Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence'. PMLR assigns no DOI.
- **zhang2019adacos** (corrected). checked against Crossref 10.1109/CVPR.2019.01108 and OpenAlex; pages changed from the CVF PDF numbering 10823-10832 to the DOI record's 10815-10824; all other fields match.
- **wang2017normface** (corrected). checked against Crossref 10.1145/3123266.3123359 (title plus subtitle L<sub>2</sub>, pages 1041-1049, MM '17); fourth author's given names changed to 'Alan Loddon' as in the record.
- **wei2022logitnorm** (correct). every field (six authors in order, title, Proceedings of the 39th ICML, PMLR vol. 162, pp. 23631-23644, 2022) matches the PMLR page v162/wei22d. PMLR assigns no DOI.
- **wei2023logitclip** (correct). every field (seven authors in order, title, Proceedings of the 40th ICML, PMLR vol. 202, pp. 36868-36886, 2023) matches the PMLR page v202/wei23e. PMLR assigns no DOI.
- **wang2018cosface** (corrected). checked against Crossref 10.1109/CVPR.2018.00552 (pages 5265-5274 confirmed) and the CVF page; booktitle aligned with the DOI record's 'IEEE/CVF Conference on Computer Vision and Pattern Recognition'.
- **ranjan2017l2** (corrected). checked against the arXiv API and abstract page (cs.CV; v3 of 7 Jun 2017 is the latest) and DataCite 10.48550/arXiv.1703.09507; no peer-reviewed version in Crossref or OpenAlex (DBLP and Semantic Scholar unreachable); added eprint, archivePrefix, primaryClass and doi.
- **meng2022dsr** (corrected). checked against Crossref 10.1109/CVPR52688.2022.01212 and OpenAlex; pages changed from the CVF numbering 12444-12453 to the DOI record's 12434-12443; all other fields match.
- **chen2025knowing** (correct). every field (Jiechen Chen, Sangwoo Park, Osvaldo Simeone; exact title; IEEE Journal of Selected Topics in Signal Processing 19(1):88-102, January 2025; DOI) matches Crossref 10.1109/JSTSP.2024.3431996 and OpenAlex.
- **atarashi2025bounding** (corrected). authors and title checked against arXiv API and abs page for 2506.10572. The title is v2's (23 Feb 2026); v1 (12 Jun 2025) was 'Box-Constrained Softmax Function and Its Application for Post-Hoc Calibration'. Primary class stat.ML. No peer-reviewed version found on Semantic Scholar, OpenReview/DBLP-CoRR, Crossref or the Kyoto University researcher database (2026-11-12).
- **ncaltech101mirror** (corrected). DataCite 10.17632/cy6cvx3ryv.1 gives creator Garrick Orchard (Jayawant, Thakor and Cohen as contributors), title N-Caltech101, issued 2022-01-12. The Mendeley Data landing page and API cite four authors (Orchard, Jayawant, Cohen, Thakor), Mendeley Data, v1. Deposited by Garrick Orchard and linked to the Frontiers 2015 paper as its source; the record does not call itself a mirror.
- **orchard2015converting** (corrected). authors, title, journal, volume 9, year and DOI checked against Crossref 10.3389/fnins.2015.00437. The article number 437 is not in Crossref; it is taken from the Frontiers citation line ('Front. Neurosci. 9:437') and the JATS elocation-id, and is given as pages = {437}.
- **li2023dynamic** (corrected). authors and title checked against Crossref 10.1109/ICCV51070.2023.01228 (2023 IEEE/CVF ICCV, pp. 13304-13314, IEEE) and the CVF open-access BibTeX (pp. 13350-13360). DOI added and IEEE pagination used for consistency. arXiv 2303.10276 title differs ('by Dynamic Confidence').
- **chen2024agreeing** (correct). every field checked against Crossref 10.3390/e26020126. Authors Jiechen Chen, Sangwoo Park and Osvaldo Simeone; title exact; Entropy 26(2), article 126; published 2024-01-31; MDPI.
- **guo2017calibration** (corrected). checked against PMLR v70 (guo17a). Proceedings of the 34th International Conference on Machine Learning, PMLR 70, pp. 1321-1330, eds. Precup and Teh, 2017. No DOI exists.
- **horowitz2014energy** (corrected). checked against Crossref 10.1109/ISSCC.2014.6757323. Mark Horowitz, 2014 IEEE ISSCC Digest of Technical Papers, pp. 10-14, IEEE, San Francisco; Crossref prefixes the title with the paper number '1.1'.
- **fang2023spikingjelly** (correct). every field checked against Crossref 10.1126/sciadv.adi1480. All ten authors in order (Fang, Chen Y., Ding J., Yu, Masquelier, Chen D., Huang, Zhou, Li, Tian); title exact (plainnat's sentence case reproduces the published form); Science Advances 9(40), article eadi1480; published 2023-10-06.

The arXiv identifier stays in `note` for every arXiv entry because `plainnat` prints neither
`eprint` nor `archivePrefix`. N-Caltech101: the Mendeley Data record is the 2022 deposit, by
the dataset's first author, of the data described in Orchard et al. (2015); it links that
paper as its source and does not call itself a mirror. The paper calls it a redistribution
of the original data. Its DataCite metadata names one creator (Garrick Orchard) and three
further contributors; the repository's own citation, which `refs.bib` follows, lists all four
as authors.

## Still to verify

1. **Amir et al. CVPR 2017 — full author list** for DVS-Gesture. Only "Amir et al." and the
   title are confirmed, via SpikingJelly's docstring.
2. **Li et al. 2017 — full author list** for CIFAR10-DVS. Same situation.
3. **ATan surrogate gradient — originating paper. Still open; checked 2026-09-12.**
   SpikingJelly 0.0.0.0.14 documents `surrogate.ATan` with its formula,
   `g'(x) = alpha / (2 (1 + (pi/2 alpha x)^2))`, and **attaches no citation** — while the same
   module does name a proposing paper for other surrogates (S2NN, q-PseudoSpike, Log-tailed
   ReLU). The only lead in the package is `activation_based/model/parametric_lif_net.py`,
   whose header names **Fang, Yu, Chen, Masquelier, Huang & Tian, "Incorporating Learnable
   Membrane Time Constant to Enhance Learning of Spiking Neural Networks", arXiv:2007.05785**
   (authors from the arXiv record: Wei Fang, Zhaofei Yu, Yanqi Chen, Timothee Masquelier,
   Tiejun Huang, Yonghong Tian; 11 July 2020) and whose examples use `ATan`. Read: that
   paper **uses** `sigma(x) = (1/pi) arctan(pi x) + 1/2` in its experimental settings — exactly
   SpikingJelly's ATan at `alpha = 2`, the value this study uses — but **does not claim to
   propose it**. So its origin is unconfirmed. `paper/main.tex` describes the surrogate as the
   ATan function as implemented in SpikingJelly 0.0.0.0.14, gives the formula, and cites no
   paper for it. The attribution of surrogate-gradient training in general to other authors
   is not checked here and is not relied on.
4. **DVS-Gesture 1,176 / 288 split counts.** From SpikingJelly's docstring; not built.
5. **CIFAR10-DVS 10,000 sample count.** Same.
6. **The AOT-SNN classification paper**, if one exists, and whether temperature scaling is
   its calibration baseline.
   *Resolved 2026-10-13:* it is Sun, Yin & Bohté, arXiv:2304.10191, and it does compare against
   temperature scaling; see "Literature check, 2026-10-13". Its venue, given by a search as
   ICANN 2023, is not confirmed.
7. **Loihi compartment-state access cost** and **SpiNNaker recording bandwidth for a small
   output layer** — see "Could NOT be confirmed", item 6.
8. **Adam `lr = 1e-3`** — an unjustified constant, never tuned.
9. **LIF `tau = 2.0` as a readout parameter.** Adopted as a neuron default, but the leak
    experiment shows it sets the membrane readout's evidence window and therefore drives a
    reported result. No sweep has been run. Every statement about "a membrane readout" is a
    statement about `tau = 2`.
10. **N-Caltech101 class semantics** — the 101 = 100 + background composition is from the
    dataset homepage; only the counts were measured here.
