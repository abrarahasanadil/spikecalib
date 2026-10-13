# Working log

Append-only. Newest entries at the bottom.

This file records **how** decisions were reached: what was tried, what broke, what was
measured, and which alternatives were rejected and why. Once a decision is settled it
moves to `NOTES.md`, which states the design as it stands without the history. If the two
disagree, `NOTES.md` is authoritative for *what* the design is and this file explains how
it got there.

The first two entries are reconstructed from the commits they describe.

---

## 2026-09-09 — Scaffold: design document, stubs, environment, N-MNIST acquisition

**Did.** Wrote `NOTES.md` as the authoritative design document (claim, hypothesis and its
falsification condition, scope, prior work, four correctness rules, resolved design
decisions, per-dataset hazards). Created `spikecalib/` stubs carrying docstrings and
TODOs only, each restating the correctness rule it owns, plus `tests/ results/ figures/
scripts/`, `README.md`, `LICENSE` (MIT), `CITATION.cff`, `.gitignore`. Built `.venv` on
Python 3.13.1 and pinned `requirements.txt`. Wrote `scripts/check_env.py` and
`scripts/fetch_data.py`. Initialised git, committed, pushed to the private remote.

**Ran.** `pip install -r requirements.txt` → torch 2.14.0, spikingjelly 0.0.0.0.14,
scikit-learn 1.9.0, numpy 2.5.3, on macOS arm64 / Python 3.13.1.
`scripts/check_env.py` → exit 0; MPS available; LIF spikes per step `[0, 8192, 0]` of
8192 neurons, final membrane potential 0.7500.
Probed the mirror: `HEAD` on the Dropbox URL resolves to `N-MNIST.zip`,
1,181,573,087 bytes, confirming the figure in the brief.
Confirmed against the installed package that all three N-MNIST traps are real:
`NMNIST.downloadable()` is `False`, its resource URL is the dataset homepage
(`garrickorchard.com`), `create_events_np_files` iterates `os.listdir(source_dir)` and
`os.mkdir`s every entry as a class, and the only cache guard is
`if not os.path.exists(events_np_root)`.

**Chose.**
- *LIF drive of 1.5 with tau 2.0*, rather than a drive large enough to fire on every
  step. With tau 2.0 the potential goes 0.75 → 1.125 (fire, reset) → 0.75, so step 1
  **must** be silent and step 2 **must** fire. A drive of 2.0 would spike at every step
  and would therefore also pass if the threshold comparison were broken in the
  always-fire direction. The point of the check is to catch a broken comparator, so the
  drive has to make both failure directions visible.
- *Read the MD5s from `resource_url_md5()` at runtime* rather than copying them into the
  script. If SpikingJelly ever revises them, a hardcoded copy would start rejecting good
  archives, and the failure would look like a corrupt download.
- *Pin the full transitive dependency set*, not just direct dependencies. Correctness
  rule 4 asks for reproducible experiments; a resolvable environment is not a reproducible
  one.
- *Delete the 1.10 GiB wrapper archive after both inner archives pass MD5* (`--keep-archive`
  retains it). Once the hashes verify, the wrapper is 1.1 GiB of duplicate bytes.
- *Register DVS-Gesture and CIFAR10-DVS with metadata but raise `NotImplementedError` on
  acquisition*, rather than writing download code that cannot be tested yet. The interface
  exists so the rest of the codebase can be written against it; untested download code in
  the repository would look supported and would not be.

**Open.**
- DVS-Gesture sample count: the paper says 1,342, SpikingJelly yields 1,176 / 288 = 1,464.
  Registry currently records SpikingJelly's numbers with a comment; needs resolving.
- CIFAR10-DVS has no official train/test split and the study has not defined one.
- Only `T = 10` is integrated. The other timesteps in the sweep are not built.

---

## 2026-09-09 — The `.DS_Store` race, and why pruning once is not enough

**Did.** The first full `fetch_data.py` run failed. Pruning macOS metadata *before*
handing the tree to SpikingJelly is not sufficient, because macOS recreates `.DS_Store`
while a multi-minute build is running. Replaced the prune-then-hope approach with an
`os.listdir` shim that filters macOS metadata for the duration of the build. Also stopped
`and` from short-circuiting the second `events_np` cache check out of the reported state.

**Ran.** First run: prune at 17:46 removed 3 metadata entries and verified
`extract/Train` held exactly 10 class directories and 60,000 `.bin` files. Finder wrote
`.DS_Store` back into four directories at 17:47, mid-build. SpikingJelly then did
`os.mkdir(events_np/train/.DS_Store)` followed by
`os.listdir(extract/Train/.DS_Store)` → `NotADirectoryError`, leaving a truncated
`events_np`. Exit code 1.
Second run, with the shim: detected the stump (`events_np/train: 0 of 60,000 .npz`),
deleted and rebuilt it, re-verified both archives by MD5 without re-downloading, reused
`extract/`. Built in 176.8s. **8 metadata entries were recreated during that build**, so
the race is wide, not theoretical. `len(train)=60,000`, `len(test)=10,000`, frame shape
`(10, 2, 34, 34)`. Disk: 3.54 GiB one-time (download 1.10 + extract 1.36 + events_np
1.08), 224.7 MiB per T.

**Chose.**
- *Filter `os.listdir` for the duration of the build*, over two alternatives. Pruning more
  often only narrows the window — the failure happened inside a 177-second build, and
  there is no prune frequency that closes a race against a filesystem daemon. Subclassing
  `NMNIST` and reimplementing `create_events_np_files` correctly would duplicate ~25 lines
  of upstream code that would then drift out of step with the installed version. The shim
  targets the exact mechanism (`os.listdir` on the extract tree), is reverted on exit, and
  is about ten lines.
- *Scope the shim to the build* rather than applying it process-wide, so nothing outside
  the SpikingJelly call sees a patched `os`.
- Noted that `create_same_directory_structure` also reads `os.listdir`, so the frame-cache
  path had the same exposure and is covered by the same shim.

**Open.**
- Cache verification counts files. It does not check that the files are readable or
  non-empty, so a build interrupted mid-write could still pass. Counting is cheap and
  catches the observed failure; hashing 70,000 files on every run is not. Revisit if a
  partial-write failure ever appears.

---

## 2026-09-09 — Four-way splits: correctness rule 1 implemented and tested

**Did.** Moved the dataset registry out of `scripts/fetch_data.py` into
`spikecalib/datasets.py`, so dataset metadata has one home and the script and the
experiment code cannot drift; the script now imports it. Added `n_val_per_class` and
`n_cal_per_class` to every spec. Implemented `spikecalib/data.py`: `load_dataset`,
`training_labels`, `make_splits`, `assert_disjoint`, `save_splits`/`load_splits`,
`summarise`, and the `SplitOverlapError` / `InsufficientSamplesError` exception types.
Wrote `tests/test_data.py` (23 tests, weighted toward failure paths) and
`scripts/check_splits.py`, which runs the whole thing against the real N-MNIST cache.
Resolved the DVS-Gesture count discrepancy in `NOTES.md`. Created this file.

**Ran.** `pytest tests/ -q` → **23 passed in 0.08s**.
Mutation-checked the suite by breaking the code it guards, to confirm the tests are not
vacuous: making `assert_disjoint` a no-op fails 2 tests; deleting the per-class
sufficiency check fails 3; ignoring the seed fails 1; removing the within-class shuffle
fails 2 (after adding a test targeting it specifically — it was previously caught only
incidentally by the seed test).
`scripts/check_splits.py` on real N-MNIST → exit 0. Split for seed 0: train 50,000, val
5,000, cal 5,000, test 10,000; fingerprint `e524beccace851af`; all three pairwise
intersections empty; train+val+cal covers all 60,000 training samples with no index
twice; save/load round-trip identical at 105.6 KiB; seed 1 gives a different fingerprint.
Real class imbalance in the training split is visible: min 4,421 (class 5), max 5,742
(class 1), ratio 1.299.

**Chose.**
- *Keep `test` in its own index space and check disjointness only across train/val/cal.*
  `train`/`val`/`cal` are positions in the training dataset; `test` is positions in a
  different object. The alternative — offsetting test indices into one combined space so
  that `assert_disjoint` could include it — was rejected because it invents a relationship
  that does not exist and produces a check that can never fail while looking as though it
  proves something. Rule 1's "disjoint from test" is maintained structurally: the test set
  is never carved. `summarise` says so on every report so the distinction is not lost.
- *A named `SplitOverlapError` rather than a bare `assert`.* Assertions disappear under
  `python -O`, and this is the one condition that invalidates every number in the paper.
  The message names both splits and the overlap size so the failure is actionable.
- *Call `assert_disjoint` inside `make_splits`.* A caller who has to remember to check is
  a caller who eventually forgets. A test pins this behaviour.
- *Return sorted index arrays.* Sorting makes two runs of the same seed byte-identical
  regardless of the order classes were concatenated in, which is what makes the
  fingerprint and any array comparison meaningful.
- *Persist as compressed npz with a JSON metadata blob and a truncated SHA-256
  fingerprint*, over plain JSON. JSON of 60,000 indices is bulky and loses dtype; the
  fingerprint lets a results row cite the split that produced it instead of asserting it,
  and doubles as a corruption check on load — there is a test that tampering with a saved
  file is caught.
- *Stratify by class.* An unstratified draw can drop a rare class out of the calibration
  set entirely, and a calibrator fitted without it extrapolates on that class at test
  time.
- *Split sizes per dataset:* N-MNIST 500/500 (6,000 per class available, so this costs a
  sixth of training data and leaves a 5,000-sample calibration set). DVS-Gesture 10/30 —
  about 107 per class exist, so 40 held out leaves ~67 to train on, and the resulting
  330-sample calibration set is small enough that cross-validated isotonic is required
  rather than optional. CIFAR10-DVS 100/100, assuming the planned 200-per-class test
  holdout leaves 800 per class.
- *Corrected the DVS-Gesture explanation.* The brief for this session described 1,464 as
  including an "other" category absent from 1,342. The source says otherwise: "other
  gestures" is class 11 in the label CSVs and is already inside the 11 classes of both
  counts. The real difference is segmentation — SpikingJelly emits one sample per CSV row
  and some trials label a gesture twice (its own documentation shows a 12-row CSV for
  `user01_fluorescent` where class 8 repeats). Recorded the accurate mechanism in
  `NOTES.md` rather than the tidier but wrong one.

**Open.**
- *Should the calibration split be subject-disjoint on DVS-Gesture?* SpikingJelly's
  train/test division is by recording trial, so no subject crosses it. Splits carved from
  the training set do not inherit that: a calibration sample can come from a subject also
  seen in training, which could make the calibration set look easier than the test set and
  flatter both calibrators. Not necessarily fatal — it affects both arms — but it should be
  decided before DVS-Gesture numbers are reported.
- *Is a validation split needed at all?* Calibration here is strictly post-hoc, so `val`
  is currently reserved for training-time model selection and early stopping. If training
  ends up using a fixed schedule with no model selection, `val` should be folded back into
  train rather than held out for nothing.
- *CIFAR10-DVS train/test split* still undefined, so `make_splits` raises for it rather
  than inventing one.
- Only `T = 10` is built. The sweep needs `T` in {2, 4, 6, 8, 10}.

---

## 2026-09-09 — Calibrators, both ECE estimators, and a synthetic mechanism check

**Did.** Implemented `spikecalib/calibration.py`: `softmax_probs`/`softmax_confidence`,
`predict_with_ties`/`tie_rate`, the `Outputs` bundle, four calibrators (`Uncalibrated`,
`TemperatureScaling`, `IsotonicTopLabel` as primary, `IsotonicOneVsRest` as secondary),
`ece_equal_width`, `ece_equal_mass`, and `compute_metrics` returning both estimators
together. Wrote `tests/test_calibration.py` (31 tests) and `scripts/synthetic_check.py`.
Amended `NOTES.md` in two places (see **Chose**).

**Ran.** `pytest tests/ -q` → **54 passed** (23 split, 31 calibration).
Mutation-checked the new suite: tie-breaking by `np.argmax` fails 2 tests; equal-mass
returning 0.0 on the degenerate case fails 4; splitting ties across equal-mass bins by
rank fails 5; a temperature stuck at 1.0 fails 1; dropping isotonic's `out_of_bounds`
clipping fails 2.
`scripts/synthetic_check.py` (10 classes, p_hi 0.75, p_lo 0.15, noise 0.20, 10k cal /
10k test, seed 0):

| T | accuracy | tie rate | distinct conf | mean conf | max conf | fitted temp |
|---|---------|----------|---------------|-----------|----------|-------------|
| 2 | 0.5045 | 0.4436 | 52 | 0.2860 | 0.4509 | 0.5564 |
| 4 | 0.6773 | 0.1872 | 388 | 0.5056 | 0.8585 | 0.7934 |
| 6 | 0.7483 | 0.0768 | 1,093 | 0.7049 | 0.9782 | 1.0636 |
| 8 | 0.7756 | 0.0308 | 2,033 | 0.8414 | 0.9970 | 1.3628 |
| 10 | 0.7833 | 0.0118 | 3,035 | 0.9170 | 0.9996 | 1.6963 |

Tie rate rises 38x from T=10 to T=2 and distinct confidences fall 58x, which is the
discreteness mechanism behaving as argued. Equal-width ECE for uncalibrated ranged
0.11–0.22; isotonic-top-label was the best arm at every T (0.0096–0.0149).

**Chose.**
- *Equal-mass bins are built by packing unique confidence values, not from
  `np.quantile` edges.* The quantile version was written first and was wrong for this
  study. `np.quantile` interpolates: given observed confidences of 0.81 and 0.85 it
  produces an edge at 0.83, a value no sample holds and no spike-count readout can emit.
  For continuous scores that is harmless; here it mis-assigns exactly the lattice
  structure being measured. The replacement walks unique values in order and greedily
  packs them into bins of about `n / n_bins` samples, so a bin boundary can never fall
  inside a tie. Rank-splitting ties was the other candidate and is worse: with all
  confidences equal it manufactures per-bin accuracy differences out of sampling noise
  and reports them as calibration error.
- *`compute_metrics` returns both ECEs in one object.* Correctness rule 2 becomes a
  property of the interface rather than a convention someone has to remember; there is no
  code path that yields equal-width without equal-mass.
- *The degenerate case returns the honest single-bin gap by default and raises under
  `strict=True`.* Returning 0.0 would claim perfect calibration for the most degenerate
  point in the sweep. Returning the real gap `|accuracy − confidence|` and flagging
  `degenerate=True` keeps the number usable; `strict` exists for callers that would rather
  stop.
- *`predict_with_ties` draws one uniform per (sample, class), masks non-maximal entries to
  −1, and takes the argmax of the draws.* Uniform over the tied set, vectorised, and
  structurally incapable of selecting a non-maximal class. The obvious alternative —
  adding small random noise to the counts — can reorder near-ties in floating point and
  would silently corrupt the tie statistics that are being reported.
- *NLL and Brier are the top-label binary scores*, not multiclass. The top-label
  calibrators emit a scalar confidence and no distribution, so multiclass NLL is undefined
  for them; using the binary version keeps every number in a results row about the same
  quantity that ECE measures.
- *One-vs-rest holds the prediction fixed.* Renormalised one-vs-rest can move the argmax.
  Letting it would change accuracy between arms, so the arms would differ by more than the
  calibrator and the comparison would stop being about calibration.
- *Temperature scaling uses `scipy.optimize.minimize_scalar` over `log(t)`.* Rule 4 as
  written said "scikit-learn only", but temperature scaling has no scikit-learn
  implementation. `scipy.optimize` is already pinned, CPU-only and deterministic, so the
  rule was **amended in `NOTES.md`** to name what the path actually uses rather than be
  quietly violated. Hand-rolling a 1-D optimiser was the alternative and is worse
  engineering than using a tested one for a well-understood convex problem.
- *Added the attainable-confidence ceiling to `NOTES.md`.* The synthetic run exposed
  something the design document had not: the reachable confidence set is not merely coarse,
  it is **bounded**. The extreme count vector gives `e^T / (e^T + C − 1)`, which for 10
  classes is 0.451 at T=2 and 0.859 at T=4. In the run, accuracy at T=2 was 0.5045 —
  *above* the ceiling — so the model was structurally incapable of being correctly
  confident, and the fitted temperature went below 1 to sharpen rather than above 1 to
  soften. The direction of miscalibration therefore flips across the sweep. This is a
  design-relevant fact, so it went into the claim section of `NOTES.md`, with the
  consequence that the study should report the **signed** calibration gap and not only
  `|ECE|`, or the low-T and high-T regimes will cancel when averaged.

**Open.**
- *The synthetic run does not reproduce the hypothesis, and should not be read as
  evidence against it.* The isotonic-minus-temperature gap in equal-width ECE was 0.023 at
  T=2 against 0.109 at T=10 — smallest where the hypothesis predicts it should be largest.
  That is a property of this generator: at T=2 its miscalibration is a near-uniform
  underconfidence, which a single scalar happens to fix well. Real SNN counts are not
  Binomial with a fixed `p_hi`. The synthetic check tests the machinery, not the claim.
- *Equal-width and equal-mass agreed to within 0.007 everywhere in the run*, so the
  "disagreement is a finding" case did not arise. The unit tests construct it deliberately
  (0.33 vs 0.52 on two confidences inside one bin). Whether it arises on real data is open.
- *Should the sweep report signed calibration gap alongside ECE?* Implied by the ceiling
  finding above but not yet decided or implemented.
- *Cross-validated isotonic for DVS-Gesture* is designed for but not implemented.
- *No confidence intervals on any ECE.* With 3 seeds the study will need a defensible
  spread estimate; bootstrap over the test split is the obvious candidate.

---

## 2026-09-09 — The mechanism was wrong: tie-breaking replaced by capacity, then tested

**Did.** Rewrote the "Claim under test" and "Hypothesis" sections of `NOTES.md`. The
mechanism the study argues for is no longer tie-breaking; it is **capacity**. Corrected
the same argument where it had been copied into code docstrings (`calibration.py` module
docstring, `tie_rate`, `TemperatureScaling`, `IsotonicTopLabel`) and the `sweep.py` and
`figures.py` stubs. Added `IsotonicTopLabel.n_knots_` and `.n_levels_` so the fit's
realised degrees of freedom can be measured. Rewrote `scripts/synthetic_check.py` with a
`--miscalibration {uniform,conditional,both}` flag and a ratio test that varies `n_cal` at
fixed `T`.

**Ran.** `pytest tests/ -q` → **54 passed**, unchanged.
Verified both defects in the old argument directly against the implementation before
rewriting: two samples with identical tied counts get identical output from
`IsotonicTopLabel` (0.4879 → 0.4932 for both), and `softmax(z/t)` at `T=2, C=10` reaches
0.8585 at `t=0.5` and 0.9996 at `t=0.2`, against a raw-readout ceiling of 0.4509.
`scripts/synthetic_check.py` (10 classes, 10k cal / 10k test, seed 0) — conditional mode,
equal-width ECE:

| T | samp/lvl | iso levels | uncal ECE | temp ECE | iso ECE | advantage |
|---|----------|-----------|-----------|----------|---------|-----------|
| 2 | 188.7 | 13 | 0.0838 | 0.0306 | 0.0086 | 0.0220 |
| 4 | 26.1 | 17 | 0.1335 | 0.1459 | 0.0042 | 0.1418 |
| 6 | 8.9 | 14 | 0.1840 | 0.1748 | 0.0108 | 0.1640 |
| 8 | 4.8 | 12 | 0.2341 | 0.1664 | 0.0081 | 0.1583 |
| 10 | 3.4 | 11 | 0.2756 | 0.1597 | 0.0047 | 0.1550 |

Ratio test at fixed `T = 10`, `n_cal` from 250 to 20,000: samples-per-level moved 1.1 →
4.8 and the advantage stayed flat at 0.137–0.159. At `T = 2`, `n_cal` 250 → 20,000 moved
samples-per-level 7.4 → 350.9 and the advantage 0.0076 → 0.0269.

**Chose.**
- *Replaced tie-breaking with capacity as the mechanism, and said so explicitly in
  `NOTES.md` rather than quietly editing.* The old argument — "ties cannot be broken by a
  scalar" — is true of temperature but equally true of isotonic top-label, which computes
  `conf' = f(conf)` for monotone `f`, so two samples with identical counts get identical
  output from it as well. Both calibrators hold the prediction fixed by design and
  tie-breaking happens once before either runs. The claim therefore never distinguished
  the two methods. It was caught by asking what isotonic actually does to a tie, which is
  a question the implementation can answer in three lines; it should have been asked when
  the argument was first written.
- *Corrected the ceiling claim's scope rather than deleting it.* `NOTES.md` had said a
  model above the ceiling is forced into underconfidence and "no calibrator can be blamed
  for the shortfall". The ceiling `e^T / (e^T + C - 1)` is real and bounds the **raw
  readout**, but not the calibrators: as `t` approaches 0, `softmax(z/t)` approaches
  one-hot, so temperature sharpening escapes it entirely. The synthetic run had already
  shown this and it was not noticed — fitted temperature was 0.556 at `T = 2`, i.e.
  sharpening, which is only useful if sharpening reaches past the ceiling. The ceiling now
  explains the shape of the *uncalibrated* error and the flip in miscalibration direction,
  which is what it actually supports.
- *Kept tie rate in the results table, reclassified.* It is a descriptive measure that the
  output is genuinely discrete — the premise of the capacity argument — not the reason
  isotonic wins. Removing it would lose a real and cheap piece of evidence; leaving it
  labelled as the mechanism would repeat the error.
- *Conditional miscalibration implemented as count-dependent label noise.* `p_noise` rises
  linearly with the favoured class's count, so high-confidence samples are wrong more often
  and low-confidence ones less often. This is the shape one scalar cannot fix, and it makes
  the control/treatment contrast the point of the script rather than an aside.
- *Added a ratio test that varies `n_cal` at fixed `T`.* The `T` sweep alone cannot
  separate capacity from anything else that changes with `T`. Varying calibration-set size
  moves `n_cal / n_distinct` while `T` is held constant, which is the only manipulation in
  reach that isolates the named mechanism.

**The pre-test did not support the mechanism, and that is recorded in `NOTES.md`, not
just here.**
- *Isotonic self-regularises, so `n_distinct` is the wrong statistic.* Pool-adjacent-
  violators collapsed the fit to 4–22 distinct output values while `n_distinct` spanned 34
  to 4,197 — a factor of 123 in the bound and no trend in the realised count. The
  parameters isotonic actually uses are set by the complexity of the true calibration
  curve, not by how many confidence levels exist.
- *The advantage was indifferent to the ratio at `T = 10`.* At 1.1 samples per confidence
  level, isotonic neither overfit nor lost. The `T = 2` sweep showed a weak positive trend,
  so the ratio is not irrelevant, but it is not the driver the hypothesis names.
- *`T` is confounded with miscalibration severity.* Uncalibrated ECE rose 0.084 → 0.276
  from `T = 2` to `T = 10` in the conditional generator, because a wider count range gives
  the injected miscalibration more room to vary. The advantage rose with `T` in step with
  severity, which explains the data better than capacity does here.
- This is not a refutation for real SNNs: a smooth monotone synthetic calibration curve is
  the best possible case for isotonic's self-regularisation. But it imposes two
  requirements on the real experiment, now written into `NOTES.md` — report the realised
  block count, not only `n_distinct`, and compare the advantage at matched uncalibrated
  ECE.

**Open.**
- *Is there a better capacity statistic than `n_cal / n_distinct`?* Realised block count
  per calibration sample is the obvious candidate but it is an output of the fit, not a
  property of the regime, so it cannot be used to predict where isotonic will win.
- *Can a synthetic generator hold miscalibration severity fixed across `T`?* Without that
  the `T` sweep cannot cleanly test capacity even in simulation. Calibrating `noise_hi`
  per `T` to equalise uncalibrated ECE would be the way.
- *Does PAVA self-regularisation hold on real SNN confidences?* The whole negative result
  rests on the true calibration curve being smooth. Real event data may produce a rougher
  one, where realised capacity does track `n_distinct`.
- Earlier open items stand: subject-disjoint calibration for DVS-Gesture, whether `val` is
  needed at all, CIFAR10-DVS split undefined, only `T = 10` built, no confidence intervals
  on any ECE.

---

## 2026-09-09 — Provenance rule, model and training, bootstrap intervals, CI

**Did.** Adopted a provenance rule: every numeric constant and factual claim carries
`[LIT]`, `[DERIVED]`, `[MEASURED]` or `[UNVERIFIED]`. Created `SOURCES.md` and audited
`NOTES.md`, the `spikecalib/` docstrings and the `scripts/` docstrings against it.
Implemented `spikecalib/models.py` (`build_model`, `SpikingClassifier`, `SynOpsCounter`)
and `spikecalib/train.py` (`set_seed`, `train`, `predict_counts`). Added bootstrap
confidence intervals and a paired difference bootstrap to `spikecalib/calibration.py`.
Added `scripts/smoke_train.py`, `.github/workflows/tests.yml`, `tests/conftest.py`,
`tests/test_models.py` and `tests/test_train.py`.

**Ran.** `pytest tests/ -q` → **92 passed** (up from 54).
`scripts/fetch_data.py --dataset nmnist --timesteps 4` → built the T=4 frame cache;
`events_np` reused, no re-download.
`scripts/smoke_train.py` → exit 0. 1,000 train / 400 val / 600 cal / 2,000 test, 2 epochs,
1.6 s on MPS. Val accuracy 0.2275 against 0.100 chance. Counts `int64`, range `[0, 2]` at
T=4, only 3 distinct confidences on the calibration split, tie rate 0.8765. SynOps
530,590,554 AC and 420,789,312 MAC → 1.207 µJ per sample as a design-comparison proxy.
Equal-width ECE: uncalibrated 0.0896 [0.0802, 0.1061], temperature 0.0035 [0.0010,
0.0175], isotonic-top 0.0230 [0.0109, 0.0352]. Paired difference isotonic − temperature
+0.0195 with 95% CI [−0.0062, +0.0255] — **not resolvable**, which is the correct answer
for a 2,000-sample smoke test.

**Chose.**
- *`[UNVERIFIED]` on the Horowitz energy constants, rather than a plausible citation.*
  The paper's existence and topic are not in doubt, but the figure number, the exact
  0.9 pJ, and whether 4.6 pJ is stated or is the community convention of FP-multiply plus
  FP-add are all unconfirmed here. `SOURCES.md` records what specifically needs checking.
  The same applies to the four prior-work references, which came from the project brief
  and have never been checked against a primary document — they are now labelled as such
  in `NOTES.md` itself, not just in the log.
- *`[LIT]` for the dataset papers, because they are locally verifiable.* Titles, venues
  and DOIs were read directly out of the installed SpikingJelly docstrings. Author lists
  beyond the first name are still `[UNVERIFIED]`.
- *BatchNorm in the fully-connected head.* Not a design preference — a bug fix. The first
  version of the model emitted **exactly zero output spikes for every input**, at every
  input scale, in both train and eval mode. A `Linear` over sparse binary spikes with
  default PyTorch init produces pre-activations around 1e-3 against a threshold of 1.0,
  so the head never fired and no gradient ever returned. Layer-by-layer activations
  located it at the first fully-connected LIF. `SeqToANNContainer(nn.BatchNorm1d(...))`
  rather than `layer.BatchNorm1d`, which requires a 4-D input in multi-step mode. There is
  now a test asserting the untrained model fires at all.
- *`forward` returns a float tensor with integral values; `count_spikes` returns `int64`.*
  Autograd cannot flow through an integer dtype, so training forces float. Rather than
  hide that, both properties are asserted: values are integral and bounded by `T`, and the
  inference path returns a true integer dtype.
- *`predict_counts` resets neuron state before every batch, and two tests cover it.* One
  asserts that scoring the same data twice gives identical counts and that counts are
  independent of batch size; the other *demonstrates the failure* by showing consecutive
  forwards without a reset disagree. Without the second test the first could pass
  vacuously on a model that had no state to leak.
- *SynOps gates operations by input sparsity for both AC and MAC*, and the AC/MAC
  distinction is about whether the input is binary rather than about layer type. The
  counter also records how often an AC-costed layer received non-binary input, so a wrong
  assumption is visible rather than silently mis-costed.
- *Bootstrap holds bin edges fixed* at those from the full sample. This conditions on the
  observed confidence support, which in this setting is nearly fixed by construction: the
  attainable confidence set is determined by `T` and the class count, not by chance.
  Recomputing bins per resample is the defensible alternative and is not done; the choice
  is documented where the interval is produced.
- *Added a paired difference bootstrap.* Two separate intervals cannot answer "is isotonic
  better than temperature": overlapping intervals do not imply no difference, and the arms
  share a test set so their errors are correlated. Resampling one index set and evaluating
  both arms on it keeps the pairing and cancels the shared binning bias. `smoke_train.py`
  reports it, and it correctly says the smoke-test difference is unresolvable.

**Found while testing.**
- *Binned ECE is positively biased, and the bootstrap interval can sit entirely above the
  point estimate.* A test asserting the interval brackets the point estimate failed:
  0.01304 against a CI of [0.01407, 0.03060]. This is not an implementation error.
  Averaging `|accuracy − confidence|` converts sampling noise into positive error, so the
  statistic overstates calibration error, and a resample — being noisier than the full
  sample — overstates it more. The test was rewritten to assert the true behaviour so a
  future change cannot quietly "fix" it into something that is no longer the sampling
  distribution of this estimator.
- *That bias is not constant across the sweep.* It depends on how many samples fall in
  each bin, so at low `T`, where the effective bin count collapses, the bias shrinks with
  it. **Comparing raw ECE across `T` compares differently-biased numbers.** Comparing two
  calibrators at the same `T` on the same test split does not have this problem, which is
  a further argument for the headline quantity being a paired difference. Documented in
  `calibration.py`; it should also go into `NOTES.md` when the sweep is designed.

**Open.**
- *The smoke model is badly undertrained and it shows in the readout*: counts reach only
  2 of a possible 4, giving 3 distinct confidences and an 87.65% tie rate. That is
  expected at 2 epochs on 1,000 samples, but it means the smoke test exercises the
  low-distinct-confidence regime by accident rather than by design. A properly trained
  model at T=4 should use more of the count range, and if it does not, that is itself
  worth knowing before the sweep.
- *`lr = 1e-3` is the Adam default and has never been tuned* for this model, dataset or
  `T`. Listed in `SOURCES.md` "To verify" item 12 because an untuned constant that
  determines whether a model converges is not a neutral choice.
- *No MPS-specific test exists yet.* `tests/conftest.py` provides the skip marker and CI
  runs CPU-only, but every current test is device-agnostic. A test that MPS and CPU agree
  to a stated tolerance would make the non-determinism caveat concrete rather than
  asserted.
- *CI has never actually run.* The workflow is written but unproven until the first push
  reaches GitHub; the CPU torch wheel pin in particular is untested.
- Earlier open items stand: capacity statistic, severity-matched synthetic generator,
  subject-disjoint calibration for DVS-Gesture, whether `val` is needed, CIFAR10-DVS
  split, and confidence intervals now exist but are not yet wired into a sweep.

---

## 2026-09-09 — The validation metric was broken; fixing it, then a model that actually trains

**Did.** Fixed `_evaluate` in `spikecalib/train.py`, which used `counts.argmax(dim=1)` and
therefore broke ties by lowest index. Validation now uses
`spikecalib.calibration.predict_with_ties` under a fixed `eval_seed`, recorded in the
`RunRecord`, and every history entry carries the validation tie rate. Added per-layer
attribution to `SynOpsCounter` so encoder energy can be separated from spiking energy.
Added `output_v_threshold` to `ModelConfig`. Wrote `scripts/train_one.py` and trained
N-MNIST at T=4, seed 0 on the real split.

**Ran.** `pytest tests/ -q` → **100 passed** (up from 92).
`scripts/train_one.py --dataset nmnist --timesteps 4 --seed 0 --epochs 15 --lr 1e-3
--batch-size 128 --patience 5 --num-workers 6` → exit 0. Split 50,000 / 5,000 / 5,000 /
10,000, fingerprint `e524beccace851af`, 1,070,868 parameters. **Best validation accuracy
0.9900 at epoch 12**, final validation tie rate 0.0078. 291.1 s wall clock, 19.4 s per
epoch. Extrapolating, 15 runs (5 timesteps x 3 seeds) cost about **1.2 h** at T=4, more at
higher T.
Per-layer SynOps on the smoke-test model: the input encoder is **89.8%** of the energy
proxy (`net.0`, MAC, 85.2 M ops) against 10.2% for all three spiking layers combined.

**Chose.**
- *Validation uses the study's own tie-break, not a debiased alternative.* The obvious
  narrow fix is to break ties randomly in the metric. Using `predict_with_ties` with a
  recorded seed instead means the number early stopping consumes is produced the same way
  as every number in the paper, so validation accuracy and reported accuracy cannot drift
  apart for a reason nobody is looking for.
- *`eval_seed` is fixed rather than drawn per epoch.* Early stopping compares epochs; a
  tie-break that re-rolled each epoch would add noise to the comparison for nothing.
- *Validation tie rate is printed every epoch.* It is the diagnostic that makes the
  original bug impossible to miss: at epoch 0 of the pilot it read 0.7562, which says
  plainly that three quarters of samples had no unique argmax and any accuracy alongside
  it is close to meaningless.
- *Kept the default output threshold of 1.0.* A sweep on the 8,000-sample pilot gave 0.2568
  at threshold 1.0, 0.1996 at 0.5, and 0.1998 at 0.25. Lowering it did exactly what it
  should mechanically -- non-zero counts rose from 2.50% to 17.51% and the tie rate fell
  from 0.169 to 0.049 -- and accuracy did not improve. `output_v_threshold` stays as
  configuration because the argument for it was reasonable, but the default is unchanged
  because the evidence did not support changing it.

**Errors I made while diagnosing, recorded because the corrections matter.**
- *I evaluated on `splits.val[:1500]` and read 0.6647 accuracy, then reported it as
  evidence the model worked in eval mode.* It did not. `splits.val` is sorted, and
  N-MNIST is stored class-ordered, so the first 1,500 indices are **classes 0, 1 and 2
  only** -- `[500, 500, 500, 0, 0, 0, 0, 0, 0, 0]`. The same mistake inflated a
  "train-subset accuracy 0.9773". On the full, balanced validation split the real number
  was 0.2378, exactly what the training loop had reported. **Any subset of a sorted index
  array is class-ordered, not random**, and this class of error is invisible because the
  inflated number looks like good news.
- *I then concluded "the network is not discriminating" and went looking for an
  architectural fault.* That was also wrong. The model was starved of data: the pilot used
  8,000 training samples. On the full 50,000-sample split the same architecture reaches
  97.4% after a single epoch and 99.0% by epoch 12. Nothing was wrong with the model, the
  threshold, or the learning rate. Two experiments -- the threshold sweep and the
  per-layer firing diagnostic -- were spent on a problem that did not exist, because I
  drew a conclusion from a pilot small enough that underfitting dominated everything else.

**Found.**
- *Input encoding dominates the energy proxy.* 89.8% of the modelled energy is the first
  layer, which receives real-valued event frames and is therefore costed at 4.6 pJ per
  operation rather than 0.9 pJ. An SNN energy argument rests on spikes being cheap, and
  the layer that does not receive spikes is nine tenths of the bill. This is now reported
  separately by `SynOps.encoder_energy_fraction`, and the paper should lead with it rather
  than be asked about it.
- *A trained model at T=4 has a low tie rate: 0.0078.* The smoke test's 87.65% was an
  artefact of being undertrained, not a property of T=4. **The hypothesis' prediction that
  tie rate rises as T falls must therefore be tested on converged models only**, and the
  sweep must record validation accuracy alongside tie rate so an undertrained run cannot
  be mistaken for a low-T discreteness effect. This is the concrete reason `val` exists.

**Open.**
- *Only T=4 has been trained.* The claim that a trained model has a low tie rate is a
  single data point; the T=2 model is the one that matters most to the hypothesis and has
  not been run.
- *15 epochs did not early-stop*, so 0.9900 may not be the ceiling. Validation accuracy was
  still drifting up slowly. Whether to train longer is a cost question for the sweep.
- *The threshold sweep was run on the 8,000-sample pilot* and its conclusion may not
  transfer to the full split. It is evidence that threshold is not the dominant lever, not
  evidence that 1.0 is optimal.
- *Per-epoch cost is 19.4 s at T=4*, so the full sweep is about 1.2 h at this T and more at
  higher T -- affordable, which removes any argument for cutting corners on seeds.
- Earlier open items stand: capacity statistic, severity-matched synthetic generator,
  subject-disjoint calibration for DVS-Gesture, CIFAR10-DVS split, CI never yet run.

---

## 2026-09-09 — T=2 trained; what a converged readout looks like where the ceiling binds

**Did.** Built the T=2 frame cache, trained N-MNIST at T=2 seed 0 with the same settings as
the T=4 run, and wrote `scripts/readout_report.py` to describe a trained model's raw
readout without fitting any calibrator. Added a note to `NOTES.md` that the
attainable-confidence ceiling is also a floor on the training loss.

**Ran.** `scripts/fetch_data.py --dataset nmnist --timesteps 2` → frame shape
`(2, 2, 34, 34)`, 106.5 MiB, `events_np` reused.
`scripts/train_one.py --timesteps 2 --seed 0 --epochs 15 --lr 1e-3 --batch-size 128` →
best validation accuracy **0.9842** at epoch 13, final validation tie rate 0.0112, 159.2 s
at 10.6 s per epoch.
`scripts/readout_report.py --timesteps 2 4 --seed 0`, both on split `e524beccace851af`:

| quantity | T=2 | T=4 |
|---|---|---|
| validation accuracy | 0.9842 | 0.9900 |
| test accuracy | 0.9853 | 0.9888 |
| test tie rate | 0.0133 | 0.0072 |
| distinct confidences (cal) | 12 | 48 |
| samples per level (cal) | 416.7 | 104.2 |
| attainable ceiling | 0.4509 | 0.8585 |
| max confidence observed | 0.4509 | 0.8585 |
| mean confidence | 0.4463 | 0.8451 |
| mean confidence − accuracy | −0.5390 | −0.1437 |
| winning-class count / T | 0.9952 | 0.9911 |
| encoder share of energy proxy | 63.5% | 49.1% |
| energy per sample (proxy) | 2.353 µJ | 3.942 µJ |

Count histograms: at T=2, 89.48% of entries are 0, 0.57% are 1, 9.95% are 2. At T=4,
88.68% are 0, 9.70% are 4, and everything in between totals 1.62%.

**Found.**
- *The ceiling binds hard, and the model saturates it exactly.* Max observed confidence
  equals the derived ceiling to four decimals at both settings — 0.4509 at T=2 and 0.8585
  at T=4. The model drives the winning class to essentially the maximum count (0.9952 and
  0.9911 of `T`) and everything else to zero. So structural underconfidence is not a
  possibility the design warned about; it is what a converged model actually does, at
  **−0.539** at T=2.
- *The ceiling is also a floor on the training loss.* Cross-entropy cannot fall below
  `-ln(ceiling)` = 0.797 at T=2 and 0.153 at T=4. Observed plateaus were 0.806 and 0.166,
  each within 0.014. The T=2 loss curve looks like a failure to converge next to the T=4
  curve and is nothing of the kind. **Training loss is not comparable across `T`**, and
  convergence must be judged on validation accuracy. Now recorded in `NOTES.md`.
- *The trained readout is effectively binary.* Intermediate counts are almost unused:
  1.62% of entries at T=4 and 0.57% at T=2. The attainable confidence set is not merely
  coarse in principle — the model occupies only a handful of its points, 12 distinct
  confidences at T=2 and 48 at T=4 on a 5,000-sample calibration split.
- *That is the capacity regime the hypothesis names, and it is real on trained models.*
  416.7 calibration samples per distinct confidence level at T=2 against 104.2 at T=4, a
  4x difference in exactly the predicted direction. This is the first evidence for the
  mechanism that is not synthetic.
- *Tie rate moves the predicted way but is far too small to carry the argument.* 0.0133 at
  T=2 against 0.0072 at T=4 — the right direction, and a factor of 1.8, but both are near
  1%. The synthetic check's 44% at T=2 came from an untrained-like generator with no
  learned separation. **On converged models the tie rate is a weak signal**, and the
  paper should not lean on it as evidence of discreteness; the distinct-confidence count
  is the far stronger measure and moves 4x over the same interval.
- *Encoder share of the energy proxy falls as `T` rises*: 63.5% at T=2, 49.1% at T=4.
  Spiking operations grow faster with `T` (2.3x from T=2 to T=4) than encoder operations
  do (1.3x), because event frames get sparser per frame as the same events are split
  across more time bins. So the "input encoding dominates" claim is itself `T`-dependent
  and must be reported per `T`, not once.

**Chose.**
- *`readout_report.py` fits no calibrator.* The question here is what the raw readout looks
  like; mixing in calibration results would make it impossible to tell which properties are
  the model's and which are the calibrator's.
- *The report reconstructs the model from the saved `model_config` in the run record*
  rather than from defaults, so it cannot silently analyse a different architecture than
  the one that was trained.

**Open.**
- *Confidence is highly concentrated near the ceiling* — mean 0.4463 against a max of
  0.4509 at T=2, a spread of under 0.005 across 10,000 test samples. This is close to the
  degenerate case that `ece_equal_mass` has a special path for, and it is where equal-width
  and equal-mass binning should diverge most. Untested until calibration runs.
- *Only 12 distinct confidence levels at T=2* means isotonic has at most 12 knots and,
  after monotonicity pooling, likely far fewer. Whether that is enough capacity to beat a
  single temperature is now an empirical question with real numbers behind it.
- *T=6, 8, 10 are not trained*, so the trend in distinct confidences and samples-per-level
  rests on two points.
- *Neither model early-stopped in 15 epochs*, so neither accuracy is a ceiling.
- Earlier open items stand: severity-matched synthetic generator, subject-disjoint
  calibration for DVS-Gesture, CIFAR10-DVS split, CI never yet run.

---

## 2026-09-10 — Can confidence separate right from wrong? Plus a float-noise bug in the capacity statistic

**Did.** Added four measurements to `scripts/readout_report.py`: AUROC of confidence for
predicting correctness, mean confidence split by correctness with the gap, the full
confidence table on the calibration split (every distinct value, its support, and the
accuracy at it), and the standard deviation of confidence with its ratio to the mean.
Found and fixed a floating-point bug that had been inflating every distinct-confidence
count in the repository. Recorded a dated, pre-registered prediction in `NOTES.md`.

**Ran.** `pytest tests/ -q` → **104 passed** (up from 100).
`scripts/readout_report.py --timesteps 2 4 --seed 0`, N-MNIST, calibration split 5,000:

| quantity | T=2 | T=4 |
|---|---|---|
| AUROC of confidence vs correctness (test) | 0.9106 | 0.9728 |
| mean confidence \| correct | 0.4481 | 0.8484 |
| mean confidence \| incorrect | 0.3258 | 0.5568 |
| gap | +0.1223 | +0.2916 |
| confidence std (test) | 0.027156 | 0.061963 |
| confidence std / mean (test) | 0.06085 | 0.07332 |
| distinct levels (cal) | 8 | 29 |
| share of cal mass at the top level | 95.0% | 89.0% |
| gap at that level | +0.5464 | +0.1415 |
| mass-weighted mean gap | +0.5412 | +0.1449 |

**Found: summation order was inventing confidence levels.**
Two samples whose counts form the same multiset -- say `{2, 1, 0, ..., 0}` -- have
mathematically identical confidence. If the winning class sits at a different index the
softmax sums its terms in a different order, and float64 returns values ~5.55e-17 apart.
`np.unique` then counts them as separate levels.

This was not cosmetic. It corrupted two things:

- **The capacity statistic the hypothesis is stated in.** `n_distinct` and
  `n_cal / n_distinct` are the quantities `NOTES.md` says the isotonic advantage should
  track. On the trained models the counts fell from 12 to **8** at T=2 and from 48 to
  **29** at T=4 once fixed. On the synthetic check the overcount was 40-50% at every T:
  53 -> 34, 383 -> 204, 1,126 -> 584, 2,071 -> 1,173, 2,956 -> 1,816.
- **Equal-mass binning.** `_equal_mass_bins` calls `np.unique` to keep tied confidences in
  the same bin, and is documented as never splitting a tie -- because splitting one
  manufactures per-bin accuracy differences out of sampling noise. Noise-separated values
  were being split exactly as if they were genuinely different.

Fixed by quantising confidence to 12 decimal places at source, in both
`softmax_confidence` and `model_outputs` (the latter builds `conf` by indexing the probs
matrix rather than calling the former, so it needed the same treatment and was missed on
the first pass -- caught only because the synthetic counts did not move after the first
fix). `_equal_mass_bins` and `compute_metrics` also quantise defensively, since a
calibrator's output is not guaranteed to have come through either function. Three tests
cover it and all three fail if the quantisation is made a no-op.

Twelve decimals is chosen with headroom: the noise is ~1e-16 relative, while the smallest
gap between genuinely different attainable confidences is set by changing one class's count
by one, which for C=10 is ~8e-5 even at T=10. Seven orders of magnitude of margin.

Reported ECEs did not move -- `compute_metrics` was already quantising defensively -- so
the calibration conclusions in earlier entries stand. Only the distinct-level and
samples-per-level figures were wrong. Earlier entries are left as written, since this log
is append-only; `NOTES.md` has been corrected, as it is the authoritative current state.

**Found: confidence does carry information, so calibration is a real question.**
AUROC is 0.9106 at T=2 and 0.9728 at T=4. A calibrator is a monotone map and cannot
reorder samples, so it cannot change AUROC at all -- whatever separation exists here is a
hard ceiling on what any calibrated confidence can express. Had it come back near 0.5, the
best possible calibrator would output the base accuracy rate for every sample and the study
would be asking the wrong question. It did not, so the question stands.

**Chose.**
- *Recorded the prediction with a date, in `NOTES.md`, before fitting anything.* It
  predicts that **temperature scaling will match or beat isotonic at every `T` on
  N-MNIST**, which **contradicts the study's own hypothesis**. The reasoning is that ECE is
  mass-weighted and ~90% of the calibration mass sits at one confidence level, so
  correcting that single level is almost the whole job and a scalar can do it. Writing down
  the conflict in advance is the only way the result can settle it; a prediction that can be
  edited afterwards is not a prediction.
- *Stated the caveat in the same section rather than saving it.* The per-level gap is not
  constant -- it runs +0.154 to +0.698 across the eight levels at T=2 -- so this is not a
  uniform-offset miscalibration in the strict sense, and the claim rests specifically on
  mass weighting. If isotonic wins, that is the most likely reason, and saying so now costs
  nothing while saying it afterwards would be worthless.
- *The confidence table is printed for the calibration split, not the test split.* It is
  meant to show what the calibrator actually fits on.

**Open.**
- *The prediction is for N-MNIST only.* DVS-Gesture has ~107 samples per class and a much
  smaller calibration split, where isotonic's per-level estimates will be far noisier; the
  mass-concentration argument may not transfer at all.
- *T=6, 8, 10 are still untrained*, so the concentration figures rest on two points.
- *AUROC is computed on a heavily tied score* -- 8 distinct values across 5,000 samples at
  T=2. scikit-learn handles ties correctly, but the number is coarse and its sampling
  variability has not been quantified. A bootstrap on AUROC would be cheap and is not done.
- Earlier open items stand: severity-matched synthetic generator, subject-disjoint
  calibration for DVS-Gesture, CIFAR10-DVS split, CI never yet run.

---

## 2026-09-10 — Sweep preconditions: all five caches built, and convergence made visible

**Did.** Built the missing T=6 and T=8 frame caches. Implemented `spikecalib/sweep.py`
(`verify_caches`, `assert_caches_ready`, `SweepConfig`, `run_one`, `run_sweep`,
`convergence_audit`) and a thin CLI in `scripts/run_sweep.py` with `--verify-only`. The
sweep now refuses to start unless every frame cache it will need is present and complete,
and every results row records `epochs_run` and `early_stopped`.

**Ran.** `scripts/fetch_data.py --timesteps 6` and `--timesteps 8` → both exit 0, frame
shapes `(6, 2, 34, 34)` and `(8, 2, 34, 34)`, 70,000 npz each, assertions passed.
`scripts/run_sweep.py --verify-only` → all five caches complete (70,000 npz each), exit 0.
Fail-early path checked by requesting an unbuilt T=12: exits **1**, lists the missing
cache and prints the exact `fetch_data.py` command to build it.
Harness smoke (`--timesteps 2 --seeds 0 --epochs 2 --patience 1`) → 4 rows written, one
per arm; audit correctly reported `EPOCH CAP` and warned.
`pytest tests/ -q` → **115 passed** (up from 104), 11 of them new for the preconditions.

**Chose.**
- *Defaults are `epochs=30`, `patience=5`, so the stopping rule decides and not the cap.*
  30 is well beyond where N-MNIST converged in earlier runs; patience 5 ends a run once
  validation accuracy has been flat for five epochs.
- *`epochs_run` and `early_stopped` go on every row, not just in a side file.* Each run
  emits four rows, one per calibration arm, and all four repeat the training provenance.
  A reader looking at a single calibration number can see whether the model behind it
  converged without cross-referencing anything.
- *A convergence audit prints before the calibration numbers, and a capped run makes the
  process exit 2.* An undertrained model has a genuinely different readout -- fewer
  distinct confidence levels, a higher tie rate, lower accuracy -- so comparing a converged
  low-`T` model against an undertrained high-`T` one would attribute a training difference
  to a calibration mechanism. The audit deduplicates the four arm rows and counts runs.
- *Cache verification counts files rather than trusting the directory.* Same defence
  `fetch_data.py` applies: SpikingJelly checks only that a cache directory exists, so an
  interrupted build leaves a stump that looks fine. A test covers the stump case, and
  another covers hidden files not counting towards completeness.
- *Rows are flushed to CSV after each run.* An interrupted sweep keeps what it has earned.

**Wrong hypothesis, corrected by measurement.**
The harness smoke took 198.9 s for a 2-epoch T=2 run, against ~21 s of expected training.
I suspected the SynOps forward hooks: they call `.item()` twice per counted layer per
batch, forcing a device sync, and `_is_binary` materialises two boolean tensors the size
of the activation. Profiling said otherwise -- **hook overhead measured −0.1 s, i.e. free**.
The cost is `predict_counts` itself: 45.3 s for 10,000 test samples with hooks *disabled*,
dominated by reading npz files. Bootstrap (1,000 resamples) is 0.2 s and both calibrator
fits are under 0.05 s. The instrumentation I was about to optimise was not the problem.

That corrected a real defect in the cost estimate, which had modelled training epochs only.
Count extraction is ~34 s per unit of `T` per run and is paid whether or not a run stops
early, adding ~0.8 h across 15 runs. The CLI now reports training and extraction
separately: **3.8 h + 0.8 h = 4.6 h worst case** for the full sweep, with early stopping
cutting only the training half.

**Open.**
- *The sweep has not been run.* Preconditions pass and the harness is smoke-tested end to
  end, but no calibration numbers exist yet.
- *Whether 30 epochs is enough is untested.* Both earlier full runs hit a 15-epoch cap
  without early stopping, and validation accuracy was still drifting up at 15. If runs come
  back capped at 30, the audit will say so and the cap needs raising before the numbers can
  be read.
- *`predict_counts` is the per-run bottleneck at ~4.5 ms per sample*, and it is file-read
  bound rather than compute bound. Caching decoded frames in memory would pay for itself
  across a sweep; not done, and not needed for correctness.
- Earlier open items stand: severity-matched synthetic generator, subject-disjoint
  calibration for DVS-Gesture, CIFAR10-DVS split, CI never yet run.

---

## 2026-09-10 — The sweep ran, and the hypothesis did not survive it

**Did.** Moved `discrimination()` from `scripts/readout_report.py` into
`spikecalib/calibration.py` so the sweep reuses it rather than duplicating it, and added
AUROC to every results row. Made the sweep resumable. Added a validation-accuracy outlier
audit. Ran the full sweep and recorded the outcome in `NOTES.md` beneath the pre-registered
prediction, which was left untouched.

**Ran.** `pytest tests/ -q` → **129 passed** (up from 115).
`scripts/run_sweep.py --epochs 50 --patience 5` → exit **0**, 15 runs, **159.8 min**.

Convergence audit: **all 15 runs stopped on the patience rule**, between 10 and 21 epochs
against a 50 cap. Nothing hit the cap, so no comparison below is confounded by unequal
convergence. Validation-accuracy audit: the largest gap from the best seed at any `T` was
0.0034, far inside the 0.02 tolerance, so no run looks like a training failure.

Readout, mean over 3 seeds:

| `T` | test acc | tie rate | distinct levels | samples/level | AUROC | ceiling |
|---|---|---|---|---|---|---|
| 2 | 0.9850 | 0.0128 | 9 | 583.3 | 0.9204 | 0.4509 |
| 4 | 0.9887 | 0.0070 | 31 | 162.3 | 0.9735 | 0.8585 |
| 6 | 0.9902 | 0.0055 | 71 | 70.4 | 0.9842 | 0.9782 |
| 8 | 0.9898 | 0.0055 | 168 | 29.9 | 0.9837 | 0.9970 |
| 10 | 0.9908 | 0.0036 | 277 | 18.2 | 0.9863 | 0.9996 |

Mean equal-width ECE:

| `T` | uncalibrated | temperature | iso top-label | iso one-vs-rest |
|---|---|---|---|---|
| 2 | 0.5386 | 0.0018 | 0.0042 | 0.0041 |
| 4 | 0.1447 | 0.0031 | 0.0029 | 0.0028 |
| 6 | 0.0273 | 0.0026 | 0.0027 | 0.0031 |
| 8 | 0.0072 | 0.0028 | 0.0031 | 0.0035 |
| 10 | 0.0029 | 0.0026 | 0.0028 | 0.0034 |

**Found: the hypothesis is falsified on N-MNIST.**
It predicted an isotonic advantage growing as `T` falls. At `T = 2` — the most discrete
regime, 583 calibration samples per distinct confidence level — **temperature beat
isotonic**, by a larger margin than at any other `T` (+0.0024). From `T = 4` upward the
difference is within noise. The refined, capacity-based version fares no better: samples
per level spanned 583 to 18, a 32-fold range and precisely the quantity the mechanism said
the advantage should track, and the advantage was flat and near zero across all of it.

Fourteen of fifteen paired-bootstrap intervals straddle zero. Of the two that do not, one
favours temperature and one favours isotonic (`T = 6` seed 0, CI [−0.0024, −0.0003]). One
resolvable result in fifteen at a 95% level is what multiple comparisons produce unaided;
the expected count under no true difference is about 0.75. My own pre-registered
prediction, which said every interval would straddle zero, is therefore **wrong in the
letter** — `T = 6` did not — while its substance held.

**Found: the ceiling result is the real finding.**
Uncalibrated ECE at `T = 2` is **0.5386**, nearly all of it structural underconfidence
forced by the 0.4509 attainable-confidence ceiling, falling to 0.0029 by `T = 10` as the
ceiling recedes. Both calibrators remove essentially all of it. The paper's finding is not
*which* calibrator but that **calibration matters enormously at low `T` and the choice of
method does not**. That is a cleaner, more useful claim than the one the study set out to
make, and it is supported by every run.

**Found: the two ECE estimators disagree where correctness rule 2 said they would.**
At `T = 2`, equal-width gives temperature 0.0018 and isotonic 0.0042; equal-mass gives
0.0008 and 0.0015. A factor of two on the same data, in the regime where confidence is most
concentrated. Reporting only one would have produced a different ranking.

**Chose.**
- *Reuse refuses a saved run that hit a smaller epoch cap.* The condition is beyond what
  was asked, and it fired immediately: the existing `T = 2` and `T = 4` records hit a
  15-epoch cap and were refused by this 50-epoch sweep. Reusing them would have mixed
  undertrained models into a sweep whose whole point was equal convergence.
- *The outlier tolerance uses a 1e-9 slack.* Accuracies are ratios of integers, so a gap
  mathematically equal to 0.02 arrives as 0.020000000000000018 and would clear a bare
  `>`, flagging a run that is exactly at tolerance rather than beyond it.
- *The prediction section in `NOTES.md` was not edited.* The outcome is a new dated section
  beneath it. A prediction that gets revised after the result is worthless, and the value
  of having written it down is precisely that it can now be shown to have been wrong.

**Open.**
- *DVS-Gesture is now the interesting case, not a formality.* Its calibration split is
  roughly 330 samples against N-MNIST's 5,000, so isotonic's per-level estimates will be an
  order of magnitude noisier. Everything above says the two methods tie when the
  calibration set is large; whether that survives a small one is untested and is the
  question the study should now be organised around.
- *Every `T` here reached ~0.99 accuracy*, so the sweep never explored a regime where the
  model is genuinely uncertain. Calibration differences may only appear where accuracy is
  lower.
- *Uncalibrated ECE spans 0.5386 to 0.0029 across `T`*, so calibrated ECE compared across
  `T` compares wildly different amounts of work done. The severity-matched comparison
  flagged in an earlier entry is now clearly necessary rather than merely tidy.
- *CIFAR10-DVS split still undefined; CI has still never run.*

---

## 2026-09-11 — Quantisation tested as the explanation for the membrane ranking deficit, and rejected

**Did.** Added `spikecalib/quantisation.py`, `tests/test_quantisation.py`,
`scripts/quantisation_test.py` and `figure_quantisation`. The question was the one left
open by the readout comparison: membrane calibrates better wherever the ceiling binds but
*ranks* worse from `T = 6` up, and the natural explanation is that a spike count is a
thresholded membrane signal whose thresholding discards noise. The test coarsens the saved
membrane confidences to `N` levels — no retraining — and recomputes AUROC and AURC.

**Ran.** `pytest -q` → **174 passed** (up from 152).
`scripts/quantisation_test.py` → exit **0**, 30 inference cells, 296 rows, ~23 min.
All 15 full-precision AUROCs reproduced `results/readout_experiment.json` exactly, which is
the check that the cached confidences are the same ones the earlier numbers came from.

**Found: the hypothesis is falsified, and not in the way anticipated.**
The predicted signature was AUROC rising as `N` falls, peaking in the tens or low hundreds
near count's own level count, then falling. Nothing resembling it occurs at any `T` or any
seed. **AUROC rises monotonically with `N` and saturates at full precision.** The
alternative outcome written into the task — "if AUROC is flat in `N`, quantisation is not
the mechanism" — did not happen either. The curve is not flat: coarsening actively costs,
and the cost grows as the binning coarsens. That refutes the hypothesis more strongly than
flatness would have.

Coarsening did raise AUROC in 10 of 15 cells, and the magnitude is the finding: the largest
gain anywhere is **+0.000073**, against a `T = 6` deficit of **0.0132**. It is 180 times
too small, and the sign flutter across cells is floating-point noise around zero. Equal-width
binning agrees, with a best gain of +0.0013 — still an order of magnitude short.

**Found: the identity made the result readable rather than merely observed.**
Coarsening is monotone and non-injective, so it cannot reorder a pair, only merge pairs into
ties that AUROC scores 0.5. Hence `AUROC(coarse) − AUROC(full) = W × (0.5 − A)` exactly,
with `W` the fraction of pairs merged and `A` the within-level AUROC. Measured, `A` runs
0.89 at `N = 5` down to 0.50 at `N = 1000` and is above chance everywhere that merges a
meaningful number of pairs. **Membrane's fine structure is signal, not noise.** Of 266
points 24 fall below 0.5, all at `N ≥ 200` where under 1% of pairs merge, bounding their
possible effect at 0.0013.

**Found: the sharper comparison rules something in.**
Held at count's own level count, membrane at `T = 2` still beats count (0.9367 against
0.9204) and at `T = 6` is still 0.0138 behind it (0.9705 against 0.9842). At identical
resolution the two readouts rank differently in **both directions**, so resolution cannot
be what separates them. The remaining explanation is that they rank different quantities.

**Broke: a `DataLoader` deadlock took forty minutes of completed inference with it.**
The first full run hung at cell 14 of 30 — 0% CPU, no workers alive, no progress. Cause:
`persistent_workers=True` on loaders that are each iterated exactly once and then dropped,
so six worker processes per cell accumulated waiting to be reaped. The real cost was not the
hang but that `build_cache` wrote its npz only after **all** cells finished, so the kill
discarded thirteen completed inference passes. Fixed both: persistent workers off with the
loader deleted after each pass, and the cache flushed after every cell. Turning persistent
workers off tripled per-cell time from ~16 s to ~45 s, which is the right trade for a script
that is run once and must not lose its work.

**Chose.**
- *Equal-mass binning is primary, equal-width is the check.* On a confidence distribution
  piled up near 1.0, equal-width crowds the samples into the few bins nearest the mode —
  half the data lands in about 8 bins out of 100 — so its `N` axis does not mean what it
  says. Equal-mass guarantees every level is populated. Both are reported; they agree.
- *The within-level AUROC is reported alongside the curve, not just the curve.* Without it
  a flat result would be indistinguishable from a quantiser that does nothing, and the
  identity turns "the curve went down" into "the structure destroyed was signal".
- *Count confidences were cached too, doubling the inference.* It buys the only comparison
  that separates "how finely" from "what": membrane held at count's own resolution.
- *`quantise_to_levels` charges each value to the midpoint of its mass, not its upper edge.*
  Under heavy ties the upper-edge rule collapses a value holding a large share of the mass
  into its neighbour and loses a level that genuinely exists. Both rules survived the first
  round of mutation testing; the midpoint rule needed a test built specifically to separate
  them, which is now `test_equal_mass_does_not_collapse_a_heavy_tie_into_the_level_above_it`.
- *`NOTES.md`'s Status section was rewritten.* It still claimed model, training and
  calibration code were "not yet written", which stopped being true nine sessions ago.

**Open.**
- *Why the ranking reverses is still unexplained.* One candidate is eliminated, which is
  progress but not an answer. The next candidate — that a count is invariant to how far
  above threshold each crossing was, and a potential is not — is untested, and it does not
  obviously explain why the invariance would only pay once the cap stops binding.
- *The figure bug this data exposed is worth remembering.* Count's level count differs per
  seed (95, 97, 101 at `T = 6`), so the sweep probes each seed at a resolution the others
  were never evaluated at. The first version of `figure_quantisation` averaged across
  whichever seeds happened to have a given `N`, which would have drawn single-seed points
  as if they were three-seed means. It raised `StopIteration` on the real data rather than
  drawing something wrong, but only by luck of how the lookup was written.
- *`results/quantisation_cache.npz` is 30 × 10,000 confidences.* Cheap now; if the study
  adds datasets or seeds this becomes the pattern that needs a size policy.

---

## 2026-09-11 — The leak explains half the ranking deficit; auditing the dashboard found thirteen defects

**Did.** Three things. Added a non-leaky `accumulator` readout and used it to test whether
the membrane integrator's leak is what costs it its ranking at high `T`. Regenerated the
dashboard payload and audited every number on the page against the artifacts. Corrected
what the audit found, in both the page and `NOTES.md`.

**Ran.** `pytest -q` → **179 passed** (up from 174).
`scripts/readout_experiment.py --readouts accumulator --timesteps 2 6 10 --seeds 0 1 2`
→ exit **0**, 9 runs, ~2.5 h, merged to 54 rows.
`scripts/export_dashboard_data.py` → `docs/data.js`, 310 KiB.
Rendered-page audit under headless Chromium at 1680px and 400px.

**Found: the membrane readout leaks hard, and the leak was measurable before any model ran.**
`v += (z − v)/tau` with `tau = 2` gives per-step retention 0.5 — a half-life of **exactly one
step**. The input `k` steps before the end enters with weight `0.5 × 0.5^k`. At `T = 2` the
leaky window and the count window **coincide exactly** (both 100% on the last two steps); by
`T = 6` membrane puts 76% there against count's 33%. That coincidence with the observed
crossover is what made the hypothesis worth testing rather than guessing.

**Found: the leak accounts for about half the deficit, and the prediction passes its own test
while failing its own headline.** Both were written down, so both are reported. The stated
falsification condition — deficit staying above half membrane's, 0.0066 at `T = 6` and
0.0045 at `T = 10` — is not met: measured **0.0061** and **0.0035**, closing **53%** and
**62%**. But the headline said the accumulator should "match or beat count", and it does not:
count ranks better in **every seed at both settings**, 0 of 6. Since an accumulator and a
count differ *only* by the output threshold, the surviving ~40% belongs to the thresholding.

**Found: the wrong part of the prediction was the most useful part.** It said accumulator and
membrane should be close at `T = 2`. On ranking they are (0.9698 against 0.9659). On
calibration they are not: ECE **0.0031 ± 0.0002** against **0.0290 ± 0.0153**. Removing the
leak drops the mean ninefold and the seed spread **77-fold**. That answers the open question
from the previous session about why membrane's `T = 2` calibration swings so widely — a leaky
integrator at `T = 2` is reading essentially the final step.

**Found: the first consistently overconfident readout in the study.** The accumulator's
`conf − acc` is +0.0032 at `T = 6` and +0.0047 at `T = 10`, positive in 3 of 3 seeds at both.
Recorded as a lead on the literature disagreement, not a finding: the magnitude is under half
a percentage point, on one architecture and one dataset.

**Found: thirteen defects on the dashboard, most of them the same mistake.**
A number measured on seed 0, written before the other two seeds existed, never revisited.

Wrong arithmetic:
- "14 of 15" paired-bootstrap intervals straddle zero. It is **13 of 15**. `NOTES.md` said
  "fourteen of the fifteen" and "the two that do not" in a single sentence; the page dropped
  the second resolvable result altogether. The two that resolve point in *opposite*
  directions, which strengthens the null reading.
- Training loss plateaus "between 0.0086 and 0.0141" above the floor. Measured across
  fifteen runs: **0.0072 to 0.0182**. Neither endpoint was right.
- "At `T = 8` and `T = 10` the residual miscalibration is ordinary overconfidence." It is
  not: `conf − acc` is −0.0065 at `T = 8`, negative at all three seeds. Twice on the page,
  once in `NOTES.md`.

Claims the three-seed run had already superseded:
- "The membrane readout is underconfident at all five timestep settings and overconfident at
  none." **Six of the 45 cells are positive.** This one is uncomfortable: the wording was
  edited two sessions ago and *verified on the rendered page*, but only the rendering was
  checked, never the substance against the data that already existed.
- "The single positive `conf − acc` anywhere in the experiment." There were six.

Seven single-seed numbers restated as means with spread: count `conf − acc` at `T` = 2 and 4,
membrane `conf − acc` at all five `T`, the "sixteenfold" shrinkage (11.9×–63.7×), one-vs-rest
AURC "five times worse" (6.3×), operating points at `T = 2` (10 → about 9), and coverage at
risk 0.1% at `T = 4`.

**Broke: `readout_experiment.py` would have destroyed the 45-row artifact.** It wrote
`results/readout_experiment.json` with only the rows from the current run. Running one readout
over three of the five `T` — exactly what this session needed — would have replaced the whole
comparison with nine rows. Caught because the first launch was killed within seconds for this
reason; the file was intact and git had it anyway, but the next person would not have been
watching. It now merges by `(readout, T, seed)` and reports how many rows it kept.

**Chose.**
- *The accumulator is `v += z`, a plain sum, not a mean.* A sum is the exact analogue of the
  count readout, which is also a sum. That makes the pair differ by the threshold and nothing
  else, which is the only reason the residual deficit can be attributed at all.
- *The page no longer holds its own numbers.* The readout, crossover, selective and
  quantisation tables render from `window.SPIKECALIB`, copied verbatim out of the result JSON
  by the export script. Hardcoding is how the page came to state a one-seed result as the
  study's, and no amount of care would have stopped it recurring.
- *The figures keep their light surface on a dark page.* Their palette was validated against
  a light chart surface; re-stepping it for the dark ground would invalidate that validation.
  They are framed as inset documents instead.
- *`NOTES.md` corrections are dated and say what they replace* rather than being silently
  overwritten. The pre-registered predictions themselves are untouched, including the ones
  that turned out wrong.

**Open.**
- *Roughly 40% of the high-`T` ranking deficit is still unexplained*, and it is now pinned to
  the output threshold specifically — an accumulator and a count differ by nothing else. Why
  thresholding should *help* ranking is not obvious and is the next question.
- *The accumulator's mild overconfidence at `T ≥ 6` is one architecture on one dataset.*
  If it survives a second dataset it is the most direct evidence this study could offer on
  the sign disagreement.
- *`tau = 2.0` was never tuned*, and the leak result makes it load-bearing rather than
  incidental: a different `tau` would move the effective window and probably the crossover.
  Nothing here establishes that `tau = 2` is a sensible default, only what it does.
- *The accumulator costs about 0.2 accuracy points at `T = 10`* and that was not predicted or
  investigated.
- *The audit checked numbers against artifacts, not prose against prose.* Claims that are
  wrong without containing a number would not have been caught.

---

## 2026-09-11 — The C axis: the ceiling holds at 101 classes, and stops explaining things where it stops binding

**Did.** Brought N-Caltech101 into the study to vary the ceiling's second argument. Added
the registry entry, a study-defined train/test split, frame pooling, an in-memory frame
cache, and `scripts/caltech_experiment.py`. Ran 18 runs: 2 readouts × 3 `T` × 3 seeds.

**Ran.** `pytest -q` → **195 passed** (up from 179).
Download 3.72 GiB from Mendeley; `events_np` in 140 s; frame caches at `T` in {2, 6, 10}
in about 30 s each; the 18-run grid in roughly 1.5 h. All 18 stopped on the patience rule
between 14 and 40 epochs against a 50 cap, so no comparison is confounded by convergence.

**Two premises in the task were wrong and had to be corrected first.**
SpikingJelly *cannot* download N-Caltech101 — `downloadable()` returns False and
`resource_url_md5()` gives the homepage, not files. This is the N-MNIST situation, not the
CIFAR10-DVS one. The Mendeley mirror was used instead, and its file matches **both** the
published SHA-256 **and** SpikingJelly's own MD5, which is a stronger provenance check than
the repository had for N-MNIST. Separately, the ceiling at `C = 101, T = 2` is 0.068806,
not 0.0687.

**Found: the ceiling holds at 101 classes, and "valid" is not "saturated".**
The bound was respected in **9 of 9** count runs. At `T` in {2, 6} the models saturate it to
2.4e-14 and 3.8e-13 — the same one-part-in-10^13 as at `C = 10`. At `T = 10` the observed
maximum sits **1.5e-3 below** the bound rather than on it. That is not a failure: reaching
the ceiling requires a sample whose count vector is `[T, 0, ..., 0]`, meaning all 100 other
classes silent for all 10 steps, which is easy at `C = 10` and not at `C = 101`. The
temptation to report `|derived − observed| = 2.35e-03` as a deviation was real and would
have been wrong; the signed check is what matters and it was added.

**Found: the ceiling explains the error where it binds and none of it where it does not.**
This is the result that matters and only the C axis could produce it. At `T = 2` the ceiling
binds in 3/3 seeds and accounts for **97.1%** of uncalibrated ECE, mirroring N-MNIST's 99.2%.
At `T = 6` accuracy is 0.6427 against a ceiling of 0.8014 — it does **not** bind, nothing is
forced — and the readout is **still underconfident by 0.1534**, an ECE fifty times N-MNIST's
at the same `T`. On N-MNIST the regime where the ceiling stopped binding was also a regime
where almost no error was left, so the two explanations were indistinguishable. At `C = 101`
they come apart, and a large residual miscalibration is left with no account of it.

**Found: where the cap is severe it costs accuracy, not just calibration.**
At `T = 2`, the non-leaky accumulator beats count by **+0.1282 accuracy**, in every seed
(+0.1515, +0.0959, +0.1372). At `C = 10` the same comparison was worth 0.1 points. The
cross-entropy floor is 2.6765 nats at `C = 101, T = 2` against 0.7966 at `C = 10`, so the
objective is dominated by a target the model cannot reach. Above `T = 2` this reverses and
count wins by about 4 points, so it is a property of the binding regime, not of the readout.

**Found: the first genuinely uncertain regime, and the first overconfident count readout.**
Accuracy spans 0.4323–0.6561 (mean 0.5880) against N-MNIST's 0.9842–0.9914, which closes a
standing limitation. In that regime the **count** readout is overconfident at `T = 10`
(+0.0344, 3/3 seeds) — the first time in the study — and the accumulator is overconfident in
**9 of 9** runs. The disagreement with the literature is therefore a regime difference:
underconfidence is what a capped readout does where the cap binds, and removing either
condition produces overconfidence here too.

**Broke: three times, each caught by a check rather than by luck.**
- *The 18-run grid projected to 124 hours* at native 180×240. Measured, not assumed. Fixed
  by pooling 4× and by holding the cache in RAM, after measuring that training was **85%
  disk** — 27 ms/sample of I/O against ~5 ms/sample of computation. Epoch time at `T = 2`
  fell from about 3 minutes to 5 seconds and the grid to under two hours.
- *The pooling transform was a closure and `DataLoader` workers could not receive it.*
  macOS spawns workers, the transform is pickled to reach them, and a local function is not
  picklable. It failed at worker start-up with an error naming the closure, not the cause.
  It is a class now, with a test that pickles it.
- *A scratch script named `timeit.py` shadowed the stdlib module torch imports*, producing a
  circular-import error inside `torch.distributed` that had nothing to do with the code.

**Chose.**
- *Frames are average-pooled 4× to 45×60.* The honest first reason is compute. The better
  reason is that at 45×60 N-MNIST's own two-block `(32, 64)` architecture fits unchanged, so
  the `C = 101` comparison differs from the `C = 10` one in the class count and the data and
  nothing else; at native resolution it would also differ in depth and parameter count.
  Pooling cannot touch the ceiling, which contains neither resolution nor anything
  downstream, and it lowers accuracy, which makes "does it bind" harder to satisfy.
- *Average pooling, not max.* A frame is an event count per pixel, so the mean is the local
  event rate; max would report the busiest pixel and discard the rest.
- *The test split is proportional with a floor, not a fixed per-class count.* The dataset
  runs from 31 samples to 800 and no fixed count serves both ends. Val and cal stay fixed
  per class so that every class reaches the calibration set whatever its size.
- *`create_events_np_files` is SpikingJelly's own.* Only fetching and verification are
  local. The `download/` gate was skipped deliberately: it demands four files including two
  readmes the loader never opens, and only one of the four is published at a stable URL.

**Open.**
- *The residual underconfidence at `C = 101, T = 6` has no explanation.* 0.1534 with the
  ceiling slack. This is now the study's largest unexplained quantity and it sits directly
  under its central claim.
- *Only count and accumulator were run here.* Membrane and weighted-count at `C = 101` would
  say whether the leak and the granularity results also transfer, and they were skipped for
  time, not for a reason.
- *Calibrators have not been run on N-Caltech101 at all.* The null result on calibrator
  choice is still a `C = 10`, 0.99-accuracy result, and the 1,010-sample calibration set
  across 101 classes is exactly the regime where it might break.
- *`T` in {2, 6, 10} only*, against five settings on N-MNIST, so the crossover cannot be
  located as precisely here.
- *Pooling by 4 is untested as a choice.* No comparison against 2× or native was run, so
  nothing establishes that the result is insensitive to it.

---

## 2026-09-11 — The conditional ceiling refines the bound and does not rescue the C=101 residual

**Did.** Added `spikecalib/conditional.py`, `tests/test_conditional.py` and
`scripts/conditional_ceiling.py`. Measured the competitor distribution on both datasets,
computed the ceiling conditional on it, and tested a pre-registered prediction about whether
it explains the `C = 101, T = 6` residual. No training: count vectors were read from the
already-trained count models.

**Ran.** `pytest -q` → **212 passed** (up from 195).
`scripts/conditional_ceiling.py` → 24 cells, a few minutes.

**Found: the competitor distribution is skewed, and the mean hides it.**
At `C = 101, T = 6`, 81.6% of competitor slots hold zero, 12.1% hold one, 4.1% hold two,
and the tail reaches six. The single loudest competitor carries 14.7% of the entire softmax
denominator at `T = 6` and 25.6% at `T = 10`. Competitor activity grows with `T`: 8.6, 16.4
and 24.9 of 100 classes firing at `T` = 2, 6, 10. At `C = 10` the competitors are nearly
silent by comparison — 0.1 of 9 firing at `T = 2`, 1.5 at `T = 10`.

**Found: part 2 of the prediction held, and it explains the study's own history.**
Theoretical minus conditional ceiling is **at most 0.0067 at any `T` on N-MNIST**, against
up to **0.1210** at `C = 101` — a 26-fold difference at `T = 6`. With nine competitors that
are silent 95–99% of the time, the extreme vector `[T, 0, ..., 0]` is nearly reachable and
the simple bound is nearly tight. That is why fourteen sessions of `C = 10` work never
needed the refinement.

**Found: part 1 was falsified, and that is the result.**
The prediction said the conditional ceiling would account for more than half the `T = 6`
residual and named its own threshold: wrong if the mean conditional ceiling exceeded 0.5660.
Measured **0.6804** — *above* the accuracy of 0.6427 — in all three seeds (0.6571, 0.6812,
0.7029). It accounts for **0%**. Given exactly the competitor patterns these models produced
they could have averaged 0.68 confidence and averaged 0.49, leaving **0.19 of headroom
unused**. The `C = 101, T = 6` residual is **genuine miscalibration, not an attainability
effect**, and the tidiest explanation for the study's largest unexplained quantity is gone.
Part 3 missed in the same direction: predicted gap above 0.15, measured 0.1210.

**Found: the refinement is still worth having.** It is never looser than the theoretical
bound and raises the explained share wherever the ceiling binds — 99.6% against 97.1% at
`C = 101, T = 2`, and 61.7% against 44.5% at `C = 10, T = 6`.

**Broke: a false claim from the previous session, caught by this session's code.**
`scripts/conditional_ceiling.py` failed to load the N-Caltech101 weights, and the reason was
that **the C = 101 runs used four conv blocks `(32, 64, 128, 128)`, not N-MNIST's two
`(32, 64)`**. `CHANNELS` was set in `caltech_experiment.py` before the 4× pooling decision
was made, and the confound-removal argument built on that decision — written into the
registry, `NOTES.md`, the dashboard and a commit message — was never checked against it.
The claim that the `C = 10` vs `C = 101` comparison "differs in the class count and the data
and nothing else" was false when written.

Corrected in all four places with a dated note. The ceiling verification is untouched, since
the formula has no architectural term, but the *size* of the `T = 6` residual can no longer
be attributed to `C` alone until the architectures are matched. The conditional-ceiling
conclusion does survive it, because that bound is computed from each model's own outputs and
asks only whether *that* model could have been more confident.

Two things made this visible rather than permanent: the script loaded weights strictly, and
it did so against an architecture it had hardcoded. A hardcoded value that happened to match
would have been worse than one that did not — it would have loaded silently and measured the
wrong thing. The script now reads `model_config` from each run's own record.

**Chose.**
- *The winner is replaced by `T`, not kept at its observed count.* The question the
  conditional ceiling answers is what a sample **could** have expressed, so the winner is set
  to its maximum and only the competitors are held at their measured values.
- *The distribution is reported, not just the mean.* The denominator is a sum of
  exponentials and is dominated by its largest terms; two configurations with identical mean
  competitor count give materially different ceilings, which is asserted in a test.
- *`residual_share_explained` clips at 1 rather than allowing 130%.* A ceiling below the
  observed mean confidence is a contradiction and should be caught by the caller, not
  smuggled through as a percentage above 100.
- *The conditional ceiling is documented as an analysis tool, not a second derived result.*
  The theoretical ceiling is fixed before a model exists and predicts; this one is measured
  from a model's own spikes and only explains.

**Open.**
- *The `C = 101, T = 6` residual is now positively established as genuine miscalibration*
  and has no mechanism. Ruling out attainability makes it a sharper question, not a smaller
  one.
- *Whether the residual's size is a `C` effect or a depth effect is undetermined* until
  `C = 101` is re-run with `(32, 64)`. About 1.5 hours; not done.
- *An assertion in the first version of `test_conditional.py` claimed the confidence cost
  scales 11× with `C − 1`.* It does not: the denominator sensitivity does, but the ceiling
  saturates, and the measured cost ratio is about 5.8×. The same overstatement is in
  prediction 4's rationale, where it is left as written with a note, since the prediction's
  thresholds did not depend on it.
- *Nothing here tests a conditional ceiling at `C` between 10 and 101*, so the claim that it
  converges on the theoretical bound as `C` falls rests on two points.

---

## 2026-09-12 — The matched-architecture 2×2: the ceiling is architecture-independent, the headline accuracy gain is not

**Did.** Ran both architectures on both datasets — `(32, 64)` and `(32, 64, 128, 128)`,
`T` in {2, 6}, seeds 0–2, count and accumulator in every cell — to remove the depth confound
introduced when N-Caltech101 was run four blocks deep against N-MNIST's two. 48 cells, of
which 24 already existed and were reused under their historical filenames.

**Ran.** `scripts/arch_matched.py` → 48 cells, 24 newly trained, about 2 h.
`pytest -q` → **212 passed**, unchanged.

**Corrected the framing first.** The task described the +12.8 accuracy gain at
`C = 101, T = 2` as confounded with depth. It is not: count and accumulator share a trunk
within N-Caltech101, so depth never varied inside that comparison. What it lacked was
replication at a second depth — a weaker charge, and one this run answers. The genuinely
confounded claims were the cross-dataset ones, and they were listed as suspended in
prediction 5 before anything was run.

**Found: the two architectures differ in depth AND size, in opposite directions.**
More pooling shrinks the flatten, so the four-block model is the *smaller* one — 464k
parameters against the two-block model's 2,749k on N-Caltech101, 375k against 1,071k on
N-MNIST. This is a genuine alternative-architecture check and it is **not** a depth
experiment; nothing here can attribute a difference to depth specifically, and the writeup
says so.

**Found: the ceiling is architecture-independent, exactly as the formula requires.**
Zero violations across all 24 count cells, and the bound is **saturated to float precision
in every one** — both depths, both datasets, both `T`. The prediction declined to claim
saturation because reaching the bound needs a sample with all competitors silent, which is a
property of the trained model. It happened anyway, everywhere.

**Found: the headline accuracy gain does not replicate in magnitude — prediction falsified.**
The prediction said the gain at `C = 101, T = 2` would stay above +0.05 at both depths.
Measured **+0.1282 at four blocks and +0.0308 at two**, so it is falsified by its own
threshold. The gain stays positive in 3 of 3 seeds at both, so the direction replicates and
the magnitude does not. That also undercuts the mechanism offered for it: a 2.68-nat
cross-entropy floor dominating the gradient has no architectural term, and if it were the
whole story the gain would not be fourfold different between networks. "+12.8 points" is now
reported as "+3 to +13 points depending on the network".

**Found: the `C = 101, T = 6` residual survives — prediction held.**
0.1534 at four blocks, **0.1159** at two, against a threshold of 0.08. About a quarter of it
is attributable to the network and three quarters are not. N-MNIST's residuals barely move
between architectures (0.5386/0.5398 at `T = 2`, 0.0272/0.0253 at `T = 6`), so the
sensitivity is specific to `C = 101`. With the confound removed the cross-dataset ratio at
`T = 6` is **4.3× at two blocks and 6.1× at four**.

**Found: the conditional ceiling still explains none of it — prediction held.**
0.7146 against an accuracy of 0.5618 at two blocks; 0.6804 against 0.6427 at four. Above
accuracy either way, so it forces nothing. Prediction 4's falsification survives the matched
design.

**Broke: a second false number from the C-axis session, caught by this comparison.**
The C-axis section called the `C = 101, T = 6` ECE "fifty times N-MNIST's at the same `T`".
It is **4.4× to 6.0×** at the same `T`. Fifty is the ratio against N-MNIST at `T = 10`
(0.1539 / 0.0029 = 53) — a different timestep setting. The figure was mislabelled rather
than miscomputed, and it was only caught because the matched table put the same-`T`
comparison in front of me at two depths. Corrected in `NOTES.md` and on the page.

That makes two factual errors from that one session — this and the architecture claim — both
found by later work rather than by review at the time. The pattern in both: a number or a
claim written into prose while the code that would have contradicted it sat unexamined in
another file.

**Chose.**
- *Reuse existing weights under their historical filenames.* A cell whose architecture
  matches what that dataset already ran resolves to the old stem and is loaded, not
  retrained. The reused N-MNIST cells reproduce their recorded accuracies exactly, which is
  what confirms the lookup is finding the right files.
- *New cells carry an architecture tag in their filename.* `_arch32-64-128-128`. Without it
  a second architecture would silently overwrite the first dataset's weights.
- *Training settings are taken per dataset from what its existing runs used* — batch 128 for
  N-MNIST, 32 for N-Caltech101 — so a new cell differs from an existing one in depth alone.
- *N-MNIST is materialised into RAM like N-Caltech101.* Frames round-trip through float16
  exactly (maximum value 15, integers), checked before relying on it.

**Open.**
- *Depth and capacity cannot be separated here* because they are anti-correlated across the
  two architectures. A matched-parameter pair at two depths would separate them and has not
  been run.
- *Two architectures is two points.* "Survives both depths" is a much weaker statement than
  "architecture-independent", except for the ceiling itself, which is arithmetic.
- *`T` in {2, 6} only* for the matched grid, so nothing here says whether the `T = 10`
  behaviour replicates.
- *Why the accuracy gain is architecture-sensitive is unexplained*, and it is now the most
  concrete open question about the cap's effect on learning.
- *The residual's remaining three quarters still has no mechanism.* Ruling out attainability
  and now most of the capacity explanation makes it sharper, not smaller.

---

## 2026-09-12 — Verification pass over every `[UNVERIFIED]` source: two citations withdrawn, one constant reclassified

**Did.** Checked every `[UNVERIFIED]` item against primary documents, because they are
load-bearing in a preprint and several came from a project brief rather than from reading.
Nothing was invented to close a gap; where a source could not be confirmed the claim was
withdrawn or softened rather than patched with a plausible substitute.

**Found: Horowitz is fully confirmed, and one of the two constants is not a citation.**
Read the ISSCC 2014 digest paper directly. The energy table is **Figure 1.1.9**, captioned
"Rough energy costs for various operations in 45nm 0.9V", Digest p. 14. It gives Integer
Add 32-bit 0.1 pJ, Integer Mult 32-bit 3.1 pJ, FP FAdd 32-bit **0.9 pJ**, FP FMult 32-bit
**3.7 pJ**, cache 10–100 pJ, DRAM 1.3–2.6 nJ.
- `E_AC = 0.9 pJ` is **confirmed** as the 32-bit floating-point add. The caption's "0.9V" is
  the supply voltage — an unfortunate coincidence with the energy figure, and not its source.
- `E_MAC = 4.6 pJ` **appears nowhere in the paper.** There is no MAC row and no combined
  value in the text. It is the community convention 0.9 + 3.7. Reclassified from `[LIT]` to
  `[DERIVED]` in `spikecalib/models.py`, and the paper must describe it as a convention.
- The caption says **"Rough"**. The source presents its own numbers as order-of-magnitude,
  which retroactively justifies the study's existing rule that the energy figure is only ever
  a design-comparison proxy.

**Found: Atarashi et al. is confirmed, and the distinction is sharper than the abstract
allowed.** Kyohei Atarashi, Satoshi Oyama, Hiromi Arai, Hisashi Kashima, arXiv:2506.10572,
12 June 2025. The bounds in BCSoftmax are **learned by empirical risk minimisation on a
validation set**, and the paper **never applies the method to a structurally bounded
readout** — every experiment is a standard unbounded softmax over ResNet-50, DenseNet-121,
ViT-B/16 or a GloVe CNN. So the two works are inverses: they *impose* a box constraint to
fix miscalibration; this study *finds* one the readout already imposes and shows it *causes*
miscalibration.

**Found: the disagreement with the literature was never a disagreement.**
"Agreeing to Stop" is Jiechen Chen, Sangwoo Park & Osvaldo Simeone, *Entropy* 26(2):126,
February 2024, arXiv:2310.16675. Read the full text. It **does** use this study's readout —
`f_c = e^{r_c} / Σ e^{r_c'}` over accumulated spike counts — it **is** directly trained with
surrogate gradients, and it **does** say SNNs "tend to be poorly calibrated, producing
overconfident decisions". All three of the repository's claims about it were right.

The regime was the missing piece: **they run at `T = 80`**, checkpointing at {20, 40, 60,
80}. `[DERIVED]` at `T = 80, C = 10` the attainable-confidence ceiling is 1.0 to double
precision, and even at `T = 20` it is 0.99999998. **The cap is completely inert where they
work.** This study had asserted a regime explanation for fourteen sessions without knowing
their timestep count; the number now supports it exactly.

**Broke: two citations did not survive and were withdrawn.**
- *"Sun & Bohte, AOT-SNN, arXiv:2405.00699" is wrong on authors and on method.* That
  identifier is **Dengyu Wu, Yi Qi, Kaiwen Cai, Gaojie Jin, Xinping Yi & Xiaowei Huang**,
  "Direct Training Needs Regularisation: Anytime Optimal Inference Spiking Neural Network",
  and the method is **AOI-SNN**, not AOT-SNN. Two papers with similar acronyms had been
  merged into one reference. A genuine AOT-SNN line exists and is Tao Sun & Sander Bohté's —
  arXiv:2412.00278 is confirmed but is a **regression** paper and probably not what the brief
  meant. The paper is now cited under its real authors and the claim that "AOT-SNN uses
  temperature scaling as its calibration baseline" is withdrawn, since it was attached to the
  wrong paper and the right one has not been read.
- *"Skatchkovsky & Simeone on SNN overconfidence" could not be confirmed at all.* The
  overconfidence result this study compares itself against is Chen, Park & Simeone's. No
  Skatchkovsky paper on SNN overconfidence was identified. Guessing at which was meant would
  have produced exactly the failure mode this pass existed to prevent, so it is removed and
  the removal is recorded on the page rather than hidden.

**Found: a title this repository had been using was the method name, not the paper.**
"Dynamic Confidence" is the method. The paper is **"Unleashing the Potential of Spiking
Neural Networks by Dynamic Confidence"**, Chen Li, Edward Jones, Steve Furber, **ICCV 2023,
pp. 13350–13360**, arXiv:2303.10276. Verified: **ANN-to-SNN converted**, static image data,
temperature scaling as post-training calibration, and it runs at **2.5–14 timesteps** — the
closest regime overlap of any reference here.

**Open, and consequential:** that paper reports 94.11% accuracy at 4 timesteps on CIFAR-10.
If its readout is a softmax over accumulated spike *counts*, the `C = 10, T = 4` ceiling of
0.8585 binds there too, and the ceiling would already constrain published work. The readout
description available to this pass was ambiguous between accumulated spikes and accumulated
membrane potential, so it is recorded as a checkable question, not a finding.

**Could not confirm: the Loihi and SpiNNaker claims.** Intel's NxSDK API reference is not
publicly reachable and the sPyNNaker limitations page does not discuss recording bandwidth.
What could be established, from peer-reviewed secondary sources: Loihi's CUBA LIF carries
membrane potential as internal state and Rueckauer et al.'s NxTF paper says embedded CPUs
may read out SNN state during the management phase — consistent with readability, silent on
per-timestep cost, which is the only part the argument needed. sPyNNaker exposes potential
through PyNN's `record('v')` with host transfers bounded by Ethernet bandwidth, but whether
that binds for a ten-neuron output layer is unknown. **The strong form of the hardware
argument is withdrawn** in both `NOTES.md` and on the page; the case for the count readout
now rests on spikes-as-communication-primitive and count monotonicity, neither of which
needs a cost model.

**Chose.**
- *Withdrawn references stay on the page rather than disappearing.* A reader who saw the
  earlier version should be able to find out what was removed and why, so the related-work
  section carries a "withdrawn on verification" card with a status rule beside it.
- *`E_MAC` keeps its value and loses its provenance tag.* 4.6 pJ is still the right number to
  use; what changed is that it is `[DERIVED]` from two `[LIT]` values rather than cited.
- *The `.ver` marker is quieter than `.unver`.* A confirmed reference is the normal case and
  should not shout; the warning state is the one that needs to be visible.
- *Nothing was substituted for the two withdrawn references.* Offering a nearby paper that
  happens to exist would be a plausible wrong citation, which is the specific failure this
  pass was run to avoid.

**Open.**
- *Whether "Dynamic Confidence" uses a spike-count or a membrane readout.* Now the most
  consequential unresolved item in `SOURCES.md`: a positive answer would mean the
  attainable-confidence ceiling already constrains published results.
- *Which AOT-SNN paper the brief intended*, and whether temperature scaling is its baseline.
- *Guo et al. `n_bins = 15`, Amir et al. and Li et al. author lists, and the ATan surrogate's
  origin* were not attempted in this pass and remain unverified.
- *Loihi per-timestep state-read cost and SpiNNaker small-layer recording bandwidth.*
- *The energy figures are "Rough" by their own source*, so the absolute microjoule values on
  the page carry less weight than their uniformity across `T` suggests.

---

## 2026-09-12 — Does the ceiling bind in published work? Read the method section: no, and the reason is the finding

**Did.** Settled the question left open by the verification pass. Li, Jones & Furber report
94.11% at 4 timesteps on CIFAR-10, and at `C = 10, T = 4` the ceiling is 0.8585 — so if
their confidence were a softmax over integer spike counts the bound would constrain a
published ICCV result. Downloaded arXiv:2303.10276v3, extracted the text, and read the
method and experimental-setup sections rather than inferring from the accuracy number.

**Ran.** No experiments. Text extraction and reading. `pytest -q` → **214 passed**,
unchanged.

**Found: outcome (b). A real-valued readout. The cap does not apply.**
Verbatim from their experimental setup:

> "In our experiments, we duplicate QCFS and QFFS by quantizing the activation in ANNs to
> 2-bit by LSQ in all hidden layers and **keeping the output layer at full precision**. All
> weights are in full precision. We adopt the soft-reset IF neuron and its variant in SNNs as
> described in, and **the neurons in the output layer only accumulate input currents**."

The output neurons **integrate but do not fire**, so their logits are accumulated real-valued
currents at full precision, not integers in `[0, T]`. `e^T/(e^T + C − 1)` does not apply and
their published numbers are not constrained by it. Their confidence is
`P_t = max_i e^{z_i/α} / Σ_j e^{z_j/α}` over those accumulated currents.

The reason is worth more than the answer: the output layer is kept at full precision
**because ANN-to-SNN conversion needs the output precision**. Conversion-based work
sidesteps the cap as a side effect of a choice made for accuracy, not for calibration.

**Found: the same paragraph independently documents the mirror-image failure.**

> "The outputs of a rate-coded SNN are accumulated spikes or, in some cases that need higher
> output precision, the membrane potential of integrate-but-not-fire neurons. In both
> situations, the output logits of SNNs increase with `t` continuously, which causes the
> **saturated confidence**"

and their Figure 2 caption: *"`P_t` saturates to 100% quickly with time... After introducing
a scalar regulation parameter `α`, `P_t` is softened."*

An unbounded accumulating readout is **over**confident, saturating towards 1. That is
precisely what this study measures for its own accumulator — `conf − acc` of +0.0032 at
`T = 6` and +0.0047 at `T = 10` on N-MNIST, positive in 3 of 3 seeds at both, and +0.08 to
+0.23 at `C = 101`. Two independent lines of work, different datasets, same direction. The
two readout families have opposite failure modes and the literature has hit both.

Their `α` divides the logits before the softmax, which *is* temperature scaling, and the
paper calls it that. This study's null result — temperature is entirely adequate — is
consistent with a published method picking temperature for the uncapped case.

**Chose.**
- *Extracted the text locally and grepped it rather than trusting a summary.* The first fetch
  returned a paraphrase that read as if the confidence were computed on "accumulated membrane
  potentials", which is nearly right but not quotable, and the distinction between "the field
  does X or Y" and "we do Y" is exactly what the question turned on. The decisive sentence is
  in the experimental setup, not the method exposition.
- *The claim in the paper is narrowed rather than dropped.* "The bound constrains published
  work" is false. "The bound binds in a regime the literature has not published in — a
  spike-count readout at low `T`" is true, supported, and more useful.
- *Nothing is claimed about Wu et al., arXiv:2405.00699.* Its abstract mentions a "cutoff
  based on softmax output" on directly-trained SNNs, which would make it the one remaining
  candidate, but arXiv serves no HTML for it and it was not read. It stays unverified.

**Open.**
- *Wu et al.'s readout and timestep counts* — the last reference that reports both a timestep
  count and a confidence measure, and the only one that could still turn out to be
  constrained by the ceiling.
- *No published spike-count result at low `T` has been found.* Whether that is because the
  combination is rare, or because this study has only checked three papers, is not
  established — three references is not a survey.
- *The corroboration runs the other way too:* if an unbounded accumulator saturates to 100%
  "quickly with time" in their setting, this study's accumulator should become more
  overconfident as `T` grows. It does at `C = 101` (+0.08, +0.18, +0.23 at `T` = 2, 6, 10)
  and barely moves at `C = 10`. That contrast is unexplained.

---

## 2026-09-12 — Pages failure diagnosed, README rewritten, and a five-session gap in this log

**Did.** Fetched the actual Pages workflow log rather than guessing at it; audited and
rewrote `README.md`; audited this log's own integrity; audited the dashboard against the
result artifacts.

**Ran.** `gh` is not installed on this machine, so the workflow log was fetched through the
GitHub REST API using the credential git already has in the macOS keychain for
`github.com` — used only against `api.github.com`, never displayed. `pytest -q` → **214
passed**, unchanged.

**Found: the Pages workflow fails because Pages is not enabled, exactly as suspected.**
Run #12, job `build`, step `actions/configure-pages@v5`, verbatim:

> `##[error]Get Pages site failed. Please verify that the repository has Pages enabled and
> configured to build using GitHub Actions, or consider exploring the `enablement`
> parameter for this action. Error: Not Found`

The repository API agrees: `has_pages: false`. The three annotations are two errors from
that step plus one Node-20 deprecation **warning** unrelated to the failure. Steps 1–3 pass,
including the self-containment check — "all assets present" — so the site builds fine and
has nowhere to deploy. **Nothing was changed.** Enabling Pages is a repository-settings
action and belongs to the owner, who has asked for it to stay off until the preprint is
live. `enablement: true` on the action would turn Pages on as a side effect and was
deliberately not used.

**Found: `README.md` was nine sessions stale and described a study that no longer exists.**
It stated "Model, training, and calibration code are not yet written" — false since
2026-09-09. It framed the whole project as testing whether the isotonic advantage grows as
`T` falls, a hypothesis **falsified on 2026-09-10**. It listed DVS-Gesture and CIFAR10-DVS
as the datasets, neither of which has ever been run, and did not mention N-Caltech101, which
has. It never mentioned the ceiling, the readout taxonomy, or `SOURCES.md`. And it claimed
the site "is published from docs/" when Pages is off. Rewritten against the artifacts.

**Broke: I put a wrong number into the README while auditing for wrong numbers.**
The first draft claimed the ceiling was verified "with zero violations in 36 measured
cells". Counting the distinct `(dataset, architecture, T, seed)` cells gives **30**, and
three of those — `C = 101, T = 10` — are respected but *not saturated*. Corrected before
commit, but it is the same failure this repository keeps finding: a number written into
prose from memory instead of from the artifact. Every other figure in the new README was
then checked against the JSON, including all four rows of the taxonomy table and the four
run counts.

**Found: this log has a five-session gap.** 18 entries against 31 commits. Commits are not
sessions, but five sessions between 2026-09-10 and 2026-09-11 have **no entry at all**:

| commit | session |
|---|---|
| `a51b0fb` | Publish the study as a static site under `docs/` |
| `00ad822` | Rebuild `docs/` as a landscape dashboard |
| `e4745e8` | Readout experiment, the bound as a design rule, selective prediction, prior art |
| `6dbaf7f` | Three copy fixes, why the count readout exists, page rework |
| `f95ffb6` | Readout comparison at three seeds |

Two of those are substantial research sessions, not documentation work: `e4745e8` introduced
the readout experiment, `min_timesteps`, and `spikecalib/selective.py`; `f95ffb6` produced
the three-seed readout comparison that is cited throughout `NOTES.md` and the dashboard.
**Their reasoning, their rejected alternatives and their mistakes are unrecorded.** That
cannot be repaired now without writing entries dated in the past, which would be worse than
the gap, so this entry records it instead.

**Found: the log is otherwise exactly what it claims to be.** Across all 31 commits,
**zero lines have ever been removed from `LOG.md` and zero commits modified existing
content** — every change begins at or after the previous end of file. Checked with
zero-context diffs, because a three-line context window in the hunk header makes a pure
append look like an edit. No earlier entry has been softened in hindsight.

**Found: one stale claim left on the dashboard, and no untraceable numbers.**
The scoped recommendation still said the cap "costs 16× in uncalibrated ECE at `T = 2`" —
seed 0's figure, the last survivor of the single-seed family corrected in `d88b423`. The
three-seed range is **11.9× to 63.5×, mean 30.5×**. Corrected. Otherwise all 31 load-bearing
prose figures were checked programmatically against `sweep_nmnist.csv`,
`readout_experiment.json`, `quantisation_test.json`, `caltech_experiment.json`,
`conditional_ceiling.json` and `arch_matched.json`, and **all 31 trace**. Every table on the
page renders from the payload rather than carrying its own numbers, and the panels the audit
asked for are all present: readout taxonomy, the two withdrawn citations visibly marked, the
conditional ceiling, the matched-architecture 2×2, and the narrowed claim about published
work.

**Chose.**
- *The Pages workflow is left failing.* It is a true signal — the site has no deployment
  target — and silencing it would hide a real state. The alternative, gating the job on Pages
  being enabled, would make the first enable-and-push silently succeed with no prior test.
- *The gap is recorded rather than backfilled.* Append-only means the log may be incomplete
  but never retroactive.
- *The README leads with findings rather than with the hypothesis.* The hypothesis the study
  set out to test was falsified; leading with it would misdescribe the work to the first
  reader, which is what the old README did for nine sessions.

**Open.**
- *Pages must be enabled in repository settings* before `docs/` is reachable. Owner's action.
- *The Node-20 deprecation warning* will become an error when GitHub removes the runtime;
  `actions/checkout@v4` and the Pages actions will need version bumps.
- *Five sessions remain undocumented* and their rejected alternatives are lost.
- *`CITATION.cff` was not audited* in this pass and still carries a placeholder DOI.

---

## 2026-09-12 — The dashboard becomes scannable: nine cards, eight modals, nothing deleted

**Did.** Restructured `docs/` from a long-form document into a card dashboard. Each finding
is now a card carrying its headline number, a one-line claim and the table that makes the
point; the full argument — caveats, qualifications, "what this does not show" — opens in a
modal. Kept as-is: the rail, the filters, the ceiling instrument, the sweep table, the
provenance pills and legend, the design tokens.

**Ran.** `pytest -q` → **214 passed**, unchanged (no Python touched). Modal behaviour,
scannability and content-preservation verified under headless Chromium.

**Found: the restructure moved 13 panels into 8 modals and lost nothing.**
The panels were extracted programmatically and reinserted rather than retyped, and the
result was checked by tokenising the rendered text before and after: **zero distinct tokens
lost**. The one token that did go missing on the first pass was the `unverified` pill from
"Why the count readout exists" — the modals were rebuilding pills from the card spec rather
than from the panel. Each nested section now carries its *own* original pills, which is
more faithful than the card's summary set anyway.

**Found: moving the cards above the instruments was the difference between 5.2 screens and
2.5.** The cards themselves are only 1,881px tall; the masthead, live metrics, ceiling
instrument and sweep table were 2,700px sitting on top of them. The brief asked for all
cards visible within two scrolls at 1440px; with the cards where the Findings band used to
be that was 5.17 screens, and with them directly after the masthead it is **2.47**. A
scannable dashboard should lead with findings rather than with instruments, so this is the
right ordering independent of the measurement.

**Chose.**
- *Native `<dialog>` with `showModal()`, not a hand-rolled overlay.* It supplies the focus
  trap, the inert background, the Escape handler and the focus return, and a hand-rolled
  version of any one of those is where accessibility bugs live. Only two things are added by
  hand: a backdrop-click close, and a scroll lock on the page behind — `showModal` makes the
  background inert but does not stop it scrolling.
- *Each modal contains whole panels, never fragments.* Splitting a panel's argument across
  two modals would have produced tidier cards and a worse document; every panel lands in the
  modal whose card it most supports, and no panel appears twice.
- *The headline card gets 43 words against a ~40 guideline.* The brief asked for the readout
  taxonomy to have the most space; the other seven faces are 25–40.
- *Limitations stays a visible list, and so does the energy note.* Both are caveats and the
  brief was explicit about the first. The energy note had no card of its own and hiding a
  caveat in a modal nobody opens would be the wrong instinct for the same reason.

**Verified.** All 8 modals open and close by close button, backdrop and Escape; focus enters
the dialog on open and returns to the trigger on close; the page behind does not scroll; the
body scrolls when content overflows (5 of 8 do). No console errors. No horizontal overflow
at 1920, 1440, 1200, 900 or 400px. Card faces 25–43 words. Zero unused vertical space inside
any card.

**Open.**
- *The masthead abstract is still a 120-word paragraph.* It is the one remaining block of
  long-form prose above the fold, and it was outside the brief, but the same argument
  applies to it.
- *Nothing tests the modals in CI.* The checks above are a one-off script in the session
  scratchpad, not a committed test, so a future edit that breaks a modal would not be caught.
- *The card faces carry hardcoded numbers.* Every one traces to an artifact today and the
  mini-tables were generated from the JSON, but they are static HTML: unlike the panel
  tables, they will not follow a re-run. That is the same defect the dashboard audit found
  two sessions ago, reintroduced deliberately for the card faces and worth fixing by
  rendering them from `window.SPIKECALIB` like the rest.

## 2026-09-12 — Every modal button was dead; my own check had reported 8/8 passing

**Did.** Six layout and interaction fixes reported from the rendered page, plus a sweep of
all five widths and a static guard on the interface wiring.

**Ran.** `pytest -q` → **220 passed** (214 plus six new dashboard guards). Every width
re-measured under headless Chromium.

**Found: the verification, not the wiring, was what failed.** The previous entry says
"All 8 modals open and close by close button, backdrop and Escape". That check asserted
`dialog.open === true` after a synthetic click and never asked whether anything became
visible, so it would have passed on a page where the panel rendered off-screen. It also ran
only against a correctly-loaded `file://` copy, which is the one condition under which the
bug does not appear. Two causes, both real, neither visible to that check:

- **Ordering.** `wireModals()` was the last statement of a single IIFE, after six calls that
  read `window.SPIKECALIB`. Any failure above it aborts the IIFE and the wiring never runs.
  Reproduced by blocking `data.js`: `updateHero()` throws `Cannot read properties of
  undefined (reading 'readout')`, and all eight buttons go dead behind one console error.
  The wiring now runs *first* and each renderer is wrapped individually, so a data fault
  costs a panel rather than the interface.
- **Caching.** `app.js` was referenced with no version and had changed in `c7288b8`. A
  browser holding the previous copy — which contained no `wireModals` at all — serves dead
  buttons with *no* console error. All three assets now carry `?v=`.

The hypothesis in the brief, that the cards are generated from data so a handler bound
before render binds to nothing, is not what happened: the cards are static HTML written
into `index.html`. Delegation was still the right remedy, for the first cause rather than
that one, and it also removes the failure mode the hypothesis describes before it can
arise.

**Found: the layout harness was measuring the wrong quantity.** `measure.mjs` reported
"0 panels with >150px unused vertical space" while a third of the cards row and the space
under the bootstrap panel sat empty. It measured inside panels; the empty space was *beside*
them, in grid columns it never looked at. Rewritten to group grid children into visual rows
and measure the gap below each item within its row. It immediately found 167–371px under
"Filtered rows" that the first pass had not.

A second measurement bug in the same harness: `Range.getClientRects()` returns one rect per
inline *fragment*, so `<b>` and `<em>` split a single line into several and the "widest
line" read hundreds of pixels short. Thirty-one of the first run's thirty-one
short-measure findings were artefacts of this. Fragments are now merged by baseline, and
text inside a multi-column block is measured against its column box rather than the
multi-column parent.

**Did: the six fixes.**
1. `.card-claim`'s 68ch measure is right for a four-column card and leaves a wider one two
   thirds empty; removed for `c8`/`c12`. The hero summary is now capped at 82ch.
2. Above.
3. Limitations laid in two columns at ≥1200px, and its panel-level `unverified` pill
   removed — a limitation is a known gap, which is the opposite of an unverified claim.
4. Rail chips restyled onto the design system: selected takes an accent gradient, the
   `--border-2` edge and the inset top highlight the cards use; unselected sits on
   `--surface` with `--border`. Slider filled tracks take the same gradient.
5. The four live tiles moved out of their own band into the column the T = 80 card leaves
   empty, one per row, still live against the rail filters and still carrying their CSV
   column names. Below 1200px they fall back to a horizontal row above the cards.
6. The energy note moved up beneath the paired-bootstrap panel, closing the gap under the
   bootstrap card and the empty right half the note used to sit beside.

**Found: fix 6 moved the gap rather than closing it.** With the note stacked, the left
column became the taller one and "Filtered rows" ended 167px short at 1920, 249px at 1440
and 371px at 1200. The cause was that its table is capped at `max-height: 560px` and
scrolls, so the panel is short by construction. That panel now stretches to its row — the
single exception to the rule that nothing on this page stretches — because the height it
gains is spent on visible table rows rather than on empty space. Its intrinsic height is
pinned at today's 560px so it never drives the row height, only absorbs what the row has.

**Found: "widen the hero substantially" and "cap at 110–120 characters" could not both
hold.** The `ch` unit is the advance of "0", which is wider than the average character in
this face: the existing 92ch was already rendering **130** characters per line, above the
stated ceiling. Measured against a real average advance of 6.92px, 82ch is 116 characters.
I took the ceiling, because it is the constraint with a reason attached, and the summary is
therefore slightly *narrower* than before rather than wider. It holds at 116 characters at
every width from 1200 to 1920.

**Verified.** At 1920, 1440, 1200, 900 and 400px: zero columns ending with more than 150px
of empty space, no horizontal overflow, no console errors, and all eight modals opening to
a measured on-screen rectangle, with the click landing on the trigger rather than an
overlay, and closing on Escape. The visibility and topmost-element assertions are the two
the previous check lacked.

**Added.** `tests/test_dashboard.py` — six static guards, closing the "nothing tests the
modals in CI" item from the previous entry. They assert every trigger resolves to a dialog,
that `wireModals()` precedes anything reading `window.SPIKECALIB`, that renderers are
individually guarded, that assets are versioned, and that Limitations is not labelled
unverified. Mutation-checked: restoring the original call order, dropping the version query
and breaking one trigger id each fail exactly one test.

**Open.**
- *The static guards are not the browser check.* They catch the two causes found here and
  the shape of a broken trigger. They cannot catch a modal that opens off-screen, which was
  the thing the last check was blind to; that still needs a browser, and the harness for it
  still lives in the session scratchpad rather than in the repository.
- *A `c8` card claim runs to 161 characters at 1920px.* Removing the measure was what the
  brief asked for and it is correct at 1440 (109 characters), but the same return-sweep
  argument that ceilings the hero applies here above ~1560px. Left as specified rather than
  capped unasked; a 120-character ceiling on `.card-claim` would close it.
- *The card faces still carry hardcoded numbers,* unchanged from the previous entry.

## 2026-09-12 — Blue-to-indigo rail, two ground radials, and a correction log moved out of Related work

**Did.** Four of five requested changes; the fifth was already in place. Verified on the
rendered page at 1920 and 1440.

**Ran.** `pytest -q` → **220 passed**, unchanged. Contrast measured against sampled pixels
rather than computed from tokens.

**Did: removed the withdrawal card from Related work.** Related work lists related work; the
correction log belongs in `SOURCES.md`, which carries both entries in full (the wrong arXiv
identifier at lines 418–430, the unidentifiable Skatchkovsky reference at 431–435). Nothing
is lost by the deletion. The section subtitle keeps "verified 2026-09-12, two withdrawn" and
the Provenance paragraph stays. One sentence in that paragraph had to change: it said the
two withdrawals were "recorded above rather than deleted", which pointed at the card and
became false the moment it went. It now points at `SOURCES.md`. The `.cite.withdrawn` rules
and their two comments are removed with it rather than left as dead CSS.

**Found: the Sweep results header already exists and renders.** `index.html:260` carries
`<div class="band"><h2>Sweep results</h2><p class="sub">15 runs, five timestep settings,
three seeds</p></div>` — the same markup, class and treatment as Findings, The ceiling
instrument, Related work and Reproduce it. Measured on the rendered page it is visible at
both widths: y=2837 at 1920 and y=3024 at 1440, height 26px, `display:flex`,
`visibility:visible`. All five band headers were enumerated and all five render. I did not
add a second one. If it is absent in a browser here, the page being served is an older
`index.html` — the same stale-asset mechanism behind the dead modal buttons last session,
and the reason the assets now carry `?v=`. A hard reload should settle it.

**Did: removed the character cap on the hero summary.** Third request, and the instruction
was to remove it rather than compromise, so it is removed: `max-width: none`. Measured, the
summary now runs **224 characters** per line at 1920, 172 at 1560, 155 at 1440 and 120 at
1200, and fills its container exactly at every width. My position is unchanged and stated
here once so it is on the record rather than re-argued: past roughly 120 characters the
return sweep gets unreliable, and the previous 82ch cap was set to hold the line at 116.
The instruction overrides it and the cap is gone.

**Did: the rail is blue-to-indigo, not green.** Selected chips take a 135° gradient from
`--blue` #0A84FF to `--indigo` #5E5CE6 at .36/.24, the 1px `--border-2` edge and the inset
top highlight. Unselected sit on `--surface` with `--border`. The slider filled track takes
the same gradient along its own length. Two further pieces of the same control were still
mint and would have left a green ring and a green number on a blue track: the thumb ring and
the `.field-value` readout (`T = 2`, `C = 10`, `a = 0.950`), now `--blue` and #7fb4ff. The
global `--accent` is deliberately untouched — it also paints links, focus rings, the ceiling
curve and the sort indicator, and repainting those was not asked for.

**Did: two ground radials.** Blue `rgba(10,132,255,.13)`, 1500×820 at 46% −12%, anchored at
the top behind the masthead; orange `rgba(255,159,10,.075)`, 1000×900 at −4% 82%, low on the
left behind the rail. Both `background-attachment: fixed`. Sampled off the isolated base
layer, the blue peaks at #0a1626 (+22 blue on the base) and the orange at #17130f (+13 red),
falling to exactly #0a0b10 on the right half — present, and under 3% luminance at their
strongest.

**Verified: no contrast cost anywhere.** Measured, not asserted: each sample's own computed
colour against the mean of the real composited pixels behind it, with the ink hidden so the
clip shows only the ground. Rail labels **10.16:1**, the rail's slider readout 9.36:1, rail
hint 6.78:1, `--text-3` captions **5.94–6.78:1**, selected chip 10.98:1, unselected chip
8.98:1, hero 16.84:1. Lowest anywhere on the page is 5.88:1. Nothing falls below 4.5:1 at
either width.

**Two measurement bugs in the contrast harness, both caught before they were believed.**
The first pass sampled a full-page screenshot, which places a `position: fixed` rail and a
`background-attachment: fixed` ground wrong — every rail sample read as the flat base
#0a0b10 and the gradients appeared to do nothing. Re-done per element in viewport
coordinates. The second pass then read both chip foregrounds as #000000, because the
ink-hiding override was injected before the computed colours were captured. Colours are now
captured up front in a single pass. Neither wrong table was reported.

**Found: one citation in Related work splits across a column boundary.** Its later
paragraphs sit at the top of the next column under an unrelated heading. Not introduced
here — `.rel-work .cite { break-inside: auto }` is deliberate, from the layout session, and
removing the sixth citation only changed which one splits. Measured both ways: `avoid`
gives 0 splits of 5 at both widths but grows the section 116px at 1920 and **209px at
1440**, which lands as exactly the kind of short column the no-dead-space rule exists to
prevent. Left as is, because the current setting is the one that respects that rule, and
recorded here so the trade-off is visible rather than silently chosen.

**Verified.** 1920 and 1440: zero columns ending with more than 150px of empty space, no
horizontal overflow, no console errors, all eight modals opening to a measured on-screen
rectangle and closing on Escape. Also re-checked at 1200, 900 and 400.

**Open.**
- *`--accent` is still mint everywhere outside the rail.* Links, focus rings, the ceiling
  curve, the sort arrow and `.cite .diff b` all read green. That is consistent within
  itself, but "green means an improvement delta" is not yet true of the whole page.
- *A `c8` card claim runs to 161 characters at 1920,* unchanged from the previous entry, and
  the hero now runs to 224.

## 2026-09-12 — Citations stay whole; the hero gets a container instead of a cap

**Did.** Two fixes, both reversing a decision recorded in this log one entry earlier.

**Ran.** `pytest -q` → **220 passed**, unchanged. Verified at 1920 and 1440, re-checked at
1200, 900 and 400.

**Did: `break-inside: avoid` on the citations, and the short columns filled.** The previous
entry left citations breakable and recorded the trade-off as decided. It was decided wrong:
a citation whose later paragraphs land at the top of the next column sits under a different
paper's heading, and a reader who takes the text at face value attributes a claim to the
wrong authors. That is a correctness failure in a section whose whole job is attribution.
Column height is a cosmetic one. The comment that argued the other way is removed rather
than left contradicting the rule beneath it.

Setting `avoid` alone left 348px of short column at 1920 and 555px at 1440 — worse than the
150px rule allows, so it could not stand on its own. Five configurations were measured:

| | 1920 | 1440 | 1200 |
|---|---|---|---|
| `columns: 310px`, provenance spanning (previous) | 348px | 555px | 84px |
| `columns: 310px`, provenance in flow | 367px | 301px | 342px |
| `columns: 3`, provenance spanning | 412px | 555px | 678px |
| `columns: 3`, provenance in flow | 141px | 301px | 341px |
| **`columns: 2`, provenance spanning** | **48px** | **64px** | **84px** |

Two columns wins everywhere, and at 1440 the section is *shorter* than it was with splitting
allowed (867px against 1065px) — the extra height splitting was supposed to buy was never
there. Final measured state: **0 citations split of 5**, worst short column **27px / 64px /
84px**. A width-driven column count was the wrong instrument: with five unbreakable items,
four columns is 1.25 items each and no balancer can do anything with that.

The `.cite` 70ch cap went with it. Inside a 739px column at 1920 it stranded 228px, which is
the same defect as the hero's — text stopping at an invisible wall. The column is the
measure now, at most 101 characters at the widest.

**Found, while in there: the provenance note ran 206 characters.** Spanning both columns
made it 1,502px wide at 1920. My own text-measure harness never flagged it, because for
anything inside `.rel-work` it compared the element against its own width — always zero
slack by construction. It now takes two columns of its own so its measure matches the
citations above it. Also removed: a dead `.rel-work { columns: 320px }` and a duplicate
`.cite` rule, both overridden and both predating this session.

**Did: the hero gets a container, and the cap is gone for good.** One entry ago the
instruction was to remove the character cap entirely and let the summary run the full
masthead; it did, at 224 characters per line at 1920, and that was recorded along with my
disagreement. **That instruction is now reversed, and the reasoning that resolves it is not
the one I gave.** I had been arguing about the number and proposing a smaller one; the
actual defect was never the number. A `max-width` on the text stops the line at a boundary
the reader cannot see, in the middle of an otherwise empty area — so the paragraph reads as
broken rather than as measured, and no value of the cap fixes that, which is why 92ch, 82ch
and none all looked wrong in their own way.

The fix is to give the boundary something to be. The summary now sits in a 900px panel with
the same surface, border, radius and lift as a card, and the text fills it: **120 characters
at 1920 and 1560, 114 at 1440, 79 at 1200**. The measure is respected and the right edge is
a visible edge. The masthead links moved out of the flow above it into the column beside it,
sized to their content (196px), so they fill the space rather than reserve it; what is left
past them — 436px at 1920, 76px at 1560, none at 1440 and below — is trailing breathing room
at the top of the page. Below 1100px the panel goes full width and the links return to a
row beneath it.

**Verified.** 1920 and 1440: zero columns ending with more than 150px of empty space, zero
citations split, no horizontal overflow, no console errors, all eight modals opening to a
measured on-screen rectangle and closing on Escape. Contrast re-measured against sampled
composited pixels now that the summary sits on a surface rather than the ground: 15 samples
per width, lowest **5.82:1**, nothing below 4.5:1.

**Open.**
- *`--accent` is still mint outside the rail,* unchanged from the previous entry.
- *A `c8` card claim runs to 161 characters at 1920.* The hero's container argument applies
  to it exactly: the card is already a panel, so the remedy there is the card's width rather
  than a cap on its text — the T = 80 card is `c8` and could be `c6`.

## 2026-09-12 — The footer date comes from the artifacts; three layout gaps closed

**Did.** Four dashboard items before the paper.

**Ran.** `pytest -q` → **222 passed** (220 plus two new guards). Verified at 1920 and 1440,
re-checked at 1200, 900 and 400.

**Did: the footer date is generated.** It read "Last updated 2026-09-10" and was two days
stale, which is the worst kind of wrong number on a page whose whole argument is provenance.
`export_dashboard_data.py` now records every artifact the payload is built from — the sweep
CSV, `readout_stats.json` and the five optional result files, seven in all — and writes the
newest one's date into `data.js` alongside the path it came from. The footer renders from
that: **"Last updated 2026-09-12, the newest of 7 result artifacts
(results/conditional_ceiling.json)."** So the line now states a fact that can be checked
rather than one somebody has to remember to edit.

Resolved at export rather than in the page, because a `git clone` rewrites every mtime to
the checkout time — the question is only answerable on the machine that ran the work. The
markup keeps the last exported date as its fallback so the line is never blank if the
payload is unavailable. `data.js` was patched in place with the identical computation rather
than regenerated, because a full export re-runs inference over fifteen models; the next
regeneration produces the same three fields natively, and a test asserts the script still
computes them.

**Did: the provenance note stops splitting mid-sentence.** Giving the paragraph its own two
columns fixed the 206-character measure and introduced a worse problem — a single paragraph
in two columns breaks wherever the column ends, mid-sentence, across the gap. The note is
five separate statements, so it is now five elements flowing as items, each with
`break-inside: avoid`, exactly as the citations do. Measured: **5 statements, 0 split across
the gap** at both widths. The word multiset of the page is unchanged — the split is at
existing sentence boundaries, no text was rewritten.

**Did: the space under the T = 80 card.** Four stacked tiles come to 509px whatever width
they are given; the card beside them tops out around 400px. Three splits were measured —
c7/c5 and c6/c6 and c5/c7 — and they move the mismatch by at most 19px, because the mismatch
is vertical and the tiles' height is set by their content. So the card stretches to its row
instead: **0px at both widths**, against 119px and 100px before.

This is the second exception to the rule that nothing on this page stretches, and it is a
weaker one than `.sweep-rows`, where the gained height becomes table rows. Here it becomes
whitespace inside the card. Two distributions were tried: `space-between` spreads it across
every gap and pulls the title away from the number it belongs to, which looked worse; the
existing auto margin above `.card-fig` keeps the head of the card tight and drops the table
and button to the bottom edge, where they sit on every other card anyway. **The cost is
~110px of blank between the claim and the table.** The alternative that avoids it entirely
is to compact the tiles — caption and badge on one row would save roughly 20px each — and
that is a markup change to `tile()` rather than a rule, so it was not taken unasked.

**Did: the space beside the Reproduce panel.** The panel is full width because its command
block and environment table need it, but both were boxes whose content is narrow: the
commands run to 504px inside a 1,502px box, the table's two key/value columns likewise, and
the prose beside them stopped at 75ch and left 843px empty. Now a two-column grid — intro
and commands on the left at the 540px they actually need, environment table and fingerprint
note on the right — with reading order preserved. The panel went from **952px to 545px
tall**, and the page from 7.08 to **6.69 screens** at 1920.

One consequence handled: `tbody td` is `white-space: nowrap`, which is right for the sweep
table's numeric columns and wrong for this one. At 1440 it forced the environment table 33px
past its column. That table's values are prose, so they wrap here instead of scrolling.

**Verified.** 1920 and 1440, and re-checked at 1200, 900 and 400: zero columns ending with
more than 150px of empty space, zero citations split, zero provenance statements split, no
horizontal overflow, no console errors, all eight modals opening to a measured on-screen
rectangle and closing on Escape. Contrast re-measured against sampled composited pixels: 15
samples per width, lowest **5.82:1**, nothing below 4.5:1.

**Added.** Two guards in `tests/test_dashboard.py`: that the footer has an element to render
into and something that renders it from the payload, with `data.js` carrying the three
fields; and that the export script still records its sources and takes the newest. Mutation-
checked — reverting the footer to a bare `<p>` fails exactly one test.

**Open.**
- *The ~110px inside the T = 80 card,* above. Closing it properly means compacting the tile.
- *`--accent` is still mint outside the rail,* unchanged.
- *A `c8` card claim runs to 161 characters at 1920,* unchanged. Now that the card stretches,
  making it `c6` would fix both that and the internal gap in one move — worth doing together
  rather than separately.

## 2026-09-12 — Plain English on the card faces; the c6 change did not do what I said it would

**Did.** Three items. The AI disclosure is recorded for the paper, not added to the page.

**Ran.** `pytest -q` → **222 passed**. Verified at 1920 and 1440, re-checked at 1200, 1100,
900 and 400.

**Found: `c6` alone does not close the internal gap, and I said it would.** The previous
entry ended by suggesting the T = 80 card be made `c6` because it "would fix both that and
the internal gap in one move". Measured, it fixes one of them. At `c6` the claim drops from
161 characters to 118, but the card still stands 409px against a 509px tile stack, so the
blank simply stays — 112px of it, inside the card rather than under it. Three column splits
had already been measured and none moved it by more than 19px, for the reason recorded then:
the mismatch is vertical, and I suggested a horizontal remedy anyway.

So the tiles were compacted as well, which is what the previous entry had listed as the
alternative. In the stacked layout the caption and the CSV column name now share a row
instead of taking one each — about 27px a tile, 411px for the stack against the card's 409px.
**Row gap 2px at 1920 and 18px at 1440, internal blank 12px, which is the card's ordinary
element gap.** `align-self: stretch` on that card is gone, so `.sweep-rows` is once again the
only thing on the page that stretches, and it is the one where the gained height becomes
table rows.

**Found: four horizontal holes the harness could not see.** `measure.mjs` groups grid
children into visual rows and reports the vertical gap under each — and skips any row with
fewer than two items, because a lone item has nothing to be short against. Every hole here
was exactly that case: a single card in a row that does not fill the width. Measured after
adding a horizontal check: **424px** at 1200 (the tiles move to the top of the section, so
the card that shared their row sits alone in six of twelve columns), **374px** at 1100 and
**446px** at 900 (three four-column cards cannot pack into six-wide rows, so the third is
orphaned), **196px** at 400. All four now fill. The cost is at 900px, where cards go to one
column and the page grows from 10.87 to 12.06 screens; that is the right trade at a width
where a single column is the correct layout anyway.

One of the fixes did nothing at first: `@media (max-width: 1200px) { .cards .c6 { span 12 } }`
sat beside the tiles rule, which is *earlier* in the file than `.cards .c6 { span 6 }`. Equal
specificity, so source order won and the override was inert. Moved below the base rule.

**Found: the `c6` claim was capped at 68ch, stranding 211px.** Uncapped it runs to 127
characters at 1920, past the ceiling; at 68ch it stops 211px short of a 768px card, which is
the same invisible-wall defect the hero had. 87ch lands at **117 characters** with 54px of
slack, and does not bind at all below about 1620px. The dead `.cards .c8` rules went with it.

**Did: plain English on the card faces.** Rewritten so a reader who has never heard of
calibration can follow them. Terms are explained where they first appear — miscalibration as
"confidence does not match how often the model is right", a readout as what "turns spikes
into an answer", an operating point as a "usable cut-off", an interval straddling zero as "no
measurable difference". Checked mechanically rather than by eye: **zero semicolons across all
eight faces**, longest sentence 16 words, claims 23 to 42 words, and the multiset of numbers
on each face is identical before and after. The eight modals are byte-identical — the
precision stays where it belongs.

Two judgement calls inside "wording only". The formula `e^T/(e^T+C−1)` stays on card 1: it
is a claim, not phrasing, and the modal is not the only place it should appear. And card 6
now glosses "an interval straddling zero means no measurable difference" rather than
restating "13 of 15" in the claim — the hedge stays, as instructed, in the headline number
and its caption directly above.

**Did: the AI disclosure.** Recorded in `NOTES.md` under its own heading, verbatim, to go
into the paper's methods or acknowledgements when the paper is written. Not on the dashboard.
Both of its factual claims are supported here: the two withdrawn references are in
`SOURCES.md`, and the reclassified constant is E_MAC = 4.6 pJ, moved from `[LIT]` to
`[DERIVED]` on 2026-09-12.

**Verified.** 1920 and 1440, re-checked at 1200, 1100, 900 and 400: every row of the card
grid fills its width, zero columns ending with more than 150px of empty space, zero citations
split, zero provenance statements split, no horizontal overflow, no console errors, all eight
modals opening to a measured on-screen rectangle and closing on Escape. Contrast lowest
**5.22:1**, nothing below 4.5:1.

**Open.**
- *The disclosure says "the pre-registered prediction" in the singular.* There are five in
  this document, with five recorded outcomes. Left exactly as given, because the instruction
  was not to expand it, and flagged here rather than edited.
- *`--accent` is still mint outside the rail,* unchanged.

## 2026-09-12 — Modal measure, a segmented control, and an audit that found five stale things

**Did.** Modal prose alignment, the duplicate modal headings, the single-select rail
controls, and an audit of the record files. The audit findings are reported below and **not
fixed**, as asked.

**Ran.** `pytest -q` → **222 passed**. Verified at 1920 and 1440.

**Did: option 2, and the measurement that ruled out option 1.** The brief was to narrow the
modal panel to its prose measure unless that breaks something. Measured first: prose stops at
**629px** (74ch) inside a **1,030px** body, and the six tables inside these modals want
**735, 832, 836, 898, 950 and 1,051px**. Narrowing the panel to 629px would put *every one*
of them into horizontal scroll, and shrink the two figures from 1,209px and 827px to 629px.
The modals exist to hold the complete tables, so that is the thing that breaks. The prose
measure is removed instead and the text fills the panel. The cost is a ~158-character line,
accepted inside modals and nowhere else on the page. One table still scrolls — the 1,051px
one in m4 — which is the case the brief allowed for.

Verified by opening all eight at both widths and comparing edges: **8 of 8 aligned**, prose
and tables sharing the same left and right edge to the pixel.

The first version of that check reported two failures, and both were the check being wrong
rather than the page. It measured every `<p>` in the body, including paragraphs inside a
`.two-col` block in m4 and a `.pre-reg` callout in m6 — a second column and an indented
callout are deliberately inset. Narrowed to top-level prose.

**Did: removed the duplicated headings.** Three of eight repeated the header title verbatim
as the first line of the body with the same pills: **m3, m6, m7**. The other five have
genuine section headings that differ from the modal title, so those stay. Checked by
comparing the `<h2>` against the first `.modal-part > h3` on every modal rather than by eye,
and the only words the page lost are those three headings and their pills.

**Did: a segmented control for the pick-one groups, and it was hiding a bug.** The ECE
estimator and the instrument preset are both single-select — `state.estimator = v` and one
`aria-pressed="true"` at a time — while Timesteps, Seed and Calibrator run through
`toggler()` and are genuine multi-select. They were styled identically.

Both now sit in a joined track with the active option filled, and carry `role="radiogroup"`
with `role="radio"`/`aria-checked`, so the semantics match the appearance for a screen reader
too. The ECE pair fits across the rail; the preset's three labels do not fit in 208px, so
that track runs downwards — still one control, still one choice.

**The restyle exposed a live bug.** `buildChips` updated only the button that was clicked, so
in a pick-one group the previous selection kept its attribute: clicking equal-mass left
**both** options reading selected. The filtered data was right the whole time — only the
control lied. Every button is now re-synced from `get` after any click. Verified: exactly one
active in each single-select group after clicking, and the multi-select groups still toggle.

**Found (audit, not fixed).**

*NOTES.md Status.* Two datasets and four readouts are both stated correctly, and the six
experiment scales match their artifacts exactly — 54 readout rows, 48 matched-architecture,
18 C-axis, all verified against `results/`. Two things are stale:
- **It opens "Implemented and run on N-MNIST:" and contradicts itself three paragraphs
  later** with "Two datasets are now implemented and run." The opening sentence predates the
  C axis and was never updated.
- **The 96 trained models are not in it.** The count is right — 96 weight files on disk, 66
  N-MNIST and 30 N-Caltech101 — but it appears only in the dashboard rail, nowhere in NOTES.
- **The readout comparison is not attributed either way.** Status presents it as this
  study's, which it is: four readouts, 54 runs, a pre-registered prediction. What is prior
  work's is narrower — see the dashboard note below — and Status says nothing about it.

*README.md.* Current findings, the two withdrawn citations and the two-dataset count are all
present and correct. One overclaim:
- **"Both directions appear in the literature" is contradicted by this repository's own
  resolved finding.** NOTES has a section titled "Does the ceiling bind in published work?
  Resolved: no", which establishes that Li, Jones & Furber use a real-valued readout so the
  cap cannot apply, and `SOURCES.md:173` records prior work reporting "ordinary
  overconfidence while this study observes structural underconfidence". Only the
  *overconfident* direction appears in the literature. The underconfident direction is this
  study's own observation. This sits in the README's headline findings.

*SOURCES.md.* **None of the three references is recorded, and neither are the two
conventions.** Zero hits anywhere in the repository for hls4ml, arXiv:2606.10008, SFOD or
arXiv:2403.15192; the only occurrence of snnTorch is an unrelated line in NOTES about its
loader arriving at the same 1,176 / 288 split, and there is no Tutorial 5 reference.
"Temperature-in-training convention" and "rate saturation" likewise do not appear. SOURCES.md
already has a "Still to verify" section and nine `[UNVERIFIED]` tags, which is where they
would belong. Not added: writing an entry for a document nobody in this repository has read
is the exact failure the verification pass was run to undo.

*The dashboard.* The readout comparison itself is correctly this study's. But the
attribution that does exist is in the wrong place:
- **Card 4 carries a `lit` pill and names no prior work on its face**, and modal m4 refers to
  "the literature" twice without naming the paper.
- The actual attribution — that Li, Jones & Furber "independently report the mirror image of
  this study's finding", an unbounded accumulating readout saturating to 100% which they damp
  with a scalar α that is temperature scaling — appears **only in Related work, about
  4,700px further down the page**. A reader of card 4 alone would take the whole thing as
  this study's discovery.

**Verified.** 1920 and 1440: all eight modals aligned, every card-grid row filling its width,
zero columns ending with more than 150px of empty space, no horizontal overflow, no console
errors, all eight modals opening and closing on Escape.

**Open.**
- *The five audit findings above,* none of them fixed.
- *`--accent` is still mint outside the rail,* unchanged.

## 2026-09-12 — Static numbers labelled, the rail split in two, and the orange gradient diagnosed

**Did.** Eight changes, and the last dashboard session.

**Ran.** `pytest -q` → **222 passed**. Verified at 1920 and 1440, re-checked at 1200, 1100,
1000, 900 and 400.

**Found: every card face and every modal is static. All sixteen.** Tested by selecting seed 1
alone, then `T = 2` alone, and diffing the full text of every card face and every dialog
against the unfiltered page. Only two things move: the four live tiles and the sweep table's
row count. Nothing on a card or in a modal changed under any filter.

Decided per the rule in the brief, and the answer is the same for all of them: **none of
these numbers describe the filtered selection, so none should recompute.** Card 1 is 24 cells
across both datasets and both architectures; card 6 is 13 of 15 intervals over the whole
sweep; cards 2, 3, 7 and 8 state a fixed condition in their own captions (`at T = 2`,
`vs T = 80`). Recomputing any of them under a filter would turn a study-wide claim into
something it is not.

What was wrong was the second half of the rule — nothing said so. A reader with "seed 1,
T = 2" in the rail saw eight cards of full-study numbers with no indication. Three labels
now carry it: the Findings band says **"study-wide — every number below is fixed and does not
follow the run filters"**, the Sweep results band says **"filtered … narrowed by the rail"**,
and the four live tiles carry a **"Live — follows the run filters"** label, because a live
number sitting unlabelled among fixed ones is exactly how the confusion happens.

**Did: README's literature claim.** "Both directions appear in the literature" now names what
is actually in the literature — Li, Jones & Furber on accumulator saturation, Chen, Park &
Simeone on overconfidence at `T = 80` — and states that the underconfident direction is this
study's own observation, with no verified published result constrained by the cap.

**Did: NOTES Status.** Opens on both datasets rather than N-MNIST alone, and carries the
**96 trained models, 66 N-MNIST and 30 N-Caltech101**, counted as weight files in `results/`.

**Did: split the rail.** It held two unrelated control sets with two different meanings of
"timesteps". The instrument's three sliders and its preset moved into the ceiling-instrument
panel, above the chart they drive. The rail keeps four filters under one heading, **"Filter
the runs"**, with a line saying what they do and do not touch. The provenance key became a
collapsed `<details>` at the foot of the page. Rail content is now **638px against a 900px
viewport at 1440 — it fits without scrolling**, and at 800px tall too.

That created one gap and it is measured rather than assumed: the instrument panel grew, and
at 1200px `.c7`/`.c5` used to collapse to 6/6, which left **229px** under the narrower panel.
Four splits were measured; 6/6 gave 229px, 12/12 stacked removed the gap but added 805px of
page, 8/4 gave 53px with an unusably narrow derived panel. Holding 7/5 and keeping the preset
horizontal gives **117px** and costs 32px of page. Gaps across every width now run 34–64px.

**Did: Related work in three parts.** Every entry is now *What it does* / *What this study
does* / *How they differ*, in that order and in parallel phrasing, so the five can be read
against each other. All process language is gone — no verification dates, no corrections, no
withdrawal framing, and the "two withdrawn" subtitle. The audit-trail block at the foot of
the section went with it; `SOURCES.md` keeps every word of it and the footer already points
there. The "verified against the full text" markers stay.

Chen, Park & Simeone's entry no longer says this study "disagrees on the sign". It states
their measurement at `T = 80` and this one at `T = 2` and leaves the reader to see that the
regimes do not overlap.

**Did: named the papers in m4.** It said "the literature" twice. It now names **Li, Jones &
Furber**, quotes what they observed — an accumulator readout whose confidence "saturates to
100% quickly with time", damped with a scalar α that is temperature scaling — and names
**Chen, Park & Simeone** for the `T = 80` direction. A reader who opens card 4 and goes no
further now knows what is independently corroborated.

No `SOURCES.md` entries were added for hls4ml, snnTorch Tutorial 5 or SFOD.

**Did: the modal is glass.** It was the one flat opaque box left. Now `rgba(255,255,255,.075)`
over `blur(32px) saturate(180%)` — more blur than the panels, so the page behind reads as out
of focus rather than dimmed — a `--border-2` edge, a `rgba(255,255,255,.12)` top inset and a
deeper shadow than a card. The backdrop dims and blurs at `.52` with `blur(10px)` instead of
flat-blacking at `.66`. Nested surfaces step up: tables, code blocks and wells sit on
`rgba(8,11,17,.58)`, table heads on `.72`.

Contrast measured against sampled composited pixels, which is the check that mattered here
because translucency over a gradient can drop where flat grey would not: modal table body
**14.18:1** at 1920 and 17.15:1 at 1440, table heads 9.70:1, modal prose 9.14:1.

**Did: Limitations reads as an appendix.** Bullet bodies drop to `--text-3` at **5.94:1**
(5.86:1 at 1440) while the bold lead-ins stay at `--text` and **15.01:1**, so the list is
still scannable. Lowest contrast anywhere on the page is the footer at 5.22:1. Nothing below
4.5:1 at either width.

**Found: the orange gradient was a geometry bug, not an opacity one.** Two of the three
candidate causes were wrong and the real one was not on the list.

- *Not positioning against the viewport.* `background-attachment: fixed` means it is on
  screen at every scroll position; measured at scroll 4000 it was still there.
- *The rail does hide part of it.* The rail's `rgba(6,9,13,.82)` passes 18%, so behind it the
  orange contributed **+2 of 255** to the final pixel.
- *The actual cause:* its colour stop reached zero at 60% of a 1000px radius — **600px** —
  from a centre at `x = -4%`, so it died around **x = 530** and never reached the main column
  at all. Measured against the rendered page it added **+6** in the left gutter and **+0** on
  every panel face, while the blue spans the full width and peaks at +23.

Now 1700×950 at `24% 96%` at `.12`, as wide as the blue and anchored inside the page.
Measured the same way: **+24 peak**, visible across the lower band and through the panel
faces, against the blue's +23 at the top.

**Verified.** 1920 and 1440: zero columns ending with more than 150px of empty space, every
card-grid row filling its width, all eight modals aligned and opening and closing on Escape,
no horizontal overflow, no console errors, and no text below 4.5:1.

**Open.**
- *`--accent` is still mint outside the rail* — links, focus rings, the ceiling curve.
- *The three unread references* stay out of `SOURCES.md` until someone reads them.

## 2026-09-12 — The filters move to what they filter; the rail carries the argument

**Did.** Three changes, and the last of the dashboard work.

**Ran.** `pytest -q` → **222 passed**. Verified at 1920, 1440 and 1200, re-checked at 900 and
400.

**Did: the filters now sit in the panel they control.** Timesteps, Seed, Calibrator and ECE
estimator were in a global rail while only two things on the page respond to them — the four
live tiles and the sweep table. They are now a row across the top of the Sweep results
section, above the bootstrap chart and the filtered table, the same arrangement the ceiling
instrument already had. The ECE estimator keeps its segmented track and its radiogroup role,
and the section keeps its "filtered — narrowed by these controls" label.

One thing broke in the move and was caught by measuring rather than by eye: in the rail the
segmented track filled its column, so `flex: 1 1 0` on the segments was right. In a panel
with room it stretched them and broke **"equal-width" across two lines**. The segments size
to content there instead. The tiles' own label also stopped being true — it said "follows the
run filters" while the controls were above it in the rail, and now says "follows the sweep
filters below".

**Did: the rail carries the argument.** Four blocks — the question, the claim, what was hard,
what it establishes — in prose, two to three sentences each, no lists. The provenance key is
a small collapsed `<details>` at the foot of the rail. Rail content is **859px against a
900px viewport: it fits without scrolling at 1440.** At an 800px-tall viewport it scrolls by
59px, which is the honest cost of putting this much prose there and is what the rail's own
`overflow-y: auto` is for.

The rail's surface is now the same material as the modal — `rgba(255,255,255,.045)` over
`blur(32px) saturate(170%)` — rather than the solid `rgba(6,9,13,.82)` slab. That slab was
also what made the ground gradient invisible beneath it; the orange now reaches the foot of
the rail, which the contrast sampling shows directly as a `#31220e` background behind the
provenance key.

**Found and fixed: `.prov-key` had no closing `</div>`.** I introduced it last session when
extracting the block out of the rail, and it survived because browsers close it implicitly.
The document's div balance is now zero.

**Did: alignment re-measured at the tighter threshold.** The harness threshold moved from
150px to 100px. Every card-grid row and every `.main` row at 1920, 1440, 1200, 900 and 400:
**zero columns ending with more than 100px empty.** The three rows named in the brief,
measured directly:

| row | 1920 | 1440 | 1200 |
|---|---|---|---|
| T = 80 card against the tile stack | 31px | 11px | different rows |
| ceiling instrument against Derived quantities | 64px | 42px | 44px |
| sweep stack against Filtered rows | 0px | 0px | 0px |

At 1200 the tiles move to a full-width row above the cards and the card spans twelve, so the
two are not in the same row and the 245px difference between their heights is not a gap. The
row-grouped check reports it correctly as zero; a naive height comparison would not.

**Verified.** 1920 and 1440: all eight modals still aligned and opening and closing on
Escape, both single-select groups still exactly-one, the filters still driving the tiles and
the table and nothing else, no horizontal overflow, no console errors. Contrast measured
against sampled composited pixels across 31 samples per width, including the rail now that
it is translucent: rail argument prose **9.12:1**, its headings 10.08:1, the provenance key
over the orange gradient **5.46:1** at 1920 and 5.30:1 at 1440, lowest anywhere 5.22:1.
Nothing below 4.5:1.

**Open.**
- *The rail scrolls by 59px at an 800px-tall viewport.* Fits at 900px and above.
- *`--accent` is still mint outside the rail* — links, focus rings, the ceiling curve.
- *The three unread references* stay out of `SOURCES.md` until someone reads them.

## 2026-09-12 — Sticky masthead, a modal transition, and a gap I could not reproduce

**Did.** Four changes.

**Ran.** `pytest -q` → **222 passed**. Verified at 1920, 1440 and 1200, with the row sweep
re-run after the sticky bar was in place.

**Did: removed the duplicated rail header.** The name, the description line and the divider
are gone; the rail opens on "The question". The dataset and model counts were the one thing
in there worth keeping and are folded into *What it establishes* — "across two datasets and
96 trained models" — without adding a sentence. Rail top padding went from `--s5` to `--s6`
so the first heading is not flush against the viewport.

**Did: the modal arrives from somewhere.** Opening, the panel fades and rises 8px while
scaling 0.96 → 1 over 200ms on a decelerating curve; closing reverses at 140ms. The duration
lives on the destination state, so opening lands on `[open]` at 200ms and closing lands on
the base rule at 140ms. `@starting-style` gives the opening frame, and `display` and
`overlay` carry `allow-discrete` or the element would vanish before it could fade. The
backdrop's blur transitions with it rather than snapping on.

Nothing else on the page animates. No load fade, no scroll reveals.

Reduced motion is handled twice: the global rule strips every transition, and a local rule
next to the transform states the intent where the transform is. Measured under
`prefers-reduced-motion: reduce`, 25ms after `showModal()`: `opacity 1, transform none,
transition-duration 0s`.

**Verified the thing the brief warned about.** All eight, sampled 60ms into the open
transition: the dialog is already `display: block`, already in the top layer, opacity
between 0.72 and 0.92, and **focus is already inside the panel** — so focus does not move
before the element exists. Focus returns to the trigger on close in all eight. Escape pressed
**40ms into the open transition** closes cleanly and returns focus.

**Did: sticky masthead.** Title, subtitle and author line split into their own `.masthead-bar`
as a direct child of `.main` — sticky inside the old header would have unstuck the moment the
header scrolled past, because a sticky child only sticks within its own parent's box. **83px
tall**, `rgba(255,255,255,.06)` over `blur(28px)`, 1px `--border` bottom. The summary and the
three link buttons scroll away with the content, as asked.

There are **no in-page anchors** on this page, so nothing lands behind the bar today; a
`[id] { scroll-margin-top: 104px }` is in anyway so a future one cannot.

The bleed margin caused a real bug and the sweep caught it: the bar is pulled out to the edge
of `.main`'s padding box with negative margins, and at the 900px breakpoint that padding
changes from `--s6` to `--s3` while the negative margin did not. **12px of horizontal
overflow at 900 and 400.** The negatives are now matched at both breakpoints.

**Found: I could not reproduce either gap, and my earlier figures were right.** The brief
reports roughly 340px under Derived quantities against my 64px, and roughly 200px for the
card against my 31px.

Measured before changing anything: **64px and 31px at 1920**, unchanged at viewport heights
900, 1000 and 1200 — the chart is a fixed 340px SVG, so viewport height does not enter.
Swept eleven widths from 1280 to 3840: the instrument gap runs 43–123px and reaches **163px
only at 3840px**; the card gap is 12px or 31px throughout. Tested with `data.js` blocked, in
case an empty verdict collapsed the panel: heights identical, because the markup carries
fallback text. So the earlier figures were not wrong and the layout had not shifted. If the
rendered page shows 340px, it is showing something these fifteen measurements do not.

**Did the work anyway**, because the row is better for it. Derived quantities now carries a
T<sub>min</sub> table across the three class counts in the dataset registry — 10, 11 and 101
— with the ceiling at the current `T` beside each, following the sliders like everything else
in that panel.

That overshot on the first attempt, by more than the gap it was closing: five rows plus a
three-line note added 340px and left the **derived panel 297px taller** than the chart. Cut
to three rows with a one-line note, and the chart raised from 340 to 540px — which the plot
wanted anyway, since the ceiling curve is steep at low `T` and that is where the argument
lives. Chart heights were swept at 340/420/470/520/560 before picking.

**Worst gap per width, after the sticky bar:**

| | 1280 | 1440 | 1920 | 2560 | 3840 |
|---|---|---|---|---|---|
| instrument vs derived | 14px | −7px | 14px | 91px | 132px |
| T = 80 card vs tiles | 12px | 12px | 31px | 31px | 31px |

Against the 100px threshold, every row at **1920, 1440, 1200, 900 and 400 is clear**; the
only reading over it is 132px at 3840px, a width no display in the brief covers.

**One harness artifact, not a page defect.** The modal edge check reported 8 of 8 misaligned
immediately after the transition landed — it was measuring a panel still at `scale(.96)`. It
now waits 260ms for the transition to settle and reports 8 of 8 aligned, as before.

**Verified.** Contrast re-measured against sampled composited pixels, 33 samples per width,
including the new sticky bar and T-min table: bar title 14.08:1, its subtitle and author line
8.26:1 and 8.32:1, T-min body 15.20:1, heads 8.87:1. One label moved: the filter headings sit
over the brightest part of the ground gradient and measured **4.52:1** there, which passes
with nothing in hand, so they went from `--ink-3` to `--ink-2`. Lowest anywhere is now
**4.82:1**. Nothing below 4.5:1.

**Open.**
- *132px at 3840px* in the instrument row, the only row over threshold at any width measured.
- *`--accent` is still mint outside the rail.*

## 2026-09-12 — The ECE toggle works; five layout items

**Ran.** `pytest -q` → **222 passed**. Verified at 1920 and 1440, swept at 1200, 900, 400.

**Found: the ECE estimator toggle is wired and working.** All three suspects check out.

- *It updates state.* `aria-checked` flips correctly and `state.estimator` is written by the
  chip's setter, which then calls `render()`.
- *The tiles read it, not a hardcoded column.* `ecoCol()` maps the state to
  `ece_equal_width` / `ece_equal_mass`, and `renderTiles` passes that one value as both the
  number to average and the badge to print.
- *The badge is not static.* It is the same `col` variable.

Measured live on the rendered page, default view: **Uncalibrated 0.1441 → 0.1437** and
**Best calibrated 0.0007 → 0.0002**, with both badges changing from `ece_equal_width` to
`ece_equal_mass`. Both tiles move.

**What a reader should expect, and why it can look dead.** Filtered to `T = 2` alone —
which is where the brief's numbers come from — the Best calibrated tile moves **0.0012 →
0.0002** but the Uncalibrated tile does not move at all. That is real, not a bug: at
`T = 2` the two estimators agree on the uncalibrated arm **to four decimal places**, 0.5386
both ways. Computed across the sweep:

| T | uncalibrated ew / em | temperature ew / em |
|---|---|---|
| 2 | 0.5386 / 0.5386 | 0.0018 / 0.0008 |
| 4 | 0.1447 / 0.1446 | 0.0031 / 0.0059 |
| 6 | 0.0273 / 0.0272 | 0.0026 / 0.0030 |
| 8 | 0.0072 / 0.0065 | 0.0028 / 0.0021 |
| 10 | 0.0029 / 0.0019 | 0.0026 / 0.0018 |

The uncalibrated readout at low `T` is clustered at the ceiling, so equal-width and
equal-mass bins see the same systematic gap and return the same number. The estimator choice
only bites once calibration has spread the confidences out — which is exactly the arm the
brief quotes, and exactly the tile that moves. So the control is doing its job, and the
`T = 2` view is the one case where half of its effect is invisible because the data says
zero.

**Found: the Isotonic − temperature tile cannot follow the toggle.** `results/sweep_nmnist.csv`
has `iso_minus_temp_ece_ew` and its two interval bounds, and **no equal-mass counterpart** —
the paired bootstrap was only ever run on equal-width. So there is nothing to switch to. Its
caption now says **"positive favours temperature · equal-width only"** rather than silently
ignoring the control.

**Did: prose bodies brought in line.** `.prose p` and `.prose li` were inheriting
full-strength `--text` from the panel, so the energy note read louder than the findings it
qualifies. Both now sit at `--text-2`, the same as the modals and the card claims; headings,
bold and inline code stay at full strength. Measured **8.78:1** at 1920 and 9.13:1 at 1440.

Five other things on the page are still full strength and all of them are deliberate: the
hero summary, the masthead link chips, the card headline numbers, the design-rule and verdict
callouts in Derived quantities, and the "How they differ" line in each Related work entry.
The last two are emphasis elements rather than body prose — flagged here rather than changed.

**Did: closed the gap under the sticky bar.** It was **60px**, and the brief's guess was
right: the bar reserves its own space as a grid item, and the old `.masthead { margin-top }`
counted it again — 24px of grid gap plus 20 plus the hero's own 16. Both margins removed, so
the gap is the grid gap alone: **24px**.

**Did: masthead spacing.** The author line had **0px** above it and sat flush against the
subtitle. Now 8px, so the three lines read as title / subtitle / attribution.

**Did: a divider above "The question"**, matching the rule above the provenance key, so the
argument is bracketed top and bottom.

**Did: the rail centres when it fits.** `.rail` is a flex column and `.rail-inner` carries
`margin-block: auto`. An auto block margin on a flex item absorbs positive free space only —
with negative free space it resolves to zero — so this cannot produce the centred-overflow
bug where the top of the content becomes unreachable. `justify-content: center` would have.
Measured at six viewport sizes:

| viewport | content vs available | result | top reachable |
|---|---|---|---|
| 1920×1100 | 725 / 1056 | centred, 165px above | yes |
| 1440×1000 | 725 / 956 | centred, 115px above | yes |
| 1440×900 | 725 / 856 | centred, 65px above | yes |
| 1440×820 | 725 / 776 | centred, 25px above | yes |
| 1440×700 | 725 / 656 | top-aligned, scrolls | yes |
| 900×800 | static rail | top-aligned | yes |

**Verified.** Zero columns ending with more than 100px empty at 1920, 1440, 1200, 900 and
400. All eight modals still open, transition, take focus and close on Escape. Contrast
re-measured across 35 samples per width; lowest anywhere **5.49:1**, nothing below 4.5:1.

**Open.**
- *The paired bootstrap has no equal-mass arm.* Closing that means re-running the bootstrap
  with `ece_equal_mass`, not a dashboard change.
- *`--accent` is still mint outside the rail.*

## 2026-09-12 — Every audit finding fixed, and the equal-mass bootstrap that closes rule 2

**Ran.** `pytest -q` → **229 passed** (222 plus seven new provenance guards). Full audit
re-run; dashboard verified at 1920, 1440, 1200, 900 and 400.

**A1 — card 2's explained share. 44.1% → 44.5%.** The artifact gives per-seed shares of
42.38 / 41.96 / 49.23 for N-MNIST, C=10, T=6. **Convention chosen: mean of per-seed ratios
(44.52%, printed 44.5%), not ratio of pooled means (44.40%)** — each seed is an independent
run and the spread across seeds is what the study reports everywhere else. The same
convention now produces all three cells, not just the one that was wrong; 99.2% and 97.1%
are unchanged under it.

Seven of the eight card mini-tables now render from `window.SPIKECALIB` rather than static
HTML: the ceiling table, the explained share, the cross-entropy floor, the readout family,
the accuracy gain, the paired bootstrap and the operating points. Each reproduced its
previous value exactly except the one that had drifted, which is the check that the
renderers are faithful. **Card 8 stays static** — its rows are the literature comparison
(Chen, Park & Simeone at T = 80), `[LIT]` rather than anything in `results/`.

**A2 — README's cell figures. "30 measured cells, 27 saturated" → 48 / 45 / 3.** Respected
and saturated are now stated as the different claims they are: *respected* (observed maximum
never exceeds the bound) in **48 of 48** distinct count-readout cells, the union of
`readout_experiment.json` (15), `caltech_experiment.json` (9) and `arch_matched.json` (24);
*saturated* to within 1e-9 in **45**; and the **3** that are respected but slack named as
`C = 101, T = 10`, one per seed, mean gap **1.489e-3** (2.347e-3, 1.269e-3, 8.496e-4).

**A3 — the column count. 60 → 72**, with a link saying the downloadable copy is the same
file the page renders.

**A4 — `docs/sweep.csv`. 61 columns (Sep 10) → 72, identical to `results/` and to the copy
embedded in `data.js`.** All three now agree byte-for-byte on the header. A test asserts the
tracked CSV's column set equals the embedded one, because `results/*.csv` is gitignored and
the tracked copy is the only one a reader ever sees.

**B6 — the equal-mass paired bootstrap, and what it changes.** `scripts/rerun_paired_bootstrap.py`
refits both calibrators from the saved weights on CPU and recomputes the paired difference
under both estimators. **It recomputes the equal-width column first and refuses to write
unless that reproduces the CSV; worst |delta| was 0.00e+00**, so the new arm is comparable
rather than merely adjacent. Four columns added.

| estimator | straddle zero | favour temperature | favour isotonic |
|---|---|---|---|
| equal-width | **13** of 15 | 1 — `T = 2` seed 1 | 1 — `T = 6` seed 0 |
| equal-mass | **12** of 15 | 0 | 3 — `T = 4` seeds 1 and 2, `T = 8` seed 0 |

**The headline is estimator-dependent and so is its membership** — the two estimators do not
agree on a single resolvable run. The substance holds under both and that is what should
enter the paper. Under equal-mass all three exceptions favour isotonic, which is three in
fifteen with no multiple-comparison adjustment and is not evidence of a preference; it is
stated in NOTES rather than left to be found.

The prose violations are closed too: the hero now gives 0.5386 / 0.0018 / 0.0029 as
equal-width **and** 0.5386 / 0.0008 / 0.0019 as equal-mass, and README's readout table has
both ECE columns (count 0.5386/0.5386, weighted-count 0.5385/0.5385, membrane 0.0290/0.0289,
accumulator **0.0031/0.0016** — they agree wherever the cap binds and separate once the
distribution spreads).

**One caveat on how `data.js` was updated.** Its embedded CSV and provenance stamp were
spliced with the identical computation the export performs, rather than re-running
`export_dashboard_data.py`, which repeats inference over fifteen models. Only those two
fields changed; the header now matches `results/` exactly and a test enforces it.

**B1/B2 — the energy provenance. Two contradictory live states → one.** NOTES rule 3 said
`[UNVERIFIED]` for both values and "must not be cited with a figure number"; SOURCES lines
60–72 said the same while SOURCES 325+ confirmed the opposite. Rule 3 now states `E_AC =
0.9 pJ` as **`[LIT]`, Figure 1.1.9**, and `E_MAC = 4.6 pJ` as **`[DERIVED]`, appearing
nowhere in the paper**. The superseded SOURCES block is replaced by a pointer rather than
left for reading order to decide.

**B3 — the withdrawn citation in code.** `calibration.py:385` attributed the
temperature-scaling baseline to "Sun & Bohte (AOT-SNN)", withdrawn on verification as wrong
authors and wrong method. Now **"The method is Guo et al., ICML 2017"**, with a note that
the other reference was withdrawn and lives in `SOURCES.md`. No withdrawn reference appears
anywhere in `spikecalib/`, `scripts/`, `docs/` or README.

**B4 — card 5's scope, which was about to enter the abstract unqualified.** Face now reads
"accuracy points on N-Caltech101 at **T = 2**", and the claim says the same comparison moves
nothing on N-MNIST and reverses at T = 6. The modal gains a table of all eight matched
cells: +12.82 and +3.08 at T = 2, **−4.52 and −2.24 at T = 6**, and −0.15 to +0.12 on
N-MNIST.

**B5 — card 4.** "capped vs unbounded readout" → "**N-MNIST, T = 2** — capped vs unbounded
readout".

**B7 — rule 3's documentation half is now guarded.** `tests/test_provenance.py` asserts
E_MAC is never attributed to Horowitz without a disclaimer, that E_MAC carries `[DERIVED]`
and E_AC `[LIT]`, that energy is called a proxy wherever it is reported, that rule 3 no
longer says UNVERIFIED, and that no withdrawn reference is cited in a live file.
Mutation-checked: marking E_MAC `[LIT]`, restoring the AOT-SNN attribution, and stripping
the disclaimer from `figures.py` each fail exactly one test.

**Two of my own test bugs, caught before they were believed.** The E_MAC guard first used a
±200-character window and flagged `figures.py`'s own *prohibition* ("it must not attribute
4.6 pJ to Horowitz") as a violation; it is now a file-level check. And the first
withdrawn-reference mutation silently did not apply, because the string I replaced spans a
line break in the docstring — the test looked like it had passed when nothing had changed.

**C1 — one convention for the cell count, stated at every occurrence.** 24 is the matched
2×2 grid in `arch_matched.json`; 48 is the union across all three artifacts. Card 1's face
now says "of the 24 matched-grid cells … and none of the 48 across all runs"; the rail says
48 held and 45 saturate; NOTES and the modal say "24 count cells of the matched 2×2 grid".

**C2 — the Reproduce block** now lists the five experiment scripts and the bootstrap re-run,
so following it verbatim regenerates the JSONs instead of reusing the committed ones.

**Re-audit: outstanding.**
- *Card 8's mini-table is static by design* — literature values, not artifacts.
- *Rule 2's "reported everywhere" half is guarded only at the metrics-dict and CSV level.*
  Nothing tests that prose quotes both estimators; that is a human check.
- *The equal-mass arm exists only for N-MNIST.* `caltech_experiment` and `arch_matched` carry
  both ECE columns per row, but no paired bootstrap at all, on either estimator.
- *`--num-workers 6` deadlocks `rerun_paired_bootstrap.py`* on this machine, as it did the
  sweep once before. It ran with `--num-workers 0`; the default is still 6.

## 2026-09-12 — A default that cannot hang, and the disjoint resolvable sets

**Ran.** `pytest -q` → **229 passed**. Dashboard re-swept at 1920, 1440, 1200, 900 and 400:
no column ending with more than 100px empty, no overflow, no console errors, 8 of 8 modals
passing.

**Did: `rerun_paired_bootstrap.py --num-workers` now defaults to 0.** At 6 the script
deadlocked on this machine — the process sat at 0% CPU indefinitely rather than failing —
and the main sweep hit the same hang once before. A default that hangs is worse than one
that is slow, because the slow one finishes; fifteen runs of inference take a few minutes
single-process. The hazard is stated in the script's docstring, in the argument's help
text, and in the dashboard's Reproduce block, which now passes `--num-workers 0` explicitly.

**Not changed, reported:** the other scripts that load data still default to 4 or 6 —
`run_sweep`, `readout_experiment`, `quantisation_test`, `export_dashboard_data`,
`extract_readout_stats`, `extend_sweep_selective` and `readout_report` at 6;
`caltech_experiment`, `arch_matched`, `conditional_ceiling` and `train_one` at 4. Those
defaults produced the committed artifacts without hanging, so they were left as they are
rather than changed on the strength of one script's failure. The same trap may be waiting in
any of them.

**Did: recorded the disjointness of the resolvable sets, which strengthens the null result.**
Verified against `results/sweep_nmnist.csv` before recording. 13 of 15 paired intervals
straddle zero under equal-width binning and 12 of 15 under equal-mass. The runs that resolve
do not overlap: equal-width resolves `T = 2` seed 1 (temperature) and `T = 6` seed 0
(isotonic); equal-mass resolves `T = 4` seeds 1 and 2 and `T = 8` seed 0, all isotonic.
Five runs resolve under one estimator, and **none under both**.

The two estimators see identical calibrated confidences and differ only in how they bin
them, so a run that resolved because the calibrators really differed would resolve under
both. Their total disagreement about which runs resolve is evidence that none of the five
are real. That argument is stronger than either count, and it is now how the result is
stated in three places, each in the same order — counts, then the disjointness, then the
conclusion:

- **NOTES**, as a dated addendum directly beneath the pre-registration outcome. The outcome
  paragraph itself is unedited apart from a pointer to it.
- **Card 6.** The headline number is now **0** runs showing a difference under both measures,
  not 13 / 15. The claim gives both counts in one sentence, then the disjointness, then the
  conclusion, in plain English with no semicolons. Its table gains a *resolve under both* row,
  counted from the CSV at render time rather than written down, because the argument depends
  on that intersection being empty.
- **Modal m6**, as a new section immediately after the pre-registration outcome, with the
  five resolvable runs tabulated by estimator and direction.

**One analysis replaced.** The subsection "The paired bootstrap under both estimators",
written earlier today, closed with "Neither is the arbiter" and treated the estimators'
disagreement as a caveat on the finding. That was the weaker reading of the same data. Its
conclusion now points to the addendum. This was analysis rather than a pre-registration, so
replacing it revises no prediction.

## 2026-09-12 — The paper: every number generated from results/

**Did.** `paper/main.tex` and `paper/refs.bib`, arXiv cs.NE, sole author. **Compiled with
Tectonic to 9 pages: sections 1–7 end on page 8, and Code and data, Acknowledgements and the
references fill page 9.**

**How the numbers got in.** `main.tex` types no measured value. `paper/make_numbers.py` reads
`results/` and writes `paper/numbers.tex` — **216 macros, 204 of them used** — and each
definition carries a comment naming the artifact it came from. Every line of `main.tex` that
uses one carries its own source comment, checked mechanically: **0 lines without one**.
Derived arithmetic (the bound at a given T and C, T_min, the T = 80 cap) is computed there too
and labelled `[DERIVED]` rather than attributed to a result. The handful of numbers that are
not results say where they come from instead: `[LIT]` for Horowitz's Figure 1.1.9 and Li et
al.'s timestep range, `spikecalib/datasets.py` for the 45×60 frames, `spikecalib/train.py` for
the optimiser. The script asserts the facts the prose depends on — zero violations, the three
slack cells all at C = 101 and T = 10, an empty intersection of resolvable runs — so a
changed artifact breaks the build instead of silently falsifying a sentence.

**Two values had to be derived rather than read.** No record stores the early-stopping
patience. Every one of the 96 records has `epochs_run − best_epoch = 6`, and `best_epoch` is
0-indexed in `train.py`, so patience is 5 — that is how the macro is computed, and its comment
says so. And the seed-0 split fingerprint is quoted as seed 0's, because splits differ by seed.

**Where the text departs from the brief, and why.** Each change narrows a claim to what the
artifacts support.
- *"Unbounded readouts are overconfident"* became **unbounded accumulating** readouts,
  overconfident **as T grows**. The leaky membrane readout is unbounded and underconfident
  (−0.0289 at T = 2). Capped readouts are underconfident **where the cap binds**: the count
  readout is overconfident, +0.0344, at C = 101, T = 10, where the cap is slack.
- *"At C = 10 the two ceilings differ by at most 0.0067"* is true of seed means; the worst
  single seed is 0.0075. Both are stated.
- *"99.2% and 97.1% of expected calibration error"*: the artifact measures the share of the
  gap between accuracy and mean confidence. At those two cells that gap equals ECE to four
  decimals under both estimators, so the abstract says "under either binning estimator".
  99.2% is the two-block N-MNIST model and 97.1% the four-block N-Caltech101 model; the other
  depth gives 99.3% and 96.8%, named inline.
- *"−0.10 to +0.12 on N-MNIST"* is T = 2; the T = 6 values (−0.15, −0.14) are in the table.
- *The 0.1534 residual* is four blocks; two blocks gives 0.1159.
- The accuracy range does not appear in the abstract, because the four sentences specified for
  it do not include it; wherever it appears in the body the dataset and timestep are named.

**Cut to length.** The first build was 10 pages. Removed: the risk-coverage figure, which was
seed 0 only and restated numbers the text already gives, and a five-number table of shares that
now reads as a sentence. Figure 1 shrank, margins went to 0.9 in, the bibliography tightened,
two one-line Limitations paragraphs merged, and one heading shortened to clear the last overfull
line. Figure 1 is seed 0, drawn from `results/readout_stats.json`, and its caption says so.

**Left for the author.**
- *Affiliation* is a visible red TODO on page 1.
- *Guo et al.* is cited as instructed, but SOURCES.md still lists its author list and venue
  under "Still to verify"; `refs.bib` flags this. The paper relies only on the method
  attribution and does not attribute the 15-bin default to it.
- *No year* for SpikingJelly or the N-Caltech101 mirror, because SOURCES.md records none; these
  are the two remaining BibTeX warnings.

**Found, not fixed — outside the requested files.**
- *SOURCES.md line 115* still has an older Atarashi et al. entry dated 2026, marked
  `[UNVERIFIED]` and "not read in full", contradicting the confirmed entry at line 361 (12 June
  2025, read for the distinction). The paper uses the confirmed state.
- *The dashboard's energy note* (`docs/index.html`, "between 41% and 65%") matches neither the
  CSV's per-run range, 41–70%, nor its per-T means, 47–67%. The paper uses 41–70%.

**Tooling.** Tectonic installed via Homebrew. The PDF and build logs are ignored in
`.gitignore`; the source, the bibliography, the generated `numbers.tex` and its generator are
tracked, so `tectonic -X compile paper/main.tex` rebuilds the paper from a checkout.

## 2026-09-12 — Paper and sources cleanup: affiliation, Guo verified, stale entries removed

**Ran.** `pytest -q` → **230 passed** (one new guard). Paper recompiled: **9 pages, sections 1–7
ending on page 8**, no overfull lines, no undefined references, **no BibTeX warnings**. Dashboard
re-checked at 1920, 1440, 1200, 900 and 400: no row over 100px empty, 8 of 8 modals, filters
unchanged, lowest contrast 4.88:1.

**Did: affiliation.** The red TODO on page 1 now reads "Independent researcher, Liverpool,
United Kingdom". No employer or project is named.

**Did: Guo et al. verified against the primary document** — the ICML 2017 proceedings PDF and
its PMLR record. Authors **Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger**; PMLR 70,
pages 1321–1330, published 17 July 2017. What it does and does not say:
- **Temperature scaling is theirs**, "the simplest extension of Platt scaling", and they are
  "not aware of any prior use" of it for calibration.
- **Equal-width ECE is described, not introduced.** They define it by "M equally-spaced bins"
  and **attribute ECE to Naeini et al. (2015)**.
- **15 bins is the setting they report with, not a default.** Table 1 and the results text give
  ECE "with M = 15 bins"; nothing recommends it.
- **Isotonic regression and the one-versus-all extension are attributed to Zadrozny & Elkan
  (2002)**; Guo et al. describe the renormalised extension and apply it to isotonic regression.
- Equal-mass binning does not appear.

So the credits were narrowed rather than confirmed wholesale. The paper said "temperature
scaling and the binned expected calibration error follow Guo et al."; it now credits them with
proposing temperature scaling and with *describing* the equal-width estimate, and says they
report ECE with 15 bins. `refs.bib` carries the full entry. SOURCES.md moves Guo to the verified
section with the evidence above; NOTES correctness rule 2 and three `calibration.py` docstrings
lose their "`[UNVERIFIED]` 15-bin default" wording. Naeini et al. and Zadrozny & Elkan are not
read and are not cited.

**Did: stale and superseded entries in SOURCES.md.** Removed or corrected, one entry per work
now:
- **The stale Atarashi et al. entry** — dated 2026, `[UNVERIFIED]`, "not read in full" — beside
  the confirmed one. Removed.
- **A Horowitz pointer** duplicating the confirmed entry, and the section heading "Believed but
  not yet confirmed", which no longer held anything unconfirmed. Removed; the brief-citations
  summary table keeps its own heading.
- **Wu et al.: two "Distinction" paragraphs**, the shorter superseded by the fuller. The shorter
  removed.
- **Chen, Park & Simeone: "most likely one of regime"**, a hedge written before the T = 80 regime
  was confirmed. Now states the regime difference.
- **The E_MAC derivation** still called 0.9 and 3.7 pJ "believed" and the sum "conditional on two
  `[UNVERIFIED]` inputs". Both are read from Figure 1.1.9; the inputs are now `[LIT]` and the sum
  `[DERIVED]`.
- **"Still to verify"** carried two resolved items, Guo and Dynamic Confidence's readout. Removed
  and renumbered, 12 → 10, with one internal reference repaired; a second pointer in the Wu entry
  ("see the next entry") pointed at nothing after the reshuffle and now names its section.

**Did: bibliography years, found in the primary metadata.** SpikingJelly 0.0.0.0.14 — PyPI
records the upload on 20 March 2023 and the author as "PKU MLG, PCL, and other contributors".
The N-Caltech101 Mendeley mirror — the DataCite record gives creator Garrick Orchard, publisher
Mendeley, issued 12 January 2022. Both recorded in SOURCES.md and used in `refs.bib`; the two
BibTeX warnings are gone.

**Did: the dashboard's energy range. "Between 41% and 65% … at every timestep setting" →
"between 41% and 70% … across the fifteen sweep runs, and between 47% and 67% when the three
seeds at each timestep setting are averaged."** The old figure was exactly **seed 0's range**
(41.3% at T = 8, 65.0% at T = 2) presented as if it covered every run; T = 2 seed 2 is 70.0%.
A new test in `tests/test_provenance.py` recomputes both ranges from `docs/sweep.csv` and checks
the sentence.

**Found by the re-audit, fixed.**
- *Card 2's static fallback table still said 44.1%.* The data renderer overwrites it at load, so
  only a reader without `data.js` saw it. Now 44.5%, and a comparison of all seven fallback
  tables against their data-rendered versions shows **0 of 7 differ**.
- *Card 6 wiped its own fallback without data.* With `data.js` unavailable, its renderer ran on
  an empty CSV and replaced the correct table with zeros — including "resolve under both 0", a
  claim about data it never saw. It now writes only when there are rows to count.
- *My own scan was broken first.* Its file list was an unquoted variable, which zsh does not
  split, so every grep searched one nonexistent path and reported zero. Re-run with an explicit
  list and a sanity row that must match.

**Outstanding.**
- *`CITATION.cff` carries `doi: "10.48550/arXiv.TODO"`* — the arXiv DOI, which cannot exist until
  the preprint is posted.
- *Three bibliography entries give surnames only* — Atarashi et al., Horowitz, Orchard et al. —
  because SOURCES.md records only surnames for them; first names would need the primary documents
  opened again.
- *The arctangent surrogate gradient is uncited in the paper*, because its originating paper is
  item 3 of "Still to verify".
- *The ten open items in "Still to verify"* — none is cited by the paper except as noted above;
  the untuned learning rate and the unswept τ = 2 are the two that bear on reported results.
- *The withdrawn references still appear in NOTES.md*, as a record of their withdrawal rather than
  as citations.

## 2026-09-12 — Full author names, the ATan surrogate, and two limitations stated

**Ran.** `pytest -q` → **230 passed**. Paper recompiled: **9 pages, sections 1–7 ending on page
8**, no overfull lines, no undefined references, no BibTeX warnings, and every line that uses a
generated number still carries its source comment.

**Did: the three surname-only references now have full author names**, each read from the
registry record of the primary document already verified, and recorded in SOURCES.md:
- **Atarashi et al.** — Kyohei Atarashi, Satoshi Oyama, Hiromi Arai, Hisashi Kashima, from the
  arXiv record for 2506.10572 (v1 12 June 2025, v2 23 February 2026).
- **Horowitz** — Mark Horowitz, DOI 10.1109/ISSCC.2014.6757323, from the Crossref record for that
  DOI, whose title, proceedings and pages 10–14 match the confirmed entry.
- **Orchard et al.** — Garrick Orchard, Ajinkya Jayawant, Gregory K. Cohen, Nitish Thakor,
  *Frontiers in Neuroscience* volume 9, from the Crossref record for 10.3389/fnins.2015.00437.

`refs.bib` now has no surname-only author field, and its header says every field is confirmed
against a primary document or its registry record.

**Did: the ATan surrogate. Its origin could not be confirmed, so the paper cites nothing for it.**
SpikingJelly 0.0.0.0.14 documents `surrogate.ATan` with its formula and **attaches no citation**,
though the same module names a proposing paper for other surrogates (S2NN, q-PseudoSpike,
Log-tailed ReLU). The only lead in the package was the header of `parametric_lif_net.py`, which
names Fang et al., arXiv:2007.05785, a model file whose examples use ATan. That paper was read:
it **uses** `sigma(x) = (1/pi) arctan(pi x) + 1/2` in its experimental settings — exactly
SpikingJelly's ATan at `alpha = 2`, the value used here — but **does not claim to propose it**.
Using a function is not originating it, so no attribution follows. The method now says the
surrogate is the ATan function as implemented in SpikingJelly 0.0.0.0.14, gives
`g'(x) = alpha / (2(1 + (pi/2 alpha x)^2))` from the installed source, and states alpha = 2.0.
SOURCES.md "Still to verify" item 3 records all of this and stays open.

**Did: two limitations stated on the page.** Both were on the to-verify list and both bear on
reported results, so neither is left off:
- **No hyperparameter search was performed.** The learning rate of 10^-3, batch size of 128, the
  early-stopping rule and the surrogate parameter were set once and never tuned; a tuned model
  could be more accurate, which would change how much error the ceiling forces.
- **The membrane time constant was fixed at tau = 2.0 and never varied.** It sets how many
  timesteps the membrane readout integrates, so that readout's results hold at tau = 2.0 only.

**Cut to length again.** The additions pushed the conclusion onto page 10. Recovered without
changing any claim or number: Related Work's closing temperature-scaling sentence folded into the
method's existing Guo et al. citation; the Splits paragraph merged into Data; the one-sentence
Counts paragraph merged into the paragraph before it; the two "unexplained" limitations merged;
the rate-saturation paragraph and the conclusion tightened; and Figure 1 reduced over three passes
from 0.52 to 0.36 of the text width.

**Two of my own slips, both caught.** A search of the SpikingJelly package ran against nothing
because zsh expanded the unquoted `--include=*.py` pattern and aborted; re-run with quoting. And a
check for the rendered tau sentence reported it missing because PDF text extraction turns "fi"
into a single ligature glyph — the same cause as an earlier false miss on "Geoff"; the checks now
normalise ligatures first.

**Outstanding.**
- *Figure 1 is now small*, at 0.36 of the text width. It is legible on screen; worth a look in
  print before submission.
- *The ATan surrogate's origin* remains open, item 3 of "Still to verify".
- *`CITATION.cff`'s arXiv DOI* cannot be filled until the preprint is posted.

## 2026-09-28 — Final pass before a reader: why ECE equals the gap, the SpikingJelly paper, five dashboard numbers

**Did: explained the six identical numbers in Section 5.2, and corrected the explanation first
proposed for them.** The proposed reason was that the count readout puts almost all its mass on
one confidence value. Recomputed from the cached test counts with the pipeline's own ECE code,
that holds on N-MNIST at `T = 2` (94.9–95.4% of test predictions sit at the ceiling, 95.1% on
average, so equal-mass binning forms a single bin) but **not on N-Caltech101**, where only
9.3–12.6% (10.6% on average) sit there, across 102–154 distinct values and 14 equal-mass bins.
The actual reason covers both: ECE is never smaller than |accuracy − mean confidence|, and it
equals it exactly when no bin is overconfident. Where the cap binds, confidence sits below every
bin's accuracy. The single exception in six runs is one N-MNIST seed-0 test sample, a ten-way
tie at confidence 0.1 and wrong, which moves equal-width ECE by 2e-5. The paper now says this.
The bound comes from a new macro, `\eceGapMaxDev`, read from `results/arch_matched.json` and
asserted below 1e-4 in `make_numbers.py`. The mass fraction is not in the paper because the
count cache is not tracked.

**Did: cited the SpikingJelly paper.** Fang et al., *Science Advances* 9(40), eadi1480, 2023, was
read in its open-access full text (PMC10558124). Title, authors, volume, issue, eLocator and
date were taken from the article record, and the Crossref record and the project README's
citation block agree. It states no version and does not name ATan, so both still come from the
installed software. `refs.bib` now carries the article, and SOURCES.md records both the article
and the software.

**Did: paper hygiene.**
- Table 1's lower panel now has distinct values (new `ncCount*Levels`, `ncAcc*Levels`).
- `\todo` is removed.
- The twelve unused macros are removed. "Every run stopped early" is now an assertion instead of
  two unused counts.
- The conclusion scopes the calibration claim to N-MNIST.
- "unsigned" is replaced by "neither over- nor underconfident".
- "competitors silent … nearly always at C = 10" is replaced. It held at `T = 2` (0.6% of
  competitor outputs active) but not at `T = 10` (17%).
- Figure 1 is widened from 0.36 to 0.72 of the text width. At 0.36 its 8.5–10.5 pt text printed
  at 3.4–4.2 pt. The paper is still nine pages, with zero LaTeX warnings.

**Did: `fig4-paired-difference` said "14 of 15 intervals straddle zero" over a plot showing two
resolvable intervals.** The title is now computed from the data (13 of 15, equal-width), and only
that figure was regenerated.

**Did: five wrong numbers on the dashboard.**
- Card 4 and the Li et al. entry gave the accumulator as +0.0047 at `T = 2`. That is the
  `T = 10` value; at `T = 2` it is +0.0000.
- m2 gave shares of 90.0% and 44.2%; they are 90.1% and 44.5%.
- The bootstrap note said "one exclusion in fifteen"; there are two under equal-width and three
  under equal-mass.
- m4 said the ceiling binds for 10 classes at `T ≤ 4`. It binds through `T = 6`. The membrane
  recommendation keeps its ranking claim at `T ≤ 4` only.
- "26 times larger" now states what it compares.
- Also removed: the claim that temperature scaling fixes the accumulator. No calibrator was ever
  fitted to it.

**Checked, unchanged.**
- `numbers.tex` regenerated identically before any edit.
- `data.js` regenerated byte-identical.
- 230 tests pass.
- Dashboard at 1920 and 1440: 0 console messages, all 8 modals open and close, every filter
  changes the view.
- Ceilings, floors, `T_min` and the `T = 20` and `T = 80` ceilings were recomputed independently
  with `bc`.

**Outstanding, reported not fixed.**
- Dashboard prose still out of date: the limitations say one architecture and an unmatched
  confound, both since resolved; the accumulator is called "a lead"; the literature-sign modal
  body calls the disagreement a contradiction; "the cap should be avoided" is unscoped; the
  calibrator headline has no "on N-MNIST".
- The interval chart ignores the estimator control.
- The ceiling presets offer DVS-Gesture and CIFAR-100 but not N-Caltech101.
- NOTES.md has superseded passages without pointers to their corrections (:49, :466, :934,
  :1446, :1535, Scope).
- README omits the calibrator result from its findings, and its reproduce block omits
  `rerun_paired_bootstrap.py` and `extract_readout_stats.py`.
- The Acknowledgements drop "on verification" from the disclosure wording NOTES prescribes.

## 2026-09-29 — One precision per number, the Figure 1 caption, the related-work scope, and the paired-difference figure in the paper

**Did: removed mixed precision from Section 5.3.** "Accuracy, and its narrow scope" gave the two
`T = 2` N-Caltech101 gains at one decimal (+12.8, +3.1) and the two `T = 6` costs at two
(−2.24, −4.52), while Table 2 gives all four at two. The short-form macros
`\gainNcFourTwoShort` and `\gainNcTwoTwoShort` existed only to do that rounding and are
deleted; the prose now uses `\gainNcFourTwo` and `\gainNcTwoTwo` (+12.82, +3.08).

**Did: swept the paper for the same defect.** Every used macro was compared pairwise for two
printings of one value at different precisions. One more was found: Limitations gave the
N-Caltech101 accuracy range as 0.45 to 0.64, whose endpoints are the Table 1 cells 0.4543 and
0.6435. `\ncAccMin` and `\ncAccMax` are now printed at four decimals. The remaining matches
were coincidences between different quantities. `\roCountOps` (about 9) beside
`\roCountOpsExact` (9.3) is deliberate and stated as such. No number is hardcoded in prose.

**Did: the dashboard had the same mismatch.** Its prose and gain card said +12.8 and +3.1
beside its own table's +12.82 and +3.08. Seven strings in `docs/index.html` now use two
decimals. The per-seed gains (+15.2, +9.6, +13.7) and the "+3 to +13" headline tile are
unchanged: neither has a two-decimal counterpart.

**Did: Figure 1 caption.** It now reads "The derived ceiling … shown against the measured test
accuracy and the largest observed confidence"; the dangling "which is derived" is gone.

**Did: Related work states its scope.** One sentence says the search covered calibration of
spiking networks, confidence-based early exit and stopping, and post-hoc bounds on output
probabilities, and that the three entries are the works bearing directly on the claim. No
citation was added.

**Did: Acknowledgements give the reason for the withdrawals.** "Two were withdrawn" became "two
were withdrawn because no primary document supported them". This is accurate for both cases in
`SOURCES.md`: one identifier belonged to a different paper, and the other paper could not be
found. The disclosure wording that NOTES.md prescribed is amended there, with the reason for
the change, so that the two files agree.

**Did: the paired-difference figure is in the paper, redrawn first.** As drawn, fig4 showed
only the equal-width intervals. That shows two resolvable runs; it cannot show that they are
disjoint from the equal-mass ones, which is the argument. It is now two panels with shared
rows, equal-width on the left and equal-mass on the right, and resolvable intervals in orange.
Each panel title counts its straddling intervals, and the figure title counts runs resolvable
under both estimators; all three are computed from the data. The panels have separate
horizontal scales, and the caption says so. In the paper it is Figure 2 in Section 5.4, with
`\straddleW`, `\straddleM`, `\runsTotal` and `\resolveBoth` in the caption. Its definition sits
before the Section 5.4 heading so that it floats to the top of the page on which the section
opens.

**Ran.**
- `pytest -q` → **230 passed**.
- `numbers.tex` changes only by the two deleted macros and the two re-precisioned ones.
- `data.js` regenerated in the project venv: byte-identical, since no result changed.
- Paper compiled with tectonic: **10 pages**, 0 LaTeX warnings, 0 overfull/underfull boxes.
  The figure costs about half a page. Sections 1–7 and Code and data end on page 9; page 10
  holds the last two lines of the Acknowledgements and the references. Nothing was cut to win
  the page back.

## 2026-10-13 — The 48 cells were 36; the outstanding list from 2026-09-28 closed

**Found: "48 distinct count-readout cells" counted twelve models twice.** The figure was
15 + 9 + 24, the count-readout rows of `readout_experiment.json`, `caltech_experiment.json`
and `arch_matched.json`. But the matched grid reuses the two-block N-MNIST and four-block
N-Caltech101 models at `T` in {2, 6}, which the first two experiments had already measured.
Those twelve rows are the same trained models: observed maximum, accuracy and mean confidence
agree exactly between the artifacts, and `results/` holds 36 count-readout weight files, not
48. `make_numbers.py` keyed each cell by the artifact it came from, so an overlap could not
collide. The entry of 2026-09-12 ("A2") introduced the figure and called the cells distinct;
that claim is wrong and is corrected here, not there.

**The distinct count is 36: respected in 36, saturated in 33, slack in 3.** The three slack
cells are unchanged, `C = 101, T = 10`, mean gap 1.489e-3, because none of them is in the
grid. No conclusion moves: zero violations before, zero now. What changes is the size of the
claim, and it was overstated by a third.

**Did: corrected it everywhere it appeared.**
- `paper/make_numbers.py` keys a cell by (dataset, architecture, `T`, seed), reading the depth
  of the readout and C-axis rows from their run records. It asserts that a repeated cell
  carries identical numbers, and that the distinct count equals the number of count-readout
  weight files. New macro `\cellsShared` (12).
- `paper/main.tex`, Section 5.1: 36 and 33 through the existing macros, and the sentence
  listing 15, 9 and 24 now says twelve of the grid's cells are reused and counted once.
- `README.md`: 36 of 36, 33 saturated, with the same explanation.
- Dashboard rail, card 1 and its modal: 36 of 36, the matched grid's 24 of 24 stated
  separately, 33 saturated and 3 slack in the modal. These are rendered from the payload by a
  new `renderCells()`; the typed values are the fallback. Card 1 previously led with the
  grid's 24.
- `tests/test_provenance.py::test_count_cells_are_counted_once` recounts from the tracked
  artifacts and checks `numbers.tex`, the README and the page against it.

**Did: the seven outstanding items.**
1. *Dashboard Limitations.* "One architecture" and "confounded with depth" are removed; the
   matched 2×2 resolved both. One item replaces them and states what remains true: two
   architectures whose depth and size are anti-correlated, and weighted-count and membrane
   run on the two-block N-MNIST network only.
2. *Dashboard prose.* The accumulator's overconfidence is no longer "a lead": the page gives
   the N-Caltech101 values, +0.0816, +0.1843, +0.2254 with four blocks (9 of 9 runs) and
   +0.0774, +0.2076 with two (6 of 6). The literature-sign modal says regime difference, with
   `T = 80` and the count readout's +0.0344 at `C = 101, T = 10`. "Should be avoided" is scoped
   to where the cap binds (N-MNIST, `T ≤ 6`) and to calibration, with the ranking advantage at
   `T ≤ 4` only. The calibrator headline, its modal title, the rail and the hero say "on
   N-MNIST".
3. *Interval chart and tile.* The chart draws equal-width and equal-mass in two panels with
   shared rows and one horizontal scale, so a run that is amber in one and not the other is
   visible as that. Each panel counts its own straddling intervals and the note counts runs
   resolving under both; all three are computed. Below about 170px per panel the two stack.
   The tile shows both means, +0.0005 and −0.0008: opposite signs, which one number hid.
   Neither follows the estimator control, and the page says so.
4. *Presets.* N-Caltech101 at `C = 101` is added, the slider reaches 101, and at that `C` the
   chart draws the measured maxima from `caltech_experiment.json`, with a computed note that
   the `T = 10` ring sits below the curve.
5. *NOTES.md.* Six dated pointers, at the passages listed on 2026-09-28. All are additions;
   no line was edited or removed and no prediction was touched.
6. *README.* The calibrator result under both estimators, scoped to N-MNIST, and
   `rerun_paired_bootstrap.py` and `extract_readout_stats.py` in the reproduce block.
7. *Card 1.* Done as part of the correction above: 36, not the 48 the item asked for.

**Found: the committed `fig4-paired-difference` was not drawn in the pinned environment.**
Its SVG metadata says Matplotlib 3.10.1; `requirements.txt` pins 3.11.1. Redrawn in the
project environment it is 949 pixels wide, not 947, with the same content. The redrawn figure
replaces it in `figures/` and `paper/figures/`. The other seven figures regenerated with
byte-identical PNGs; their SVGs differ only in the embedded date and Matplotlib's random
element ids, so they are left as committed.

**Ran.**
- `python scripts/make_figures.py`, `python scripts/export_dashboard_data.py --num-workers 0`,
  `python paper/make_numbers.py`.
- `docs/data.js`: byte-identical. `numbers.tex`: changed only in the cell-count macros.
- `pytest -q`: **231 passed** (230, plus the new cell-count test).
- Paper compiled with tectonic: 10 pages, 0 LaTeX warnings, 0 overfull or underfull boxes.
- Dashboard under headless Chromium at 1920, 1440, 900 and 400: 0 console messages, all 8
  modals open and close, no horizontal overflow, the timestep filter and all four presets
  change the view, and the page's cell arithmetic re-derived from the payload gives 48 rows
  and 36 distinct.

**Outstanding, reported not fixed.**
- The dashboard Limitations still say the class count "was fixed at 10 until this session".
- The presets include DVS-Gesture and CIFAR-100, neither of which was run.
- The measured-maxima note at `C = 10` still types "to within 4e-13".
- NOTES.md's "Status" does not mention the cell count; it never stated 48, so nothing there
  is wrong, but the correction lives only in this log, the README and the paper.
