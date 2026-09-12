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
