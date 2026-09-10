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
