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
