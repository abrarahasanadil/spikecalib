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
