# SpikeCalib

**Post-hoc confidence calibration for event-driven spiking neural networks.**
A preprint describing this work is in preparation.

## What the study found

**The bound is not new. Its consequences for spike counts are what is measured here.** A
softmax over logits confined to an interval of width `W` cannot exceed
`e^W / (e^W + C − 1)`. That is a known result for normalised and clipped logits (AdaCos,
NormFace, LogitNorm, LogitClip; see `SOURCES.md`). A count readout gives each class an integer
in `[0, T]`, so its width is the number of timesteps and the largest confidence it can emit is
`e^T / (e^T + C − 1)`: 0.4509 at `T = 2` with 10 classes, 0.0688 with 101. **Respected and
saturated are different claims and both are reported.**

- *Respected* — the observed maximum never exceeds the bound: **36 of 36** distinct
  count-readout cells, a cell being one dataset, architecture, `T` and seed.
  `readout_experiment.json` holds 15, `caltech_experiment.json` 9 and `arch_matched.json` 24;
  12 of the grid's 24 are models it reuses from the other two, and each is counted once.
  They span two datasets, two architectures, five timestep settings and both class counts.
  Zero violations.
- *Saturated* — the observed maximum reaches the bound to within `1e-9`: **33 of those 36**.
- The **3 that are respected but slack** are `C = 101, T = 10`, one per seed, sitting a mean
  of `1.489e-3` below the bound (2.347e-3, 1.269e-3, 8.496e-4). Reaching it there would need
  one class firing at every step while **all 100 others stayed silent for all 10**, which is
  a statement about tightness, not validity.

The dashboard renders the same count, **36 of 36**, from its payload. **24** of them form
the matched 2×2 grid in `arch_matched.json`, the subset where architecture is controlled.

**Where the bound binds it accounts for almost all of the miscalibration, and where it does
not bind it accounts for none.** The gap between accuracy and the ceiling is 99.2% of
uncalibrated ECE at `T = 2, C = 10` and 97.1% at `T = 2, C = 101`. At `T = 6, C = 101` the
ceiling is slack — accuracy 0.64 against a ceiling of 0.80 — and 0.1159 to 0.1534 of
underconfidence survives that the ceiling explains nothing of. A ceiling conditioned on
measured competitor activity was pre-registered as the explanation and **falsified**: it sits
*above* accuracy.

**A softmax over firing rates is capped at `e / (e + C − 1)` at every `T`.** `[DERIVED]`
Dividing the count by `T` puts the logits in `[0, 1]` whatever `T` is, so the cap is 0.2320
for ten classes and 0.0265 for 101, and more timesteps add no confidence. No rate-readout
model was trained here; this is arithmetic.

**Where the cap sits, not the readout's name, sets the direction of the failure.** A readout
whose cap is below its accuracy is *under*confident. One whose cap is far above it, because it
is unbounded or because its logit scale was learned, is *over*confident on N-Caltech101 in
every run measured. Early underconfidence of a spike-count softmax is a published observation:
Chen, Park & Simeone (arXiv:2305.11322) report that its decisions "tend to be first
under-confident and then over-confident" as time passes. This study gives the reason, the cap
on a count read after `t` steps, and measures the early regime directly. Li, Jones & Furber
report the overconfident side for an accumulating readout.

**A learnable logit scale lifts the cap. It does not calibrate.** `[MEASURED]` A count
readout with one trainable scale, initialised at 1, raises it above the minimum the bound
requires in 9 of 9 runs. On N-MNIST at `T = 2` the gap goes from −0.5386 to +0.0044. On
N-Caltech101 it goes from −0.3969 to **+0.1259** at `T = 2` and from −0.1534 to **+0.1655** at
`T = 6`: overconfident. It recovers 33% of the accuracy the cap costs at `T = 2` there, where
half was predicted.

**Which calibrator to use depends on the regime.** Both ECE estimators are given, and two
calibrators are said to differ on a run only if the paired 95% bootstrap interval excludes
zero under both, in the same direction.

- *N-MNIST, 5,000 calibration samples: a tie.* Mean ECE at `T = 2` falls from 0.5386 to 0.0018
  with temperature scaling and 0.0042 with top-label isotonic regression under equal-width
  binning, and to 0.0008 and 0.0015 under equal-mass. The interval straddles zero in **13 of
  15** runs under equal-width and **12 of 15** under equal-mass, and the runs that resolve are
  disjoint, so all 15 are ties.
- *N-Caltech101: isotonic regression is better where the cap binds.* At `T = 2` temperature
  scaling leaves an ECE of 0.1449 (0.1523 equal-mass) and isotonic 0.0320 (0.0349). The two
  differ in **5 of 9** runs, all at `T ≤ 6` and all favouring isotonic; at `T = 10` they tie.
  A tie in at least 7 of 9 was predicted, and that prediction is **falsified**.
- *N-MNIST with 100 calibration samples: a lean to temperature scaling.* It is better in 26 of
  60 subsample draws and isotonic in 3. At least half the draws in two of three seeds was
  predicted; no seed reaches half, so that prediction is **falsified** too.

**The number of confidence values is fixed by `T` and `C`; the few values match an unbounded
readout's resolution but rank errors worse.** `[DERIVED]` A count readout emits at most `binomial(T + C − 1, C − 1)` distinct
confidences at any logit scale, 55 at `T = 2, C = 10`; observed counts are within the bound
in 45 of 45 cells. `[MEASURED]` A prediction that the count readout's Brier resolution would
be below the accumulator's on N-MNIST was **falsified**: it is below in 4 of 12 comparisons,
three of them an artefact of equal-mass binning. What the count readout gives up is ranking
(AUROC 0.9204 against 0.9698) and deterministic operating points, not resolution.

Seven predictions were recorded before these last experiments. Three held and four were
falsified, and each is reported as it fell (`NOTES.md`, "Outcome of predictions 6a to 11").

**A pilot: two capped readouts that both saturate cannot be weighted against each other.**
This is a corollary of known decision-fusion theory applied to spike counts, not a new
theorem, and the test is a pilot: one dataset, the second input a degraded sample of the
same class from the same sensor rather than a second modality, no fusion network trained,
no reference-frame transformation and no closed loop.
`[DERIVED]` A count readout that fires its winner at every step and nothing else reports one
fixed vector, so where two inputs both do this the product rule is an unweighted vote, a
coin flip on every disagreement, and it gives up `0.5 · P(S) · (a_1 − a_2)` in accuracy on
those pairs. A per-input scale restores the weighting. `[MEASURED]` On N-MNIST, fusing a
clean sample with an event-dropped sample of the same class and training nothing, calibrating
each input before the product rule lifts fused accuracy from 0.9911 to 0.9954 at `T = 2`,
which is 48% of the fused errors and 0.43 points of accuracy, and the identity matches the
measured part in 3 of 3 seeds. From `T = 4` the gain is not measurable. Four predictions were
recorded before the run. Two held; two were **falsified**: one asked for a strict ordering
among values that turned out to be indistinguishable from zero, and one allowed no
accumulator run to show an effect where 1 of 9 did, in the direction against calibration
(`NOTES.md`, "Outcome of predictions 12 to 15").

## The readout taxonomy

Same architecture, same data, same training — only the final aggregation differs.
`[MEASURED]` N-MNIST, `T = 2`, `C = 10`, mean over 3 seeds:

| readout | aggregation | bound | ECE equal-width | ECE equal-mass | conf − acc | distinct levels |
|---|---|---|---|---|---|---|
| `count` | `Σ_t s_t` | capped at `e^T/(e^T+C−1)` | 0.5386 | 0.5386 | −0.5386 | 9 |
| `weighted-count` | `Σ_t w_t s_t`, `Σw = T` | **same cap**, finer grain | 0.5385 | 0.5385 | −0.5385 | 17 |
| `membrane` | leaky integrator, `τ = 2` | unbounded, one-step half-life | 0.0290 | 0.0289 | −0.0289 | 9,998 |
| `accumulator` | `Σ_t z_t`, no leak | unbounded, equal weight per step | 0.0031 | 0.0016 | +0.0000 | 10,000 |
| `scaled-count` | `s·Σ_t s_t`, `s` learned | capped at `e^(sT)/(e^(sT)+C−1)` | 0.0054 | 0.0044 | +0.0044 | 9 |

Both estimators are given because correctness rule 2 requires it. They agree to four
decimals wherever the cap binds — the confidences are clustered against it, so equal-width
and equal-mass bins see the same systematic gap — and separate only once a readout or a
calibrator has spread the distribution out, as `accumulator` shows.

`weighted-count` is the control that matters: it shares the cap exactly while taking roughly
twice as many values, and its ECE differs from `count`'s by 0.0001. **Granularity buys
nothing; the bound does all the work.** `scaled-count` is the opposite control: the same nine
values as `count`, moved by a learned scale of 4.44 to 5.20, and the error is gone.

The sign flips with the family. `conf − acc` for `count` is −0.5386, −0.0272, −0.0008 at
`T` = 2, 6, 10; for `accumulator` it is +0.0000, +0.0032, +0.0047 — positive in 3 of 3 seeds
at `T` ≥ 6, and +0.08 to +0.23 at `C = 101`.

## Results dashboard

A chart-first summary of the paper is in [docs/](docs/): four headline numbers, then one
band per finding (the cap, calibration against ranking, the direction of failure by
readout, the learned scale, the calibrators, the fusion pilot), a ledger of every
pre-registered prediction with its verdict, and the limitations. Each chart has a
collapsible table view. Every number is rendered from `docs/data.js`, which
`scripts/export_dashboard_data.py` writes from the result artifacts; none is typed into the
page. It has a light theme by default and a selectable dark theme. **GitHub Pages is not enabled**, so it is not served publicly;
open `docs/index.html` directly in a browser, which works from `file://` with no server and no
network access.

## Reproduce

Python 3.13 on Apple silicon (MPS). Every pinned version is in `requirements.txt`.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q          # 414 tests
```

Verify the environment. This runs a real forward pass through a SpikingJelly `LIFNode` on
MPS and fails hard if it produces zero spikes:

```sh
python scripts/check_env.py
```

Fetch data. Acquisition is idempotent — a re-run verifies the caches rather than refetching.
N-MNIST and N-Caltech101 are implemented; DVS-Gesture and CIFAR10-DVS are registered with
their metadata and raise `NotImplementedError`:

```sh
python scripts/fetch_data.py --dataset nmnist       --timesteps 2   # repeat for 4 6 8 10
python scripts/fetch_data.py --dataset ncaltech101  --timesteps 2   # repeat for 6 10
```

Then the experiments, in the order they were run:

```sh
python scripts/run_sweep.py --epochs 50 --patience 5        # calibrators x T x seed
python scripts/readout_experiment.py --seeds 0 1 2          # the four readouts
python scripts/quantisation_test.py                         # is coarseness the mechanism?
python scripts/caltech_experiment.py --epochs 50            # the C axis, 101 classes
python scripts/conditional_ceiling.py                       # ceiling given real competitors
python scripts/arch_matched.py --epochs 50                  # both depths, both datasets
python scripts/rerun_paired_bootstrap.py --num-workers 0    # equal-mass paired bootstrap
python scripts/scaled_count_experiment.py                   # the learned-scale arm, 9 runs
python scripts/resolution_experiment.py                     # bound check and decomposition
python scripts/caltech_calibrators.py                       # calibrators at 101 classes
python scripts/calibration_size_experiment.py               # 100 / 1,000 / 5,000 samples
python scripts/fusion_experiment.py --select-rate           # fix the event-dropout rate
python scripts/fusion_experiment.py                         # fusion of two readouts
```

`rerun_paired_bootstrap.py` trains nothing: it reuses the sweep's weights and adds the
equal-mass columns to the sweep CSV. It runs single-process because six DataLoader workers
deadlock it on Apple silicon. Of the last six, only `scaled_count_experiment.py` trains. The
others reuse weights, hold the data in memory with no worker processes, and cache each
model's outputs in `results/` so a second run is quick.

Then regenerate the figures, the dashboard payload and the paper's numbers.
`extract_readout_stats.py` comes first: it writes `results/readout_stats.json`, the
observed maximum confidence at each `T`, which the figures and the dashboard payload read.

```sh
python scripts/extract_readout_stats.py
python scripts/make_figures.py
python scripts/export_dashboard_data.py --num-workers 0
python paper/make_numbers.py
```

Training runs need an MPS device; everything else is CPU and deterministic. Splits are
reproducible from the seed alone and every result row carries the fingerprint of the split
that produced it.

## The documents, and what each is for

| file | purpose |
|---|---|
| [NOTES.md](NOTES.md) | The authoritative research design. The claim under test, twelve dated pre-registered predictions with their outcomes recorded unedited, the correctness rules every number depends on, scope and per-dataset hazards. |
| [LOG.md](LOG.md) | The append-only working log — what was run, what broke, what was chosen and why, what remains open. Strictly append-only: no line has ever been edited or removed. |
| [SOURCES.md](SOURCES.md) | Provenance for every number. Each is tagged `[LIT]`, `[DERIVED]`, `[MEASURED]` or `[UNVERIFIED]`, with a record of what was checked against a primary document and what is still unverified. |

Predictions that turned out wrong are left exactly as written, with their outcomes recorded
beneath them. Two of the four prior-work references supplied at the outset did not survive
verification and were withdrawn rather than replaced with a plausible substitute; both
withdrawals are recorded.

## Status

Implemented and run on N-MNIST and N-Caltech101: acquisition, splits, model, training, both
ECE estimators, four calibrators, bootstrap intervals, selective prediction, five readouts,
the Brier decomposition, and the figures. 60 sweep rows, 54 readout runs, 18 C-axis runs, 48
matched-architecture cells, 9 learned-scale runs, and calibrators fitted on N-Caltech101. 105
models trained. DVS-Gesture and CIFAR10-DVS are registered but not implemented, and no
rate-readout model has been trained.

**A preprint is in preparation.**

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff).
The arXiv DOI is a placeholder until the preprint is posted.
