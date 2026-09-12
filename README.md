# SpikeCalib

**Post-hoc confidence calibration for event-driven spiking neural networks.**
A preprint describing this work is in preparation.

## What the study found

**Softmax over spike counts is bounded.** A count readout gives each class an integer in
`[0, T]`, so the largest confidence the network can emit is `e^T / (e^T + C − 1)` — 0.4509
at `T = 2` with 10 classes, 0.0688 with 101. This is arithmetic, not an empirical trend, and
it holds: across two datasets, two architectures, five timestep settings and both class
counts, **the bound is respected in all 30 measured cells** and saturated to within `4e-13`
in 27 of them. The three exceptions are `C = 101, T = 10`, where the bound is respected but
sits 1.5e-3 above the observed maximum — reaching it there would need one class firing at
every step while **all 100 others stayed silent for all 10**, which is a statement about
tightness, not validity.

**Where the bound binds it accounts for almost all of the miscalibration, and where it does
not bind it accounts for none.** The gap between accuracy and the ceiling is 99.2% of
uncalibrated ECE at `T = 2, C = 10` and 97.1% at `T = 2, C = 101`. At `T = 6, C = 101` the
ceiling is slack — accuracy 0.64 against a ceiling of 0.80 — and 0.1159 to 0.1534 of
underconfidence survives that the ceiling explains nothing of. A ceiling conditioned on
measured competitor activity was pre-registered as the explanation and **falsified**: it sits
*above* accuracy, so the residual is genuine miscalibration.

**The readout family, not the network, determines the direction of the failure.** A capped
readout is *under*confident wherever the cap binds; an unbounded accumulating readout is
*over*confident, saturating towards 1 as `T` grows. Both directions appear in the literature,
and which one a paper reports is predicted by its readout and its timestep count rather than
by anything about spiking networks.

## The readout taxonomy

Same architecture, same data, same training — only the final aggregation differs.
`[MEASURED]` N-MNIST, `T = 2`, `C = 10`, mean over 3 seeds:

| readout | aggregation | bound | ECE | conf − acc | distinct levels |
|---|---|---|---|---|---|
| `count` | `Σ_t s_t` | capped at `e^T/(e^T+C−1)` | 0.5386 | −0.5386 | 9 |
| `weighted-count` | `Σ_t w_t s_t`, `Σw = T` | **same cap**, finer grain | 0.5385 | −0.5385 | 17 |
| `membrane` | leaky integrator, `τ = 2` | unbounded, one-step half-life | 0.0290 | −0.0289 | 9,998 |
| `accumulator` | `Σ_t z_t`, no leak | unbounded, equal weight per step | 0.0031 | +0.0000 | 10,000 |

`weighted-count` is the control that matters: it shares the cap exactly while taking roughly
twice as many values, and its ECE differs from `count`'s by 0.0001. **Granularity buys
nothing; the bound does all the work.**

The sign flips with the family. `conf − acc` for `count` is −0.5386, −0.0272, −0.0008 at
`T` = 2, 6, 10; for `accumulator` it is +0.0000, +0.0032, +0.0047 — positive in 3 of 3 seeds
at `T` ≥ 6, and +0.08 to +0.23 at `C = 101`.

## Results dashboard

An interactive summary — every table rendered from the result JSON rather than typed into the
page — is in [docs/](docs/). **GitHub Pages is not enabled**, so it is not served publicly;
open `docs/index.html` directly in a browser, which works from `file://` with no server and no
network access.

## Reproduce

Python 3.13 on Apple silicon (MPS). Every pinned version is in `requirements.txt`.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q          # 214 tests
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
```

Then regenerate the figures and the dashboard payload:

```sh
python scripts/make_figures.py
python scripts/export_dashboard_data.py
```

Training runs need an MPS device; everything else is CPU and deterministic. Splits are
reproducible from the seed alone and every result row carries the fingerprint of the split
that produced it.

## The documents, and what each is for

| file | purpose |
|---|---|
| [NOTES.md](NOTES.md) | The authoritative research design. The claim under test, five dated pre-registered predictions with their outcomes recorded unedited, the correctness rules every number depends on, scope and per-dataset hazards. |
| [LOG.md](LOG.md) | The append-only working log — what was run, what broke, what was chosen and why, what remains open. Strictly append-only: no line has ever been edited or removed. |
| [SOURCES.md](SOURCES.md) | Provenance for every number. Each is tagged `[LIT]`, `[DERIVED]`, `[MEASURED]` or `[UNVERIFIED]`, with a record of what was checked against a primary document and what is still unverified. |

Predictions that turned out wrong are left exactly as written, with their outcomes recorded
beneath them. Two of the four prior-work references supplied at the outset did not survive
verification and were withdrawn rather than replaced with a plausible substitute; both
withdrawals are recorded.

## Status

Implemented and run on N-MNIST and N-Caltech101: acquisition, splits, model, training, both
ECE estimators, four calibrators, bootstrap intervals, selective prediction, four readouts,
and the figures. 60 sweep rows, 54 readout runs, 18 C-axis runs, 48 matched-architecture
cells. DVS-Gesture and CIFAR10-DVS are registered but not implemented.

**A preprint is in preparation.**

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff).
The arXiv DOI is a placeholder until the preprint is posted.
