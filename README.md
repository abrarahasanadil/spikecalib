# SpikeCalib

**A preprint describing this work is in preparation.**

Post-hoc confidence calibration for event-driven spiking neural networks.

Temperature scaling assumes a continuous logit space. An event-driven SNN read out by
spike counts does not have one: each class output is an integer in `[0, T]`, so at low
timestep counts the attainable confidence set is small and coarse, and ties between
classes cannot be broken by dividing by a scalar. Isotonic regression is non-parametric
and monotone, so discreteness is native to it rather than a violated assumption.

This repository benchmarks the two calibrators across `T` in `{2, 4, 6, 8, 10}`, 3 seeds,
and multiple event datasets (N-MNIST, DVS-Gesture, CIFAR10-DVS), to test whether the
isotonic advantage grows as `T` falls — that is, whether isotonic regression is the
correct method precisely in the low-timestep regime where SNNs are energy-competitive.

A summary of the results, with figures, is published from [docs/](docs/) as a single
static page. The full research design, including the correctness rules that every
reported number depends on, is in [NOTES.md](NOTES.md). [LOG.md](LOG.md) is the append-only working log:
what was tried, what broke, and which alternatives were rejected and why.

## Install

Python 3.13 on Apple silicon (MPS). From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Verify the environment first. This runs a real 3-timestep forward pass through a
SpikingJelly `LIFNode` on MPS and fails hard if it produces zero spikes:

```sh
python scripts/check_env.py
```

Then fetch a dataset. Acquisition is idempotent — re-running verifies the existing
caches rather than refetching:

```sh
python scripts/fetch_data.py --dataset nmnist --timesteps 10
```

N-MNIST is implemented; DVS-Gesture and CIFAR10-DVS are registered with their metadata
but not yet downloadable.

Build and verify the four-way train / validation / calibration / test split. This asserts
correctness rule 1 — that the calibration split is disjoint from the others — on the real
index sets, and round-trips the split through disk:

```sh
python scripts/check_splits.py --dataset nmnist --timesteps 10 --seed 0
```

Run the tests:

```sh
python -m pytest tests/ -q
```

Model, training, and calibration code are not yet written.

## Citation

If you use this software, please cite it using the metadata in
[CITATION.cff](CITATION.cff). The arXiv DOI is a placeholder until the preprint is
posted.
