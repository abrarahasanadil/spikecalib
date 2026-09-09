#!/usr/bin/env python3
"""End-to-end smoke test: train a tiny SNN on a small N-MNIST subset and calibrate it.

This is the check that the whole pipeline runs before any sweep is launched -- data, model,
training, count extraction, tie-breaking, SynOps, and both ECE estimators with confidence
intervals. It is deliberately small: a 2,000-sample subset at T=4 for a couple of epochs.

**It is a smoke test, not a result.** The accuracy will be poor, the calibration numbers
are computed on a few hundred samples, and no hyperparameter has been tuned. Nothing
printed here belongs in the paper.

Exit codes: 0 everything ran and the invariants held, 1 an invariant failed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicOneVsRest,
    IsotonicTopLabel,
    TemperatureScaling,
    Uncalibrated,
    bootstrap_ece_difference,
    compute_metrics,
    model_outputs,
    tie_rate,
)
from spikecalib.data import load_dataset, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import E_AC_JOULES, E_MAC_JOULES, ModelConfig, build_model  # noqa: E402
from spikecalib.train import TrainConfig, predict_counts, resolve_device, set_seed, train  # noqa: E402

ARMS = [
    ("uncalibrated", Uncalibrated),
    ("temperature", TemperatureScaling),
    ("isotonic-top", IsotonicTopLabel),
    ("isotonic-ovr", IsotonicOneVsRest),
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"    ok   {message}")
    else:
        print(f"    FAIL {message}")
        failures.append(message)


def stratified_subset(labels: np.ndarray, per_class: int, n_classes: int, rng) -> np.ndarray:
    picks = []
    for cls in range(n_classes):
        idx = np.flatnonzero(labels == cls)
        picks.append(rng.permutation(idx)[:per_class])
    return np.sort(np.concatenate(picks))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--n-train-pool", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-cal", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()

    print("=" * 84)
    print("SMOKE TEST -- tiny model, tiny subset, untuned. Not a result.".center(84))
    print("=" * 84)
    print(f"  dataset {spec.display_name}   T={args.timesteps}   device={device}   seed={args.seed}")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    section("Data")
    with contextlib.redirect_stdout(io.StringIO()):
        train_set, test_set = load_dataset(args.dataset, args.timesteps, args.data_root)
    labels = training_labels(train_set)
    print(f"  full train / test    : {len(train_set):,} / {len(test_set):,}")

    per_class = args.n_train_pool // spec.n_classes
    pool = stratified_subset(labels, per_class, spec.n_classes, rng)
    # Disjoint by construction and asserted, exactly as correctness rule 1 requires.
    shuffled = rng.permutation(pool)
    val_idx = np.sort(shuffled[: args.n_val])
    cal_idx = np.sort(shuffled[args.n_val : args.n_val + args.n_cal])
    fit_idx = np.sort(shuffled[args.n_val + args.n_cal :])
    test_idx = np.sort(rng.permutation(len(test_set))[: args.n_test])

    for a, b, na, nb in [
        (fit_idx, val_idx, "train", "val"),
        (fit_idx, cal_idx, "train", "cal"),
        (val_idx, cal_idx, "val", "cal"),
    ]:
        overlap = np.intersect1d(a, b).size
        check(overlap == 0, f"{na} and {nb} are disjoint ({overlap} shared)")
    print(f"  subset train/val/cal : {fit_idx.size:,} / {val_idx.size:,} / {cal_idx.size:,}")
    print(f"  test subset          : {test_idx.size:,} (from the dataset's own test set)")

    def loader(dataset, idx, shuffle=False):
        return DataLoader(
            Subset(dataset, idx.tolist()),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    loaders = {
        "train": loader(train_set, fit_idx, shuffle=True),
        "val": loader(train_set, val_idx),
    }

    section("Training")
    cfg = ModelConfig(channels=(16, 32), hidden=128)
    model = build_model(spec, args.timesteps, cfg)
    print(f"  parameters           : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  epochs               : {args.epochs}  lr 1e-3 (Adam default, untuned)")
    started = time.time()
    state, record = train(
        model,
        loaders,
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.epochs, log_every=1),
        seed=args.seed,
        dataset=spec.key,
        timesteps=args.timesteps,
        split_fingerprint="smoke-test-no-saved-split",
    )
    model.load_state_dict(state)
    print(f"  wall clock           : {time.time() - started:,.1f}s on {record.device}")
    print(f"  best val accuracy    : {record.best_val_accuracy:.4f} (epoch {record.best_epoch})")
    print(f"  MPS nondeterministic : {record.mps_nondeterministic}  (seed bounds, not eliminates, variation)")
    check(record.best_val_accuracy > 1.5 / spec.n_classes,
          f"val accuracy {record.best_val_accuracy:.4f} beats chance ({1/spec.n_classes:.3f})")

    section("Spike counts")
    cal_counts, cal_labels, cal_synops = predict_counts(model, loader(train_set, cal_idx), device)
    test_counts, test_labels, test_synops = predict_counts(model, loader(test_set, test_idx), device)

    print(f"  dtype                : {test_counts.dtype}")
    print(f"  shape                : {test_counts.shape}")
    print(f"  range                : [{test_counts.min()}, {test_counts.max()}]  (T={args.timesteps})")
    print(f"  distinct count values: {np.unique(test_counts).tolist()}")
    check(np.issubdtype(test_counts.dtype, np.integer), "counts are an integer dtype")
    check(test_counts.min() >= 0 and test_counts.max() <= args.timesteps,
          f"counts lie in [0, {args.timesteps}]")

    section("Readout")
    cal = model_outputs(cal_counts, cal_labels, np.random.default_rng(args.seed))
    test = model_outputs(test_counts, test_labels, np.random.default_rng(args.seed + 1))
    n_distinct = int(np.unique(cal.conf).size)
    print(f"  test accuracy        : {test.accuracy:.4f}")
    print(f"  tie rate             : {tie_rate(test_counts):.4f}  (descriptive: output discreteness)")
    print(f"  distinct conf (cal)  : {n_distinct}")
    print(f"  samples per level    : {cal.n / max(n_distinct, 1):.1f}")

    section("Energy proxy (SynOps)")
    print(f"  AC operations        : {test_synops.ac:,.0f}")
    print(f"  MAC operations       : {test_synops.mac:,.0f}")
    print(f"  E_AC / E_MAC         : {E_AC_JOULES*1e12:.1f} pJ / {E_MAC_JOULES*1e12:.1f} pJ")
    print(f"  energy per sample    : {test_synops.energy_per_sample_joules*1e6:.3f} uJ")
    print("  DESIGN-COMPARISON PROXY, NOT MEASURED POWER. Constants [UNVERIFIED]; see SOURCES.md.")
    check(test_synops.non_binary_ac_inputs == 0, "every AC-costed layer received binary spikes")

    section(f"Calibration (bootstrap {args.bootstrap} resamples, seed {args.seed})")
    print(f"  {'arm':<15}{'ECE ew':>9}{'95% CI':>22}{'ECE em':>9}{'95% CI':>22}")
    results = {}
    for name, cls in ARMS:
        fitted = cls().fit(cal.conf, cal.correct, outputs=cal)
        calibrated = fitted.transform(test.conf, outputs=test)
        m = compute_metrics(calibrated, test.correct, n_bootstrap=args.bootstrap, bootstrap_seed=args.seed)
        results[name] = (calibrated, m)
        print(
            f"  {name:<15}{m.ece_equal_width:>9.4f}"
            f"{m._fmt_ci(m.ece_equal_width_ci):>22}"
            f"{m.ece_equal_mass:>9.4f}"
            f"{m._fmt_ci(m.ece_equal_mass_ci):>22}"
        )
    notes = {n for _, m in results.values() for n in m.notes}
    for n in sorted(notes):
        print(f"  note: {n}")

    section("Is the isotonic-vs-temperature difference real?")
    diff = bootstrap_ece_difference(
        results["isotonic-top"][0], results["temperature"][0], test.correct,
        n_bootstrap=args.bootstrap, seed=args.seed,
    )
    print(f"  isotonic - temperature (equal-width ECE): {diff['difference']:+.4f}")
    print(f"  95% CI on the paired difference        : [{diff['ci'][0]:+.4f}, {diff['ci'][1]:+.4f}]")
    print(f"  isotonic lower in {diff['a_better_fraction']*100:.1f}% of resamples")
    if diff["resolvable"]:
        print("  -> the interval excludes zero: the difference is resolvable at this test size")
    else:
        print("  -> the interval straddles zero: NOT resolvable at this test size, whatever")
        print("     the point estimate says. This is a 2,000-sample smoke test, so that is")
        print("     the expected outcome, not a problem.")

    section("Result")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        return 1
    print("  Pipeline ran end to end; all invariants held.")
    print("  Reminder: tiny subset, 2 epochs, untuned. Not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
