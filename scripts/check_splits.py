#!/usr/bin/env python3
"""Build and verify the four-way split on a real dataset.

Loads the built frame cache, reads the training labels, carves the seeded split, prints
the summary, and confirms correctness rule 1 holds on the real index sets rather than on
synthetic test fixtures. The unit tests prove the logic; this proves it against the
actual data on disk, where class counts are whatever N-MNIST happens to contain.

Also round-trips the split through save/load, so the persistence path is exercised on
real sizes rather than only on toy arrays.

Exit codes: 0 everything verified, 1 a check failed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.data import (  # noqa: E402
    CARVED_SPLITS,
    SplitOverlapError,
    assert_disjoint,
    load_dataset,
    load_splits,
    make_splits,
    save_splits,
    summarise,
    training_labels,
)
from spikecalib.datasets import get_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--save", default=None, help="also write the split to this path")
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    print(f"Split check: {spec.display_name}  T={args.timesteps}  seed={args.seed}")
    print("=" * 60)

    print()
    print("Loading the built cache")
    print("-" * 22)
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        # SpikingJelly narrates cache reuse on stdout; the counts below are the check.
        train_set, test_set = load_dataset(args.dataset, args.timesteps, args.data_root)
    labels = training_labels(train_set)
    print(f"  train_set          : {len(train_set):,} samples")
    print(f"  test_set           : {len(test_set):,} samples")
    print(f"  labels read        : {labels.size:,} in {time.time() - started:.2f}s (no frames decoded)")
    print(f"  classes present    : {sorted(set(labels.tolist()))}")

    if len(train_set) != spec.n_train or len(test_set) != spec.n_test:
        print(f"  cache does not match the registry ({spec.n_train:,}/{spec.n_test:,})")
        return 1

    print()
    print("Building the split")
    print("-" * 18)
    print(f"  from the registry  : {spec.n_val_per_class} val/class, {spec.n_cal_per_class} cal/class")
    splits = make_splits(labels, spec, seed=args.seed, n_test=len(test_set))
    print(f"  built              : {splits.fingerprint}")

    print()
    print(summarise(splits, labels, spec=spec))

    print()
    print("Correctness rule 1")
    print("-" * 18)
    try:
        assert_disjoint(splits)
    except SplitOverlapError as exc:
        print(f"  FAILED: {exc}")
        return 1
    print("  assert_disjoint(train, val, cal) passed on the real index sets")

    # Spell the pairwise result out rather than trusting a silent pass.
    for a, b in [("train", "val"), ("train", "cal"), ("val", "cal")]:
        n = np.intersect1d(getattr(splits, a), getattr(splits, b)).size
        print(f"    {a:<5} vs {b:<5}: {n} shared indices")
    covered = sum(getattr(splits, s).size for s in CARVED_SPLITS)
    print(f"  train+val+cal covers {covered:,} of {labels.size:,} training samples, no index twice")
    print("  test is the test set's own range and was never carved from the training set")

    print()
    print("Persistence round-trip")
    print("-" * 22)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_splits(splits, os.path.join(tmp, "split.npz"))
        size = os.path.getsize(path)
        restored = load_splits(path)
        identical = all(
            np.array_equal(getattr(splits, s), getattr(restored, s))
            for s in (*CARVED_SPLITS, "test")
        )
        print(f"  saved              : {size / 1024:,.1f} KiB")
        print(f"  reloaded           : fingerprint {restored.fingerprint}")
        print(f"  indices identical  : {identical}")
        if not identical or restored.fingerprint != splits.fingerprint:
            print("  FAILED: round-trip changed the split")
            return 1

    if args.save:
        written = save_splits(splits, args.save)
        print(f"  written to         : {written}")

    print()
    print("Reproducibility")
    print("-" * 15)
    again = make_splits(labels, spec, seed=args.seed, n_test=len(test_set))
    other = make_splits(labels, spec, seed=args.seed + 1, n_test=len(test_set))
    print(f"  same seed  ({args.seed}) -> {again.fingerprint}  same={again.fingerprint == splits.fingerprint}")
    print(f"  other seed ({args.seed + 1}) -> {other.fingerprint}  same={other.fingerprint == splits.fingerprint}")
    if again.fingerprint != splits.fingerprint or other.fingerprint == splits.fingerprint:
        print("  FAILED: seeding is not behaving as rule 4 requires")
        return 1

    print()
    print("Result")
    print("-" * 6)
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
