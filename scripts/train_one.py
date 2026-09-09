#!/usr/bin/env python3
"""Train one (dataset, T, seed) on the real split and save everything needed to reuse it.

Uses :func:`spikecalib.data.make_splits`, not an ad-hoc subset, so the calibration split
this run reserves is the same one the calibration stage will later fit on -- and the
disjointness assertion of correctness rule 1 runs on the real index sets.

Saves the weights, the run record, and the split, all keyed by the split fingerprint so a
calibration result can prove which training run and which split produced it.

Prints per-epoch training loss, validation accuracy and **validation tie rate**. The tie
rate is there because it is the diagnostic that catches a model which is not separating
classes: a tie rate near 1.0 means almost every sample has a tied argmax, and any accuracy
reported alongside it is close to meaningless whatever its value.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.data import load_dataset, make_splits, save_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.train import TrainConfig, resolve_device, set_seed, train  # noqa: E402


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0, help="0 = use the full split")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    set_seed(args.seed)

    print("=" * 78)
    print(f"TRAIN  {spec.display_name}  T={args.timesteps}  seed={args.seed}  device={device}")
    print("=" * 78)

    section("Data and split")
    with contextlib.redirect_stdout(io.StringIO()):
        train_set, test_set = load_dataset(args.dataset, args.timesteps, args.data_root)
    labels = training_labels(train_set)
    splits = make_splits(labels, spec, seed=args.seed, n_test=len(test_set))
    print(f"  train / val / cal / test : {splits.train.size:,} / {splits.val.size:,} / "
          f"{splits.cal.size:,} / {splits.test.size:,}")
    print(f"  split fingerprint        : {splits.fingerprint}")
    print("  disjointness asserted inside make_splits (correctness rule 1)")

    train_idx = splits.train
    if args.limit_train:
        train_idx = train_idx[: args.limit_train]
        print(f"  LIMITED to {train_idx.size:,} training samples (pilot mode)")

    def loader(dataset, idx, shuffle=False):
        return DataLoader(
            Subset(dataset, idx.tolist()),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    loaders = {
        "train": loader(train_set, train_idx, shuffle=True),
        "val": loader(train_set, splits.val),
    }

    section("Model")
    cfg = ModelConfig(channels=tuple(args.channels), hidden=args.hidden)
    model = build_model(spec, args.timesteps, cfg)
    print(f"  channels {cfg.channels}  hidden {cfg.hidden}")
    print(f"  feature shape            : {model.feature_shape}")
    print(f"  parameters               : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  lr {args.lr}  batch {args.batch_size}  epochs {args.epochs}  patience {args.patience}")

    section("Training")
    print("  val tie rate near 1.0 means the model is not separating classes, whatever")
    print("  the accuracy column says.")
    started = time.time()
    state, record = train(
        model,
        loaders,
        TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            log_every=1,
            eval_seed=args.eval_seed,
        ),
        seed=args.seed,
        dataset=spec.key,
        timesteps=args.timesteps,
        split_fingerprint=splits.fingerprint,
    )
    elapsed = time.time() - started

    section("Result")
    print(f"  epochs run               : {record.epochs_run} of {record.epochs_requested}"
          f"{'  (early stopped)' if record.early_stopped else ''}")
    print(f"  best val accuracy        : {record.best_val_accuracy:.4f}  (epoch {record.best_epoch})")
    print(f"  final val tie rate       : {record.final_val_tie_rate:.4f}")
    print(f"  wall clock               : {elapsed:,.1f}s  ({elapsed/max(record.epochs_run,1):,.1f}s per epoch)")
    print(f"  MPS nondeterministic     : {record.mps_nondeterministic}")

    os.makedirs(args.out, exist_ok=True)
    stem = f"{spec.key}_T{args.timesteps}_seed{args.seed}"
    weights_path = os.path.join(args.out, f"{stem}_weights.pt")
    record_path = os.path.join(args.out, f"{stem}_record.json")
    split_path = os.path.join(args.out, f"{stem}_split.npz")

    torch.save(state, weights_path)
    save_splits(splits, split_path)
    payload = record.as_dict()
    payload["model_config"] = {"channels": list(cfg.channels), "hidden": cfg.hidden,
                               "tau": cfg.tau, "v_threshold": cfg.v_threshold,
                               "surrogate_alpha": cfg.surrogate_alpha}
    payload["split_sizes"] = splits.sizes
    payload["limit_train"] = args.limit_train
    payload["wall_clock_seconds"] = elapsed
    with open(record_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    section("Saved")
    print(f"  weights   : {os.path.relpath(weights_path, REPO_ROOT)}")
    print(f"  record    : {os.path.relpath(record_path, REPO_ROOT)}")
    print(f"  split     : {os.path.relpath(split_path, REPO_ROOT)}  (fingerprint {splits.fingerprint})")

    section("Sweep estimate")
    per_epoch = elapsed / max(record.epochs_run, 1)
    runs = 5 * 3  # T in {2,4,6,8,10} x 3 seeds
    print(f"  {per_epoch:,.1f}s per epoch at T={args.timesteps}")
    print(f"  a {record.epochs_run}-epoch run costs {elapsed/60:,.1f} min")
    print(f"  {runs} runs (5 timesteps x 3 seeds) at this cost: {runs*elapsed/3600:,.2f} h")
    print("  cost scales roughly with T, so lower T is cheaper and higher T dearer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
