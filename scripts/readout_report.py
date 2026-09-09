#!/usr/bin/env python3
"""Describe what a trained model's spike-count readout actually looks like.

Loads trained weights for one or more timestep settings and reports the properties the
calibration study depends on -- **before any calibrator is fitted**. The point is to see
the raw readout: how much of the count range ``[0, T]`` a converged model uses, how often
the argmax is tied, how many distinct confidence values exist for isotonic to work with,
and where the model sits relative to the attainable-confidence ceiling
``e^T / (e^T + C - 1)``.

That ceiling is the reason this report exists. At T=2 with 10 classes it is 0.451, which
is below any respectable accuracy, so a converged model is *structurally* unable to be
correctly confident (``NOTES.md``, claim section). This script measures whether that
actually happens rather than assuming it.

No calibration is performed here. Counts, confidences and energy only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import predict_with_ties, softmax_confidence, tie_rate  # noqa: E402
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.train import predict_counts, resolve_device  # noqa: E402


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def ceiling(timesteps: int, n_classes: int) -> float:
    """[DERIVED] softmax of the extreme count vector [T, 0, ..., 0]. See SOURCES.md."""
    return float(np.exp(timesteps) / (np.exp(timesteps) + n_classes - 1))


def analyse(spec, timesteps, seed, data_root, results_dir, batch_size, num_workers):
    stem = f"{spec.key}_T{timesteps}_seed{seed}"
    record_path = os.path.join(results_dir, f"{stem}_record.json")
    weights_path = os.path.join(results_dir, f"{stem}_weights.pt")
    for path in (record_path, weights_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Train it first:\n"
                f"    python scripts/train_one.py --dataset {spec.key} "
                f"--timesteps {timesteps} --seed {seed}"
            )
    with open(record_path) as fh:
        record = json.load(fh)

    with contextlib.redirect_stdout(io.StringIO()):
        train_set, test_set = load_dataset(spec.key, timesteps, data_root)
    labels = training_labels(train_set)
    splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))

    mc = record["model_config"]
    cfg = ModelConfig(
        channels=tuple(mc["channels"]), hidden=mc["hidden"], tau=mc["tau"],
        v_threshold=mc["v_threshold"], surrogate_alpha=mc["surrogate_alpha"],
    )
    device = resolve_device()
    model = build_model(spec, timesteps, cfg)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def loader(dataset, idx):
        return DataLoader(
            Subset(dataset, idx.tolist()), batch_size=batch_size,
            num_workers=num_workers, persistent_workers=num_workers > 0,
        )

    cal_counts, cal_labels, _ = predict_counts(
        model, loader(train_set, splits.cal), device, count_synops=False
    )
    test_counts, test_labels, synops = predict_counts(
        model, loader(test_set, splits.test), device
    )

    rng = np.random.default_rng(seed + 1)
    predictions, tie_mask = predict_with_ties(test_counts, rng)
    test_conf = softmax_confidence(test_counts)
    cal_conf = softmax_confidence(cal_counts)

    return {
        "timesteps": timesteps,
        "record": record,
        "split_fingerprint": splits.fingerprint,
        "val_accuracy": record["best_val_accuracy"],
        "val_tie_rate": record["final_val_tie_rate"],
        "test_accuracy": float((predictions == test_labels).mean()),
        "test_tie_rate": tie_rate(test_counts),
        "tie_mask": tie_mask,
        "n_test": int(test_labels.size),
        "n_cal": int(cal_labels.size),
        "n_distinct_cal": int(np.unique(cal_conf).size),
        "n_distinct_test": int(np.unique(test_conf).size),
        "counts": test_counts,
        "conf": test_conf,
        "ceiling": ceiling(timesteps, spec.n_classes),
        "synops": synops,
    }


def report(spec, r) -> None:
    T = r["timesteps"]
    print()
    print("=" * 78)
    print(f"T = {T}   {spec.display_name}   seed {r['record']['seed']}   "
          f"split {r['split_fingerprint']}")
    print("=" * 78)

    section("Accuracy")
    print(f"  validation accuracy (best epoch) : {r['val_accuracy']:.4f}")
    print(f"  validation tie rate (final epoch): {r['val_tie_rate']:.4f}")
    print(f"  TEST accuracy                    : {r['test_accuracy']:.4f}  (n={r['n_test']:,})")
    print(f"  TEST tie rate                    : {r['test_tie_rate']:.4f}")

    section(f"Count histogram -- how much of [0, {T}] the readout actually uses")
    counts = r["counts"]
    total = counts.size
    hist = np.bincount(counts.ravel(), minlength=T + 1)
    print(f"  {'count':>7}{'entries':>14}{'share':>10}")
    for value, n in enumerate(hist):
        print(f"  {value:>7}{n:>14,}{100.0 * n / total:>9.2f}%")
    used = int((hist > 0).sum())
    print(f"  values used: {used} of {T + 1} available")
    per_sample_max = counts.max(axis=1)
    print(f"  per-sample max count: mean {per_sample_max.mean():.3f}, "
          f"histogram {np.bincount(per_sample_max, minlength=T + 1).tolist()}")
    winners = counts[np.arange(counts.shape[0]), counts.argmax(axis=1)]
    print(f"  winning-class count: mean {winners.mean():.3f} of a possible {T}")

    section("Confidence and the attainable ceiling")
    conf = r["conf"]
    print(f"  attainable ceiling e^T/(e^T + C - 1) : {r['ceiling']:.4f}   [DERIVED]")
    print(f"  max confidence observed              : {conf.max():.4f}")
    print(f"  mean confidence                      : {conf.mean():.4f}")
    print(f"  test accuracy                        : {r['test_accuracy']:.4f}")
    gap = conf.mean() - r["test_accuracy"]
    direction = "OVERconfident" if gap > 0 else "UNDERconfident"
    print(f"  mean confidence - accuracy           : {gap:+.4f}  ({direction})")
    if r["ceiling"] < r["test_accuracy"]:
        print(f"  CEILING BINDS: accuracy {r['test_accuracy']:.4f} exceeds the ceiling "
              f"{r['ceiling']:.4f},")
        print("  so no reachable output could be correctly confident. The underconfidence")
        print("  above is structural, not a training failure.")
    else:
        print("  ceiling does not bind: accuracy is below it, so correct confidence is reachable.")
    print(f"  distinct confidences on CAL  : {r['n_distinct_cal']:,} "
          f"({r['n_cal'] / max(r['n_distinct_cal'], 1):.1f} samples per level)")
    print(f"  distinct confidences on TEST : {r['n_distinct_test']:,}")

    section("Energy proxy -- encoder vs spiking layers")
    s = r["synops"]
    print(s.layer_table())
    print(f"  encoder (real-valued input, MAC) : {s.encoder_energy_joules * 1e3:.4f} mJ  "
          f"({s.encoder_energy_fraction * 100:.1f}%)")
    print(f"  spiking layers (binary, AC)      : {s.spiking_energy_joules * 1e3:.4f} mJ  "
          f"({(1 - s.encoder_energy_fraction) * 100:.1f}%)")
    print(f"  per sample                       : {s.energy_per_sample_joules * 1e6:.3f} uJ")
    print("  DESIGN-COMPARISON PROXY, NOT MEASURED POWER. Constants [UNVERIFIED]; SOURCES.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    print("=" * 78)
    print("READOUT REPORT -- trained models, no calibration performed".center(78))
    print("=" * 78)

    rows = []
    for timesteps in args.timesteps:
        r = analyse(spec, timesteps, args.seed, args.data_root, args.results,
                    args.batch_size, args.num_workers)
        report(spec, r)
        rows.append(r)

    if len(rows) > 1:
        print()
        print("=" * 78)
        print("SIDE BY SIDE")
        print("=" * 78)
        print(f"  {'quantity':<34}" + "".join(f"{'T=' + str(r['timesteps']):>14}" for r in rows))
        def line(name, fn, fmt="{:.4f}"):
            print(f"  {name:<34}" + "".join(f"{fmt.format(fn(r)):>14}" for r in rows))
        line("validation accuracy", lambda r: r["val_accuracy"])
        line("test accuracy", lambda r: r["test_accuracy"])
        line("test tie rate", lambda r: r["test_tie_rate"])
        line("distinct confidences (cal)", lambda r: r["n_distinct_cal"], "{:,.0f}")
        line("samples per level (cal)", lambda r: r["n_cal"] / max(r["n_distinct_cal"], 1), "{:,.1f}")
        line("attainable ceiling", lambda r: r["ceiling"])
        line("mean confidence", lambda r: float(r["conf"].mean()))
        line("mean conf - accuracy", lambda r: float(r["conf"].mean()) - r["test_accuracy"], "{:+.4f}")
        line("winning-class count / T", lambda r: float(
            r["counts"][np.arange(r["counts"].shape[0]), r["counts"].argmax(axis=1)].mean()
            / r["timesteps"]))
        line("encoder share of energy", lambda r: r["synops"].encoder_energy_fraction)
        print()
        print("  No calibrator has been fitted. These are raw readout properties.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
