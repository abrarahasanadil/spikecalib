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

It also asks a question that comes logically *before* calibration: does confidence
carry any information about correctness at all? AUROC of confidence against correctness
answers it. At 0.5 confidence is pure noise with respect to being right, and no
calibrator -- isotonic, temperature or anything else -- can create information that is not
there. All any of them could do is map every confidence onto the base accuracy rate. So
this number decides whether calibration is even the right question for this readout.

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

from spikecalib.calibration import (  # noqa: E402
    discrimination,
    predict_with_ties,
    softmax_confidence,
    tie_rate,
)
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


def confidence_table(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Every distinct confidence value, how many samples hold it, and their accuracy.

    This is literally the input to a top-label calibrator: it fits a monotone map from the
    left column to the right one. Printing it in full makes the capacity argument concrete
    -- the number of rows is the most knots isotonic could ever place, and the count column
    is how many samples support each.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)
    values, inverse = np.unique(conf, return_inverse=True)
    counts = np.bincount(inverse, minlength=values.size)
    hits = np.bincount(inverse, weights=correct.astype(np.float64), minlength=values.size)
    return [
        {
            "confidence": float(v),
            "n": int(c),
            "accuracy": float(h / c) if c else float("nan"),
            "gap": float(h / c - v) if c else float("nan"),
        }
        for v, c, h in zip(values, counts, hits)
    ]


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

    # Same seeding convention as the calibration stage: cal draws from seed, test from
    # seed + 1, so the tie-break here is the one a calibration run would see.
    cal_predictions, _ = predict_with_ties(cal_counts, np.random.default_rng(seed))
    predictions, tie_mask = predict_with_ties(test_counts, np.random.default_rng(seed + 1))
    test_conf = softmax_confidence(test_counts)
    cal_conf = softmax_confidence(cal_counts)
    cal_correct = cal_predictions == cal_labels
    test_correct = predictions == test_labels

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
        "cal_conf": cal_conf,
        "cal_correct": cal_correct,
        "test_correct": test_correct,
        "cal_accuracy": float(cal_correct.mean()),
        "discrimination_cal": discrimination(cal_conf, cal_correct),
        "discrimination_test": discrimination(test_conf, test_correct),
        "confidence_table_cal": confidence_table(cal_conf, cal_correct),
        "ceiling": ceiling(timesteps, spec.n_classes),
        "synops": synops,
    }


def report(spec, r, max_rows: int = 60) -> None:
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

    section("Does confidence carry information about correctness?")
    print("  AUROC of confidence as a score for 'this prediction is right'. A calibrator is")
    print("  a monotone map, so it cannot reorder samples and cannot change this number.")
    print("  Whatever separation exists here bounds what ANY calibrated confidence can say.")
    print()
    print(f"  {'split':<8}{'n':>9}{'accuracy':>10}{'AUROC':>9}"
          f"{'mean|correct':>14}{'mean|wrong':>12}{'gap':>10}")
    for name, key, accuracy in (
        ("cal", "discrimination_cal", r["cal_accuracy"]),
        ("test", "discrimination_test", r["test_accuracy"]),
    ):
        d = r[key]
        print(
            f"  {name:<8}{d['n_correct'] + d['n_wrong']:>9,}{accuracy:>10.4f}"
            f"{d['auroc']:>9.4f}{d['mean_conf_correct']:>14.4f}"
            f"{d['mean_conf_incorrect']:>12.4f}{d['gap']:>+10.4f}"
        )
    for key in ("discrimination_cal", "discrimination_test"):
        if r[key]["note"]:
            print(f"  note: {r[key]['note']}")
    auroc = r["discrimination_test"]["auroc"]
    if not np.isnan(auroc):
        if auroc < 0.55:
            print("  -> AUROC near 0.5: confidence barely orders correct above incorrect, so no")
            print("     calibrator can do better than mapping everything to the base accuracy.")
        elif auroc > 0.8:
            print("  -> confidence orders correctness well; calibration is a meaningful question.")
        else:
            print("  -> confidence carries some information, but weakly.")

    section("Confidence spread -- how much room a calibrator has")
    print(f"  {'split':<8}{'distinct':>10}{'mean':>10}{'std':>10}{'std/mean':>11}{'min':>10}{'max':>10}")
    for name, key, conf in (
        ("cal", "discrimination_cal", r["cal_conf"]),
        ("test", "discrimination_test", r["conf"]),
    ):
        d = r[key]
        print(
            f"  {name:<8}{d['n_distinct']:>10,}{d['mean']:>10.4f}{d['std']:>10.6f}"
            f"{d['cv']:>11.5f}{float(conf.min()):>10.4f}{float(conf.max()):>10.4f}"
        )
    print("  std/mean is the coefficient of variation. Near zero means confidence is")
    print("  effectively a constant, and a constant offers a calibrator nothing to")
    print("  discriminate on -- every monotone map collapses to a single output value.")

    section("Full confidence table on the CALIBRATION split -- the calibrator's actual input")
    table = r["confidence_table_cal"]
    print("  A top-label calibrator fits a monotone map from 'confidence' to 'accuracy'.")
    print("  Rows are the most knots isotonic could ever place; n is the support for each.")
    print()
    print(f"  {'#':>4}{'confidence':>13}{'n':>9}{'accuracy':>11}{'acc - conf':>13}")
    shown = table if len(table) <= max_rows else table[:max_rows]
    for i, row in enumerate(shown):
        print(
            f"  {i:>4}{row['confidence']:>13.6f}{row['n']:>9,}"
            f"{row['accuracy']:>11.4f}{row['gap']:>+13.4f}"
        )
    if len(shown) < len(table):
        print(f"  ... {len(table) - len(shown):,} further rows suppressed "
              f"(--histogram-max-rows to raise the cap)")
    print(f"  {len(table):,} distinct confidence values across {r['n_cal']:,} calibration samples")
    gaps = np.array([row["gap"] for row in table])
    weights = np.array([row["n"] for row in table], dtype=np.float64)
    print(f"  weighted mean (accuracy - confidence): "
          f"{float(np.average(gaps, weights=weights)):+.4f}")
    print(f"  spread of that gap across rows       : "
          f"{float(gaps.min()):+.4f} to {float(gaps.max()):+.4f}")
    print("  If the gap column is near-constant, the miscalibration is a uniform offset and")
    print("  a single scalar can remove it. If it varies with confidence, it cannot.")

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
    parser.add_argument("--histogram-max-rows", type=int, default=60,
                        help="cap on printed rows of the full confidence table")
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    print("=" * 78)
    print("READOUT REPORT -- trained models, no calibration performed".center(78))
    print("=" * 78)

    rows = []
    for timesteps in args.timesteps:
        r = analyse(spec, timesteps, args.seed, args.data_root, args.results,
                    args.batch_size, args.num_workers)
        report(spec, r, args.histogram_max_rows)
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
        line("AUROC conf vs correctness (test)", lambda r: r["discrimination_test"]["auroc"])
        line("mean conf | correct", lambda r: r["discrimination_test"]["mean_conf_correct"])
        line("mean conf | incorrect", lambda r: r["discrimination_test"]["mean_conf_incorrect"])
        line("gap (correct - incorrect)", lambda r: r["discrimination_test"]["gap"], "{:+.4f}")
        line("confidence std (test)", lambda r: r["discrimination_test"]["std"], "{:.6f}")
        line("confidence std/mean (test)", lambda r: r["discrimination_test"]["cv"], "{:.5f}")
        line("encoder share of energy", lambda r: r["synops"].encoder_energy_fraction)
        print()
        print("  No calibrator has been fitted. These are raw readout properties.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
