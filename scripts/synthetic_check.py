#!/usr/bin/env python3
"""Synthetic sanity experiment for the calibration machinery.

============================ THIS IS NOT A RESULT ABOUT SNNs ============================
Every number below comes from *synthetic* integer spike counts drawn from a generator
whose miscalibration is set by hand. No network is trained and no event data is read. The
purpose is to exercise the mechanism before a model exists. Nothing here supports or
refutes the study's hypothesis about spiking networks.
=========================================================================================

What this tests. NOTES.md argues that isotonic regression beats temperature scaling
because of **capacity**: temperature has one parameter and can only rescale the whole
confidence curve, while isotonic has at most one free value per distinct confidence level.
At low T there are few levels and many samples supporting each, so the usual variance
objection to isotonic weakens. The prediction is that the isotonic advantage tracks
``n_cal / n_distinct`` -- samples per distinct confidence level -- for which T is a proxy.

Two generator modes, because the mechanism only bites in one of them:

  uniform      Label noise is a constant. The resulting miscalibration is a roughly
               uniform offset, which one scalar fixes well. This is the CONTROL: capacity
               should buy little here, and an isotonic advantage that vanishes under
               uniform miscalibration confirms the mechanism rather than undermining it.

  conditional  Label noise varies with the favoured class's spike count, so the direction
               and size of the error depend on where a sample sits in the confidence
               range -- overconfident at the top, underconfident at the bottom. One scalar
               cannot correct both ends; a step function can.

The ratio test moves ``n_cal / n_distinct`` independently of T by varying the calibration
set size at fixed T. If the advantage tracks T but is indifferent to n_cal at fixed T,
the capacity explanation is wrong even though its T-prediction appeared to hold.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicOneVsRest,
    IsotonicTopLabel,
    TemperatureScaling,
    Uncalibrated,
    compute_metrics,
    model_outputs,
    tie_rate,
)

ARMS = [
    ("uncalibrated", Uncalibrated),
    ("temperature", TemperatureScaling),
    ("isotonic-top", IsotonicTopLabel),
    ("isotonic-ovr", IsotonicOneVsRest),
]

BANNER = "SYNTHETIC DATA -- mechanism check only, not a result about SNNs"


def max_attainable_confidence(timesteps: int, n_classes: int) -> float:
    """Largest softmax confidence the RAW readout can emit: ``e^T / (e^T + C - 1)``.

    A ceiling on the uncalibrated model only. Temperature scaling with ``t < 1`` sharpens
    past it, and as ``t`` approaches 0 confidence approaches 1, so no calibrator is
    trapped by it. What it explains is why the uncalibrated model is structurally
    underconfident at low T: at T=2 with 10 classes the ceiling is 0.451, below any
    respectable accuracy.
    """
    return float(np.exp(timesteps) / (np.exp(timesteps) + n_classes - 1))


def generate(n, timesteps, n_classes, p_hi, p_lo, mode, noise, noise_lo, noise_hi, rng):
    """Integer spike counts in [0, T] with miscalibration of a chosen shape.

    A favoured class is drawn uniformly and the counts are built around it, so the model's
    prediction is (almost always) the favoured class. The label is then corrupted with
    probability ``p_noise``; where it is corrupted the confident prediction is wrong, which
    is what creates miscalibration.

    ``uniform``      ``p_noise`` is constant, so the error is the same everywhere in the
                     confidence range -- a pure offset.
    ``conditional``  ``p_noise`` rises linearly with the favoured class's count ``k``, from
                     ``noise_lo`` at ``k = 0`` to ``noise_hi`` at ``k = T``. Since ``k`` is
                     what drives confidence, high-confidence samples are wrong more often
                     and low-confidence samples less often: overconfident at the top of the
                     range, underconfident at the bottom. No single temperature fixes both.
    """
    favoured = rng.integers(0, n_classes, size=n)
    probs = np.full((n, n_classes), p_lo)
    probs[np.arange(n), favoured] = p_hi
    counts = rng.binomial(timesteps, probs)

    favoured_count = counts[np.arange(n), favoured]
    if mode == "uniform":
        p_noise = np.full(n, noise)
    elif mode == "conditional":
        p_noise = noise_lo + (noise_hi - noise_lo) * (favoured_count / timesteps)
    else:
        raise ValueError(f"unknown miscalibration mode {mode!r}")

    labels = favoured.copy()
    corrupt = rng.random(n) < p_noise
    if corrupt.any():
        offsets = rng.integers(1, n_classes, size=int(corrupt.sum()))
        labels[corrupt] = (favoured[corrupt] + offsets) % n_classes
    return counts, labels


def evaluate(counts, labels, n_cal, n_test, seed, bins):
    """Fit every arm on a calibration split and score them on a disjoint test split."""
    rng = np.random.default_rng(seed + 977)
    perm = rng.permutation(counts.shape[0])
    cal_idx, test_idx = perm[:n_cal], perm[n_cal : n_cal + n_test]
    assert np.intersect1d(cal_idx, test_idx).size == 0, "cal and test must be disjoint"

    cal = model_outputs(counts[cal_idx], labels[cal_idx], np.random.default_rng(seed))
    test = model_outputs(counts[test_idx], labels[test_idx], np.random.default_rng(seed + 1))

    # n_distinct is measured on the CALIBRATION split, because that is what determines how
    # many free values the isotonic fit has and how many samples support each of them.
    n_distinct = int(np.unique(cal.conf).size)
    row = {
        "n_cal": n_cal,
        "acc": test.accuracy,
        "tie_rate": tie_rate(counts[test_idx]),
        "n_distinct": n_distinct,
        "per_level": n_cal / max(n_distinct, 1),
        "arms": {},
    }
    for name, cls in ARMS:
        model = cls().fit(cal.conf, cal.correct, outputs=cal)
        row["arms"][name] = compute_metrics(
            model.transform(test.conf, outputs=test), test.correct, n_bins=bins
        )
        if name == "temperature":
            row["temperature"] = model.temperature_
        if name == "isotonic-top":
            row["iso_knots"] = model.n_knots_
            row["iso_levels"] = model.n_levels_
    for key in ("ece_equal_width", "ece_equal_mass"):
        # Advantage = temperature ECE - isotonic ECE. POSITIVE means isotonic is better.
        row[f"adv_{key}"] = (
            getattr(row["arms"]["temperature"], key)
            - getattr(row["arms"]["isotonic-top"], key)
        )
    return row


def run_mode(mode, args):
    rows = []
    for timesteps in args.timesteps:
        rng = np.random.default_rng(args.seed * 1000 + timesteps)
        counts, labels = generate(
            args.n_cal + args.n_test, timesteps, args.classes, args.p_hi, args.p_lo,
            mode, args.noise, args.noise_lo, args.noise_hi, rng,
        )
        row = evaluate(counts, labels, args.n_cal, args.n_test, args.seed, args.bins)
        row["T"] = timesteps
        row["ceiling"] = max_attainable_confidence(timesteps, args.classes)
        rows.append(row)
    return rows


def print_mode(mode, rows, description):
    title = f"MODE: {mode}  --  {description}"
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    print()
    print("  Readout and capacity (n_distinct and samples/level measured on the cal split)")
    print(
        f"  {'T':>3} {'acc':>7} {'tie':>7} {'distinct':>9} {'samp/lvl':>9} "
        f"{'iso knots':>10} {'iso levels':>11} {'max conf':>9} {'temp':>7}"
    )
    for r in rows:
        print(
            f"  {r['T']:>3} {r['acc']:>7.4f} {r['tie_rate']:>7.4f} {r['n_distinct']:>9,} "
            f"{r['per_level']:>9.1f} {r['iso_knots']:>10,} {r['iso_levels']:>11,} "
            f"{r['ceiling']:>9.4f} {r['temperature']:>7.4f}"
        )

    for label, key in [
        ("ECE, equal-width bins (Guo et al.)", "ece_equal_width"),
        ("ECE, equal-mass bins", "ece_equal_mass"),
    ]:
        print()
        print(f"  {label}")
        print(
            f"  {'T':>3}" + "".join(f"{name:>15}" for name, _ in ARMS)
            + f"{'iso advantage':>15}"
        )
        for r in rows:
            line = f"  {r['T']:>3}"
            for name, _ in ARMS:
                line += f"{getattr(r['arms'][name], key):>15.4f}"
            line += f"{r[f'adv_{key}']:>15.4f}"
            print(line)
    print()
    print("  iso advantage = temperature ECE - isotonic-top ECE. POSITIVE means isotonic wins.")


def run_ratio_test(mode, args):
    """Move n_cal / n_distinct independently of T by varying the calibration-set size."""
    print()
    print("=" * 88)
    print(f"RATIO TEST ({mode}) -- vary n_cal at FIXED T, so the ratio moves without T moving")
    print("=" * 88)
    print()
    print("  If the capacity story is right, the isotonic advantage should follow")
    print("  samples-per-level here, even though T never changes. If it is flat in n_cal,")
    print("  the advantage tracks T for some other reason and the mechanism is wrong.")

    for timesteps in args.ratio_timesteps:
        rng = np.random.default_rng(args.seed * 1000 + timesteps + 7)
        pool = max(args.ratio_sizes) + args.n_test
        counts, labels = generate(
            pool, timesteps, args.classes, args.p_hi, args.p_lo,
            mode, args.noise, args.noise_lo, args.noise_hi, rng,
        )
        print()
        print(f"  T = {timesteps}  (test split fixed at {args.n_test:,})")
        print(
            f"  {'n_cal':>8} {'distinct':>9} {'samp/lvl':>9} {'iso levels':>11} "
            f"{'temp ECE':>10} {'iso ECE':>10} {'advantage':>10}"
        )
        for n_cal in args.ratio_sizes:
            r = evaluate(counts, labels, n_cal, args.n_test, args.seed, args.bins)
            print(
                f"  {n_cal:>8,} {r['n_distinct']:>9,} {r['per_level']:>9.1f} "
                f"{r['iso_levels']:>11,} "
                f"{r['arms']['temperature'].ece_equal_width:>10.4f} "
                f"{r['arms']['isotonic-top'].ece_equal_width:>10.4f} "
                f"{r['adv_ece_equal_width']:>10.4f}"
            )
        print("  (equal-width ECE shown; advantage = temp - iso, positive means isotonic wins)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--miscalibration",
        choices=["uniform", "conditional", "both"],
        default="both",
        help="shape of the injected miscalibration",
    )
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--n-cal", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--p-hi", type=float, default=0.75)
    parser.add_argument("--p-lo", type=float, default=0.15)
    parser.add_argument("--noise", type=float, default=0.20, help="uniform mode: constant label noise")
    parser.add_argument("--noise-lo", type=float, default=0.02, help="conditional mode: noise at count 0")
    parser.add_argument("--noise-hi", type=float, default=0.45, help="conditional mode: noise at count T")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio-timesteps", type=int, nargs="+", default=[2, 10])
    parser.add_argument(
        "--ratio-sizes", type=int, nargs="+",
        default=[250, 500, 1000, 2500, 5000, 10000, 20000],
    )
    parser.add_argument("--no-ratio-test", action="store_true")
    args = parser.parse_args()

    print("=" * 88)
    print(BANNER.center(88))
    print("=" * 88)
    print()
    print("Generator")
    print("-" * 9)
    print(f"  classes                 : {args.classes}")
    print(f"  spike prob, favoured    : {args.p_hi}      other: {args.p_lo}")
    print(f"  uniform mode noise      : {args.noise}")
    print(f"  conditional mode noise  : {args.noise_lo} at count 0  ->  {args.noise_hi} at count T")
    print(f"  cal / test              : {args.n_cal:,} / {args.n_test:,}  (disjoint by index)")
    print(f"  ECE bins                : {args.bins}        seed: {args.seed}")
    print()
    print("  Counts are integers in [0, T], drawn Binomial(T, p). Confidence is the softmax")
    print("  over those counts, which is the readout NOTES.md fixes for the real study.")

    modes = ["uniform", "conditional"] if args.miscalibration == "both" else [args.miscalibration]
    descriptions = {
        "uniform": "CONTROL: constant label noise, a uniform offset one scalar can fix",
        "conditional": "noise rises with count, so error direction varies with confidence",
    }
    results = {}
    for mode in modes:
        rows = run_mode(mode, args)
        results[mode] = rows
        print_mode(mode, rows, descriptions[mode])

    if not args.no_ratio_test:
        for mode in modes:
            run_ratio_test(mode, args)

    if len(modes) == 2:
        print()
        print("=" * 88)
        print("CONTROL vs CONDITIONAL -- isotonic advantage (equal-width ECE)")
        print("=" * 88)
        print()
        print(f"  {'T':>3} {'samp/lvl':>9} {'uniform adv':>13} {'conditional adv':>17}")
        for u, c in zip(results["uniform"], results["conditional"]):
            print(
                f"  {u['T']:>3} {u['per_level']:>9.1f} {u['adv_ece_equal_width']:>13.4f} "
                f"{c['adv_ece_equal_width']:>17.4f}"
            )
        print()
        print("  Capacity should buy little in the control (a scalar is sufficient there) and")
        print("  more in the conditional mode, growing as samples-per-level rises.")

    print()
    print("=" * 88)
    print(BANNER.center(88))
    print("  Synthetic counts from a hand-set generator. These numbers show the machinery")
    print("  measures what it claims to measure and that the capacity mechanism is")
    print("  testable. The study's hypothesis can only be tested on trained SNNs and real")
    print("  event data.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
