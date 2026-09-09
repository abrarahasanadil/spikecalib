#!/usr/bin/env python3
"""Synthetic sanity experiment for the calibration machinery.

============================ THIS IS NOT A RESULT ABOUT SNNs ============================
Every number below comes from *synthetic* integer spike counts drawn from a generator
whose miscalibration is set by hand. No network is trained and no event data is read. The
purpose is to exercise the mechanism before a model exists: to check that the tie rate
behaves as the discreteness argument predicts, that both ECE estimators are computed and
can disagree, and that the four calibrators run end to end on counts shaped like the ones
a real readout will produce. Nothing here supports or refutes the study's hypothesis.
=========================================================================================

The generator. For each sample a true class ``y`` is drawn uniformly. With probability
``--noise`` the counts are instead built around a different class ``z``, which is what
makes the model overconfident: it produces the same crisp count pattern it produces when
it is right, but the label disagrees. Per-class counts are then drawn as
``Binomial(T, p_hi)`` for the favoured class and ``Binomial(T, p_lo)`` for the rest, so
counts are integers in ``[0, T]`` exactly as a spike-count readout gives.

Calibration and test sets are disjoint by index, carved from one pool and asserted.
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


def generate(n, timesteps, n_classes, p_hi, p_lo, noise, rng):
    """Integer spike counts in [0, T], with controllable overconfidence."""
    labels = rng.integers(0, n_classes, size=n)
    favoured = labels.copy()
    corrupted = rng.random(n) < noise
    if corrupted.any():
        # Move the favoured class somewhere else, uniformly among the wrong classes.
        offsets = rng.integers(1, n_classes, size=int(corrupted.sum()))
        favoured[corrupted] = (labels[corrupted] + offsets) % n_classes
    probs = np.full((n, n_classes), p_lo)
    probs[np.arange(n), favoured] = p_hi
    counts = rng.binomial(timesteps, probs)
    return counts, labels


def max_attainable_confidence(timesteps: int, n_classes: int) -> float:
    """The largest softmax confidence any spike-count readout can produce at this T.

    The most extreme count vector is one class at T and the rest at 0, giving
    ``e^T / (e^T + C - 1)``. This is a hard ceiling: no amount of training can push
    confidence past it, because there is no larger gap available on the integer lattice.
    At T=2 with 10 classes it is 0.451, which is *below* a plausible accuracy -- so the
    model is structurally forced into underconfidence at low T. This is the "attainable
    confidence set is small and coarse" claim from NOTES.md as a number.
    """
    return float(np.exp(timesteps) / (np.exp(timesteps) + n_classes - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--n-cal", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--p-hi", type=float, default=0.75, help="spike probability for the favoured class")
    parser.add_argument("--p-lo", type=float, default=0.15, help="spike probability for every other class")
    parser.add_argument("--noise", type=float, default=0.20, help="fraction of samples whose counts favour a wrong class")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 88)
    print(BANNER.center(88))
    print("=" * 88)
    print()
    print("Generator")
    print("-" * 9)
    print(f"  classes                 : {args.classes}")
    print(f"  spike prob, favoured    : {args.p_hi}")
    print(f"  spike prob, other       : {args.p_lo}")
    print(f"  label noise             : {args.noise}  (counts favour a wrong class this often)")
    print(f"  cal / test              : {args.n_cal:,} / {args.n_test:,}  (disjoint by index)")
    print(f"  ECE bins                : {args.bins}")
    print(f"  seed                    : {args.seed}")
    print()
    print("  Counts are integers in [0, T], drawn Binomial(T, p). Confidence is the softmax")
    print("  over those counts, which is the readout NOTES.md fixes for the real study.")

    rows = []
    for timesteps in args.timesteps:
        rng = np.random.default_rng(args.seed * 1000 + timesteps)
        pool_n = args.n_cal + args.n_test
        counts, labels = generate(
            pool_n, timesteps, args.classes, args.p_hi, args.p_lo, args.noise, rng
        )

        # Disjoint cal/test by index, asserted rather than assumed (correctness rule 1).
        perm = rng.permutation(pool_n)
        cal_idx, test_idx = perm[: args.n_cal], perm[args.n_cal :]
        assert np.intersect1d(cal_idx, test_idx).size == 0

        cal = model_outputs(counts[cal_idx], labels[cal_idx], np.random.default_rng(args.seed))
        test = model_outputs(counts[test_idx], labels[test_idx], np.random.default_rng(args.seed + 1))

        row = {
            "T": timesteps,
            "acc": test.accuracy,
            "tie_rate": tie_rate(counts[test_idx]),
            "distinct": int(np.unique(test.conf).size),
            "ceiling": max_attainable_confidence(timesteps, args.classes),
            "mean_conf": float(test.conf.mean()),
            "arms": {},
        }
        for name, cls in ARMS:
            model = cls().fit(cal.conf, cal.correct, outputs=cal)
            calibrated = model.transform(test.conf, outputs=test)
            metrics = compute_metrics(calibrated, test.correct, n_bins=args.bins)
            row["arms"][name] = metrics
            if name == "temperature":
                row["temperature"] = model.temperature_
        rows.append(row)

    # ---------------------------------------------------------------------------------
    print()
    print("Table 1 -- readout behaviour (test split)")
    print("-" * 41)
    print(
        f"  {'T':>3} {'accuracy':>9} {'tie rate':>9} {'distinct':>9} "
        f"{'mean conf':>10} {'max conf':>9} {'fitted temp':>12}"
    )
    for r in rows:
        print(
            f"  {r['T']:>3} {r['acc']:>9.4f} {r['tie_rate']:>9.4f} {r['distinct']:>9,} "
            f"{r['mean_conf']:>10.4f} {r['ceiling']:>9.4f} "
            f"{r.get('temperature', float('nan')):>12.4f}"
        )
    print()
    print("  tie rate is a reported quantity, not a diagnostic: the hypothesis predicts it")
    print("  rises as T falls, because fewer attainable counts means more collisions.")
    print()
    print("  max conf is the ceiling e^T / (e^T + C - 1) -- the largest confidence ANY")
    print("  spike-count readout can emit at this T. Where it sits below accuracy, the model")
    print("  is structurally forced into underconfidence and no calibrator can be blamed for")
    print("  it. That is the coarse attainable-confidence set from NOTES.md, as a number.")

    for label, key in [
        ("Table 2 -- ECE, equal-width bins (Guo et al.)", "ece_equal_width"),
        ("Table 3 -- ECE, equal-mass bins", "ece_equal_mass"),
    ]:
        print()
        print(label)
        print("-" * len(label))
        header = f"  {'T':>3}" + "".join(f"{name:>15}" for name, _ in ARMS)
        print(header)
        for r in rows:
            line = f"  {r['T']:>3}"
            for name, _ in ARMS:
                line += f"{getattr(r['arms'][name], key):>15.4f}"
            print(line)

    print()
    print("Table 4 -- disagreement between the two estimators (|equal-mass - equal-width|)")
    print("-" * 78)
    print(f"  {'T':>3}" + "".join(f"{name:>15}" for name, _ in ARMS) + f"{'degenerate':>12}")
    for r in rows:
        line = f"  {r['T']:>3}"
        for name, _ in ARMS:
            line += f"{r['arms'][name].ece_disagreement:>15.4f}"
        flag = any(r["arms"][name].degenerate for name, _ in ARMS)
        line += f"{str(flag):>12}"
        print(line)
    print()
    print("  Disagreement between the two estimators is itself a finding (correctness rule 2).")
    print("  Both are always reported; neither is chosen because it is more convenient.")

    notes = {n for r in rows for name, _ in ARMS for n in r["arms"][name].notes}
    if notes:
        print()
        print("Binning notes raised during this run")
        print("-" * 35)
        for n in sorted(notes):
            print(f"  - {n}")

    print()
    print("=" * 88)
    print(BANNER.center(88))
    print("  Synthetic counts from a hand-set generator. The tie-rate trend here is a")
    print("  property of Binomial draws at small T, not evidence about spiking networks.")
    print("  It shows the machinery measures what it claims to measure; the study's")
    print("  hypothesis can only be tested on trained SNNs and real event data.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
