#!/usr/bin/env python3
"""A1 and A2, added after review, not pre-registered: the cap as a floor on ECE, and the bins.

For every count-readout run (36 test sets: N-MNIST two blocks at T = 2 to 10, N-MNIST four
blocks at T = 2 and 6, N-Caltech101 four blocks at T = 2, 6 and 10, N-Caltech101 two blocks at
T = 2 and 6; three seeds each) this script rebuilds the test confidences exactly as every
published number was built, checks that it reproduces the published accuracy and both ECEs,
and then reports:

- A1: per-bin signed gaps under equal-width bins, equal-mass bins and one group per distinct
  value; the number of overconfident bins, the share of test samples in them and the largest
  overconfident gap; and, where accuracy exceeds the cap, the floor ratio ``(a - p*) / ECE``
  with a test-set bootstrap 95% interval.
- A2: the effective number of equal-mass bins and their smallest and largest occupancy, the
  occupied equal-width bins, and ECE grouped by distinct value beside both binned estimates.

The settings were fixed in NOTES.md before this script was run. Writes
``results/ece_floor.json``. Reads only cached outputs; trains nothing.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import ece_equal_mass, ece_equal_width, model_outputs  # noqa: E402
from spikecalib.conditional import theoretical_ceiling  # noqa: E402
from spikecalib.floor import (  # noqa: E402
    SCHEMES, bin_assignment, ece_distinct, ece_from_gaps, overconfidence_summary, signed_bin_gaps,
)

RESULTS = os.path.join(REPO_ROOT, "results")
N_BINS = 15
N_BOOT = 1000
SEEDS = (0, 1, 2)
# (dataset, n_classes, channels label, file suffix, timesteps)
CELLS = (
    [("nmnist", 10, "32-64", "", t) for t in (2, 4, 6, 8, 10)]
    + [("nmnist", 10, "32-64-128-128", "_arch32-64-128-128", t) for t in (2, 6)]
    + [("ncaltech101", 101, "32-64-128-128", "", t) for t in (2, 6, 10)]
    + [("ncaltech101", 101, "32-64", "_arch32-64", t) for t in (2, 6)]
)
ESTIMATORS = {"equal_width": ece_equal_width, "equal_mass": ece_equal_mass,
              "distinct": lambda c, y, n_bins=N_BINS: ece_distinct(c, y)}


def published(dataset: str, channels: str, timesteps: int, seed: int) -> dict:
    """Accuracy and both ECEs as already published for this run."""
    if dataset == "nmnist" and channels == "32-64":
        with open(os.path.join(RESULTS, "sweep_nmnist.csv")) as fh:
            for r in csv.DictReader(fh):
                if (int(r["timesteps"]) == timesteps and int(r["seed"]) == seed
                        and r["arm"] == "uncalibrated"):
                    return {"accuracy": float(r["accuracy"]),
                            "ece_equal_width": float(r["ece_equal_width"]),
                            "ece_equal_mass": float(r["ece_equal_mass"]),
                            "source": "results/sweep_nmnist.csv, uncalibrated"}
    if dataset == "ncaltech101" and channels == "32-64-128-128":
        with open(os.path.join(RESULTS, "caltech_experiment.json")) as fh:
            rows = json.load(fh)["rows"]
        src = "results/caltech_experiment.json"
    else:
        with open(os.path.join(RESULTS, "arch_matched.json")) as fh:
            rows = json.load(fh)["rows"]
        src = "results/arch_matched.json"
    for r in rows:
        ch = "-".join(str(c) for c in r.get("channels", [32, 64, 128, 128]))
        if (r["dataset"] == dataset and r["readout"] == "count" and r["timesteps"] == timesteps
                and r["seed"] == seed and ch == channels):
            return {"accuracy": r["test_accuracy"], "ece_equal_width": r["ece_equal_width"],
                    "ece_equal_mass": r["ece_equal_mass"], "source": src}
    raise KeyError((dataset, channels, timesteps, seed))


def test_set(dataset, suffix, timesteps, seed):
    name = f"{dataset}_T{timesteps}_seed{seed}{suffix}"
    z = np.load(os.path.join(RESULTS, name + "_outputs.npz"))
    test = model_outputs(z["test_scores"], z["test_labels"], np.random.default_rng(seed + 1))
    return name, test.conf, (test.predictions == test.labels).astype(np.float64)


def table(g: dict) -> list:
    return [{"n": int(n), "mean_confidence": float(c), "accuracy": float(a), "gap": float(d)}
            for n, c, a, d in zip(g["n"], g["mean_confidence"], g["accuracy"], g["gap"])]


def overconfident_group_tests(g: dict) -> dict:
    """A8: are the overconfident distinct-value groups more than sampling noise?

    Every group holds one confidence value ``c``. For each group with a negative gap, the exact
    one-sided binomial p-value ``P(X <= k)``, ``X ~ Bin(n, c)``; and, over all groups, the
    number of negative-gap groups expected if every group's accuracy equalled its confidence,
    ``sum P(X < n c)``.
    """
    from scipy.stats import binom

    n = np.asarray(g["n"], dtype=np.int64)
    c = np.asarray(g["mean_confidence"], dtype=np.float64)
    k = np.rint(np.asarray(g["accuracy"]) * n).astype(np.int64)
    assert np.allclose(k / n, g["accuracy"], atol=1e-12)
    over = np.asarray(g["gap"]) < 0
    p = binom.cdf(k[over], n[over], c[over])
    # P(X < n c) = P(X <= ceil(n c) - 1)
    expected = float(np.sum(binom.cdf(np.ceil(n * c - 1e-12) - 1, n, c)))
    return {"overconfident_min_p": float(p.min()) if p.size else None,
            "overconfident_significant_005": int(np.sum(p < 0.05)),
            "overconfident_median_size": float(np.median(n[over])) if over.any() else None,
            "overconfident_with_no_correct": int(np.sum(k[over] == 0)),
            "expected_negative_gap_groups_if_calibrated": expected}


def describe(values: list) -> dict:
    return {"mean": st.mean(values), "sd": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "values": values}


def main() -> int:
    runs, cells = [], []
    for dataset, n_classes, channels, suffix, timesteps in CELLS:
        cap = theoretical_ceiling(timesteps, n_classes)
        boot = {est: [] for est in ESTIMATORS}
        cell_runs = []
        for seed in SEEDS:
            name, conf, correct = test_set(dataset, suffix, timesteps, seed)
            a, p_bar = float(correct.mean()), float(conf.mean())
            pub = published(dataset, channels, timesteps, seed)
            ece = {est: f(conf, correct, N_BINS) for est, f in ESTIMATORS.items()}
            for key in ("ece_equal_width", "ece_equal_mass"):
                if abs(ece[key[4:]] - pub[key]) > 1e-12:
                    raise RuntimeError(f"{name}: {key} {ece[key[4:]]} != published {pub[key]}")
            if abs(a - pub["accuracy"]) > 1e-12:
                raise RuntimeError(f"{name}: accuracy {a} != published {pub['accuracy']}")
            if conf.max() > cap + 1e-9:
                raise RuntimeError(f"{name}: a confidence exceeds the cap")
            row = {"name": name, "dataset": dataset, "n_classes": n_classes, "channels": channels,
                   "timesteps": timesteps, "seed": seed, "n_test": int(conf.size),
                   "accuracy": a, "mean_confidence": p_bar, "cap": cap,
                   "binds": bool(a > cap), "floor": max(0.0, a - cap),
                   "aggregate_gap": a - p_bar, "published_source": pub["source"],
                   "n_distinct": int(np.unique(np.round(conf, 12)).size), "schemes": {}}
            for scheme in SCHEMES:
                g = signed_bin_gaps(conf, correct, scheme, N_BINS)
                e = ece_from_gaps(g)
                assert abs(e - ece[scheme]) < 1e-12
                assert e >= abs(a - p_bar) - 1e-12
                if row["binds"]:
                    assert e >= a - cap - 1e-12
                s = overconfidence_summary(g)
                index, n_slots = bin_assignment(conf, scheme, N_BINS)
                occ = np.bincount(index, minlength=n_slots)
                occ = occ[occ > 0]
                if scheme == "distinct":
                    s.update(overconfident_group_tests(g))   # A8, added after the independent checks
                row["schemes"][scheme] = {
                    "ece": e, "occupied_bins": int(occ.size),
                    "min_occupancy": int(occ.min()), "max_occupancy": int(occ.max()),
                    "floor_ratio": (a - cap) / e if row["binds"] else None,
                    "excess_over_aggregate_gap": e - abs(a - p_bar),
                    **s, "bins": table(g),
                }
            # Test-set bootstrap of the floor ratio, as fixed in NOTES.md (A1).
            if row["binds"]:
                rng = np.random.default_rng(seed)
                n = conf.size
                per = {est: np.empty(N_BOOT) for est in ESTIMATORS}
                for b in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    c, y = conf[idx], correct[idx]
                    ab = float(y.mean())
                    for est, f in ESTIMATORS.items():
                        per[est][b] = (ab - cap) / f(c, y, N_BINS)
                for est in ESTIMATORS:
                    row["schemes"][est]["floor_ratio_ci95"] = [
                        float(np.percentile(per[est], 2.5)), float(np.percentile(per[est], 97.5))]
                    boot[est].append(per[est])
            runs.append(row)
            cell_runs.append(row)
            print(f"{name:<40} a {a:.4f}  cap {cap:.4f}  binds {row['binds']!s:<5}  "
                  + "  ".join(f"{s[:5]} {row['schemes'][s]['ece']:.4f}/"
                              f"{row['schemes'][s]['n_overconfident_bins']}" for s in SCHEMES),
                  flush=True)

        cell = {"dataset": dataset, "n_classes": n_classes, "channels": channels,
                "timesteps": timesteps, "cap": cap,
                "binds_in": sum(r["binds"] for r in cell_runs),
                "accuracy": describe([r["accuracy"] for r in cell_runs]),
                "mean_confidence": describe([r["mean_confidence"] for r in cell_runs]),
                "floor": describe([r["floor"] for r in cell_runs]),
                "n_distinct": describe([r["n_distinct"] for r in cell_runs]),
                "schemes": {}}
        for scheme in SCHEMES:
            sc = {
                "ece": describe([r["schemes"][scheme]["ece"] for r in cell_runs]),
                "occupied_bins": describe([r["schemes"][scheme]["occupied_bins"] for r in cell_runs]),
                "min_occupancy": min(r["schemes"][scheme]["min_occupancy"] for r in cell_runs),
                "max_occupancy": max(r["schemes"][scheme]["max_occupancy"] for r in cell_runs),
                "n_overconfident_bins": [r["schemes"][scheme]["n_overconfident_bins"] for r in cell_runs],
                "share_in_overconfident_bins": describe(
                    [r["schemes"][scheme]["share_in_overconfident_bins"] for r in cell_runs]),
                "largest_overconfident_gap": max(
                    r["schemes"][scheme]["largest_overconfident_gap"] for r in cell_runs),
            }
            if scheme == "distinct":
                sc["overconfident_min_p"] = min((r["schemes"][scheme]["overconfident_min_p"] for r in cell_runs
                                                 if r["schemes"][scheme]["overconfident_min_p"] is not None),
                                                default=None)
                sc["overconfident_significant_005"] = [r["schemes"][scheme]["overconfident_significant_005"]
                                                       for r in cell_runs]
                sc["expected_negative_gap_groups_if_calibrated"] = [
                    r["schemes"][scheme]["expected_negative_gap_groups_if_calibrated"] for r in cell_runs]
            if cell["binds_in"] == len(cell_runs):
                sc["floor_ratio"] = describe([r["schemes"][scheme]["floor_ratio"] for r in cell_runs])
                mean_boot = np.mean(np.vstack(boot[scheme]), axis=0)
                sc["floor_ratio_ci95"] = [float(np.percentile(mean_boot, 2.5)),
                                          float(np.percentile(mean_boot, 97.5))]
            cell["schemes"][scheme] = sc
        cells.append(cell)

    doc = {"analysis": "A1 and A2, added after review, not pre-registered",
           "thresholds": "NOTES.md, Post-review analyses, thresholds fixed 2026-11-12 before computation",
           "n_bins": N_BINS, "n_bootstrap": N_BOOT,
           "bootstrap": "per run: numpy default_rng(seed), 1,000 resamples of the test set, bins rebuilt "
                        "on each resample; per cell: the seed mean of the per-run resampled ratios",
           "cells": cells, "runs": runs}
    with open(os.path.join(RESULTS, "ece_floor.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote results/ece_floor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
