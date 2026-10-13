#!/usr/bin/env python3
"""Resolution is a timestep problem: the operating-point bound and the Brier decomposition.

No training. Every count-readout model on disk, every accumulator and every learned-scale
model is run over its calibration and test splits once (outputs are cached), and three
things are computed.

**The check.** The number of distinct top-label confidences a count readout emits is at
most ``binomial(T + C - 1, C - 1)`` for any logit scale (``spikecalib.theory``). The
observed count is compared with that bound in every count and scaled-count cell. This is
recorded as a check and not a prediction, because the counts were already in the result
files when the bound was derived. The bound is only informative while it is below the
number of test samples, and each row says whether it is.

**Prediction 9**, recorded in NOTES.md before this script existed: on N-MNIST at ``T = 2``
with two blocks, the count readout's resolution stays below the accumulator's under both
binned estimators, with or without the learned scale and after either calibrator. Twelve
comparisons of three-seed means; the script evaluates them against the registered condition.

**The decomposition itself.** For the count and scaled-count readouts, whose confidences
take few distinct values, the Murphy decomposition by exact value is reported, with no
binning choice. For every readout the binned decomposition is reported under both
estimators. Resolution is compared across readouts only under the same binned estimator;
exact and binned figures are never set against each other. On N-Caltech101, where the two
readouts' accuracies differ, resolution is reported as ``RES/UNC`` and nothing is predicted.

Calibrators are fitted on the calibration split, on CPU. Zero DataLoader workers.
Writes ``results/resolution_experiment.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicTopLabel, TemperatureScaling, Uncalibrated, discrimination, quantise_confidence,
)
from spikecalib.experiments import (  # noqa: E402
    DEFAULT_CHANNELS, all_cached, cell_outputs, load_data, stem,
)
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.theory import max_distinct_confidences, murphy_binned, murphy_exact  # noqa: E402
from spikecalib.train import resolve_device  # noqa: E402

TWO, FOUR = (32, 64), (32, 64, 128, 128)
READOUTS = ("count", "scaled-count", "accumulator")
STATES = (("uncalibrated", Uncalibrated), ("temperature", TemperatureScaling),
          ("isotonic-top-label", IsotonicTopLabel))
ESTIMATORS = ("equal_width", "equal_mass")
# (dataset, channels, timesteps): every cell in which at least a count model was trained.
CELLS = (
    [("nmnist", TWO, T) for T in (2, 4, 6, 8, 10)] + [("nmnist", FOUR, T) for T in (2, 6)]
    + [("ncaltech101", FOUR, T) for T in (2, 6, 10)] + [("ncaltech101", TWO, T) for T in (2, 6)]
)
SEEDS = (0, 1, 2)
KEEP = ("brier", "reliability", "resolution", "uncertainty", "within_bin_variance",
        "within_bin_covariance", "ece", "n_groups")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()
    device = resolve_device()

    rows, check = [], []
    print("=" * 108)
    print("RESOLUTION  operating-point bound and Brier decomposition; no training")
    print("=" * 108, flush=True)

    by_t: dict = {}
    for dataset, channels, T in CELLS:
        by_t.setdefault((dataset, T), []).append(channels)

    for (dataset, T), channel_sets in by_t.items():
        present = []
        for channels in channel_sets:
            for readout in READOUTS:
                for seed in SEEDS:
                    name = stem(dataset, T, seed, readout, channels)
                    if os.path.exists(os.path.join(args.out, name + "_weights.pt")):
                        present.append((channels, readout, seed, name))
        # Frames are read into RAM only if some model here has no cached outputs yet.
        data = load_data(dataset, T, args.data_root,
                         in_memory=not all_cached(args.out, [p[3] for p in present]))
        C = data.spec.n_classes
        for channels, readout, seed, name in present:
            out = cell_outputs(args.out, data, seed, device, readout, channels)
            test, cal = out.test, out.cal
            base = {"dataset": dataset, "n_classes": C, "channels": list(channels),
                    "n_blocks": len(channels), "timesteps": T, "seed": seed, "readout": readout,
                    "logit_scale": out.scale, "split_fingerprint": out.split_fingerprint,
                    "n_cal": int(cal.n), "n_test": int(test.n),
                    "test_accuracy": float(test.correct.mean())}

            if readout != "accumulator":
                bound = max_distinct_confidences(T, C)
                n_conf = int(np.unique(quantise_confidence(test.conf)).size)
                # the distinct gap multisets actually produced, counted on the integer counts
                gaps = np.sort(out.test_counts.max(axis=1, keepdims=True) - out.test_counts, axis=1)
                n_multisets = int(np.unique(gaps, axis=0).shape[0])
                check.append({**base, "n_distinct_confidences": n_conf,
                              "n_distinct_gap_multisets": n_multisets, "bound": bound,
                              "within_bound": bool(n_conf <= bound and n_multisets <= bound),
                              "bound_below_n_test": bool(bound < test.n),
                              "share_of_bound_used": n_conf / bound})

            for state, cls in STATES:
                conf = cls().fit(cal.conf, cal.correct, outputs=cal).transform(test.conf, outputs=test)
                conf = quantise_confidence(np.asarray(conf, dtype=np.float64))
                d = discrimination(conf, test.correct)
                sel = selective_summary(conf, test.correct)
                common = {**base, "state": state, "mean_confidence": float(conf.mean()),
                          "mean_conf_minus_acc": float(conf.mean() - test.correct.mean()),
                          "n_distinct_confidences": int(np.unique(conf).size),
                          "auroc": d["auroc"], "aurc": sel["aurc"]}
                for est in ("exact",) + ESTIMATORS:
                    m = murphy_exact(conf, test.correct) if est == "exact" else \
                        murphy_binned(conf, test.correct, args.bins, est)
                    row = {**common, "estimator": est, **{k: m[k] for k in KEEP}}
                    row["res_over_unc"] = m["resolution"] / m["uncertainty"] if m["uncertainty"] > 0 else None
                    row["samples_per_group"] = test.n / m["n_groups"]
                    rows.append(row)
            r = next(x for x in reversed(rows) if x["state"] == "uncalibrated" and x["estimator"] == "equal_width")
            print(f"  {dataset:<12} blocks {len(channels)} T={T:<2} seed {seed} {readout:<12} "
                  f"acc {base['test_accuracy']:.4f}  RES ew {r['resolution']:.5f}  "
                  f"RES/UNC {r['res_over_unc']:.3f}  levels {r['n_distinct_confidences']}", flush=True)

    # ---- the check ----
    violations = [c for c in check if not c["within_bound"]]
    informative = [c for c in check if c["bound_below_n_test"]]
    print(f"\n  CHECK  distinct confidences within binomial(T+C-1, C-1): "
          f"{len(check) - len(violations)} of {len(check)} cells; the bound is below the "
          f"test-set size in {len(informative)} of them")

    # ---- prediction 9 ----
    def mean_res(readout, state, est):
        vals = [r["resolution"] for r in rows
                if r["dataset"] == "nmnist" and r["channels"] == list(TWO) and r["timesteps"] == 2
                and r["readout"] == readout and r["state"] == state and r["estimator"] == est]
        return st.mean(vals) if len(vals) == len(SEEDS) else None

    comparisons = []
    for readout in ("count", "scaled-count"):
        for state, _ in STATES:
            for est in ESTIMATORS:
                a, b = mean_res(readout, state, est), mean_res("accumulator", state, est)
                comparisons.append({"readout": readout, "state": state, "estimator": est,
                                    "count_resolution": a, "accumulator_resolution": b,
                                    "count_below_accumulator": None if a is None or b is None else bool(a < b)})
    complete = all(c["count_below_accumulator"] is not None for c in comparisons)
    verdict = {"statement": "N-MNIST T=2, 2 blocks: count resolution below the accumulator's in all "
                            "twelve comparisons (2 count readouts x 3 states x 2 binned estimators)",
               "comparisons": comparisons, "complete": complete,
               "held": bool(complete and all(c["count_below_accumulator"] for c in comparisons))}
    if complete:
        print(f"  PREDICTION 9  {'HELD' if verdict['held'] else 'FALSIFIED'}: count below "
              f"accumulator in {sum(c['count_below_accumulator'] for c in comparisons)} of 12")
    else:
        print("  PREDICTION 9  not evaluated: some of the three readouts are missing at N-MNIST T=2")

    path = os.path.join(args.out, "resolution_experiment.json")
    with open(path, "w") as fh:
        json.dump({"n_bins": args.bins,
                   "registered": "NOTES.md, prediction 9 and the bound check, 2026-10-13",
                   "check": {"cells": len(check), "violations": len(violations),
                             "cells_where_bound_is_below_n_test": len(informative), "rows": check},
                   "prediction_9": verdict, "rows": rows}, fh, indent=1)
    print(f"\n  wrote {os.path.relpath(path, REPO_ROOT)}  ({len(rows)} decomposition rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
