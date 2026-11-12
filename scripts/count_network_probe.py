#!/usr/bin/env python3
"""A5 and A6, added after review, not pre-registered: what the count-trained networks do.

One inference pass per count-trained model (36: N-MNIST two blocks at T = 2 to 10, N-MNIST four
blocks at T = 2 and 6, N-Caltech101 four blocks at T = 2, 6 and 10, N-Caltech101 two blocks at
T = 2 and 6; three seeds each), on MPS, with the saved weights. Nothing is trained.

- The count logits of the pass must equal the stored ``*_outputs.npz`` exactly, or the script
  stops: the probe then describes the very networks every published number came from.
- A5: per LIF layer, over the full test set, mean spikes per neuron per timestep and the share
  of neurons that never fire. Spike rasters of test index 0 of the seed-0 split at T = 2 and
  T = 10 (main depth), with 32 neurons per hidden layer drawn by ``default_rng(0)`` and every
  output neuron. Output count patterns at T = 2 from the stored counts.
- A6: the accumulator and leaky-membrane readouts computed from the same count-trained weights
  (the output LIF bypassed; ``z_t`` aggregated), with accuracy and ECE before and after
  temperature scaling fitted on the calibration split. Exploratory.

Writes ``results/spiking_activity.json`` and ``results/same_weights_readouts.json``.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.activity import ActivityRecorder, run_split  # noqa: E402
from spikecalib.calibration import (  # noqa: E402
    TemperatureScaling, ece_equal_mass, ece_equal_width, model_outputs, quantise_confidence,
)
from spikecalib.experiments import BATCH_SIZE, load_data, load_trained, loader, splits_for  # noqa: E402

R = os.path.join(REPO_ROOT, "results")
DATA = os.path.join(REPO_ROOT, "data")
N_BINS = 15
CELLS = (
    [("nmnist", (32, 64), t) for t in (2, 4, 6, 8, 10)]
    + [("nmnist", (32, 64, 128, 128), t) for t in (2, 6)]
    + [("ncaltech101", (32, 64, 128, 128), t) for t in (2, 6, 10)]
    + [("ncaltech101", (32, 64), t) for t in (2, 6)]
)
MAIN = {"nmnist": (32, 64), "ncaltech101": (32, 64, 128, 128)}
RASTER_T = (2, 10)
RASTER_NEURONS = 32


def describe(v):
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v)}


def metrics(conf, correct):
    conf = quantise_confidence(conf)
    return {"accuracy": float(correct.mean()), "mean_conf_minus_acc": float(conf.mean() - correct.mean()),
            "ece_equal_width": ece_equal_width(conf, correct, N_BINS),
            "ece_equal_mass": ece_equal_mass(conf, correct, N_BINS)}


def readout_metrics(cal_scores, cal_labels, test_scores, test_labels, seed):
    cal = model_outputs(cal_scores, cal_labels, np.random.default_rng(seed))
    test = model_outputs(test_scores, test_labels, np.random.default_rng(seed + 1))
    correct = (test.predictions == test.labels).astype(np.float64)
    ts = TemperatureScaling().fit(cal.conf, cal.correct, outputs=cal)
    return {"uncalibrated": metrics(test.conf, correct),
            "temperature": metrics(ts.transform(test.conf, outputs=test), correct),
            "fitted_temperature": float(ts.temperature_)}


def output_patterns(counts: np.ndarray, T: int) -> dict:
    top = counts.max(axis=1)
    n_top = (counts == top[:, None]).sum(axis=1)
    others_silent = (counts.sum(axis=1) == top)
    return {"saturated": float(np.mean((top == T) & others_silent & (n_top == 1))),
            "all_silent": float(np.mean(top == 0)),
            "tie_at_top": float(np.mean(n_top > 1)),
            "top_fires_every_step": float(np.mean(top == T))}


def main() -> int:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device.type != "mps":
        raise RuntimeError("run on MPS: CPU arithmetic flips a few spikes and would not reproduce the stored outputs")
    activity_rows, same_rows, rasters, patterns = [], [], [], []
    data_cache = {}
    for dataset, channels, T in CELLS:
        if (dataset, T) not in data_cache:
            data_cache.clear()
            data_cache[(dataset, T)] = load_data(dataset, T, DATA, in_memory=False)
        data = data_cache[(dataset, T)]
        bs = BATCH_SIZE[dataset]
        for seed in (0, 1, 2):
            splits = splits_for(data, seed)
            model, record, name = load_trained(R, data, seed, "count", channels, splits)
            z = np.load(os.path.join(R, name + "_outputs.npz"))
            rec = ActivityRecorder(model)
            keep = seed == 0 and channels == MAIN[dataset] and T in RASTER_T
            with rec:
                t_counts, t_acc, t_mem, t_y = run_split(model, loader(data.test, splits.test, bs), device,
                                                        rec, keep_first=keep)
            c_counts, c_acc, c_mem, c_y = run_split(model, loader(data.train, splits.cal, bs), device)
            for got, want in ((t_counts, z["test_scores"]), (c_counts, z["cal_scores"])):
                if not np.array_equal(got, want):
                    raise RuntimeError(f"{name}: counts differ from the stored outputs "
                                       f"(max {np.abs(got - want).max()})")
            assert np.array_equal(t_y, z["test_labels"]) and np.array_equal(c_y, z["cal_labels"])
            layers = {n: {"n_neurons": s.n_neurons, "rate": s.rate, "silent_share": s.silent_share}
                      for n, s in rec.stats.items()}
            activity_rows.append({"name": name, "dataset": dataset, "n_blocks": len(channels),
                                  "timesteps": T, "seed": seed, "n_test": int(t_y.size), "layers": layers})
            if T == 2:
                patterns.append({"name": name, "dataset": dataset, "n_blocks": len(channels), "seed": seed,
                                 **output_patterns(t_counts, T)})
            if keep:
                rng = np.random.default_rng(0)
                raster = {"dataset": dataset, "timesteps": T, "seed": 0, "n_blocks": len(channels),
                          "test_index": int(splits.test[0]), "label": int(t_y[0]),
                          "counts": t_counts[0].tolist(), "layers": {}}
                for lname, spikes in rec.kept.items():
                    n = spikes.shape[1]
                    if lname == "output":
                        pick = np.arange(n)
                    else:
                        pick = np.sort(rng.choice(n, size=min(RASTER_NEURONS, n), replace=False))
                    raster["layers"][lname] = {"neurons": pick.tolist(), "n_neurons": int(n),
                                               "spikes": spikes[:, pick].astype(int).tolist(),
                                               "layer_rate_this_sample": float(spikes.mean())}
                rasters.append(raster)
            row = {"name": name, "dataset": dataset, "n_blocks": len(channels), "timesteps": T, "seed": seed,
                   "count": readout_metrics(c_counts, c_y, t_counts, t_y, seed),
                   "accumulator_same_weights": readout_metrics(c_acc, c_y, t_acc, t_y, seed),
                   "membrane_same_weights": readout_metrics(c_mem, c_y, t_mem, t_y, seed)}
            same_rows.append(row)
            print(f"{name:<40} rates " + " ".join(f"{k} {v['rate']:.3f}/{v['silent_share']:.2f}"
                                                   for k, v in layers.items())
                  + f" | acc count {row['count']['uncalibrated']['accuracy']:.4f} "
                    f"accum {row['accumulator_same_weights']['uncalibrated']['accuracy']:.4f} "
                    f"mem {row['membrane_same_weights']['uncalibrated']['accuracy']:.4f}", flush=True)

    cells = []
    for dataset, channels, T in CELLS:
        rs = [r for r in activity_rows if r["dataset"] == dataset and r["n_blocks"] == len(channels)
              and r["timesteps"] == T]
        cells.append({"dataset": dataset, "n_blocks": len(channels), "timesteps": T,
                      "layers": {n: {"n_neurons": rs[0]["layers"][n]["n_neurons"],
                                     "rate": describe([r["layers"][n]["rate"] for r in rs]),
                                     "silent_share": describe([r["layers"][n]["silent_share"] for r in rs])}
                                 for n in rs[0]["layers"]}})
    pattern_cells = []
    for dataset in ("nmnist", "ncaltech101"):
        for blocks in (2, 4):
            ps = [p for p in patterns if p["dataset"] == dataset and p["n_blocks"] == blocks]
            if ps:
                pattern_cells.append({"dataset": dataset, "n_blocks": blocks,
                                      **{k: describe([p[k] for p in ps]) for k in
                                         ("saturated", "all_silent", "tie_at_top", "top_fires_every_step")}})
    with open(os.path.join(R, "spiking_activity.json"), "w") as fh:
        json.dump({"analysis": "A5, added after review, not pre-registered", "device": str(device),
                   "cells": cells, "runs": activity_rows, "output_patterns_T2": pattern_cells,
                   "output_patterns_T2_runs": patterns, "rasters": rasters}, fh, indent=1)
    with open(os.path.join(R, "same_weights_readouts.json"), "w") as fh:
        json.dump({"analysis": "A6, added after review, not pre-registered; exploratory",
                   "method": "count-trained weights; output LIF bypassed; the final BatchNorm1d output "
                             "aggregated as the accumulator (sum) and the leaky membrane (tau = 2); "
                             "temperature scaling fitted on the calibration split",
                   "rows": same_rows}, fh, indent=1)
    print("wrote results/spiking_activity.json and results/same_weights_readouts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
