#!/usr/bin/env python3
"""Ranking with the weights held fixed. Added after review, not pre-registered.

Settings fixed in NOTES.md ("A8. Sensitivity checks", "Ranking with the weights held fixed")
before this script was run. For the three count-trained N-MNIST networks at T = 2 (two blocks),
one inference pass over the test set on MPS with the saved weights; the count logits must equal
the stored ``*_outputs.npz`` exactly. The pre-threshold output ``z_t`` is also read as the
accumulator and as the leaky membrane, as in A6. Reported per seed:

- the AUROC of confidence for correctness of the count readout, the accumulator reading and the
  membrane reading of the same weights, and of the separately trained accumulator;
- the paired difference count minus accumulator reading, with a test-set bootstrap
  (``numpy.random.default_rng(seed)``, 1,000 resamples), as in A4;
- for the count readout, the separately trained accumulator and the accumulator reading, the
  shares of correct-wrong pairs ordered correctly, tied and ordered wrongly, so that
  ``AUROC = ordered + tied / 2``.

Writes ``results/same_weights_ranking.json``.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.activity import run_split  # noqa: E402
from spikecalib.calibration import model_outputs, quantise_confidence  # noqa: E402
from spikecalib.experiments import BATCH_SIZE, load_data, load_trained, loader, splits_for  # noqa: E402

R = os.path.join(REPO_ROOT, "results")
DATA = os.path.join(REPO_ROOT, "data")
N_BOOT = 1000
T = 2
CHANNELS = (32, 64)


def pair_shares(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Shares of (correct, wrong) pairs with the correct sample's confidence above, equal to and
    below the wrong sample's. Confidences are grouped to 12 decimals, as everywhere else."""
    q = quantise_confidence(conf)
    good, bad = np.sort(q[correct == 1]), np.sort(q[correct == 0])
    n_pairs = good.size * bad.size
    below = np.searchsorted(bad, good, side="left")      # wrong samples strictly below each correct one
    not_above = np.searchsorted(bad, good, side="right")  # wrong samples at or below
    ordered = float(below.sum()) / n_pairs
    tied = float((not_above - below).sum()) / n_pairs
    return {"ordered": ordered, "tied": tied, "reversed": 1.0 - ordered - tied,
            "auroc_from_pairs": ordered + tied / 2.0, "n_pairs": int(n_pairs)}


def describe(v):
    return {"mean": st.mean(v), "sd": st.stdev(v), "min": min(v), "max": max(v), "values": v}


def main() -> int:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device.type != "mps":
        raise RuntimeError("run on MPS: CPU arithmetic flips a few spikes and would not reproduce the stored outputs")
    data = load_data("nmnist", T, DATA, in_memory=False)
    rows = []
    for seed in (0, 1, 2):
        splits = splits_for(data, seed)
        model, _, name = load_trained(R, data, seed, "count", CHANNELS, splits)
        z = np.load(os.path.join(R, name + "_outputs.npz"))
        t_counts, t_acc, t_mem, t_y = run_split(model, loader(data.test, splits.test, BATCH_SIZE["nmnist"]), device)
        if not np.array_equal(t_counts, z["test_scores"]):
            raise RuntimeError(f"{name}: counts differ from the stored outputs")
        assert np.array_equal(t_y, z["test_labels"])
        zt = np.load(os.path.join(R, f"nmnist_T{T}_seed{seed}_accumulator_outputs.npz"))
        assert np.array_equal(zt["test_labels"], t_y)
        readings = {
            "count": model_outputs(t_counts, t_y, np.random.default_rng(seed + 1)),
            "accumulator_same_weights": model_outputs(t_acc, t_y, np.random.default_rng(seed + 1)),
            "membrane_same_weights": model_outputs(t_mem, t_y, np.random.default_rng(seed + 1)),
            "accumulator_trained": model_outputs(zt["test_scores"], zt["test_labels"], np.random.default_rng(seed + 1)),
        }
        conf = {k: (o.conf if k == "count" else quantise_confidence(o.conf)) for k, o in readings.items()}
        correct = {k: (o.predictions == o.labels).astype(np.float64) for k, o in readings.items()}
        row = {"name": name, "seed": seed, "n_test": int(t_y.size),
               "accuracy": {k: float(v.mean()) for k, v in correct.items()},
               "auroc": {k: float(roc_auc_score(correct[k], conf[k])) for k in readings},
               "pairs": {k: pair_shares(conf[k], correct[k])
                         for k in ("count", "accumulator_trained", "accumulator_same_weights")}}
        for k, p in row["pairs"].items():
            assert abs(p["auroc_from_pairs"] - row["auroc"][k]) < 1e-9, (name, k)
        # Paired bootstrap of count minus the accumulator reading, on the same test images.
        rng = np.random.default_rng(seed)
        n = t_y.size
        d = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            d[i] = (roc_auc_score(correct["count"][idx], conf["count"][idx])
                    - roc_auc_score(correct["accumulator_same_weights"][idx], conf["accumulator_same_weights"][idx]))
        row["count_minus_accumulator_same_weights"] = {
            "difference": row["auroc"]["count"] - row["auroc"]["accumulator_same_weights"],
            "draws": d}
        rows.append(row)
        print(f"{name}: AUROC count {row['auroc']['count']:.4f}  accum(same) {row['auroc']['accumulator_same_weights']:.4f}  "
              f"mem(same) {row['auroc']['membrane_same_weights']:.4f}  accum(trained) {row['auroc']['accumulator_trained']:.4f}",
              flush=True)

    mean_draws = np.mean(np.vstack([r["count_minus_accumulator_same_weights"].pop("draws") for r in rows]), axis=0)
    summary = {k: describe([r["auroc"][k] for r in rows]) for k in rows[0]["auroc"]}
    diffs = [r["count_minus_accumulator_same_weights"]["difference"] for r in rows]
    summary["count_minus_accumulator_same_weights"] = {
        **describe(diffs), "ci95_seed_mean": [float(np.percentile(mean_draws, 2.5)), float(np.percentile(mean_draws, 97.5))]}
    for k in ("count", "accumulator_trained", "accumulator_same_weights"):
        for part in ("ordered", "tied", "reversed"):
            summary[f"pairs_{k}_{part}"] = describe([r["pairs"][k][part] for r in rows])
    doc = {"analysis": "Ranking with the weights held fixed; added after review, not pre-registered",
           "thresholds": "NOTES.md, A8, Ranking with the weights held fixed (descriptive)",
           "dataset": "nmnist", "timesteps": T, "channels": list(CHANNELS), "n_bootstrap": N_BOOT,
           "rows": rows, "summary": summary}
    with open(os.path.join(R, "same_weights_ranking.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps(summary["count_minus_accumulator_same_weights"]))
    print("wrote results/same_weights_ranking.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
