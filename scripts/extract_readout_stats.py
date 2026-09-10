#!/usr/bin/env python3
"""Extract per-T readout statistics the sweep CSV does not carry, and cache them.

The sweep records aggregate calibration metrics, but the site's ceiling table needs the
**observed maximum confidence** at each ``T`` -- the single largest softmax value any test
sample produced -- so it can be compared against the derived ceiling
``e^T / (e^T + C - 1)``. That is a property of the raw counts rather than of any
calibrator, so it is extracted once from the saved weights and cached to JSON. Figures and
the site read the cache; nothing re-runs inference to redraw a plot.

Writes ``results/readout_stats.json``.
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


def ceiling(timesteps: int, n_classes: int) -> float:
    """[DERIVED] softmax of the extreme reachable count vector [T, 0, ..., 0]."""
    return float(np.exp(timesteps) / (np.exp(timesteps) + n_classes - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    out: dict = {}

    for timesteps in args.timesteps:
        stem = f"{spec.key}_T{timesteps}_seed{args.seed}"
        with open(os.path.join(args.results, f"{stem}_record.json")) as fh:
            record = json.load(fh)
        mc = record["model_config"]
        cfg = ModelConfig(channels=tuple(mc["channels"]), hidden=mc["hidden"],
                          tau=mc["tau"], v_threshold=mc["v_threshold"],
                          surrogate_alpha=mc["surrogate_alpha"])

        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)
        splits = make_splits(labels, spec, seed=args.seed, n_test=len(test_set))

        model = build_model(spec, timesteps, cfg)
        model.load_state_dict(
            torch.load(os.path.join(args.results, f"{stem}_weights.pt"), map_location="cpu")
        )
        loader = DataLoader(
            Subset(test_set, splits.test.tolist()), batch_size=args.batch_size,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        )
        counts, test_labels, synops = predict_counts(model, loader, device)
        conf = softmax_confidence(counts)
        predictions, _ = predict_with_ties(counts, np.random.default_rng(args.seed + 1))

        derived = ceiling(timesteps, spec.n_classes)
        observed_max = float(conf.max())
        out[str(timesteps)] = {
            "timesteps": timesteps,
            "seed": args.seed,
            "split_fingerprint": splits.fingerprint,
            "ceiling_derived": derived,
            "max_confidence_observed": observed_max,
            "ceiling_minus_observed": derived - observed_max,
            "mean_confidence": float(conf.mean()),
            "min_confidence": float(conf.min()),
            "accuracy": float((predictions == test_labels).mean()),
            "tie_rate": tie_rate(counts),
            "n_distinct_confidences": int(np.unique(conf).size),
            "count_histogram": np.bincount(counts.ravel(), minlength=timesteps + 1).tolist(),
            "winning_count_mean": float(counts.max(axis=1).mean()),
            "encoder_energy_fraction": synops.encoder_energy_fraction,
            "energy_per_sample_joules_proxy": synops.energy_per_sample_joules,
            "train_loss_final": record["history"][-1]["train_loss"],
            "cross_entropy_floor": float(-np.log(derived)),
        }
        print(
            f"  T={timesteps:<3} ceiling {derived:.4f}  observed max {observed_max:.4f}  "
            f"diff {derived - observed_max:+.2e}  distinct {out[str(timesteps)]['n_distinct_confidences']}",
            flush=True,
        )

    path = os.path.join(args.results, "readout_stats.json")
    with open(path, "w") as fh:
        json.dump({spec.key: out}, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
