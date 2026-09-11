#!/usr/bin/env python3
"""Add selective-prediction columns to the sweep CSV, recomputed from saved counts.

No training. The spike counts are re-extracted from the fifteen saved models, the four
calibrators are refitted exactly as the sweep fitted them, and the selective numbers are
computed from the resulting confidences.

Also caches the per-sample confidence and correctness arrays to
``results/selective_cache.npz`` so the risk-coverage figure and any later selective
analysis can be produced without touching a model again.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import model_outputs  # noqa: E402
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.sweep import ARMS  # noqa: E402
from spikecalib.train import predict_counts, resolve_device  # noqa: E402

NEW_COLS = [
    "aurc", "aurc_optimal", "aurc_excess_normalised",
    "n_achievable_coverage_levels",
    "coverage_at_risk_0p01", "coverage_at_risk_0p005", "coverage_at_risk_0p001",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    computed: dict[tuple, dict] = {}
    cache: dict[str, np.ndarray] = {}

    for timesteps in args.timesteps:
        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)

        for seed in args.seeds:
            stem = f"{spec.key}_T{timesteps}_seed{seed}"
            rec = json.load(open(os.path.join(args.results, f"{stem}_record.json")))
            mc = rec["model_config"]
            cfg = ModelConfig(channels=tuple(mc["channels"]), hidden=mc["hidden"],
                              tau=mc["tau"], v_threshold=mc["v_threshold"],
                              surrogate_alpha=mc["surrogate_alpha"])
            splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))
            model = build_model(spec, timesteps, cfg)
            model.load_state_dict(
                torch.load(os.path.join(args.results, f"{stem}_weights.pt"), map_location="cpu"))

            def loader(ds, idx):
                return DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0)

            cal_counts, cal_labels, _ = predict_counts(
                model, loader(train_set, splits.cal), device, count_synops=False)
            test_counts, test_labels, _ = predict_counts(
                model, loader(test_set, splits.test), device, count_synops=False)
            cal = model_outputs(cal_counts, cal_labels, np.random.default_rng(seed))
            test = model_outputs(test_counts, test_labels, np.random.default_rng(seed + 1))

            for arm_name, arm_cls in ARMS:
                fitted = arm_cls().fit(cal.conf, cal.correct, outputs=cal)
                conf = fitted.transform(test.conf, outputs=test)
                summary = selective_summary(conf, test.correct)
                computed[(str(timesteps), str(seed), arm_name)] = summary
                key = f"{timesteps}|{seed}|{arm_name}"
                cache[key + "|conf"] = conf.astype(np.float32)
                cache[key + "|correct"] = test.correct.astype(np.int8)
            print(f"  T={timesteps:<3} seed={seed}  4 arms  "
                  f"levels={summary['n_achievable_coverage_levels']:>4}  "
                  f"aurc={summary['aurc']:.5f}", flush=True)

    csv_path = os.path.join(args.results, f"sweep_{spec.key}.csv")
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())

    for row in rows:
        s = computed.get((row["timesteps"], row["seed"], row["arm"]))
        if s is None:
            continue
        for col in NEW_COLS:
            row[col] = "" if s.get(col) is None else s[col]
    for col in NEW_COLS:
        if col not in fields:
            fields.append(col)

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(os.path.join(args.results, "selective_cache.npz"), **cache)
    print(f"\n  {len(rows)} rows now carry {len(NEW_COLS)} selective columns")
    print(f"  cached per-sample confidence and correctness for {len(cache)//2} combinations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
