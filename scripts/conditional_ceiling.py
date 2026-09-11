#!/usr/bin/env python3
"""Does a conditional ceiling explain the C = 101 residual?

At ``C = 101, T = 6`` the theoretical ceiling is 0.8014, accuracy is 0.6427 so the ceiling
is slack, and the readout is still underconfident by 0.1534. The theoretical bound assumes
all 100 competitors silent for all T steps. This measures what the competitors actually do
and recomputes the bound conditional on that.

No training. Count-readout models are loaded from disk and run over the test split once per
``(dataset, T, seed)``; the full count VECTORS are cached, because the competitor
distribution is not recoverable from the scalar confidences already stored.

The prediction this tests was recorded in NOTES.md and committed before any count was
measured.

    python scripts/conditional_ceiling.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import predict_with_ties, softmax_confidence  # noqa: E402
from spikecalib.conditional import (  # noqa: E402
    competitor_profile, conditional_ceiling, residual_share_explained, theoretical_ceiling,
)
from spikecalib.data import (  # noqa: E402
    load_dataset, make_dataset_splits, materialise, training_labels,
)
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.train import predict_outputs, resolve_device  # noqa: E402

# (dataset key, timesteps) -- the count-readout models already on disk. The architecture is
# NOT listed here: it is read from each run's own record.json. Hardcoding it is what caused
# this script to fail on its first launch, against models whose channels had been changed in
# a different file, and a hardcoded value that merely happened to match would have been
# worse -- it would have loaded silently and measured the wrong thing.
GRID = [
    ("ncaltech101", [2, 6, 10]),
    ("nmnist", [2, 4, 6, 8, 10]),
]


def model_for(spec, timesteps, seed, results):
    """Rebuild a trained model from its own record, so the architecture cannot be guessed."""
    stem = f"{spec.key}_T{timesteps}_seed{seed}"
    rpath = os.path.join(results, stem + "_record.json")
    wpath = os.path.join(results, stem + "_weights.pt")
    if not (os.path.exists(rpath) and os.path.exists(wpath)):
        return None
    mc = json.load(open(rpath))["model_config"]
    cfg = ModelConfig(channels=tuple(mc["channels"]), hidden=mc["hidden"],
                      tau=mc["tau"], v_threshold=mc["v_threshold"],
                      surrogate_alpha=mc["surrogate_alpha"],
                      readout=mc.get("readout", "count"))
    model = build_model(spec, timesteps, cfg)
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    return model


def cache_counts(args, key, timesteps_list, store, device):
    """Cache raw count vectors for every (T, seed) of one dataset's count readout."""
    spec = get_spec(key)
    for timesteps in timesteps_list:
        needed = [s for s in args.seeds
                  if f"{key}|{timesteps}|{s}|counts" not in store]
        if not needed:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            train_obj, test_obj = load_dataset(key, timesteps, args.data_root)
        if spec.defines_own_test_split:
            train_obj = materialise(train_obj, num_workers=args.num_workers)
            test_obj = train_obj
        labels = training_labels(train_obj)
        for seed in needed:
            t0 = time.time()
            splits = make_dataset_splits(labels, spec, seed=seed,
                                         n_test=None if spec.defines_own_test_split else None)
            model = model_for(spec, timesteps, seed, args.results)
            if model is None:
                print(f"    missing {key} T={timesteps} seed {seed}; skipping")
                continue
            nw = 0 if spec.defines_own_test_split else args.num_workers
            loader = DataLoader(Subset(test_obj, splits.test.tolist()),
                                batch_size=args.batch_size, num_workers=nw)
            counts, y, _ = predict_outputs(model, loader, device, count_synops=False)
            store[f"{key}|{timesteps}|{seed}|counts"] = counts.astype(np.float32)
            store[f"{key}|{timesteps}|{seed}|labels"] = y.astype(np.int64)
            np.savez_compressed(args.cache, **store)
            print(f"    {key} T={timesteps} seed {seed}: {counts.shape} in "
                  f"{time.time() - t0:.0f}s", flush=True)
    return store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "results",
                                                    "count_vectors_cache.npz"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results",
                                                  "conditional_ceiling.json"))
    args = ap.parse_args()

    device = resolve_device()
    store = dict(np.load(args.cache)) if os.path.exists(args.cache) else {}

    print("=" * 110)
    print("CONDITIONAL CEILING  e^T / (e^T + sum_j e^{c_j})  vs the theoretical all-silent bound")
    print("=" * 110)
    print("Caching count vectors (the competitor distribution is not in the stored scalars)")
    for key, ts in GRID:
        store = cache_counts(args, key, ts, store, device)

    rows = []
    for key, ts in GRID:
        spec = get_spec(key)
        for timesteps in ts:
            for seed in args.seeds:
                ck = f"{key}|{timesteps}|{seed}|counts"
                if ck not in store:
                    continue
                counts = store[ck].astype(np.float64)
                y = store[f"{key}|{timesteps}|{seed}|labels"]
                conf = softmax_confidence(counts)
                pred, _ = predict_with_ties(counts, np.random.default_rng(seed + 1))
                correct = pred == y
                acc = float(correct.mean())
                mean_conf = float(conf.mean())
                cond = conditional_ceiling(counts, timesteps)
                theo = theoretical_ceiling(timesteps, spec.n_classes)
                prof = competitor_profile(counts, spec.n_classes)
                rows.append({
                    "dataset": key, "n_classes": spec.n_classes,
                    "timesteps": timesteps, "seed": seed,
                    "test_accuracy": acc, "mean_confidence": mean_conf,
                    "observed_max_confidence": float(conf.max()),
                    "residual": acc - mean_conf,
                    "theoretical_ceiling": theo,
                    "conditional_ceiling_mean": float(cond.mean()),
                    "conditional_ceiling_median": float(np.median(cond)),
                    "conditional_ceiling_p05": float(np.quantile(cond, 0.05)),
                    "conditional_ceiling_p95": float(np.quantile(cond, 0.95)),
                    "ceiling_gap_theo_minus_cond": theo - float(cond.mean()),
                    "share_explained_conditional":
                        residual_share_explained(acc, mean_conf, float(cond.mean())),
                    "share_explained_theoretical":
                        residual_share_explained(acc, mean_conf, theo),
                    "competitors": prof,
                })

    with open(args.out, "w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, REPO_ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
