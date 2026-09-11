#!/usr/bin/env python3
"""Is the membrane readout's ranking deficit at high T explained by its fine resolution?

From ``T = 6`` upward the membrane readout has the better ECE and the *worse* AUROC and
AURC. The hypothesis under test: a spike count is a thresholded membrane signal, the
thresholding discards noise, and once the count has enough levels to express the ordering
the discarding is pure gain -- so coarsening the membrane confidence should recover the
lost ranking quality, peaking near count's own level count and falling again once the
binning is too coarse to express the ordering at all.

No training. The saved membrane weights are run over the test split once per ``(T, seed)``
and the resulting confidences are cached; every resolution is then evaluated on those
cached confidences. Count confidences are cached alongside, because the sharpest form of
the question is not "does AUROC rise" but "does membrane, coarsened to count's own number
of levels, rank as well as count does".

The cache is written after **every** cell, not at the end. A multi-worker ``DataLoader``
deadlocked partway through the first full run and took forty completed inference passes
with it; re-running now costs only the cells that are actually missing.

See :mod:`spikecalib.quantisation` for the identity that makes the outcome readable:
AUROC can only rise under coarsening if the structure being merged away was worse than
chance at ordering correct from incorrect.

    python scripts/quantisation_test.py
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

from spikecalib.calibration import (  # noqa: E402
    predict_with_ties, quantise_confidence, softmax_confidence,
)
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.quantisation import LEVEL_GRID, SCHEMES, quantisation_step  # noqa: E402
from spikecalib.selective import aurc  # noqa: E402
from spikecalib.train import predict_outputs, resolve_device  # noqa: E402

READOUTS = ("count", "membrane")


def weights_path(out_dir, key, timesteps, seed, readout):
    tag = "" if readout == "count" else f"_{readout}"
    return os.path.join(out_dir, f"{key}_T{timesteps}_seed{seed}{tag}_weights.pt")


def build_cache(args, spec, device, cache_path):
    """Per-sample confidence and correctness for every (T, seed, readout), cached to npz.

    The tie-breaking RNG is seeded exactly as ``scripts/readout_experiment.py`` seeds it,
    so the full-precision numbers recomputed here must reproduce the ones already recorded
    in ``results/readout_experiment.json``. The caller checks that they do.
    """
    store = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    wanted = [(t, s, r) for t in args.timesteps for s in args.seeds for r in args.readouts]
    missing = [k for k in wanted if f"{k[0]}|{k[1]}|{k[2]}|conf" not in store]
    if not missing:
        print(f"  cache complete: {len(wanted)} (T, seed, readout) cells")
        return store

    def flush():
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **store)

    print(f"  cache miss on {len(missing)} of {len(wanted)} cells; running inference")
    for timesteps in sorted({k[0] for k in missing}):
        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)
        for seed in sorted({k[1] for k in missing if k[0] == timesteps}):
            splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))
            for readout in [k[2] for k in missing if k[0] == timesteps and k[1] == seed]:
                t0 = time.time()
                wpath = weights_path(args.models, spec.key, timesteps, seed, readout)
                if not os.path.exists(wpath):
                    raise SystemExit(
                        f"missing weights {wpath}\n"
                        f"  build them with: python scripts/readout_experiment.py --seeds 0 1 2"
                    )
                cfg = ModelConfig(channels=(32, 64), hidden=256, readout=readout)
                model = build_model(spec, timesteps, cfg)
                model.load_state_dict(torch.load(wpath, map_location="cpu"))
                # A fresh loader per pass, with persistent workers OFF and the iterator
                # exhausted before the loader goes out of scope. Keeping workers alive
                # across a loader that is iterated exactly once buys nothing and leaves
                # six processes per cell waiting to be reaped; thirty such cells deadlocked
                # this script once already.
                loader = DataLoader(Subset(test_set, splits.test.tolist()),
                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                    persistent_workers=False)
                scores, y, _ = predict_outputs(model, loader, device, count_synops=False)
                del loader
                conf = softmax_confidence(scores)
                pred, _ = predict_with_ties(scores, np.random.default_rng(seed + 1))
                store[f"{timesteps}|{seed}|{readout}|conf"] = conf.astype(np.float64)
                store[f"{timesteps}|{seed}|{readout}|correct"] = (pred == y).astype(np.int8)
                flush()
                print(f"    T={timesteps:<3} seed {seed}  {readout:<9} "
                      f"acc {float((pred == y).mean()):.4f}  {time.time() - t0:5.1f}s", flush=True)

    print(f"  wrote {os.path.relpath(cache_path, REPO_ROOT)}  ({len(store) // 2} cells)")
    return store


def recorded_auroc(path, readout, timesteps, seed):
    """The AUROC already recorded for this cell, or None if the file is absent."""
    if not os.path.exists(path):
        return None
    for row in json.load(open(path))["rows"]:
        if (row["readout"], row["timesteps"], row["seed"]) == (readout, timesteps, seed):
            return row["auroc"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--models", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    cache_path = os.path.join(args.out, "quantisation_cache.npz")
    experiment = os.path.join(args.out, "readout_experiment.json")

    print("=" * 100)
    print(f"QUANTISATION TEST  {spec.display_name}  T={args.timesteps}  seeds={args.seeds}")
    print("  does coarsening the membrane confidence recover the ranking count gets for free?")
    print("=" * 100, flush=True)

    store = build_cache(args, spec, device, cache_path)

    rows, checks = [], []
    for timesteps in args.timesteps:
        print(f"\n  T = {timesteps}")
        for seed in args.seeds:
            conf = store[f"{timesteps}|{seed}|membrane|conf"]
            correct = store[f"{timesteps}|{seed}|membrane|correct"].astype(bool)

            c_conf = store[f"{timesteps}|{seed}|count|conf"]
            c_correct = store[f"{timesteps}|{seed}|count|correct"].astype(bool)
            n_count = int(np.unique(quantise_confidence(c_conf)).size)
            count_ref = {
                "n_distinct": n_count,
                "auroc": quantisation_step(c_conf, c_correct, None)["auroc"],
                "aurc": aurc(c_conf, c_correct),
            }

            # Count's own level count is the resolution the hypothesis predicts a peak at,
            # so it is evaluated explicitly rather than interpolated off the grid.
            grid = list(LEVEL_GRID)
            if n_count not in grid:
                grid = sorted([n for n in grid if n is not None] + [n_count]) + [None]

            for scheme in SCHEMES:
                for n_levels in grid:
                    step = quantisation_step(conf, correct, n_levels, scheme)
                    step.update(timesteps=timesteps, seed=seed, readout="membrane",
                                at_count_levels=(n_levels == n_count),
                                count_n_distinct=count_ref["n_distinct"],
                                count_auroc=count_ref["auroc"], count_aurc=count_ref["aurc"])
                    rows.append(step)
                    if n_levels is None and scheme == SCHEMES[0]:
                        was = recorded_auroc(experiment, "membrane", timesteps, seed)
                        if was is not None:
                            checks.append((timesteps, seed, was, step["auroc"]))

            full = next(r for r in rows if r["seed"] == seed and r["timesteps"] == timesteps
                        and r["n_levels_requested"] is None and r["scheme"] == "equal-mass")
            best = max((r for r in rows if r["seed"] == seed and r["timesteps"] == timesteps
                        and r["scheme"] == "equal-mass" and r["n_levels_requested"] is not None),
                       key=lambda r: r["auroc"])
            at_count = next(r for r in rows if r["seed"] == seed and r["timesteps"] == timesteps
                            and r["scheme"] == "equal-mass" and r["at_count_levels"])
            print(f"    seed {seed}   membrane full {full['auroc']:.4f}   "
                  f"best-quantised {best['auroc']:.4f} at N={best['n_levels_requested']:<5} "
                  f"at-count-N({n_count}) {at_count['auroc']:.4f}   "
                  f"count {count_ref['auroc']:.4f}", flush=True)

    for timesteps, seed, was, now in checks:
        if abs(was - now) > 1e-9:
            print(f"  MISMATCH T={timesteps} seed {seed}: recorded AUROC {was!r} != {now!r}",
                  file=sys.stderr)
            return 1
    print(f"\n  {len(checks)} full-precision AUROCs reproduce readout_experiment.json exactly")

    path = os.path.join(args.out, "quantisation_test.json")
    with open(path, "w") as fh:
        json.dump({"dataset": spec.key, "n_classes": spec.n_classes,
                   "levels": [n for n in LEVEL_GRID], "schemes": list(SCHEMES),
                   "rows": rows}, fh, indent=2)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
