#!/usr/bin/env python3
"""Export everything the dashboard needs into a single loadable JS file.

Two problems this solves.

**Reliability diagrams need per-run detail the sweep CSV does not carry.** The CSV records
aggregate ECE; a reliability diagram needs, per bin, the sample count, the mean confidence
and the observed accuracy. Those are recomputed here from the saved weights, once, and
cached.

**The page must not carry numbers of its own.** Every figure the page states in prose or
a table is rendered from this payload at runtime, so a result that changes in ``results/``
changes on the page the next time this script runs. Hardcoding a number into
``index.html`` is how the page went stale before: it kept stating a membrane advantage
that three seeds had already qualified.

**``fetch()`` is blocked under ``file://``.** A page that loaded the CSV over fetch would
work when served and silently fail when opened from disk. So the CSV text is embedded
verbatim in a ``<script>``-loadable file and parsed by the page at runtime. The page still
computes every number from the CSV rather than carrying hardcoded values -- the CSV is
just delivered by a mechanism that works in both contexts.

Writes ``docs/data.js``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    _equal_mass_bins,
    model_outputs,
    quantise_confidence,
)
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.sweep import ARMS  # noqa: E402
from spikecalib.train import predict_counts, resolve_device  # noqa: E402


def reliability(conf, correct, n_bins, scheme):
    """Per-bin count, mean confidence and observed accuracy.

    Bin counts travel with the curve deliberately: a reliability diagram whose points are
    drawn at uniform size hides the fact that a point at the far left may rest on three
    samples. The dashboard scales each marker by its count and prints it on hover.
    """
    conf = quantise_confidence(np.asarray(conf, dtype=np.float64))
    correct = np.asarray(correct).astype(np.float64)
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        n_eff = n_bins
        lo_hi = [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]
    else:
        idx, n_eff = _equal_mass_bins(conf, n_bins)
        if idx is None:  # degenerate: one attainable confidence
            idx = np.zeros(conf.size, dtype=np.int64)
            n_eff = 1
        lo_hi = None

    bins = []
    for b in range(n_eff):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        entry = {
            "n": n,
            "conf": float(conf[m].mean()),
            "acc": float(correct[m].mean()),
            "lo": float(conf[m].min()),
            "hi": float(conf[m].max()),
        }
        if lo_hi is not None:
            entry["lo"], entry["hi"] = lo_hi[b]
        bins.append(entry)
    return bins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "data.js"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    rel: dict = {}

    for timesteps in args.timesteps:
        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)

        for seed in args.seeds:
            stem = f"{spec.key}_T{timesteps}_seed{seed}"
            with open(os.path.join(args.results, f"{stem}_record.json")) as fh:
                record = json.load(fh)
            mc = record["model_config"]
            cfg = ModelConfig(channels=tuple(mc["channels"]), hidden=mc["hidden"],
                              tau=mc["tau"], v_threshold=mc["v_threshold"],
                              surrogate_alpha=mc["surrogate_alpha"])
            splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))
            model = build_model(spec, timesteps, cfg)
            model.load_state_dict(
                torch.load(os.path.join(args.results, f"{stem}_weights.pt"), map_location="cpu")
            )

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
                calibrator = arm_cls().fit(cal.conf, cal.correct, outputs=cal)
                calibrated = calibrator.transform(test.conf, outputs=test)
                key = f"{timesteps}|{seed}|{arm_name}"
                rel[key] = {
                    "equal_width": reliability(calibrated, test.correct, args.bins, "equal_width"),
                    "equal_mass": reliability(calibrated, test.correct, args.bins, "equal_mass"),
                }
            print(f"  T={timesteps:<3} seed={seed}  4 arms binned", flush=True)

    with open(os.path.join(args.results, f"sweep_{spec.key}.csv")) as fh:
        csv_text = fh.read()
    with open(os.path.join(args.results, "readout_stats.json")) as fh:
        stats = json.load(fh)[spec.key]

    # The result artifacts the page renders tables from. Each is optional so a checkout
    # that has not run every experiment still produces a working page; the page hides the
    # corresponding section rather than showing an empty table.
    def optional(name, key="rows"):
        path = os.path.join(args.results, name)
        if not os.path.exists(path):
            print(f"  {name} absent -- the page will hide that section")
            return None
        with open(path) as fh:
            return json.load(fh)[key]

    readout_rows = optional("readout_experiment.json")
    quantisation_rows = optional("quantisation_test.json")

    payload = {
        "dataset": spec.key,
        "display_name": spec.display_name,
        "n_classes": spec.n_classes,
        "n_bins": args.bins,
        "readout": readout_rows,
        "quantisation": quantisation_rows,
        "measured": [
            {
                "T": int(t),
                "ceiling": s["ceiling_derived"],
                "observed_max": s["max_confidence_observed"],
                "accuracy": s["accuracy"],
                "train_loss_final": s["train_loss_final"],
            }
            for t, s in sorted(stats.items(), key=lambda kv: int(kv[0]))
        ],
        "reliability": rel,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("// Generated by scripts/export_dashboard_data.py -- do not edit by hand.\n")
        fh.write("// The CSV text below is verbatim results/sweep_%s.csv; the page parses it\n" % spec.key)
        fh.write("// at runtime so the page and the data cannot drift apart.\n")
        fh.write("window.SPIKECALIB_CSV = " + json.dumps(csv_text) + ";\n")
        fh.write("window.SPIKECALIB = " + json.dumps(payload) + ";\n")
    size = os.path.getsize(args.out) / 1024
    print(f"\n  wrote {os.path.relpath(args.out, REPO_ROOT)}  ({size:,.0f} KiB, "
          f"{len(rel)} reliability curves, {len(readout_rows or [])} readout rows, "
          f"{len(quantisation_rows or [])} quantisation rows)")

    # Two figures are drawn by matplotlib, not in the browser. They live in figures/,
    # which is outside the published site root, so they are copied in.
    figs = os.path.join(os.path.dirname(args.out), "figures")
    os.makedirs(figs, exist_ok=True)
    copied = []
    for name in ("fig7-risk-coverage", "fig8-quantisation"):
        for ext in ("svg", "png"):
            src = os.path.join(REPO_ROOT, "figures", f"{name}.{ext}")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(figs, f"{name}.{ext}"))
                copied.append(f"{name}.{ext}")
    print(f"  copied {len(copied)} figure files into {os.path.relpath(figs, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
