#!/usr/bin/env python3
"""Is the attainable-confidence ceiling a property of SNNs, or of the readout?

Trains the SAME architecture with three different final aggregations and compares them.
Everything up to and including the last ``Linear -> BatchNorm1d`` is identical; only the
readout differs, because changing anything else would confound the comparison.

  count            sum of output spikes over T. Integer, capped at T.
  weighted-count   sum of w_t * s_t with weights summing to T. Real-valued, SAME cap,
                   far finer grained -- this separates "capped" from "coarse".
  membrane         final potential of a non-spiking LIF integrator. Real-valued, uncapped,
                   and leaky: a one-step half-life, so it reads mostly the last two steps.
  accumulator      the same integrator with the leak removed. Uncapped, and weights every
                   step equally as count does -- so it differs from count only by the
                   threshold, and from membrane only by the leak.

The prediction was recorded in NOTES.md on 2026-09-11, before this script was run.

Count models are reused from the main sweep rather than retrained; the others are trained
here with identical settings and cached, so a re-run does not repeat the work.
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
    compute_metrics, discrimination, predict_with_ties, quantise_confidence,
    softmax_confidence, tie_rate,
)
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model, readout_ceiling  # noqa: E402
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.train import (  # noqa: E402
    TrainConfig, predict_outputs, resolve_device, set_seed, train,
)

READOUTS = ("count", "weighted-count", "membrane", "accumulator")


def paths(out_dir, key, timesteps, seed, readout):
    tag = "" if readout == "count" else f"_{readout}"
    stem = f"{key}_T{timesteps}_seed{seed}{tag}"
    return os.path.join(out_dir, stem + "_weights.pt"), os.path.join(out_dir, stem + "_record.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    rows = []
    print("=" * 96)
    print(f"READOUT EXPERIMENT  {spec.display_name}  T={args.timesteps}  seeds={args.seeds}")
    print("  identical architecture; only the final aggregation differs")
    print("=" * 96, flush=True)

    for timesteps in args.timesteps:
        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)

        for seed in args.seeds:
            splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))

            def loader(ds, idx, shuffle=False):
                return DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                                  shuffle=shuffle, num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0)

            for readout in args.readouts:
                t0 = time.time()
                cfg = ModelConfig(channels=(32, 64), hidden=256, readout=readout)
                model = build_model(spec, timesteps, cfg)
                wpath, rpath = paths(args.out, spec.key, timesteps, seed, readout)

                if os.path.exists(wpath) and os.path.exists(rpath):
                    model.load_state_dict(torch.load(wpath, map_location="cpu"))
                    rec = json.load(open(rpath))
                    trained = "reused"
                else:
                    set_seed(seed)
                    state, record = train(
                        model,
                        {"train": loader(train_set, splits.train, shuffle=True),
                         "val": loader(train_set, splits.val)},
                        TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                    patience=args.patience),
                        seed=seed, dataset=spec.key, timesteps=timesteps,
                        split_fingerprint=splits.fingerprint,
                    )
                    model.load_state_dict(state)
                    torch.save(state, wpath)
                    rec = record.as_dict()
                    rec["model_config"] = {"channels": list(cfg.channels), "hidden": cfg.hidden,
                                           "tau": cfg.tau, "v_threshold": cfg.v_threshold,
                                           "surrogate_alpha": cfg.surrogate_alpha,
                                           "readout": readout}
                    with open(rpath, "w") as fh:
                        json.dump(rec, fh, indent=2)
                    trained = "trained"

                scores, y, _ = predict_outputs(model, loader(test_set, splits.test), device,
                                               count_synops=False)
                conf = softmax_confidence(scores)
                pred, _ = predict_with_ties(scores, np.random.default_rng(seed + 1))
                correct = pred == y
                acc = float(correct.mean())
                m = compute_metrics(conf, correct, n_bins=args.bins, n_bootstrap=0)
                d = discrimination(conf, correct)
                derived = readout_ceiling(readout, timesteps, spec.n_classes)

                # Selective prediction per readout: this is what decides the
                # recommendation. Calibrating better is worth less than ranking better,
                # and the number of achievable operating points is a second, independent
                # deployment consequence of the same readout choice.
                sel = selective_summary(conf, correct)

                row = {
                    "readout": readout, "timesteps": timesteps, "seed": seed,
                    "status": trained,
                    "epochs_run": rec.get("epochs_run"), "early_stopped": rec.get("early_stopped"),
                    "val_accuracy": rec.get("best_val_accuracy"),
                    "test_accuracy": acc,
                    "ceiling_derived": derived,
                    "max_confidence_observed": float(conf.max()),
                    "mean_confidence": float(conf.mean()),
                    "mean_conf_minus_acc": float(conf.mean()) - acc,
                    "ece_equal_width": m.ece_equal_width,
                    "ece_equal_mass": m.ece_equal_mass,
                    "n_distinct": int(np.unique(quantise_confidence(conf)).size),
                    "auroc": d["auroc"],
                    "tie_rate": tie_rate(scores),
                    "aurc": sel["aurc"],
                    "aurc_optimal": sel["aurc_optimal"],
                    "aurc_excess_normalised": sel["aurc_excess_normalised"],
                    "n_achievable_coverage_levels": sel["n_achievable_coverage_levels"],
                    "coverage_at_risk_0p01": sel.get("coverage_at_risk_0p01"),
                    "coverage_at_risk_0p005": sel.get("coverage_at_risk_0p005"),
                    "coverage_at_risk_0p001": sel.get("coverage_at_risk_0p001"),
                    "seconds": time.time() - t0,
                }
                rows.append(row)
                cap = f"{derived:.4f}" if derived is not None else "none"
                print(f"  T={timesteps:<3} {readout:<15} {trained:<8} acc {acc:.4f}  "
                      f"cap {cap:<7} max {row['max_confidence_observed']:.4f}  "
                      f"conf-acc {row['mean_conf_minus_acc']:+.4f}  "
                      f"ECEew {m.ece_equal_width:.4f}  levels {row['n_distinct']:>5}  "
                      f"AUROC {d['auroc']:.4f}  AURC {sel['aurc']:.5f}  "
                      f"ops {sel['n_achievable_coverage_levels']:>5}", flush=True)

    # Merge rather than overwrite. Running a subset -- one readout, three of the five T --
    # used to replace the whole file with just those rows, which would have silently
    # discarded the 45-row comparison the moment anyone added a fourth readout.
    path = os.path.join(args.out, "readout_experiment.json")
    keyed = {}
    if os.path.exists(path):
        for row in json.load(open(path))["rows"]:
            keyed[(row["readout"], row["timesteps"], row["seed"])] = row
    kept = len(keyed)
    for row in rows:
        keyed[(row["readout"], row["timesteps"], row["seed"])] = row
    merged = [keyed[k] for k in sorted(keyed, key=lambda k: (k[1], k[2], k[0]))]
    with open(path, "w") as fh:
        json.dump({"dataset": spec.key, "n_classes": spec.n_classes, "rows": merged},
                  fh, indent=2)
    print(f"\n  wrote {os.path.relpath(path, REPO_ROOT)}  "
          f"({len(merged)} rows: {len(rows)} from this run, {len(merged) - len(rows)} kept "
          f"from {kept} already on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
