#!/usr/bin/env python3
"""The C axis: does the attainable-confidence ceiling hold at 101 classes?

Every other measurement in this study is at ``C = 10``. The ceiling ``e^T/(e^T + C - 1)``
has two variables and only ``T`` has been varied, which leaves the formula half tested.
N-Caltech101 has 101 classes, where the ``T = 2`` ceiling is **0.068806** rather than
0.450853 -- an order of magnitude more binding.

Two things are measured here.

**The ceiling itself.** Derived against observed maximum confidence, at ``T`` in
{2, 6, 10}, exactly as was done at ``C = 10``. The formula is arithmetic, so it should hold
to float precision. If it does not, the study's first claim is wrong and that is the most
important result in it -- so the comparison is reported with its absolute error rather than
as a pass/fail, and the script does not quietly succeed.

**Whether the ceiling binds.** A ceiling only forces underconfidence while accuracy exceeds
it. At ``C = 10`` on N-MNIST accuracy was 0.99 against a ``T = 2`` ceiling of 0.4509, so it
bound hard. Here accuracy will be far lower -- this is a 101-way problem with about 56
training samples per class -- and 0.0688 is a low bar, but whether accuracy clears it is a
measurement, not an assumption.

The count and accumulator readouts are run because they bracket the question: count is
capped, the accumulator is not, and the gap between them at ``C = 101`` is the same
quantity the readout experiment measured at ``C = 10``.

    python scripts/caltech_experiment.py
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
    softmax_confidence,
)
from spikecalib.data import (  # noqa: E402
    load_dataset, make_dataset_splits, materialise, save_splits, training_labels,
)
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model, readout_ceiling  # noqa: E402
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.train import (  # noqa: E402
    TrainConfig, predict_outputs, resolve_device, set_seed, train,
)

# [DESIGN] Four pooling blocks, not N-MNIST's two. 180x240 is 37 times the pixel count of
# 34x34; two blocks leave a 45x60x64 feature map and a 44M-parameter first Linear, which
# would put almost every weight in one fully-connected layer. Four blocks give 11x15x128,
# a 21,120-wide flatten and 5.7M parameters -- the same shape of model as N-MNIST's, scaled
# for the input. Not tuned.
CHANNELS = (32, 64, 128, 128)
HIDDEN = 256
READOUTS = ("count", "accumulator")


def paths(out_dir, key, timesteps, seed, readout):
    tag = "" if readout == "count" else f"_{readout}"
    stem = f"{key}_T{timesteps}_seed{seed}{tag}"
    return (os.path.join(out_dir, stem + "_weights.pt"),
            os.path.join(out_dir, stem + "_record.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ncaltech101")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 6, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    rows = []
    print("=" * 104)
    print(f"C-AXIS EXPERIMENT  {spec.display_name}  C={spec.n_classes}  "
          f"T={args.timesteps}  seeds={args.seeds}")
    print(f"  ceiling at T=2 is {readout_ceiling('count', 2, spec.n_classes):.6f} "
          f"against {readout_ceiling('count', 2, 10):.6f} at C=10")
    print("=" * 104, flush=True)

    for timesteps in args.timesteps:
        with contextlib.redirect_stdout(io.StringIO()):
            disk_set, _ = load_dataset(spec.key, timesteps, args.data_root)
        # Read once into RAM. Measured on this dataset, reading and pooling frames costs
        # about 27 ms/sample against roughly 5 ms/sample of computation, so training from
        # disk is 85% I/O and every epoch re-pays it for data that never changes. There is
        # one flat object here, so train and test are the same materialised set indexed by
        # different arrays.
        train_set = materialise(disk_set, num_workers=args.num_workers,
                                label=f"T={timesteps}")
        test_set = train_set
        labels = training_labels(train_set)

        for seed in args.seeds:
            splits = make_dataset_splits(labels, spec, seed=seed)
            # The split is the study's own, so it is saved rather than merely derived --
            # a result is only reproducible if the split that produced it is on disk.
            split_path = os.path.join(args.out, f"{spec.key}_T{timesteps}_seed{seed}_split")
            if not os.path.exists(split_path + ".npz"):
                save_splits(splits, split_path)

            # num_workers=0: the data is already in RAM, so a worker process would only
            # add pickling of a multi-hundred-megabyte tensor.
            def loader(ds, idx, shuffle=False):
                return DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                                  shuffle=shuffle, num_workers=0)

            for readout in args.readouts:
                t0 = time.time()
                cfg = ModelConfig(channels=CHANNELS, hidden=HIDDEN, readout=readout)
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
                        TrainConfig(epochs=args.epochs, lr=args.lr,
                                    batch_size=args.batch_size, patience=args.patience),
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
                sel = selective_summary(conf, correct)
                derived = readout_ceiling(readout, timesteps, spec.n_classes)
                observed = float(conf.max())

                row = {
                    "dataset": spec.key, "n_classes": spec.n_classes,
                    "readout": readout, "timesteps": timesteps, "seed": seed,
                    "status": trained,
                    "epochs_run": rec.get("epochs_run"),
                    "early_stopped": rec.get("early_stopped"),
                    "val_accuracy": rec.get("best_val_accuracy"),
                    "test_accuracy": acc,
                    "n_train": int(splits.train.size), "n_cal": int(splits.cal.size),
                    "n_test": int(splits.test.size),
                    "split_fingerprint": splits.fingerprint,
                    "ceiling_derived": derived,
                    "max_confidence_observed": observed,
                    "ceiling_abs_error": None if derived is None else abs(derived - observed),
                    "ceiling_binds": None if derived is None else bool(acc > derived),
                    "mean_confidence": float(conf.mean()),
                    "mean_conf_minus_acc": float(conf.mean()) - acc,
                    "ece_equal_width": m.ece_equal_width,
                    "ece_equal_mass": m.ece_equal_mass,
                    "n_distinct": int(np.unique(quantise_confidence(conf)).size),
                    "auroc": d["auroc"],
                    "aurc": sel["aurc"], "aurc_optimal": sel["aurc_optimal"],
                    "n_achievable_coverage_levels": sel["n_achievable_coverage_levels"],
                    "seconds": time.time() - t0,
                }
                rows.append(row)
                cap = f"{derived:.6f}" if derived is not None else "none"
                err = "n/a" if derived is None else f"{row['ceiling_abs_error']:.2e}"
                print(f"  T={timesteps:<3} {readout:<12} {trained:<8} acc {acc:.4f}  "
                      f"cap {cap:<9} max {observed:.6f}  |err| {err:<9} "
                      f"binds {str(row['ceiling_binds']):<5} "
                      f"conf-acc {row['mean_conf_minus_acc']:+.4f}  "
                      f"ECEew {m.ece_equal_width:.4f}  AUROC {d['auroc']:.4f}", flush=True)

                # Merge-on-write: a partial run must never replace a complete one.
                path = os.path.join(args.out, "caltech_experiment.json")
                keyed = {}
                if os.path.exists(path):
                    for r in json.load(open(path))["rows"]:
                        keyed[(r["readout"], r["timesteps"], r["seed"])] = r
                for r in rows:
                    keyed[(r["readout"], r["timesteps"], r["seed"])] = r
                merged = [keyed[k] for k in sorted(keyed, key=lambda k: (k[1], k[2], k[0]))]
                with open(path, "w") as fh:
                    json.dump({"dataset": spec.key, "n_classes": spec.n_classes,
                               "rows": merged}, fh, indent=2)

    print(f"\n  wrote results/caltech_experiment.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
