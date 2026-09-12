#!/usr/bin/env python3
"""The matched-architecture 2x2: remove the depth confound between the two datasets.

N-Caltech101 was run with four conv blocks and N-MNIST with two, so every cross-dataset
comparison in the study is confounded with depth. This runs **both architectures on both
datasets**, ``T`` in {2, 6}, seeds 0-2, count and accumulator in every cell, so the readout
comparison is matched as well as the dataset comparison.

Two of the four quadrants already exist and are reused rather than retrained; the weights
are found under the names the earlier sessions gave them. The 24 new cells are trained here.

It doubles as the alternative-architecture check the study never had: any claim that holds
at both depths is not an artifact of one network.

Per-run training settings are taken from whatever the dataset's existing runs used
(``batch_size`` 128 for N-MNIST, 32 for N-Caltech101), so a new cell differs from an
existing one in depth and nothing else.

The prediction this tests was recorded in NOTES.md and committed before any matched model
was trained.

    python scripts/arch_matched.py
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
from spikecalib.conditional import (  # noqa: E402
    competitor_profile, conditional_ceiling, residual_share_explained,
)
from spikecalib.data import (  # noqa: E402
    load_dataset, make_dataset_splits, materialise, training_labels,
)
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model, readout_ceiling  # noqa: E402
from spikecalib.train import (  # noqa: E402
    TrainConfig, predict_outputs, resolve_device, set_seed, train,
)

TWO = (32, 64)
FOUR = (32, 64, 128, 128)

# The architecture each dataset's EXISTING runs used. A cell at these channels is looked up
# under the historical filename so the 24 runs already on disk are reused, not retrained.
HISTORICAL = {"nmnist": TWO, "ncaltech101": FOUR}
# Batch size each dataset's existing runs used, so a new cell differs only in depth.
BATCH = {"nmnist": 128, "ncaltech101": 32}
READOUTS = ("count", "accumulator")


def arch_tag(channels) -> str:
    return "arch" + "-".join(str(c) for c in channels)


def stem(key, timesteps, seed, readout, channels) -> str:
    """Filename stem, matching the historical naming where the architecture matches."""
    rt = "" if readout == "count" else f"_{readout}"
    at = "" if tuple(channels) == HISTORICAL[key] else f"_{arch_tag(channels)}"
    return f"{key}_T{timesteps}_seed{seed}{rt}{at}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ncaltech101", "nmnist"])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    device = resolve_device()
    path = os.path.join(args.out, "arch_matched.json")

    print("=" * 116)
    print("MATCHED-ARCHITECTURE 2x2  -- both depths on both datasets, both readouts")
    print("=" * 116, flush=True)

    for key in args.datasets:
        spec = get_spec(key)
        for timesteps in args.timesteps:
            with contextlib.redirect_stdout(io.StringIO()):
                train_disk, test_disk = load_dataset(key, timesteps, args.data_root)
            # Read once into RAM: measured 85% of training time was disk on N-Caltech101,
            # and N-MNIST frames round-trip through float16 exactly (max value 15).
            train_set = materialise(train_disk, num_workers=args.num_workers,
                                    label=f"{key} T={timesteps} train")
            test_set = (train_set if spec.defines_own_test_split else
                        materialise(test_disk, num_workers=args.num_workers,
                                    label=f"{key} T={timesteps} test"))
            labels = training_labels(train_set)

            for channels in (TWO, FOUR):
                for seed in args.seeds:
                    splits = make_dataset_splits(
                        labels, spec, seed=seed,
                        n_test=None if spec.defines_own_test_split else len(test_set))

                    def loader(ds, idx, shuffle=False):
                        return DataLoader(Subset(ds, idx.tolist()),
                                          batch_size=BATCH[key], shuffle=shuffle,
                                          num_workers=0)

                    for readout in READOUTS:
                        t0 = time.time()
                        st = stem(key, timesteps, seed, readout, channels)
                        wpath = os.path.join(args.out, st + "_weights.pt")
                        rpath = os.path.join(args.out, st + "_record.json")
                        cfg = ModelConfig(channels=channels, hidden=256, readout=readout)
                        model = build_model(spec, timesteps, cfg)

                        if os.path.exists(wpath) and os.path.exists(rpath):
                            model.load_state_dict(torch.load(wpath, map_location="cpu"))
                            rec = json.load(open(rpath))
                            status = "reused"
                        else:
                            set_seed(seed)
                            state, record = train(
                                model,
                                {"train": loader(train_set, splits.train, shuffle=True),
                                 "val": loader(train_set, splits.val)},
                                TrainConfig(epochs=args.epochs, lr=args.lr,
                                            batch_size=BATCH[key], patience=args.patience),
                                seed=seed, dataset=key, timesteps=timesteps,
                                split_fingerprint=splits.fingerprint,
                            )
                            model.load_state_dict(state)
                            torch.save(state, wpath)
                            rec = record.as_dict()
                            rec["model_config"] = {
                                "channels": list(channels), "hidden": cfg.hidden,
                                "tau": cfg.tau, "v_threshold": cfg.v_threshold,
                                "surrogate_alpha": cfg.surrogate_alpha, "readout": readout}
                            with open(rpath, "w") as fh:
                                json.dump(rec, fh, indent=2)
                            status = "trained"

                        counts, y, _ = predict_outputs(
                            model, loader(test_set, splits.test), device, count_synops=False)
                        conf = softmax_confidence(counts)
                        pred, _ = predict_with_ties(counts, np.random.default_rng(seed + 1))
                        correct = pred == y
                        acc = float(correct.mean())
                        mean_conf = float(conf.mean())
                        m = compute_metrics(conf, correct, n_bins=args.bins, n_bootstrap=0)
                        d = discrimination(conf, correct)
                        derived = readout_ceiling(readout, timesteps, spec.n_classes)
                        cond = conditional_ceiling(counts, timesteps)
                        cc = float(cond.mean())

                        row = {
                            "dataset": key, "n_classes": spec.n_classes,
                            "channels": list(channels), "n_blocks": len(channels),
                            "n_parameters": int(sum(p.numel() for p in model.parameters())),
                            "readout": readout, "timesteps": timesteps, "seed": seed,
                            "status": status,
                            "epochs_run": rec.get("epochs_run"),
                            "early_stopped": rec.get("early_stopped"),
                            "test_accuracy": acc, "mean_confidence": mean_conf,
                            "residual": acc - mean_conf,
                            "max_confidence_observed": float(conf.max()),
                            "ceiling_derived": derived,
                            "ceiling_signed_excess":
                                None if derived is None else float(conf.max()) - derived,
                            "ceiling_binds": None if derived is None else bool(acc > derived),
                            "mean_conf_minus_acc": mean_conf - acc,
                            "ece_equal_width": m.ece_equal_width,
                            "ece_equal_mass": m.ece_equal_mass,
                            "n_distinct": int(np.unique(quantise_confidence(conf)).size),
                            "auroc": d["auroc"],
                            "conditional_ceiling_mean": cc,
                            "share_explained_conditional":
                                residual_share_explained(acc, mean_conf, cc),
                            "share_explained_theoretical":
                                (0.0 if derived is None else
                                 residual_share_explained(acc, mean_conf, derived)),
                            "competitors": competitor_profile(counts, spec.n_classes),
                            "seconds": time.time() - t0,
                        }

                        keyed = {}
                        if os.path.exists(path):
                            for r in json.load(open(path))["rows"]:
                                keyed[(r["dataset"], tuple(r["channels"]), r["readout"],
                                       r["timesteps"], r["seed"])] = r
                        keyed[(key, tuple(channels), readout, timesteps, seed)] = row
                        merged = [keyed[k] for k in sorted(keyed, key=lambda k: (
                            k[0], len(k[1]), k[3], k[4], k[2]))]
                        with open(path, "w") as fh:
                            json.dump({"rows": merged}, fh, indent=2)

                        cap = f"{derived:.6f}" if derived is not None else "none"
                        print(f"  {key:<12} {len(channels)}blk T={timesteps:<3} s{seed} "
                              f"{readout:<12} {status:<8} acc {acc:.4f}  cap {cap:<9} "
                              f"max {float(conf.max()):.6f}  conf-acc "
                              f"{row['mean_conf_minus_acc']:+.4f}  ECE {m.ece_equal_width:.4f}"
                              f"  condceil {cc:.4f}  {time.time()-t0:5.0f}s", flush=True)

    print(f"\n  wrote results/arch_matched.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
