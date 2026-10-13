#!/usr/bin/env python3
"""The learned-scale arm: does one trainable logit scale remove the cap's effects?

A count readout's logits are spike counts in ``[0, T]``, which caps its confidence at
``e^T/(e^T + C - 1)``. The ``scaled-count`` readout multiplies the count by a learnable
``s = exp(theta)``, ``theta`` initialised at 0, trained by the same optimiser at the same
learning rate as every weight. Its cap is ``e^(sT)/(e^(sT) + C - 1)``, so the network can
move it. Nothing else differs from the existing count runs: architecture, neuron settings,
batch size, epoch cap, patience, seeds and split are theirs, and the split's fingerprint is
checked against the count run's record before anything is trained.

Three cells, nine runs: N-MNIST at ``T = 2`` with two blocks, N-Caltech101 at ``T = 2`` and
``T = 6`` with four. They test predictions 6a, 6b, 7 and 8, which were recorded in NOTES.md
and committed before this script was run. The script evaluates each prediction against the
falsification condition written there and stores the verdict beside the numbers.

Writes ``results/scaled_count_experiment.json``. Zero DataLoader workers; data is held in
RAM. The count and accumulator reference rows are read from the existing result files and
never recomputed here.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    compute_metrics, discrimination, quantise_confidence,
)
from spikecalib.experiments import (  # noqa: E402
    BATCH_SIZE, DEFAULT_CHANNELS, HIDDEN, cell_outputs, load_data, loader, splits_for, stem,
)
from spikecalib.models import ModelConfig, build_model, readout_ceiling  # noqa: E402
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.theory import max_distinct_confidences, min_scale  # noqa: E402
from spikecalib.train import TrainConfig, resolve_device, set_seed, train  # noqa: E402

CELLS = (("nmnist", 2), ("ncaltech101", 2), ("ncaltech101", 6))
READOUT = "scaled-count"


def reference(results: str, dataset: str, timesteps: int, seed: int, readout: str) -> dict:
    """The existing count or accumulator row for this cell, from the file that owns it."""
    name = "readout_experiment.json" if dataset == "nmnist" else "caltech_experiment.json"
    with open(os.path.join(results, name)) as fh:
        rows = json.load(fh)["rows"]
    return next(r for r in rows if r["readout"] == readout and r["timesteps"] == timesteps
                and r["seed"] == seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    device = resolve_device()
    rows = []
    print("=" * 110)
    print("LEARNED-SCALE ARM  logits = s * count, s = exp(theta), theta initialised at 0")
    print("=" * 110, flush=True)

    for dataset, timesteps in CELLS:
        data = load_data(dataset, timesteps, args.data_root)
        spec, channels, bs = data.spec, DEFAULT_CHANNELS[dataset], BATCH_SIZE[dataset]
        for seed in args.seeds:
            t0 = time.time()
            splits = splits_for(data, seed)
            count_ref = reference(args.out, dataset, timesteps, seed, "count")
            accum_ref = reference(args.out, dataset, timesteps, seed, "accumulator")
            with open(os.path.join(args.out, stem(dataset, timesteps, seed) + "_record.json")) as fh:
                count_record = json.load(fh)
            # The arm is only a like-for-like comparison if it is trained on the count run's split.
            if count_record["split_fingerprint"] != splits.fingerprint:
                raise RuntimeError(f"{dataset} T={timesteps} seed {seed}: split fingerprint "
                                   f"{splits.fingerprint} is not the count run's "
                                   f"{count_record['split_fingerprint']}")
            name = stem(dataset, timesteps, seed, READOUT)
            wpath = os.path.join(args.out, name + "_weights.pt")
            rpath = os.path.join(args.out, name + "_record.json")
            status = "reused"
            if not (os.path.exists(wpath) and os.path.exists(rpath)):
                cfg = ModelConfig(channels=channels, hidden=HIDDEN, readout=READOUT)
                model = build_model(spec, timesteps, cfg)
                set_seed(seed)
                state, record = train(
                    model,
                    {"train": loader(data.train, splits.train, bs, shuffle=True),
                     "val": loader(data.train, splits.val, bs)},
                    TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=bs,
                                patience=args.patience),
                    seed=seed, dataset=spec.key, timesteps=timesteps,
                    split_fingerprint=splits.fingerprint,
                )
                torch.save(state, wpath)
                rec = record.as_dict()
                rec["model_config"] = {"channels": list(cfg.channels), "hidden": cfg.hidden,
                                       "tau": cfg.tau, "v_threshold": cfg.v_threshold,
                                       "surrogate_alpha": cfg.surrogate_alpha, "readout": READOUT}
                rec["logit_scale"] = float(np.exp(float(state["log_scale"])))
                with open(rpath, "w") as fh:
                    json.dump(rec, fh, indent=2)
                status = "trained"

            out = cell_outputs(args.out, data, seed, device, READOUT)
            rec = out.record
            conf, correct = out.test.conf, out.test.correct
            acc = float(correct.mean())
            m = compute_metrics(conf, correct, n_bins=args.bins, n_bootstrap=0)
            d = discrimination(conf, correct)
            sel = selective_summary(conf, correct)
            cap = readout_ceiling(READOUT, timesteps, spec.n_classes, scale=out.scale)
            s_min = min_scale(timesteps, spec.n_classes, acc)
            history_scale = [h.get("logit_scale") for h in rec.get("history", [])]
            row = {
                "dataset": spec.key, "n_classes": spec.n_classes, "readout": READOUT,
                "channels": list(channels), "n_blocks": len(channels),
                "timesteps": timesteps, "seed": seed, "status": status,
                "epochs_run": rec.get("epochs_run"), "best_epoch": rec.get("best_epoch"),
                "early_stopped": rec.get("early_stopped"),
                "batch_size": rec.get("batch_size"), "lr": rec.get("lr"),
                "split_fingerprint": out.split_fingerprint,
                "n_cal": int(out.cal.n), "n_test": int(out.test.n),
                "logit_scale": out.scale,
                "logit_scale_last_epoch": history_scale[-1] if history_scale else None,
                "logit_scale_by_epoch": history_scale,
                "s_min": s_min, "scale_exceeds_s_min": bool(out.scale > s_min),
                "logit_width": out.scale * timesteps,
                "ceiling_at_learned_scale": cap,
                "ceiling_at_scale_one": readout_ceiling("count", timesteps, spec.n_classes),
                "max_confidence_observed": float(conf.max()),
                "ceiling_signed_excess": float(conf.max()) - cap,
                "test_accuracy": acc,
                "mean_confidence": float(conf.mean()),
                "mean_conf_minus_acc": float(conf.mean()) - acc,
                "ece_equal_width": m.ece_equal_width, "ece_equal_mass": m.ece_equal_mass,
                "n_distinct": int(np.unique(quantise_confidence(conf)).size),
                "max_distinct_bound": max_distinct_confidences(timesteps, spec.n_classes),
                "auroc": d["auroc"], "aurc": sel["aurc"],
                "n_achievable_coverage_levels": sel["n_achievable_coverage_levels"],
                "count_test_accuracy": count_ref["test_accuracy"],
                "count_mean_conf_minus_acc": count_ref["mean_conf_minus_acc"],
                "accumulator_test_accuracy": accum_ref["test_accuracy"],
                "accumulator_mean_conf_minus_acc": accum_ref["mean_conf_minus_acc"],
                "train_seconds": rec.get("seconds"), "seconds": time.time() - t0,
            }
            rows.append(row)
            print(f"  {spec.key:<12} T={timesteps:<2} seed {seed} {status:<7} "
                  f"s {out.scale:6.3f} (s_min {s_min:5.3f})  acc {acc:.4f} "
                  f"[count {count_ref['test_accuracy']:.4f}, accum {accum_ref['test_accuracy']:.4f}]  "
                  f"conf-acc {row['mean_conf_minus_acc']:+.4f}  cap {cap:.4f} max {conf.max():.4f}  "
                  f"ECEew {m.ece_equal_width:.4f} ECEem {m.ece_equal_mass:.4f}  "
                  f"levels {row['n_distinct']}", flush=True)

    def cell(dataset, timesteps):
        return [r for r in rows if r["dataset"] == dataset and r["timesteps"] == timesteps]

    def mean(rs, key):
        return st.mean(r[key] for r in rs)

    verdicts = {}
    if len(args.seeds) == 3 and len(rows) == 9:
        nm, c2, c6 = cell("nmnist", 2), cell("ncaltech101", 2), cell("ncaltech101", 6)
        g_nm, g_c2, g_c6 = (mean(x, "mean_conf_minus_acc") for x in (nm, c2, c6))
        count_c2, accum_c2 = mean(c2, "count_test_accuracy"), mean(c2, "accumulator_test_accuracy")
        threshold = count_c2 + 0.5 * (accum_c2 - count_c2)
        verdicts = {
            "6a": {"statement": "N-MNIST T=2: |mean signed gap| < 0.05",
                   "mean_signed_gap": g_nm, "per_seed": [r["mean_conf_minus_acc"] for r in nm],
                   "held": bool(abs(g_nm) < 0.05)},
            "6b": {"statement": "N-Caltech101 T=2: mean signed gap > 0",
                   "mean_signed_gap": g_c2, "per_seed": [r["mean_conf_minus_acc"] for r in c2],
                   "held": bool(g_c2 > 0)},
            "7": {"statement": "N-Caltech101 T=2: mean accuracy >= count + half the gap to the accumulator",
                  "mean_accuracy": mean(c2, "test_accuracy"),
                  "per_seed": [r["test_accuracy"] for r in c2],
                  "count_accuracy": count_c2, "accumulator_accuracy": accum_c2,
                  "threshold_from_these_rows": threshold, "threshold_registered": 0.5184,
                  "share_of_gap_recovered": (mean(c2, "test_accuracy") - count_c2) / (accum_c2 - count_c2),
                  "held": bool(mean(c2, "test_accuracy") >= 0.5184)},
            "8": {"statement": "N-Caltech101 T=6: mean signed gap > -0.05",
                  "mean_signed_gap": g_c6, "per_seed": [r["mean_conf_minus_acc"] for r in c6],
                  "sign": "overconfident" if g_c6 > 0 else "underconfident",
                  "magnitude_below_0p05": bool(abs(g_c6) < 0.05),
                  "held": bool(g_c6 > -0.05)},
            "scale_exceeds_s_min_in": f"{sum(r['scale_exceeds_s_min'] for r in rows)} of {len(rows)} runs",
        }
        print("\n  predictions, against the conditions registered in NOTES.md:")
        for k in ("6a", "6b", "7", "8"):
            print(f"    {k:<3} {'HELD' if verdicts[k]['held'] else 'FALSIFIED':<10} {verdicts[k]['statement']}")
        print(f"    learned s > s_min in {verdicts['scale_exceeds_s_min_in']}")

    path = os.path.join(args.out, "scaled_count_experiment.json")
    with open(path, "w") as fh:
        json.dump({"readout": READOUT, "registered": "NOTES.md, predictions 6a, 6b, 7 and 8, 2026-10-13",
                   "rows": rows, "predictions": verdicts}, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
