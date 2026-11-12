#!/usr/bin/env python3
"""A7, added after review, not pre-registered: the learned scale, trained to convergence.

The nine scaled-count models of the learned-scale arm (N-MNIST at ``T = 2``; N-Caltech101 at
``T = 2`` and ``T = 6``; seeds 0, 1 and 2) stopped on validation accuracy while their scale
was still rising. This script retrains them with every setting unchanged except stopping:
training continues until ``theta = ln s`` moves by less than 0.01 over five consecutive
epochs, or 150 epochs, whichever comes first. The thresholds, the run order and the
five-hour compute limit were fixed in NOTES.md ("Post-review analyses, thresholds fixed
2026-11-12 before computation", A7) and committed before this script was run.

Steps, all recorded in ``results/scale_convergence.json``:

1. Time one epoch per cell (seed 0, a throwaway model).
2. Project the worst case (150 epochs for every run). If it exceeds five hours, train the
   three N-MNIST runs in full and then the N-Caltech101 runs in the fixed order, starting a
   run only if its projected worst case fits within five hours of the start of training.
3. Train each planned run; evaluate on the test set the checkpoint the original rule would
   have kept and the end point.

The original nine runs remain the pre-registered results and are read, never rewritten.
Rerunning resumes: a run whose row is already in the output file is not trained again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import compute_metrics, discrimination, model_outputs  # noqa: E402
from spikecalib.convergence import ConvergenceConfig, train_to_convergence  # noqa: E402
from spikecalib.experiments import (  # noqa: E402
    BATCH_SIZE, DEFAULT_CHANNELS, HIDDEN, load_data, loader, splits_for, stem,
)
from spikecalib.models import ModelConfig, build_model, readout_ceiling  # noqa: E402
from spikecalib.theory import min_scale  # noqa: E402
from spikecalib.train import predict_outputs, resolve_device, set_seed  # noqa: E402

READOUT = "scaled-count"
CELLS = (("nmnist", 2), ("ncaltech101", 2), ("ncaltech101", 6))
NMNIST_RUNS = [("nmnist", 2, s) for s in (0, 1, 2)]
CALTECH_ORDER = [("ncaltech101", 2, 0), ("ncaltech101", 6, 0), ("ncaltech101", 2, 1),
                 ("ncaltech101", 6, 1), ("ncaltech101", 2, 2), ("ncaltech101", 6, 2)]
BUDGET_SECONDS = 5 * 3600
SETTINGS = ConvergenceConfig()


def log(msg: str) -> None:
    print(msg, flush=True)


def save(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def build(spec, timesteps: int, dataset: str, seed: int):
    set_seed(seed)
    cfg = ModelConfig(channels=DEFAULT_CHANNELS[dataset], hidden=HIDDEN, readout=READOUT)
    return build_model(spec, timesteps, cfg)


def loaders_for(data, splits, dataset: str) -> dict:
    bs = BATCH_SIZE[dataset]
    return {"train": loader(data.train, splits.train, bs, shuffle=True),
            "val": loader(data.train, splits.val, bs)}


def evaluate(model, data, splits, dataset: str, timesteps: int, seed: int, device,
             bins: int, outputs_path: str) -> dict:
    """Test-set metrics of one checkpoint, built exactly as cell_outputs builds them."""
    bs = BATCH_SIZE[dataset]
    cal_scores, cal_labels, _ = predict_outputs(model, loader(data.train, splits.cal, bs), device,
                                                count_synops=False)
    test_scores, test_labels, _ = predict_outputs(model, loader(data.test, splits.test, bs), device,
                                                  count_synops=False)
    np.savez_compressed(outputs_path, cal_scores=cal_scores, cal_labels=cal_labels,
                        test_scores=test_scores, test_labels=test_labels,
                        fingerprint=splits.fingerprint, scale=model.scale)
    test = model_outputs(test_scores, test_labels, np.random.default_rng(seed + 1))
    correct = test.predictions == test.labels
    acc = float(correct.mean())
    m = compute_metrics(test.conf, correct, n_bins=bins, n_bootstrap=0)
    d = discrimination(test.conf, correct)
    s = model.scale
    n_classes = data.spec.n_classes
    cap = readout_ceiling(READOUT, timesteps, n_classes, scale=s)
    return {
        "logit_scale": s, "theta": float(np.log(s)),
        "s_min": min_scale(timesteps, n_classes, acc),
        "cap": cap, "floor": max(0.0, acc - cap),
        "test_accuracy": acc,
        "ece_equal_width": m.ece_equal_width, "ece_equal_mass": m.ece_equal_mass,
        "mean_conf": float(test.conf.mean()), "conf_minus_acc": float(test.conf.mean()) - acc,
        "auroc": d["auroc"], "n_distinct": d["n_distinct"],
        "outputs_file": os.path.relpath(outputs_path, REPO_ROOT),
    }


def time_one_epoch(dataset: str, timesteps: int, data, device_name) -> float:
    splits = splits_for(data, 0)
    model = build(data.spec, timesteps, dataset, 0)
    cfg = ConvergenceConfig(max_epochs=1, original_cap=1, batch_size=BATCH_SIZE[dataset],
                            device=device_name)
    run = train_to_convergence(model, loaders_for(data, splits, dataset), cfg, seed=0, log=log)
    return float(run.history[0]["seconds"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    device = resolve_device()
    path = os.path.join(args.out, "scale_convergence.json")
    doc = {}
    if os.path.exists(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc.setdefault("analysis", "A7, added after review, not pre-registered")
    doc.setdefault("thresholds", {
        "source": "NOTES.md, Post-review analyses, thresholds fixed 2026-11-12 before computation, A7",
        "window_epochs": SETTINGS.window, "theta_tolerance": SETTINGS.tol,
        "max_epochs": SETTINGS.max_epochs, "original_patience": SETTINGS.original_patience,
        "original_cap": SETTINGS.original_cap, "budget_seconds": BUDGET_SECONDS,
    })
    doc.setdefault("device", str(device))
    doc.setdefault("rows", [])

    datasets: dict = {}

    def data_for(dataset: str, timesteps: int):
        key = (dataset, timesteps)
        if key not in datasets:
            datasets.clear()  # hold one cell in RAM at a time
            datasets[key] = load_data(dataset, timesteps, args.data_root)
        return datasets[key]

    # 1. Timing, once.
    if "epoch_seconds" not in doc:
        timing = {}
        for dataset, timesteps in CELLS:
            log(f"timing one epoch: {dataset} T={timesteps}")
            timing[f"{dataset}_T{timesteps}"] = time_one_epoch(dataset, timesteps,
                                                               data_for(dataset, timesteps), None)
        doc["epoch_seconds"] = timing
        save(path, doc)

    # 2. The plan, fixed by the rule in NOTES.md.
    per_run = {f"{d}_T{t}": doc["epoch_seconds"][f"{d}_T{t}"] * SETTINGS.max_epochs
               for d, t in CELLS}
    projected = sum(per_run[f"{d}_T{t}"] for d, t, _ in NMNIST_RUNS + CALTECH_ORDER)
    doc["projected_worst_case_seconds"] = projected
    doc["projected_worst_case_per_run_seconds"] = per_run
    doc["over_budget"] = bool(projected > BUDGET_SECONDS)
    save(path, doc)
    log(f"projected worst case {projected / 3600:.2f} h; over budget: {doc['over_budget']}")

    done = {(r["dataset"], r["timesteps"], r["seed"]) for r in doc["rows"]}
    start = doc.get("training_started_at") or time.time()
    doc["training_started_at"] = start
    skipped = []
    for dataset, timesteps, seed in NMNIST_RUNS + CALTECH_ORDER:
        if (dataset, timesteps, seed) in done:
            continue
        if doc["over_budget"] and dataset == "ncaltech101":
            elapsed = time.time() - start
            if elapsed + per_run[f"{dataset}_T{timesteps}"] > BUDGET_SECONDS:
                skipped.append({"dataset": dataset, "timesteps": timesteps, "seed": seed,
                                "elapsed_seconds": elapsed})
                log(f"skip {dataset} T={timesteps} seed {seed}: worst case does not fit")
                continue
        data = data_for(dataset, timesteps)
        splits = splits_for(data, seed)
        name = stem(dataset, timesteps, seed, READOUT)
        with open(os.path.join(args.out, stem(dataset, timesteps, seed) + "_record.json")) as fh:
            count_record = json.load(fh)
        if count_record["split_fingerprint"] != splits.fingerprint:
            raise RuntimeError(f"{name}: split fingerprint differs from the count run's")
        log(f"train {name} to convergence")
        model = build(data.spec, timesteps, dataset, seed)
        cfg = ConvergenceConfig(batch_size=BATCH_SIZE[dataset])
        run = train_to_convergence(model, loaders_for(data, splits, dataset), cfg, seed=seed, log=log)
        row = {
            "dataset": dataset, "timesteps": timesteps, "seed": seed,
            "n_classes": data.spec.n_classes, "split_fingerprint": splits.fingerprint,
            "epochs_run": run.epochs_run, "converged_epoch": run.converged_epoch,
            "stopped_by": run.stopped_by, "original_stop_epoch": run.original_stop_epoch,
            "original_best_epoch": run.original_best_epoch, "seconds": run.seconds,
            "device": run.device, "history": run.history,
        }
        for which, state in (("original_rule", run.original_state), ("end_point", run.end_state)):
            wpath = os.path.join(args.out, f"{name}_converged_{which}_weights.pt")
            torch.save(state, wpath)
            ev_model = build(data.spec, timesteps, dataset, seed)
            ev_model.load_state_dict(state)
            row[which] = evaluate(ev_model, data, splits, dataset, timesteps, seed, device,
                                  args.bins, os.path.join(args.out, f"{name}_converged_{which}_outputs.npz"))
            row[which]["weights_file"] = os.path.relpath(wpath, REPO_ROOT)
        doc["rows"].append(row)
        doc["skipped"] = skipped
        save(path, doc)
        log(f"done {name}: {run.epochs_run} epochs, stopped by {run.stopped_by}, "
            f"s {row['end_point']['logit_scale']:.3f}")

    doc["skipped"] = skipped
    doc["training_finished_at"] = time.time()
    save(path, doc)
    log("all planned runs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
