#!/usr/bin/env python3
"""Run the T x seed sweep, or just verify that it could run.

Refuses to start unless every frame cache it will need is present and complete, because
discovering a missing cache four hours into a sweep wastes four hours. ``--verify-only``
runs that check and stops.

Epoch settings default to a cap of 30 with patience 5, so that **early stopping decides
when a run ends, not the cap**. Every results row records ``epochs_run`` and
``early_stopped``, and a convergence audit is printed before the calibration numbers: a
run that hit the cap may be undertrained, and comparing a converged low-T model against an
undertrained high-T one would attribute a training difference to a calibration mechanism.

Results are written to results/ as CSV (untracked by design -- the committed artifact is
the code that regenerates them) and rows are flushed after each run, so an interrupted
sweep keeps everything it has already earned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.sweep import (  # noqa: E402
    CacheMissingError,
    SweepConfig,
    convergence_audit,
    run_one,
    val_accuracy_audit,
    verify_caches,
)
from spikecalib.train import resolve_device  # noqa: E402

# Cost model, used only to warn before a long run. Both constants are [MEASURED] on this
# machine and are close to linear in T.
#
# Training: scripts/train_one.py, N-MNIST seed 0, 2026-09-09 -- 10.6 s/epoch at T=2 and
# 19.4 s/epoch at T=4 on MPS, so ~5 s per epoch per unit of T.
SECONDS_PER_EPOCH_PER_T = 5.0
#
# Inference: predict_counts over the 5,000-sample calibration split and the 10,000-sample
# test split, measured 2026-09-10 at ~68 s total for T=2, so ~34 s per unit of T. This is
# a per-run cost paid once, not per epoch, and it is dominated by reading npz files rather
# than by compute -- SynOps instrumentation was measured at -0.1 s, i.e. free. Leaving it
# out of the estimate understated a 15-run sweep by roughly half an hour.
INFERENCE_SECONDS_PER_T = 34.0


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_cache_table(statuses) -> None:
    print(f"  {'T':>4}  {'status':<28} path")
    for s in statuses:
        print(f"  {s.timesteps:>4}  {s.verdict:<28} {os.path.relpath(s.path, REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30, help="cap; patience should decide")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--no-resume", action="store_true",
                        help="retrain even when a matching saved run exists")
    parser.add_argument("--val-tolerance", type=float, default=0.02,
                        help="flag runs this far below the best val accuracy at their T")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the frame caches and exit without training")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    print("=" * 84)
    print(f"SWEEP  {spec.display_name}  T={args.timesteps}  seeds={args.seeds}")
    print("=" * 84)

    section("Precondition 1 -- frame caches")
    statuses = verify_caches(spec, args.timesteps, args.data_root)
    print_cache_table(statuses)
    missing = [s for s in statuses if not s.complete]
    if missing:
        print()
        print(f"  {len(missing)} of {len(statuses)} caches are not ready. Build them with:")
        for s in missing:
            print(f"    python scripts/fetch_data.py --dataset {spec.key} "
                  f"--timesteps {s.timesteps}")
        print()
        print("  Refusing to start. A sweep that dies partway through has wasted the")
        print("  training it had already done.")
        return 1
    print(f"  all {len(statuses)} caches complete")

    section("Precondition 2 -- epochs must be decided by early stopping, not the cap")
    print(f"  epoch cap : {args.epochs}")
    print(f"  patience  : {args.patience}")
    print("  Every row records epochs_run and early_stopped. A run that hits the cap may")
    print("  be undertrained, and comparing a converged low-T model against an undertrained")
    print("  high-T one would attribute a training difference to a calibration mechanism.")

    n_runs = len(args.timesteps) * len(args.seeds)
    train_seconds = sum(args.epochs * t * SECONDS_PER_EPOCH_PER_T for t in args.timesteps)
    infer_seconds = sum(t * INFERENCE_SECONDS_PER_T for t in args.timesteps)
    worst_case = (train_seconds + infer_seconds) * len(args.seeds)
    section("Cost")
    print(f"  runs                  : {n_runs}  ({len(args.timesteps)} timesteps x {len(args.seeds)} seeds)")
    print(f"  device                : {resolve_device()}")
    print(f"  training (cap, no early stopping) : {train_seconds * len(args.seeds) / 3600:,.1f} h")
    print(f"  count extraction (per run, fixed) : {infer_seconds * len(args.seeds) / 3600:,.1f} h")
    print(f"  worst case total                  : {worst_case / 3600:,.1f} h")
    print("  Early stopping cuts the training half only; the extraction cost is paid by")
    print("  every run regardless. The audit at the end says how much was actually saved.")

    if args.verify_only:
        section("Result")
        print("  --verify-only: preconditions pass, nothing trained.")
        return 0

    cfg = SweepConfig(
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_bins=args.bins, n_bootstrap=args.bootstrap,
    )
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"sweep_{spec.key}.csv")
    device = resolve_device()

    section("Running")
    rows: list[dict] = []
    writer = None
    handle = None
    started = time.time()
    try:
        for timesteps in args.timesteps:
            for seed in args.seeds:
                t0 = time.time()
                run_rows = run_one(spec, timesteps, seed, cfg, args.data_root, device,
                                   out_dir=args.out, allow_resume=not args.no_resume)
                rows.extend(run_rows)
                head = run_rows[0]
                flag = "" if head["early_stopped"] else "   <-- HIT EPOCH CAP"
                if head["resumed"]:
                    flag += "   [resumed]"
                print(
                    f"  T={timesteps:<3} seed={seed}  "
                    f"epochs {head['epochs_run']:>2}/{head['epochs_requested']}  "
                    f"val {head['best_val_accuracy']:.4f}  "
                    f"test {head['test_accuracy']:.4f}  "
                    f"levels {head['n_distinct_cal']:>4}  "
                    f"auroc {head['auroc_raw_test']:.3f}  "
                    f"{time.time() - t0:>6.1f}s{flag}",
                    flush=True,
                )
                if writer is None:
                    handle = open(csv_path, "w", newline="")
                    writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
                    writer.writeheader()
                writer.writerows(run_rows)
                handle.flush()
    finally:
        if handle is not None:
            handle.close()

    audit = convergence_audit(rows)
    section("Convergence audit -- read this before the calibration numbers")
    print(f"  {'T':>4}{'seed':>6}{'epochs':>9}{'stopped by':>14}{'best val':>11}")
    for r in audit["runs"]:
        stopped = "early stopping" if r["early_stopped"] else "EPOCH CAP"
        print(f"  {r['timesteps']:>4}{r['seed']:>6}"
              f"{r['epochs_run']:>6}/{r['epochs_requested']:<2}"
              f"{stopped:>14}{r['best_val_accuracy']:>11.4f}")
    print()
    if audit["all_converged"]:
        print(f"  All {audit['n_runs']} runs stopped on the patience rule, so every model is")
        print("  equally converged and calibration differences across T are not confounded")
        print("  by training differences.")
    else:
        print(f"  WARNING: {audit['n_capped']} of {audit['n_runs']} runs hit the epoch cap:")
        for r in audit["capped"]:
            print(f"    T={r['timesteps']} seed={r['seed']} "
                  f"({r['epochs_run']} epochs, best val {r['best_val_accuracy']:.4f})")
        print("  These may be undertrained. Any comparison involving them confounds")
        print("  calibration with convergence. Re-run with a higher --epochs before")
        print("  reading the calibration results.")

    val_audit = val_accuracy_audit(rows, tolerance=args.val_tolerance)
    section(f"Validation-accuracy audit -- outliers more than {args.val_tolerance:.2f} below best at their T")
    print(f"  {'T':>4}{'seed':>6}{'best val':>11}{'best at T':>12}{'below':>9}  flag")
    for r in val_audit["runs"]:
        mark = "  <-- OUTLIER" if r["val_outlier"] else ""
        print(f"  {r['timesteps']:>4}{r['seed']:>6}{r['best_val_accuracy']:>11.4f}"
              f"{r['best_at_timesteps']:>12.4f}{r['below_best']:>9.4f}{mark}")
    print()
    if val_audit["all_consistent"]:
        print("  All seeds agree within tolerance at every T, so no run looks like a")
        print("  training failure masquerading as seed noise.")
    else:
        print(f"  WARNING: {val_audit['n_outliers']} run(s) trail the best at their T by more")
        print(f"  than {args.val_tolerance:.2f}. A run that failed to train has a different readout")
        print("  from its siblings -- fewer confidence levels, higher tie rate -- and averaging")
        print("  it into a per-T result smears a training failure across the trend.")

    # Rewrite the CSV with the per-T outlier flags now that every run is known. The
    # incremental file written during the sweep is the crash-safe copy; this is the final
    # one, and it only replaces the incremental version once every row is in hand.
    flags = {(r["dataset"], r["timesteps"], r["seed"]): r for r in val_audit["runs"]}
    for row in rows:
        key = (row["dataset"], row["timesteps"], row["seed"])
        row["val_below_best_at_t"] = flags[key]["below_best"]
        row["val_outlier"] = flags[key]["val_outlier"]
    if rows:
        with open(csv_path, "w", newline="") as fh:
            final = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            final.writeheader()
            final.writerows(rows)

    summary_path = os.path.join(args.out, f"sweep_{spec.key}_audit.json")
    with open(summary_path, "w") as fh:
        json.dump({
            "dataset": spec.key,
            "timesteps": args.timesteps,
            "seeds": args.seeds,
            "epochs_cap": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_bootstrap": args.bootstrap,
            "wall_clock_seconds": time.time() - started,
            "audit": audit,
            "val_audit": val_audit,
        }, fh, indent=2)

    section("Saved")
    print(f"  rows  : {os.path.relpath(csv_path, REPO_ROOT)}  ({len(rows):,} rows)")
    print(f"  audit : {os.path.relpath(summary_path, REPO_ROOT)}")
    print(f"  total : {(time.time() - started) / 60:,.1f} min")
    if not audit["all_converged"]:
        return 2
    if not val_audit["all_consistent"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
