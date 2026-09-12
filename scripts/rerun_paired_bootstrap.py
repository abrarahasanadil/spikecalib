#!/usr/bin/env python3
"""Add the equal-mass paired bootstrap to the sweep, and verify by reproducing equal-width.

Correctness rule 2 says both ECE estimators are reported wherever a number is claimed. The
paired isotonic-minus-temperature difference was the one place that only ever existed for
equal-width: ``iso_minus_temp_ece_ew`` had no ``_em`` counterpart, so "13 of 15 intervals
straddle zero" was an equal-width statement presented as if it were the finding.

No training happens here. Weights are reused, inference is re-run on the calibration and
test splits, the two calibrators are refit on CPU exactly as :mod:`spikecalib.sweep` fits
them, and the paired bootstrap is computed under *both* estimators with the same seed.

**The equal-width result is recomputed too, and compared against what is already in the
CSV.** If the pipeline here is not faithful, that comparison fails and nothing is written --
a new column that silently used a different fit would be worse than no column at all.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import bootstrap_ece_difference, model_outputs  # noqa: E402
from spikecalib.data import load_dataset, make_splits, training_labels  # noqa: E402
from spikecalib.datasets import get_spec  # noqa: E402
from spikecalib.models import ModelConfig, build_model  # noqa: E402
from spikecalib.sweep import ARMS  # noqa: E402
from spikecalib.train import predict_counts, resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--tolerance", type=float, default=5e-4,
                    help="how far the recomputed equal-width difference may sit from the "
                         "CSV before this refuses to write")
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    device = resolve_device()
    out: dict[tuple[int, int], dict] = {}

    print("=" * 92)
    print(f"PAIRED BOOTSTRAP, BOTH ESTIMATORS  {spec.display_name}  "
          f"T={args.timesteps}  seeds={args.seeds}")
    print("  no training; weights reused, calibrators refit on CPU")
    print("=" * 92, flush=True)

    for timesteps in args.timesteps:
        with contextlib.redirect_stdout(io.StringIO()):
            train_set, test_set = load_dataset(spec.key, timesteps, args.data_root)
        labels = training_labels(train_set)

        for seed in args.seeds:
            splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))

            def loader(ds, idx):
                return DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                                  num_workers=args.num_workers)

            stem = f"{spec.key}_T{timesteps}_seed{seed}"
            model = build_model(spec, timesteps, ModelConfig())
            model.load_state_dict(
                torch.load(os.path.join(args.results, stem + "_weights.pt"), map_location="cpu")
            )

            cal_counts, cal_labels, _ = predict_counts(
                model, loader(train_set, splits.cal), device, count_synops=False)
            test_counts, test_labels, _ = predict_counts(
                model, loader(test_set, splits.test), device, count_synops=False)

            # Same RNG offsets as spikecalib.sweep, or the tie-breaking differs and the
            # equal-width check below would fail for a reason that is not a real mismatch.
            cal = model_outputs(cal_counts, cal_labels, np.random.default_rng(seed))
            test = model_outputs(test_counts, test_labels, np.random.default_rng(seed + 1))

            fitted = {}
            for name, cls in ARMS:
                fitted[name] = cls().fit(cal.conf, cal.correct, outputs=cal).transform(
                    test.conf, outputs=test)

            res = {}
            for est in ("equal_width", "equal_mass"):
                res[est] = bootstrap_ece_difference(
                    fitted["isotonic-top-label"], fitted["temperature"], test.correct,
                    n_bins=args.bins, n_bootstrap=args.n_bootstrap, seed=seed,
                    estimator=est,
                )
            out[(timesteps, seed)] = res
            ew, em = res["equal_width"], res["equal_mass"]
            print(f"  T={timesteps:<3} seed {seed}   ew {ew['difference']:+.6f} "
                  f"[{ew['ci'][0]:+.6f}, {ew['ci'][1]:+.6f}] resolvable {ew['resolvable']}"
                  f"   |   em {em['difference']:+.6f} "
                  f"[{em['ci'][0]:+.6f}, {em['ci'][1]:+.6f}] resolvable {em['resolvable']}",
                  flush=True)

    path = os.path.join(args.results, f"sweep_{spec.key}.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys())

    # Faithfulness check before anything is written.
    worst = 0.0
    for r in rows:
        key = (int(r["timesteps"]), int(r["seed"]))
        if key not in out:
            continue
        worst = max(worst, abs(float(r["iso_minus_temp_ece_ew"])
                               - out[key]["equal_width"]["difference"]))
    print(f"\n  equal-width recomputed vs the CSV: worst |delta| = {worst:.2e}")
    if worst > args.tolerance:
        print(f"  REFUSING TO WRITE: above the {args.tolerance:.0e} tolerance. The refit here "
              f"does not reproduce the sweep, so the equal-mass column would not be "
              f"comparable to the equal-width one already in the file.")
        return 1

    new = ["iso_minus_temp_ece_em", "iso_minus_temp_em_ci_low",
           "iso_minus_temp_em_ci_high", "iso_minus_temp_em_resolvable"]
    for c in new:
        if c not in fieldnames:
            fieldnames.append(c)
    for r in rows:
        em = out[(int(r["timesteps"]), int(r["seed"]))]["equal_mass"]
        r["iso_minus_temp_ece_em"] = em["difference"]
        r["iso_minus_temp_em_ci_low"] = em["ci"][0]
        r["iso_minus_temp_em_ci_high"] = em["ci"][1]
        r["iso_minus_temp_em_resolvable"] = em["resolvable"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    straddle_ew = sum(1 for r in rows if r["iso_minus_temp_resolvable"] != "True")
    straddle_em = sum(1 for r in rows if str(r["iso_minus_temp_em_resolvable"]) != "True")
    runs = len({(r["timesteps"], r["seed"]) for r in rows})
    n_ew = len({(r["timesteps"], r["seed"]) for r in rows
                if r["iso_minus_temp_resolvable"] != "True"})
    n_em = len({(r["timesteps"], r["seed"]) for r in rows
                if str(r["iso_minus_temp_em_resolvable"]) != "True"})
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}  (+{len(new)} columns)")
    print(f"\n  intervals straddling zero: equal-width {n_ew} of {runs}, "
          f"equal-mass {n_em} of {runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
