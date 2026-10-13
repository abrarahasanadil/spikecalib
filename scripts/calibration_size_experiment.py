#!/usr/bin/env python3
"""Does the calibrator tie depend on how much calibration data there is?

On N-MNIST the two calibrators tie with a 5,000-sample calibration set. Isotonic regression
estimates one value per confidence level and temperature scaling one value in total, so the
tie is most likely to break when calibration data is scarce. This shrinks the calibration
set and watches for it.

No training. The three N-MNIST ``T = 2`` count models are reused. For each seed and each
size in {100, 1000, 5000}, 20 subsamples of the calibration split are drawn without
replacement, stratified so each class supplies a tenth of the size. Both calibrators are
fitted on each subsample, on CPU, and compared on the full test split with the paired
bootstrap under both estimators and the tie rule fixed in NOTES.md. At 5,000 the subsample
is the whole split, so one fit is made and counted for all 20 draws.

**Prediction 11**, recorded before any subsample was drawn: at 100 samples temperature
scaling beats isotonic in at least 10 of 20 draws for at least 2 of the 3 seeds; at 1,000
and 5,000 they tie, meaning the two differ in fewer than 10 of 20 draws in every seed.

Writes ``results/calibration_size_experiment.json``. Zero DataLoader workers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicTopLabel, TemperatureScaling, ece_equal_mass, ece_equal_width, paired_verdict,
    quantise_confidence,
)
from spikecalib.experiments import (  # noqa: E402
    all_cached, cell_outputs, load_data, stem, stratified_subsample, subset,
)
from spikecalib.train import resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--timesteps", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()
    device = resolve_device()
    T = args.timesteps

    names = [stem(args.dataset, T, s) for s in args.seeds]
    data = load_data(args.dataset, T, args.data_root, in_memory=not all_cached(args.out, names))
    rows = []
    print("=" * 100)
    print(f"CALIBRATION-SET SIZE  {args.dataset} T={T}  sizes={args.sizes}  {args.draws} draws  no training")
    print("=" * 100, flush=True)

    for seed in args.seeds:
        out = cell_outputs(args.out, data, seed, device, "count")
        cal, test = out.cal, out.test
        for size in args.sizes:
            whole = size == cal.n
            for draw in range(args.draws):
                if whole and draw > 0:
                    # the subsample is the entire split: every draw is the same fit
                    rows.append({**rows[-1], "draw": draw, "repeat_of_draw": 0})
                    continue
                rng = np.random.default_rng([seed, size, draw])
                idx = np.arange(cal.n) if whole else stratified_subsample(cal.labels, size, rng)
                sub = subset(cal, idx)
                ts = TemperatureScaling().fit(sub.conf, sub.correct, outputs=sub)
                iso = IsotonicTopLabel().fit(sub.conf, sub.correct, outputs=sub)
                c_ts = quantise_confidence(np.asarray(ts.transform(test.conf, outputs=test), dtype=np.float64))
                c_iso = quantise_confidence(np.asarray(iso.transform(test.conf, outputs=test), dtype=np.float64))
                # a = isotonic, b = temperature: negative favours isotonic, as in the sweep
                v = paired_verdict(c_iso, c_ts, test.correct, n_bins=args.bins,
                                   n_bootstrap=args.n_bootstrap, seed=seed)
                ew, em = v["equal_width"], v["equal_mass"]
                rows.append({
                    "dataset": args.dataset, "timesteps": T, "seed": seed, "n_cal": size,
                    "draw": draw, "repeat_of_draw": None,
                    "split_fingerprint": out.split_fingerprint, "n_test": int(test.n),
                    "n_distinct_cal": int(np.unique(quantise_confidence(sub.conf)).size),
                    "temperature": ts.temperature_,
                    "temp_ece_ew": ece_equal_width(c_ts, test.correct, args.bins),
                    "temp_ece_em": ece_equal_mass(c_ts, test.correct, args.bins),
                    "iso_ece_ew": ece_equal_width(c_iso, test.correct, args.bins),
                    "iso_ece_em": ece_equal_mass(c_iso, test.correct, args.bins),
                    "iso_minus_temp_ece_ew": ew["difference"], "ew_ci": list(ew["ci"]),
                    "ew_excludes_zero": ew["resolvable"],
                    "iso_minus_temp_ece_em": em["difference"], "em_ci": list(em["ci"]),
                    "em_excludes_zero": em["resolvable"],
                    "differ": v["differ"],
                    "favours": {"a": "isotonic", "b": "temperature", None: None}[v["favours"]],
                })
            cell = [r for r in rows if r["seed"] == seed and r["n_cal"] == size]
            print(f"  seed {seed} n_cal {size:>5}: temperature wins {sum(r['favours'] == 'temperature' for r in cell):>2}/"
                  f"{len(cell)}, isotonic wins {sum(r['favours'] == 'isotonic' for r in cell):>2}/{len(cell)}, "
                  f"tie {sum(not r['differ'] for r in cell):>2}/{len(cell)}   "
                  f"mean ECEew temp {np.mean([r['temp_ece_ew'] for r in cell]):.4f} iso {np.mean([r['iso_ece_ew'] for r in cell]):.4f}  "
                  f"ECEem temp {np.mean([r['temp_ece_em'] for r in cell]):.4f} iso {np.mean([r['iso_ece_em'] for r in cell]):.4f}",
                  flush=True)

    half = args.draws / 2
    summary = []
    for seed in args.seeds:
        for size in args.sizes:
            cell = [r for r in rows if r["seed"] == seed and r["n_cal"] == size]
            summary.append({"seed": seed, "n_cal": size, "draws": len(cell),
                            "temperature_wins": sum(r["favours"] == "temperature" for r in cell),
                            "isotonic_wins": sum(r["favours"] == "isotonic" for r in cell),
                            "differ": sum(r["differ"] for r in cell)})
    verdict = {}
    if sorted(args.sizes) == [100, 1000, 5000] and len(args.seeds) == 3 and args.draws == 20:
        at = lambda n: [s for s in summary if s["n_cal"] == n]  # noqa: E731
        seeds_for = sum(s["temperature_wins"] >= half for s in at(100))
        tie_1000 = all(s["differ"] < half for s in at(1000))
        tie_5000 = all(s["differ"] < half for s in at(5000))
        verdict = {"statement": "at 100 temperature beats isotonic in >= 10 of 20 draws for >= 2 of 3 seeds; "
                                "at 1,000 and 5,000 the two differ in < 10 of 20 draws in every seed",
                   "seeds_where_temperature_wins_at_100": seeds_for,
                   "part_100_held": bool(seeds_for >= 2),
                   "tie_at_1000": bool(tie_1000), "tie_at_5000": bool(tie_5000),
                   "held": bool(seeds_for >= 2 and tie_1000 and tie_5000)}
        print(f"\n  PREDICTION 11  {'HELD' if verdict['held'] else 'FALSIFIED'}: temperature wins at 100 in "
              f"{seeds_for} of 3 seeds; tie at 1,000 {tie_1000}; tie at 5,000 {tie_5000}")

    path = os.path.join(args.out, "calibration_size_experiment.json")
    with open(path, "w") as fh:
        json.dump({"dataset": args.dataset, "timesteps": T, "n_bins": args.bins,
                   "draws": args.draws, "n_bootstrap": args.n_bootstrap,
                   "registered": "NOTES.md, prediction 11 and the tie definition, 2026-10-13",
                   "summary": summary, "prediction_11": verdict, "rows": rows}, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
