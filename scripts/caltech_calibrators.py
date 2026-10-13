#!/usr/bin/env python3
"""Calibrators on N-Caltech101: does the calibrator tie hold at 101 classes and 0.64 accuracy?

Every calibrator result so far is from N-MNIST: ten classes, about 0.99 accuracy and a
5,000-sample calibration set. N-Caltech101 is the regime where a tie could plausibly break:
101 classes, accuracy between 0.43 and 0.66, and a calibration set of 1,010, ten per class.

No training. The nine four-block count-readout models (``T`` in {2, 6, 10}, three seeds) are
reused, and the four calibrators are fitted on the calibration split, on CPU, exactly as the
N-MNIST sweep fits them. Top-label isotonic and temperature scaling are then compared with
the paired bootstrap under both estimators and the tie rule fixed in NOTES.md.

**Prediction 10**, recorded before any calibrator was fitted on this dataset: the two tie in
at least 7 of the 9 runs. The script evaluates it against that condition.

The accumulator and learned-scale models are calibrated too and reported, so that the
statement "no calibrator was ever fitted to an unbounded readout" stops being true; they
take no part in the prediction. One-versus-rest isotonic is reported and should not be
trusted here, with about ten calibration samples per class per one-versus-rest fit.

Writes ``results/caltech_calibrators.json``. Zero DataLoader workers.
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
    IsotonicOneVsRest, IsotonicTopLabel, TemperatureScaling, Uncalibrated, compute_metrics,
    discrimination, paired_verdict, quantise_confidence,
)
from spikecalib.experiments import all_cached, cell_outputs, load_data, stem  # noqa: E402
from spikecalib.selective import selective_summary  # noqa: E402
from spikecalib.train import resolve_device  # noqa: E402

ARMS = (("uncalibrated", Uncalibrated), ("temperature", TemperatureScaling),
        ("isotonic-top-label", IsotonicTopLabel), ("isotonic-one-vs-rest", IsotonicOneVsRest))
READOUTS = ("count", "accumulator", "scaled-count")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ncaltech101")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[2, 6, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()
    device = resolve_device()

    rows, paired = [], []
    print("=" * 112)
    print(f"CALIBRATORS ON {args.dataset}  T={args.timesteps}  seeds={args.seeds}  no training")
    print("=" * 112, flush=True)

    for T in args.timesteps:
        present = [(r, s, stem(args.dataset, T, s, r)) for r in READOUTS for s in args.seeds
                   if os.path.exists(os.path.join(args.out, stem(args.dataset, T, s, r) + "_weights.pt"))]
        data = load_data(args.dataset, T, args.data_root,
                         in_memory=not all_cached(args.out, [p[2] for p in present]))
        for readout, seed, name in present:
            out = cell_outputs(args.out, data, seed, device, readout)
            cal, test = out.cal, out.test
            fitted = {}
            for arm, cls in ARMS:
                calibrator = cls().fit(cal.conf, cal.correct, outputs=cal)
                conf = quantise_confidence(np.asarray(
                    calibrator.transform(test.conf, outputs=test), dtype=np.float64))
                fitted[arm] = conf
                m = compute_metrics(conf, test.correct, n_bins=args.bins,
                                    n_bootstrap=args.n_bootstrap, bootstrap_seed=seed)
                d = discrimination(conf, test.correct)
                sel = selective_summary(conf, test.correct)
                rows.append({
                    "dataset": args.dataset, "n_classes": data.spec.n_classes,
                    "readout": readout, "timesteps": T, "seed": seed, "arm": arm,
                    "logit_scale": out.scale, "split_fingerprint": out.split_fingerprint,
                    "n_cal": int(cal.n), "n_test": int(test.n),
                    "test_accuracy": float(test.correct.mean()),
                    "mean_confidence": float(conf.mean()),
                    "mean_conf_minus_acc": float(conf.mean() - test.correct.mean()),
                    "ece_equal_width": m.ece_equal_width, "ece_equal_mass": m.ece_equal_mass,
                    "ece_equal_width_ci": list(m.ece_equal_width_ci),
                    "ece_equal_mass_ci": list(m.ece_equal_mass_ci),
                    "nll": m.nll, "brier": m.brier,
                    "n_distinct_confidences": m.n_distinct_confidences,
                    "equal_mass_bins_effective": m.equal_mass_bins_effective,
                    "temperature": getattr(calibrator, "temperature_", None),
                    "auroc": d["auroc"], "aurc": sel["aurc"],
                    "n_bootstrap": args.n_bootstrap, "bootstrap_seed": seed,
                })
            # a = isotonic, b = temperature, as in the N-MNIST sweep: negative favours isotonic
            v = paired_verdict(fitted["isotonic-top-label"], fitted["temperature"], test.correct,
                               n_bins=args.bins, n_bootstrap=args.n_bootstrap, seed=seed)
            ew, em = v["equal_width"], v["equal_mass"]
            paired.append({
                "dataset": args.dataset, "readout": readout, "timesteps": T, "seed": seed,
                "split_fingerprint": out.split_fingerprint,
                "iso_minus_temp_ece_ew": ew["difference"], "ew_ci": list(ew["ci"]),
                "ew_excludes_zero": ew["resolvable"],
                "iso_minus_temp_ece_em": em["difference"], "em_ci": list(em["ci"]),
                "em_excludes_zero": em["resolvable"],
                "differ": v["differ"], "tie": v["tie"],
                "favours": {"a": "isotonic", "b": "temperature", None: None}[v["favours"]],
            })
            unc = next(r for r in rows[-4:] if r["arm"] == "uncalibrated")
            tmp = next(r for r in rows[-4:] if r["arm"] == "temperature")
            iso = next(r for r in rows[-4:] if r["arm"] == "isotonic-top-label")
            print(f"  T={T:<2} seed {seed} {readout:<12} acc {unc['test_accuracy']:.4f}  "
                  f"ECEew unc {unc['ece_equal_width']:.4f} temp {tmp['ece_equal_width']:.4f} "
                  f"iso {iso['ece_equal_width']:.4f} | ECEem temp {tmp['ece_equal_mass']:.4f} "
                  f"iso {iso['ece_equal_mass']:.4f} | t {tmp['temperature']:.3f} | "
                  f"ew [{ew['ci'][0]:+.4f}, {ew['ci'][1]:+.4f}] em [{em['ci'][0]:+.4f}, {em['ci'][1]:+.4f}] "
                  f"{'TIE' if v['tie'] else 'DIFFER, favours ' + paired[-1]['favours']}", flush=True)

    count = [p for p in paired if p["readout"] == "count"]
    verdict = {}
    if len(count) == 9:
        ties = sum(p["tie"] for p in count)
        verdict = {"statement": "N-Caltech101 count readout: calibrators tie in at least 7 of 9 runs",
                   "ties": ties, "runs": len(count),
                   "differ": [{k: p[k] for k in ("timesteps", "seed", "favours")} for p in count if p["differ"]],
                   "ew_excludes_zero": sum(p["ew_excludes_zero"] for p in count),
                   "em_excludes_zero": sum(p["em_excludes_zero"] for p in count),
                   "held": bool(ties >= 7)}
        print(f"\n  PREDICTION 10  {'HELD' if verdict['held'] else 'FALSIFIED'}: {ties} of 9 count runs tie "
              f"(intervals excluding zero: {verdict['ew_excludes_zero']} equal-width, "
              f"{verdict['em_excludes_zero']} equal-mass)")

    path = os.path.join(args.out, "caltech_calibrators.json")
    with open(path, "w") as fh:
        json.dump({"dataset": args.dataset, "n_bins": args.bins,
                   "registered": "NOTES.md, prediction 10 and the tie definition, 2026-10-13",
                   "tie_rule": "differ only if the paired 95% interval excludes zero under both "
                               "estimators in the same direction",
                   "rows": rows, "paired": paired, "prediction_10": verdict}, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
