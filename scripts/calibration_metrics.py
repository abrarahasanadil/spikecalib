#!/usr/bin/env python3
"""A3 and A4, added after review, not pre-registered: equivalence tests, run-level metrics,
proper scoring rules, headline intervals and the data behind the reliability diagrams.

For every saved model output (count, accumulator and scaled-count readouts on N-MNIST and
N-Caltech101; 75 runs) the calibrators are refitted on the run's own calibration split with
the existing code, exactly as the published results fitted them, and the script stops unless
the refit reproduces every published ECE to within 1e-9. Then:

- A3: for every run on which the paired bootstrap of isotonic minus temperature ECE was
  computed (N-MNIST: 15 count runs; N-Caltech101: 9 count, 9 accumulator and 6 scaled-count
  runs), both estimators: the same 1,000 resamples are regenerated, the stored 95% interval is
  reproduced to within 1e-12, and two one-sided tests at margins 0.005, 0.01 and 0.02 are read
  from the 90% interval.
- A4: per run and calibrator, accuracy, both ECEs, ECE grouped by distinct value, equal-mass
  occupancy, top-label Brier score and log loss, AUROC; multiclass NLL and Brier for the
  uncalibrated and temperature-scaled distributions; clip counts. Seed means with standard
  deviations and test-set bootstrap 95% intervals for the headline ECEs and AUROCs.
- Reliability-diagram data for seed 0 of the count readout at T = 2 and T = 10 on both
  datasets, per distinct confidence value with Wilson 95% intervals, and in equal-mass bins.

Settings fixed in NOTES.md before this script was run. Writes ``results/calibration_metrics.json``
and ``results/reliability_diagrams.json``. Reads only cached outputs.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicTopLabel, TemperatureScaling, discrimination, ece_equal_mass, ece_equal_width,
    model_outputs, quantise_confidence, softmax_probs,
)
from spikecalib.floor import bin_assignment, ece_distinct, signed_bin_gaps  # noqa: E402
from spikecalib.scoring import (  # noqa: E402
    multiclass_scores, paired_difference_draws, top_label_scores, tost_equivalent, wilson_interval,
)

R = os.path.join(REPO_ROOT, "results")
N_BINS, N_BOOT = 15, 1000
DELTAS = (0.005, 0.01, 0.02)
ARMS = ("uncalibrated", "temperature", "isotonic-top-label")
DEFAULT_BLOCKS = {"nmnist": 2, "ncaltech101": 4}
NAME = re.compile(r"^(nmnist|ncaltech101)_T(\d+)_seed(\d)(?:_(accumulator|scaled-count))?"
                  r"(?:_arch([\d-]+))?_outputs\.npz$")


def runs_on_disk() -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(R, "*_outputs.npz"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue  # the A7 checkpoints and anything else not of this form
        ds, T, seed, readout, arch = m.groups()
        blocks = len(arch.split("-")) if arch else DEFAULT_BLOCKS[ds]
        out.append({"path": path, "name": os.path.basename(path)[:-len("_outputs.npz")],
                    "dataset": ds, "timesteps": int(T), "seed": int(seed),
                    "readout": readout or "count", "n_blocks": blocks})
    return out


def load_published():
    sweep = {}
    with open(os.path.join(R, "sweep_nmnist.csv")) as fh:
        for r in csv.DictReader(fh):
            sweep[(int(r["timesteps"]), int(r["seed"]), r["arm"])] = r
    cc = json.load(open(os.path.join(R, "caltech_calibrators.json")))
    cc_rows = {(r["readout"], r["timesteps"], r["seed"], r["arm"]): r for r in cc["rows"]}
    cc_pair = {(r["readout"], r["timesteps"], r["seed"]): r for r in cc["paired"]}
    unc = {}
    for r in json.load(open(os.path.join(R, "readout_experiment.json")))["rows"]:
        unc[("nmnist", 2, r["readout"], r["timesteps"], r["seed"])] = r
    for r in json.load(open(os.path.join(R, "arch_matched.json")))["rows"]:
        unc[(r["dataset"], r["n_blocks"], r["readout"], r["timesteps"], r["seed"])] = r
    for r in json.load(open(os.path.join(R, "scaled_count_experiment.json")))["rows"]:
        unc[(r["dataset"], r["n_blocks"], "scaled-count", r["timesteps"], r["seed"])] = r
    return sweep, cc_rows, cc_pair, unc


def calibrate(run: dict) -> dict:
    z = np.load(run["path"])
    cal = model_outputs(z["cal_scores"], z["cal_labels"], np.random.default_rng(run["seed"]))
    test = model_outputs(z["test_scores"], z["test_labels"], np.random.default_rng(run["seed"] + 1))
    ts = TemperatureScaling().fit(cal.conf, cal.correct, outputs=cal)
    iso = IsotonicTopLabel().fit(cal.conf, cal.correct)
    conf = {"uncalibrated": test.conf,
            "temperature": ts.transform(test.conf, outputs=test),
            "isotonic-top-label": iso.transform(test.conf)}
    # The N-MNIST sweep scored calibrated confidences unquantised; every later script
    # quantised them. Each run is rebuilt under the convention its published numbers used.
    sweep_style = run["dataset"] == "nmnist" and run["n_blocks"] == 2 and run["readout"] == "count"
    if not sweep_style:
        conf = {k: quantise_confidence(v) for k, v in conf.items()}
    correct = (test.predictions == test.labels).astype(np.float64)
    probs = {"uncalibrated": softmax_probs(test.counts),
             "temperature": softmax_probs(test.counts / ts.temperature_)}
    return {"conf": conf, "correct": correct, "labels": test.labels, "probs": probs,
            "temperature": float(ts.temperature_), "n_test": int(test.n), "n_cal": int(cal.n)}


def check_published(run, fit, ece, sweep, cc_rows, unc) -> str:
    key = (run["dataset"], run["n_blocks"], run["readout"], run["timesteps"], run["seed"])
    if run["dataset"] == "nmnist" and run["n_blocks"] == 2 and run["readout"] == "count":
        for arm in ARMS:
            r = sweep[(run["timesteps"], run["seed"], arm)]
            for est in ("equal_width", "equal_mass"):
                assert abs(ece[arm][est] - float(r["ece_" + est])) < 1e-9, (run["name"], arm, est)
        assert abs(fit["temperature"] - float(sweep[(run["timesteps"], run["seed"], "temperature")]["temperature"])) < 1e-9
        return "results/sweep_nmnist.csv, three arms"
    if run["dataset"] == "ncaltech101" and run["n_blocks"] == 4:
        for arm in ARMS:
            r = cc_rows[(run["readout"], run["timesteps"], run["seed"], arm)]
            for est in ("equal_width", "equal_mass"):
                assert abs(ece[arm][est] - r["ece_" + est]) < 1e-9, (run["name"], arm, est)
        return "results/caltech_calibrators.json, three arms"
    r = unc[key]
    for est in ("equal_width", "equal_mass"):
        assert abs(ece["uncalibrated"][est] - r["ece_" + est]) < 1e-9, (run["name"], est)
    return "uncalibrated only (no calibrated values were published for this run)"


def describe(v):
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v)}


def seed_mean_ci(per_seed_draws: list) -> list:
    m = np.mean(np.vstack(per_seed_draws), axis=0)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ece_draws(conf, correct, seed, est) -> np.ndarray:
    f = ece_equal_width if est == "equal_width" else ece_equal_mass
    rng = np.random.default_rng(seed)
    n = conf.size
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        out[i] = f(conf[idx], correct[idx], N_BINS)
    return out


def reliability(conf, correct) -> dict:
    vals = quantise_confidence(conf)
    per_value = []
    for v in np.unique(vals):
        m = vals == v
        n, k = int(m.sum()), int(correct[m].sum())
        lo, hi = wilson_interval(k, n)
        per_value.append({"confidence": float(conf[m].mean()), "n": n, "accuracy": k / n,
                          "wilson_low": lo, "wilson_high": hi})
    g = signed_bin_gaps(conf, correct, "equal_mass", N_BINS)
    bins = []
    for n, c, a in zip(g["n"], g["mean_confidence"], g["accuracy"]):
        lo, hi = wilson_interval(int(round(a * n)), int(n))
        bins.append({"confidence": float(c), "n": int(n), "accuracy": float(a),
                     "wilson_low": lo, "wilson_high": hi})
    return {"per_value": per_value, "equal_mass": bins}


def main() -> int:
    sweep, cc_rows, cc_pair, unc = load_published()
    runs = runs_on_disk()
    assert len(runs) == 75, len(runs)
    rows, tost, fits = [], [], {}
    for run in runs:
        fit = calibrate(run)
        fits[run["name"]] = fit
        ece = {arm: {"equal_width": ece_equal_width(fit["conf"][arm], fit["correct"], N_BINS),
                     "equal_mass": ece_equal_mass(fit["conf"][arm], fit["correct"], N_BINS)}
               for arm in ARMS}
        checked = check_published(run, fit, ece, sweep, cc_rows, unc)
        for arm in ARMS:
            conf, correct = fit["conf"][arm], fit["correct"]
            idx, n_eff = bin_assignment(conf, "equal_mass", N_BINS)
            occ = np.bincount(idx)
            occ = occ[occ > 0]
            row = {k: run[k] for k in ("name", "dataset", "readout", "n_blocks", "timesteps", "seed")}
            row.update({
                "arm": arm, "n_test": fit["n_test"], "n_cal": fit["n_cal"],
                "temperature": fit["temperature"] if arm == "temperature" else None,
                "accuracy": float(correct.mean()), "mean_confidence": float(conf.mean()),
                "ece_equal_width": ece[arm]["equal_width"], "ece_equal_mass": ece[arm]["equal_mass"],
                "ece_distinct": ece_distinct(conf, correct),
                "n_distinct": int(np.unique(quantise_confidence(conf)).size),
                "equal_mass_bins": int(n_eff), "equal_mass_min_occupancy": int(occ.min()),
                "equal_mass_max_occupancy": int(occ.max()),
                "auroc": discrimination(conf, correct)["auroc"],
                **top_label_scores(conf, correct), "published_check": checked,
            })
            if arm in fit["probs"]:
                row.update(multiclass_scores(fit["probs"][arm], fit["labels"]))
            rows.append(row)

        # A3: TOST on the runs that carry a published paired interval.
        stored = None
        if run["dataset"] == "nmnist" and run["n_blocks"] == 2 and run["readout"] == "count":
            r = sweep[(run["timesteps"], run["seed"], "isotonic-top-label")]
            stored = {"equal_width": (float(r["iso_minus_temp_ci_low"]), float(r["iso_minus_temp_ci_high"])),
                      "equal_mass": (float(r["iso_minus_temp_em_ci_low"]), float(r["iso_minus_temp_em_ci_high"]))}
        elif run["dataset"] == "ncaltech101" and run["n_blocks"] == 4:
            p = cc_pair[(run["readout"], run["timesteps"], run["seed"])]
            stored = {"equal_width": tuple(p["ew_ci"]), "equal_mass": tuple(p["em_ci"])}
        if stored is not None:
            for est in ("equal_width", "equal_mass"):
                point, draws = paired_difference_draws(
                    fit["conf"]["isotonic-top-label"], fit["conf"]["temperature"], fit["correct"],
                    seed=run["seed"], n_bins=N_BINS, n_bootstrap=N_BOOT, estimator=est)
                lo95, hi95 = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
                assert abs(lo95 - stored[est][0]) < 1e-12 and abs(hi95 - stored[est][1]) < 1e-12, \
                    (run["name"], est, (lo95, hi95), stored[est])
                entry = {k: run[k] for k in ("name", "dataset", "readout", "n_blocks", "timesteps", "seed")}
                entry.update({"estimator": est, "difference": point, "ci95": [lo95, hi95],
                              "excludes_zero_95": bool(lo95 > 0 or hi95 < 0)})
                for d in DELTAS:
                    ok, (lo90, hi90) = tost_equivalent(draws, d)
                    entry["ci90"] = [lo90, hi90]
                    entry[f"equivalent_{d}"] = ok
                tost.append(entry)
        print(f"{run['name']:<48} ok  ({checked})", flush=True)

    # Per-run TOST under both estimators.
    by_run = {}
    for e in tost:
        by_run.setdefault((e["name"]), {})[e["estimator"]] = e
    tost_summary = []
    for group, sel in (("nmnist count", lambda e: e["dataset"] == "nmnist"),
                       ("ncaltech101 count", lambda e: e["dataset"] == "ncaltech101" and e["readout"] == "count"),
                       ("ncaltech101 accumulator", lambda e: e["dataset"] == "ncaltech101" and e["readout"] == "accumulator"),
                       ("ncaltech101 scaled-count", lambda e: e["dataset"] == "ncaltech101" and e["readout"] == "scaled-count")):
        names = sorted({e["name"] for e in tost if sel(e)})
        s = {"group": group, "runs": len(names)}
        for d in DELTAS:
            for est in ("equal_width", "equal_mass"):
                s[f"equivalent_{est}_{d}"] = sum(by_run[n][est][f"equivalent_{d}"] for n in names)
            s[f"equivalent_both_{d}"] = sum(by_run[n]["equal_width"][f"equivalent_{d}"]
                                            and by_run[n]["equal_mass"][f"equivalent_{d}"] for n in names)
        s["differ_registered_rule"] = int(sum(
            by_run[n]["equal_width"]["excludes_zero_95"] and by_run[n]["equal_mass"]["excludes_zero_95"]
            and np.sign(by_run[n]["equal_width"]["difference"]) == np.sign(by_run[n]["equal_mass"]["difference"])
            for n in names))
        tost_summary.append(s)

    # Headline intervals: seed means of ECE per calibrator, count readout, main depths.
    headline = {"ece": [], "auroc": {}}
    for ds, Ts in (("nmnist", (2, 4, 6, 8, 10)), ("ncaltech101", (2, 6, 10))):
        for T in Ts:
            names = [f"{ds}_T{T}_seed{s}" for s in (0, 1, 2)]
            for arm in ARMS:
                for est in ("equal_width", "equal_mass"):
                    vals = [next(r for r in rows if r["name"] == n and r["arm"] == arm)["ece_" + est] for n in names]
                    draws = [ece_draws(fits[n]["conf"][arm], fits[n]["correct"], s, est) for s, n in enumerate(names)]
                    headline["ece"].append({"dataset": ds, "timesteps": T, "arm": arm, "estimator": est,
                                            **describe(vals), "ci95_seed_mean": seed_mean_ci(draws)})
    # AUROC, N-MNIST T = 2, count against accumulator, uncalibrated, paired on the same test images.
    cnt, acc, dif = [], [], []
    vals_c, vals_a = [], []
    for s in (0, 1, 2):
        fc, fa = fits[f"nmnist_T2_seed{s}"], fits[f"nmnist_T2_seed{s}_accumulator"]
        assert np.array_equal(fc["labels"], fa["labels"])
        cc_, ca_ = fc["conf"]["uncalibrated"], fa["conf"]["uncalibrated"]
        yc, ya = fc["correct"], fa["correct"]
        vals_c.append(roc_auc_score(yc, cc_))
        vals_a.append(roc_auc_score(ya, ca_))
        rng = np.random.default_rng(s)
        dc, da = np.empty(N_BOOT), np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, yc.size, size=yc.size)
            dc[i] = roc_auc_score(yc[idx], cc_[idx])
            da[i] = roc_auc_score(ya[idx], ca_[idx])
        cnt.append(dc), acc.append(da), dif.append(dc - da)
    headline["auroc"] = {
        "count": {**describe(vals_c), "ci95_seed_mean": seed_mean_ci(cnt)},
        "accumulator": {**describe(vals_a), "ci95_seed_mean": seed_mean_ci(acc)},
        "count_minus_accumulator": {**describe([c - a for c, a in zip(vals_c, vals_a)]),
                                    "ci95_seed_mean": seed_mean_ci(dif)},
        "note": "N-MNIST, two blocks, T = 2, uncalibrated; the accumulator's confidences are scored on the same "
                "resampled test images as the count readout's",
    }

    clip = {"top_label": sum(r["n_clipped_top_label"] for r in rows),
            "multiclass": sum(r.get("n_clipped_multiclass", 0) for r in rows),
            "scores_computed_top_label": sum(r["n_test"] for r in rows),
            "scores_computed_multiclass": sum(r["n_test"] for r in rows if "nll_multiclass" in r)}
    doc = {"analysis": "A3 and A4, added after review, not pre-registered",
           "thresholds": "NOTES.md, Post-review analyses, thresholds fixed 2026-11-12 before computation",
           "n_bins": N_BINS, "n_bootstrap": N_BOOT, "deltas": list(DELTAS), "clip": 1e-12,
           "clipped": clip, "rows": rows, "tost": tost, "tost_summary": tost_summary,
           "headline": headline}
    with open(os.path.join(R, "calibration_metrics.json"), "w") as fh:
        json.dump(doc, fh, indent=1, default=lambda o: o.item())

    rel = {"note": "seed 0, count readout, main depth; per distinct value and in equal-mass bins; "
                   "Wilson 95% intervals", "panels": []}
    for ds in ("nmnist", "ncaltech101"):
        for T in (2, 10):
            fit = fits[f"{ds}_T{T}_seed0"]
            for arm in ARMS:
                rel["panels"].append({"dataset": ds, "timesteps": T, "arm": arm,
                                      "accuracy": float(fit["correct"].mean()),
                                      **reliability(fit["conf"][arm], fit["correct"])})
    with open(os.path.join(R, "reliability_diagrams.json"), "w") as fh:
        json.dump(rel, fh, indent=1, default=lambda o: o.item())
    for s in tost_summary:
        print(s)
    print("clipped:", clip)
    print("AUROC:", {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci95_seed_mean"]])
                     for k, v in headline["auroc"].items() if isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
