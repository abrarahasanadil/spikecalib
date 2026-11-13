#!/usr/bin/env python3
"""N-Caltech101 calibrators refitted to the test class distribution (S1), and the best any
single temperature can do on the test set (S2).

Added after a second internal review, not pre-registered. Settings: NOTES.md, "Second internal
review, settings fixed 2026-11-13 before computation", S1 and S2.

The N-Caltech101 calibration split holds ten samples of every class, while the test split is
proportional to class size with a floor. S1 refits temperature scaling (weighted negative
log-likelihood) and top-label isotonic regression (sample weights) with every calibration sample
weighted by the ratio of its class's test share to its calibration share, and scores both on the
unchanged test set. The unweighted refit is computed beside it and must reproduce the stored A4
values. S2 scans single temperatures on the test set itself; it uses the test labels, so it is a
bound on what one temperature could reach there, not a calibrator.

CPU only, from the stored ``*_outputs.npz``. Writes ``results/reweighted_calibration.json``.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    IsotonicTopLabel, TemperatureScaling, ece_equal_mass, ece_equal_width, model_outputs,
    paired_verdict, quantise_confidence, softmax_probs,
)

R = os.path.join(REPO_ROOT, "results")
N_BINS, N_BOOT, N_CLASSES = 15, 1000, 101
NAME = re.compile(r"^ncaltech101_T(\d+)_seed(\d)(?:_(accumulator|scaled-count))?(?:_arch([\d-]+))?_outputs\.npz$")
GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 2001))


def class_weights(cal_labels: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
    """w_i = pi_test(y_i) / pi_cal(y_i), normalised to mean one over the calibration set."""
    pt = np.bincount(test_labels, minlength=N_CLASSES) / test_labels.size
    pc = np.bincount(cal_labels, minlength=N_CLASSES) / cal_labels.size
    assert np.all(pc > 0)
    w = pt[cal_labels] / pc[cal_labels]
    return w / w.mean()


def weighted_temperature(logits: np.ndarray, labels: np.ndarray, w: np.ndarray, bounds=(1e-3, 1e3)) -> float:
    """The temperature minimising the weighted mean NLL, by the search TemperatureScaling uses."""
    logits = np.asarray(logits, dtype=np.float64)

    def nll(log_t):
        z = logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        lp = z[np.arange(labels.size), labels] - np.log(np.exp(z).sum(axis=1))
        return float(-(w * lp).sum() / w.sum())

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    assert res.success
    return float(np.exp(res.x))


def scores(conf, correct) -> dict:
    return {"ece_equal_width": ece_equal_width(conf, correct, N_BINS),
            "ece_equal_mass": ece_equal_mass(conf, correct, N_BINS),
            "mean_conf_minus_acc": float(conf.mean() - correct.mean()),
            "brier_top_label": float(np.mean((conf - correct) ** 2))}


def verdict(iso, temp, correct, seed) -> dict:
    v = paired_verdict(iso, temp, correct, n_bins=N_BINS, n_bootstrap=N_BOOT, seed=seed)
    return {"ew_difference": v["equal_width"]["difference"], "ew_ci": list(v["equal_width"]["ci"]),
            "em_difference": v["equal_mass"]["difference"], "em_ci": list(v["equal_mass"]["ci"]),
            "differ": v["differ"], "favours": {"a": "isotonic", "b": "temperature", None: None}[v["favours"]]}


def temperature_scan(test, correct) -> dict:
    """S2: equal-width and equal-mass ECE of the count logits divided by each temperature."""
    rows = np.arange(test.n)
    out = {"equal_width": [], "equal_mass": [], "mean_conf_minus_acc": []}
    for t in GRID:
        conf = quantise_confidence(softmax_probs(test.counts / t)[rows, test.predictions])
        out["equal_width"].append(ece_equal_width(conf, correct, N_BINS))
        out["equal_mass"].append(ece_equal_mass(conf, correct, N_BINS))
        out["mean_conf_minus_acc"].append(float(conf.mean() - correct.mean()))
    res = {}
    for est in ("equal_width", "equal_mass"):
        e = np.asarray(out[est])
        i = int(np.argmin(e))
        res[est] = {"min_ece": float(e[i]), "at_temperature": float(GRID[i])}

    def gap(log_t):
        conf = quantise_confidence(softmax_probs(test.counts / np.exp(log_t))[rows, test.predictions])
        return float(conf.mean() - correct.mean())

    lo, hi = np.log(GRID[0]), np.log(GRID[-1])
    assert gap(lo) > 0 > gap(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    t_match = float(np.exp(0.5 * (lo + hi)))
    conf = quantise_confidence(softmax_probs(test.counts / t_match)[rows, test.predictions])
    res["mean_matched"] = {"temperature": t_match, "ece_equal_width": ece_equal_width(conf, correct, N_BINS),
                           "ece_equal_mass": ece_equal_mass(conf, correct, N_BINS),
                           "mean_conf_minus_acc": float(conf.mean() - correct.mean())}
    return res


def main() -> int:
    stored = {r["name"] + "|" + r["arm"]: r
              for r in json.load(open(os.path.join(R, "calibration_metrics.json")))["rows"]}
    cc_pair = {(p["readout"], p["timesteps"], p["seed"]): p
               for p in json.load(open(os.path.join(R, "caltech_calibrators.json")))["paired"]}
    runs = []
    for fname in sorted(os.listdir(R)):
        m = NAME.match(fname)
        if not m:
            continue
        T, seed, readout, arch = m.groups()
        name = fname[: -len("_outputs.npz")]
        T, seed = int(T), int(seed)
        readout = readout or "count"
        blocks = len(arch.split("-")) if arch else 4
        z = np.load(os.path.join(R, fname))
        cal = model_outputs(z["cal_scores"], z["cal_labels"], np.random.default_rng(seed))
        test = model_outputs(z["test_scores"], z["test_labels"], np.random.default_rng(seed + 1))
        correct = (test.predictions == test.labels).astype(np.float64)
        w = class_weights(cal.labels, test.labels)

        # Unweighted refit, as in A4; it must reproduce the stored values.
        ts = TemperatureScaling().fit(cal.conf, cal.correct, outputs=cal)
        iso = IsotonicTopLabel().fit(cal.conf, cal.correct)
        unw = {"uncalibrated": quantise_confidence(test.conf),
               "temperature": quantise_confidence(ts.transform(test.conf, outputs=test)),
               "isotonic-top-label": quantise_confidence(iso.transform(test.conf))}
        for arm, conf in unw.items():
            ref = stored[name + "|" + arm]
            got = scores(conf, correct)
            for k in ("ece_equal_width", "ece_equal_mass"):
                assert abs(got[k] - ref[k]) < 1e-9, (name, arm, k, got[k], ref[k])
            assert abs(got["mean_conf_minus_acc"] - (ref["mean_confidence"] - ref["accuracy"])) < 1e-9, (name, arm)
        # A weight of one everywhere must give back the unweighted temperature.
        assert abs(weighted_temperature(cal.counts, cal.labels, np.ones(cal.n)) - ts.temperature_) < 1e-6 * ts.temperature_

        # Reweighted refit (S1).
        t_w = weighted_temperature(cal.counts, cal.labels, w)
        iso_w = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
            cal.conf, cal.correct.astype(np.float64), sample_weight=w)
        rows_ = np.arange(test.n)
        rew = {"temperature": quantise_confidence(softmax_probs(test.counts / t_w)[rows_, test.predictions]),
               "isotonic-top-label": quantise_confidence(np.asarray(iso_w.transform(test.conf), dtype=np.float64))}

        run = {"name": name, "readout": readout, "n_blocks": blocks, "timesteps": T, "seed": seed,
               "n_cal": int(cal.n), "n_test": int(test.n), "accuracy": float(correct.mean()),
               "weights": {"min": float(w.min()), "max": float(w.max())},
               "temperature_unweighted": float(ts.temperature_), "temperature_reweighted": t_w,
               "cal_accuracy": float(cal.correct.mean()),
               "cal_accuracy_reweighted": float((w * cal.correct).sum() / w.sum()),
               "unweighted": {arm: scores(c, correct) for arm, c in unw.items()},
               "reweighted": {arm: scores(c, correct) for arm, c in rew.items()},
               "paired_unweighted": verdict(unw["isotonic-top-label"], unw["temperature"], correct, seed),
               "paired_reweighted": verdict(rew["isotonic-top-label"], rew["temperature"], correct, seed)}
        if blocks == 4:
            p = cc_pair[(readout, T, seed)]
            assert run["paired_unweighted"]["differ"] == p["differ"], name
            assert run["paired_unweighted"]["favours"] == p["favours"], name
            assert np.allclose(run["paired_unweighted"]["ew_ci"], p["ew_ci"], atol=1e-12), name
        if readout == "count" and blocks == 4:
            run["scan"] = temperature_scan(test, correct)       # S2
        runs.append(run)
        u, r = run["unweighted"], run["reweighted"]
        print(f"{name:<48} t {ts.temperature_:.3f}->{t_w:.3f} | ECEw temp {u['temperature']['ece_equal_width']:.4f}"
              f"->{r['temperature']['ece_equal_width']:.4f} iso {u['isotonic-top-label']['ece_equal_width']:.4f}"
              f"->{r['isotonic-top-label']['ece_equal_width']:.4f} | "
              f"{run['paired_unweighted']['favours'] if run['paired_unweighted']['differ'] else 'n.r.'}"
              f"->{run['paired_reweighted']['favours'] if run['paired_reweighted']['differ'] else 'n.r.'}"
              + (f" | scan min {run['scan']['equal_width']['min_ece']:.4f}" if "scan" in run else ""), flush=True)

    summary = []
    for readout in ("count", "accumulator", "scaled-count"):
        sel = [r for r in runs if r["readout"] == readout and r["n_blocks"] == 4]
        entry = {"readout": readout, "runs": len(sel)}
        for kind in ("unweighted", "reweighted"):
            pv = [r[f"paired_{kind}"] for r in sel]
            entry[kind] = {
                "differ": sum(v["differ"] for v in pv),
                "favour_isotonic": sum(v["differ"] and v["favours"] == "isotonic" for v in pv),
                "favour_temperature": sum(v["differ"] and v["favours"] == "temperature" for v in pv),
                "underconfident_calibrated": sum(r[kind][arm]["mean_conf_minus_acc"] < 0 for r in sel
                                                 for arm in ("temperature", "isotonic-top-label")),
                "calibrated_runs": 2 * len(sel),
            }
        summary.append(entry)
    doc = {"analysis": "S1 and S2, added after a second internal review, not pre-registered",
           "settings": "NOTES.md, Second internal review, settings fixed 2026-11-13 before computation",
           "n_bins": N_BINS, "n_bootstrap": N_BOOT,
           "temperature_grid": {"min": float(GRID[0]), "max": float(GRID[-1]), "points": int(GRID.size)},
           "summary": summary, "runs": runs}
    with open(os.path.join(R, "reweighted_calibration.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    for s in summary:
        print(s)
    print("wrote results/reweighted_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
