#!/usr/bin/env python3
"""N-MNIST intervals on seed means with one shared resample of the test set (S3).

Added after a second internal review, not pre-registered. Settings: NOTES.md, "Second internal
review, settings fixed 2026-11-13 before computation", S3.

Every N-MNIST seed is tested on the same 10,000 official test samples in the same order. The
intervals on seed means of A1, A4 and A8 drew each seed's resamples from its own generator,
``default_rng(seed)``, which treats one test set as three. Here resample ``b`` draws one index
vector, the ``b``-th draw of ``default_rng(0)``, and applies it to all three seeds. The
independent construction is recomputed beside it and must reproduce the stored intervals.

Covers the floor ratio of every binding N-MNIST cell (A1; equal-width, equal-mass and
distinct-value), the ECE of the count readout per calibrator and estimator at T = 2, 6 and 10
(A4), the AUROC of the count readout, the separately trained accumulator and their difference at
T = 2 (A4), and the AUROC difference between the count readout and the accumulator reading of
the same weights (A8; the three networks are run again on MPS and must reproduce the stored
counts). Writes ``results/shared_resample_intervals.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import calibration_metrics as CM  # noqa: E402
import ece_floor as EF  # noqa: E402
from spikecalib.conditional import theoretical_ceiling  # noqa: E402

R = os.path.join(REPO_ROOT, "results")
N_BOOT, N_BINS, SEEDS = 1000, 15, (0, 1, 2)


def index_streams(n: int) -> dict:
    """The index vectors of each seed's own generator; seed 0's stream is the shared one."""
    out = {}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        out[s] = np.vstack([rng.integers(0, n, size=n) for _ in range(N_BOOT)])
    return out


def interval(per_seed: list) -> list:
    m = np.mean(np.vstack(per_seed), axis=0)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def both(stat, data: dict, streams: dict) -> dict:
    """Seed-mean intervals under the independent and the shared construction.

    ``data[s]`` holds seed s's arrays; ``stat(arrays, idx)`` is the statistic on one resample."""
    indep = [np.array([stat(data[s], streams[s][b]) for b in range(N_BOOT)]) for s in SEEDS]
    shared = [np.array([stat(data[s], streams[0][b]) for b in range(N_BOOT)]) for s in SEEDS]
    return {"independent": interval(indep), "shared": interval(shared)}


def main() -> int:
    labels = [np.load(os.path.join(R, f"nmnist_T2_seed{s}_outputs.npz"))["test_labels"] for s in SEEDS]
    assert all(np.array_equal(labels[0], x) for x in labels), "the N-MNIST seeds do not share one test set"
    n = labels[0].size
    streams = index_streams(n)
    doc = {"analysis": "S3, added after a second internal review, not pre-registered",
           "settings": "NOTES.md, Second internal review, settings fixed 2026-11-13 before computation",
           "n_bootstrap": N_BOOT, "n_test": int(n),
           "construction": "resample b applies the b-th index vector of numpy default_rng(0) to all three seeds",
           "floor_ratio": [], "ece": [], "auroc": {}, "same_weights": {}}

    # A1: floor ratio of every binding N-MNIST cell.
    stored_floor = json.load(open(os.path.join(R, "ece_floor.json")))["cells"]
    for c in stored_floor:
        if c["dataset"] != "nmnist" or c["binds_in"] != 3:
            continue
        suffix = "" if c["channels"] == "32-64" else "_arch" + c["channels"]
        cap = theoretical_ceiling(c["timesteps"], 10)
        data = {}
        for s in SEEDS:
            _, conf, correct = EF.test_set("nmnist", suffix, c["timesteps"], s)
            assert np.array_equal(np.load(os.path.join(R, f"nmnist_T{c['timesteps']}_seed{s}{suffix}_outputs.npz"))
                                  ["test_labels"], labels[0])
            data[s] = (conf, correct)
        entry = {"channels": c["channels"], "timesteps": c["timesteps"], "estimators": {}}
        for est, f in EF.ESTIMATORS.items():
            r = both(lambda d, idx: (d[1][idx].mean() - cap) / f(d[0][idx], d[1][idx], N_BINS), data, streams)
            ref = c["schemes"][est]["floor_ratio_ci95"]
            assert max(abs(a - b) for a, b in zip(r["independent"], ref)) < 1e-12, (c["channels"], c["timesteps"], est)
            r["ratio_mean"] = c["schemes"][est]["floor_ratio"]["mean"]
            entry["estimators"][est] = r
        doc["floor_ratio"].append(entry)
        print("floor", c["channels"], c["timesteps"],
              {e: [round(x, 4) for x in v["shared"]] for e, v in entry["estimators"].items()}, flush=True)

    # A4: ECE of the count readout per calibrator, two blocks, T = 2, 6 and 10.
    cm = json.load(open(os.path.join(R, "calibration_metrics.json")))
    stored_ece = {(e["timesteps"], e["arm"], e["estimator"]): e for e in cm["headline"]["ece"] if e["dataset"] == "nmnist"}
    runs = {r["name"]: r for r in CM.runs_on_disk()}
    for T in (2, 6, 10):
        fits = {s: CM.calibrate(runs[f"nmnist_T{T}_seed{s}"]) for s in SEEDS}
        for arm in CM.ARMS:
            for est, f in (("equal_width", CM.ece_equal_width), ("equal_mass", CM.ece_equal_mass)):
                data = {s: (fits[s]["conf"][arm], fits[s]["correct"]) for s in SEEDS}
                r = both(lambda d, idx: f(d[0][idx], d[1][idx], N_BINS), data, streams)
                ref = stored_ece[(T, arm, est)]
                assert max(abs(a - b) for a, b in zip(r["independent"], ref["ci95_seed_mean"])) < 1e-12, (T, arm, est)
                doc["ece"].append({"timesteps": T, "arm": arm, "estimator": est, "mean": ref["mean"], **r})
        print("ece", T, flush=True)

    # A4: AUROC at T = 2, count against the separately trained accumulator.
    fc = {s: CM.calibrate(runs[f"nmnist_T2_seed{s}"]) for s in SEEDS}
    fa = {s: CM.calibrate(runs[f"nmnist_T2_seed{s}_accumulator"]) for s in SEEDS}
    data = {s: (fc[s]["conf"]["uncalibrated"], fc[s]["correct"], fa[s]["conf"]["uncalibrated"], fa[s]["correct"])
            for s in SEEDS}
    stats = {"count": lambda d, i: roc_auc_score(d[1][i], d[0][i]),
             "accumulator": lambda d, i: roc_auc_score(d[3][i], d[2][i]),
             "count_minus_accumulator": lambda d, i: roc_auc_score(d[1][i], d[0][i]) - roc_auc_score(d[3][i], d[2][i])}
    for k, stat in stats.items():
        r = both(stat, data, streams)
        ref = cm["headline"]["auroc"][k]
        assert max(abs(a - b) for a, b in zip(r["independent"], ref["ci95_seed_mean"])) < 1e-12, k
        doc["auroc"][k] = {"mean": ref["mean"], **r}
    print("auroc", {k: [round(x, 4) for x in v["shared"]] for k, v in doc["auroc"].items()}, flush=True)

    # A8: the accumulator reading of the same weights, run again on MPS.
    import same_weights_ranking as SWR  # noqa: E402
    import torch  # noqa: E402
    from spikecalib.calibration import model_outputs, quantise_confidence  # noqa: E402
    from spikecalib.experiments import BATCH_SIZE, load_data, load_trained, loader, splits_for  # noqa: E402
    if not torch.backends.mps.is_available():
        raise RuntimeError("run on MPS: CPU arithmetic flips a few spikes and would not reproduce the stored outputs")
    device = torch.device("mps")
    nm = load_data("nmnist", 2, SWR.DATA, in_memory=False)
    data = {}
    for s in SEEDS:
        splits = splits_for(nm, s)
        model, _, name = load_trained(R, nm, s, "count", SWR.CHANNELS, splits)
        z = np.load(os.path.join(R, name + "_outputs.npz"))
        t_counts, t_acc, _, t_y = SWR.run_split(model, loader(nm.test, splits.test, BATCH_SIZE["nmnist"]), device)
        assert np.array_equal(t_counts, z["test_scores"]) and np.array_equal(t_y, labels[0]), name
        oc = model_outputs(t_counts, t_y, np.random.default_rng(s + 1))
        oa = model_outputs(t_acc, t_y, np.random.default_rng(s + 1))
        data[s] = (oc.conf, (oc.predictions == oc.labels).astype(np.float64),
                   quantise_confidence(oa.conf), (oa.predictions == oa.labels).astype(np.float64))
    r = both(stats["count_minus_accumulator"], data, streams)
    ref = json.load(open(os.path.join(R, "same_weights_ranking.json")))["summary"]["count_minus_accumulator_same_weights"]
    assert max(abs(a - b) for a, b in zip(r["independent"], ref["ci95_seed_mean"])) < 1e-12
    doc["same_weights"]["count_minus_accumulator_same_weights"] = {"mean": ref["mean"], **r}
    print("same weights", [round(x, 4) for x in r["shared"]], flush=True)

    with open(os.path.join(R, "shared_resample_intervals.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote results/shared_resample_intervals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
