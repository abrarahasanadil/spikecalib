#!/usr/bin/env python3
"""Consequence for multisensory fusion: what saturation costs a symmetric fusion rule.

A saturated count readout reports the same probability vector whichever network produced
it, so when two modalities both saturate a fusion rule that uses only their reports cannot
weight the more reliable one: on disagreement it flips a coin. This script measures how
often that happens and what it costs, against the value the proposition in NOTES.md
("Consequence for multisensory fusion") gives for it.

Inference only, on the existing N-MNIST models. No training.

**Two modalities.** Modality A is a test sample. Modality B is a different test sample of
the same class with event dropout, so the two are conditionally independent given the class
and B is the less reliable. The same trained network reads both.

**Two stages, and the order matters.** ``--select-rate`` chooses the dropout rate on the
calibration split, with the seed-0 count model at ``T = 2``, as the grid value whose accuracy
under dropout is closest to the clean accuracy minus 0.10, and writes it to
``results/fusion_dropout_rate.json``. It computes no fused prediction. The main run refuses
to start without that file and never changes the rate.

**Three rules.** R0, the product rule on the raw softmax outputs, ties broken uniformly at
random. R1, the product rule after per-modality temperature scaling fitted on the calibration
split. R2, reliability-weighted voting with weights from calibration accuracies.

**What is reported**, for count readouts at ``T`` in {2, 4, 6, 8, 10} and accumulators at
``T`` in {2, 6, 10}, three seeds, 20 pairings each: each rule's accuracy, ``P(S)``,
``delta = acc(R1) - acc(R0)`` split into its on-``S`` and off-``S`` parts, the value part (b)
of the proposition gives for the on-``S`` part (and the value its first draft gave), 95%
bootstrap intervals over test samples, and both ECE estimators for each rule's fused
confidence.

Predictions 12 to 15 were recorded in NOTES.md and committed before this script was run. It
evaluates each against the registered condition and stores the verdict beside the numbers.

Writes ``results/fusion_experiment.json``. Zero DataLoader workers; data is held in RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.calibration import (  # noqa: E402
    TemperatureScaling, ece_equal_mass, ece_equal_width, model_outputs, predict_with_ties,
    quantise_confidence, softmax_probs,
)
from spikecalib.experiments import (  # noqa: E402
    cell_outputs, forward_scores, load_data, load_trained, splits_for,
)
from spikecalib.fusion import (  # noqa: E402
    class_matched_partners, event_dropout, excess_error_as_first_drafted,
    excess_error_on_saturation, product_rule, reliability_weight, saturated, weighted_vote,
)
from spikecalib.train import resolve_device  # noqa: E402

DATASET = "nmnist"
COUNT_T = (2, 4, 6, 8, 10)
ACCUM_T = (2, 6, 10)
SEEDS = (0, 1, 2)
RATE_FILE = "fusion_dropout_rate.json"
TARGET_DROP = 0.10


def frames_of(dataset, idx: np.ndarray) -> np.ndarray:
    """Integer event-count frames for the given indices, from the in-memory dataset."""
    return dataset.frames[np.asarray(idx)].float().numpy()


def select_rate(args, device) -> int:
    """Choose the dropout rate on the calibration split. No fused prediction is computed."""
    data = load_data(DATASET, 2, args.data_root)
    splits = splits_for(data, 0)
    model, _, name = load_trained(args.out, data, 0, "count", splits=splits)
    cal_frames = frames_of(data.train, splits.cal)
    cal_labels = data.labels[splits.cal]

    def accuracy(frames, seed):
        scores = forward_scores(model, frames, device)
        pred, _ = predict_with_ties(scores, np.random.default_rng(seed))
        return float((pred == cal_labels).mean())

    clean = accuracy(cal_frames, 0)
    target = clean - TARGET_DROP
    grid = []
    for step in range(100):
        rate = step / 100.0
        dropped = event_dropout(cal_frames, rate, np.random.default_rng([2026, step]))
        grid.append({"rate": rate, "calibration_accuracy": accuracy(dropped, 1000 + step)})
        if step % 10 == 0:
            print(f"    rate {rate:.2f}  calibration accuracy {grid[-1]['calibration_accuracy']:.4f}", flush=True)
    best = min(grid, key=lambda g: (abs(g["calibration_accuracy"] - target), g["rate"]))
    out = {"dataset": DATASET, "timesteps": 2, "seed": 0, "readout": "count", "model": name,
           "split_fingerprint": splits.fingerprint, "n_cal": int(splits.cal.size),
           "clean_calibration_accuracy": clean, "target_accuracy": target,
           "target_drop": TARGET_DROP, "rate": best["rate"],
           "calibration_accuracy_at_rate": best["calibration_accuracy"], "grid": grid,
           "chosen": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(args.out, RATE_FILE)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  clean calibration accuracy {clean:.4f}; target {target:.4f}")
    print(f"  chosen rate {best['rate']:.2f}, calibration accuracy {best['calibration_accuracy']:.4f}")
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


def interval(per_sample: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """95% percentile interval on the mean of a per-sample statistic."""
    n = per_sample.size
    means = np.array([per_sample[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-rate", action="store_true",
                    help="choose the dropout rate on the calibration split and stop")
    ap.add_argument("--pairings", type=int, default=20)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()
    device = resolve_device()

    if args.select_rate:
        print("=" * 100)
        print("FUSION, STAGE 1  choose the event-dropout rate on the calibration split; no fusion")
        print("=" * 100, flush=True)
        return select_rate(args, device)

    rate_path = os.path.join(args.out, RATE_FILE)
    if not os.path.exists(rate_path):
        print(f"missing {rate_path}; run with --select-rate first", file=sys.stderr)
        return 1
    with open(rate_path) as fh:
        rate_info = json.load(fh)
    rate = rate_info["rate"]

    print("=" * 112)
    print(f"FUSION  {DATASET}  event dropout {rate:.2f} on modality B  {args.pairings} pairings  no training")
    print("=" * 112, flush=True)
    rows = []

    for T in COUNT_T:
        readouts = ("count", "accumulator") if T in ACCUM_T else ("count",)
        data = load_data(DATASET, T, args.data_root)
        C = data.spec.n_classes
        test_frames_all = data.test.frames              # [N, T, 2, H, W], float16 event counts
        assert np.array_equal(test_frames_all[:64].float().numpy(), np.rint(test_frames_all[:64].float().numpy()))

        cells = {}
        for seed in SEEDS:
            splits = splits_for(data, seed)
            for readout in readouts:
                model, _, name = load_trained(args.out, data, seed, readout, splits=splits)
                clean = cell_outputs(args.out, data, seed, device, readout)
                # the in-memory forward pass must reproduce the cached outputs it is paired with
                probe = forward_scores(model, frames_of(data.test, splits.test[:500]), device)
                assert np.allclose(probe, clean.test.counts[:500], atol=1e-3), name
                # modality B on the calibration split: one seeded dropout draw
                cal_b_frames = event_dropout(frames_of(data.train, splits.cal), rate,
                                             np.random.default_rng([7, T, seed]))
                cal_b = model_outputs(forward_scores(model, cal_b_frames, device), clean.cal.labels,
                                      np.random.default_rng([8, T, seed]))
                t_a = TemperatureScaling().fit(clean.cal.conf, clean.cal.correct, outputs=clean.cal).temperature_
                t_b = TemperatureScaling().fit(cal_b.conf, cal_b.correct, outputs=cal_b).temperature_
                acc_a_cal, acc_b_cal = float(clean.cal.correct.mean()), float(cal_b.correct.mean())
                cells[(seed, readout)] = {
                    "model": model, "name": name, "clean": clean, "splits": splits,
                    "t_a": t_a, "t_b": t_b, "acc_a_cal": acc_a_cal, "acc_b_cal": acc_b_cal,
                    "w_a": reliability_weight(C, acc_a_cal), "w_b": reliability_weight(C, acc_b_cal),
                    "stats": {k: [] for k in ("r0", "r1", "r2", "s", "d", "a", "b")},
                    "ece": {k: [] for k in ("r0_ew", "r0_em", "r1_ew", "r1_em", "r2_ew", "r2_em")},
                }
        test_idx = cells[(0, "count")]["splits"].test
        labels = cells[(0, "count")]["clean"].test.labels
        assert all(np.array_equal(c["splits"].test, test_idx) for c in cells.values())
        test_frames = frames_of(data.test, test_idx)

        for p in range(args.pairings):
            t0 = time.time()
            # one dropout draw of the test set per pairing, shared by every model at this T
            dropped = event_dropout(test_frames, rate, np.random.default_rng([11, T, p]))
            for (seed, readout), c in cells.items():
                partner = class_matched_partners(labels, np.random.default_rng([13, T, seed, p]))
                b_scores_all = forward_scores(c["model"], dropped, device)
                a_scores = c["clean"].test.counts
                b_scores = b_scores_all[partner]
                a_pred = c["clean"].test.predictions
                b_pred_all, _ = predict_with_ties(b_scores_all, np.random.default_rng([17, T, seed, p]))
                b_pred = b_pred_all[partner]
                if readout == "count":
                    s_mask = saturated(np.rint(a_scores), T) & saturated(np.rint(b_scores), T)
                else:
                    s_mask = np.zeros(labels.size, dtype=bool)      # an accumulator has no counts
                r0, p0, _ = product_rule(a_scores, b_scores, np.random.default_rng([19, T, seed, p]))
                r1, p1, _ = product_rule(a_scores, b_scores, np.random.default_rng([23, T, seed, p]),
                                         scale_a=1.0 / c["t_a"], scale_b=1.0 / c["t_b"])
                r2, p2, _ = weighted_vote(a_pred, b_pred, c["w_a"], c["w_b"], C,
                                          np.random.default_rng([29, T, seed, p]))
                stats = c["stats"]
                stats["r0"].append(r0 == labels); stats["r1"].append(r1 == labels)
                stats["r2"].append(r2 == labels); stats["s"].append(s_mask)
                stats["d"].append(a_pred != b_pred)
                stats["a"].append(a_pred == labels); stats["b"].append(b_pred == labels)
                rows_idx = np.arange(labels.size)
                for key, pred, probs in (("r0", r0, p0), ("r1", r1, p1), ("r2", r2, p2)):
                    conf = quantise_confidence(probs[rows_idx, pred])
                    c["ece"][key + "_ew"].append(ece_equal_width(conf, pred == labels, args.bins))
                    c["ece"][key + "_em"].append(ece_equal_mass(conf, pred == labels, args.bins))
            print(f"    T={T:<2} pairing {p + 1:>2}/{args.pairings}  {time.time() - t0:5.1f}s", flush=True)

        for (seed, readout), c in cells.items():
            S = {k: np.array(v, dtype=float) for k, v in c["stats"].items()}      # [pairings, n]
            rng = np.random.default_rng([31, T, seed, 0 if readout == "count" else 1])
            diff = S["r1"] - S["r0"]
            per = {"delta": diff.mean(0), "delta_on_s": (diff * S["s"]).mean(0),
                   "delta_off_s": (diff * (1 - S["s"])).mean(0),
                   "r2_minus_r0": (S["r2"] - S["r0"]).mean(0),
                   "r2_minus_r0_on_s": ((S["r2"] - S["r0"]) * S["s"]).mean(0),
                   "acc_r0": S["r0"].mean(0), "acc_r1": S["r1"].mean(0), "acc_r2": S["r2"].mean(0)}
            p_s = float(S["s"].mean())
            p_sd = float((S["s"] * S["d"]).mean())
            a_s = float(S["a"][S["s"] == 1].mean()) if p_s > 0 else None
            b_s = float(S["b"][S["s"] == 1].mean()) if p_s > 0 else None
            row = {
                "dataset": DATASET, "n_classes": C, "readout": readout, "timesteps": T, "seed": seed,
                "model": c["name"], "split_fingerprint": c["clean"].split_fingerprint,
                "n_test": int(labels.size), "pairings": args.pairings, "dropout_rate": rate,
                "n_bootstrap": args.n_bootstrap,
                "temperature_a": c["t_a"], "temperature_b": c["t_b"],
                "r1_gives_a_the_larger_scale": bool(c["t_a"] < c["t_b"]),
                "cal_accuracy_a": c["acc_a_cal"], "cal_accuracy_b": c["acc_b_cal"],
                "weight_a": c["w_a"], "weight_b": c["w_b"],
                "accuracy_a": float(S["a"].mean()), "accuracy_b": float(S["b"].mean()),
                "p_s": p_s, "p_disagree": float(S["d"].mean()), "p_s_and_disagree": p_sd,
                "accuracy_a_on_s": a_s, "accuracy_b_on_s": b_s,
                "predicted_on_s": excess_error_on_saturation(p_s, a_s, b_s) if p_s > 0 else 0.0,
                "predicted_on_s_as_first_drafted":
                    excess_error_as_first_drafted(p_sd, a_s, b_s) if p_s > 0 else 0.0,
            }
            for key, values in per.items():
                row[key] = float(values.mean())
                row[key + "_ci"] = interval(values, rng, args.n_bootstrap)
            for key, values in c["ece"].items():
                row["ece_" + key] = float(np.mean(values))
            lo, hi = row["delta_on_s_ci"]
            row["predicted_within_on_s_interval"] = bool(lo <= row["predicted_on_s"] <= hi)
            row["drafted_within_on_s_interval"] = bool(lo <= row["predicted_on_s_as_first_drafted"] <= hi)
            rows.append(row)
            print(f"  T={T:<2} seed {seed} {readout:<11} A {row['accuracy_a']:.4f} B {row['accuracy_b']:.4f}  "
                  f"R0 {row['acc_r0']:.4f} R1 {row['acc_r1']:.4f} R2 {row['acc_r2']:.4f}  P(S) {p_s:.3f}  "
                  f"delta {row['delta']:+.4f} [{row['delta_ci'][0]:+.4f}, {row['delta_ci'][1]:+.4f}]  "
                  f"on S {row['delta_on_s']:+.4f} [{lo:+.4f}, {hi:+.4f}] predicted {row['predicted_on_s']:+.4f} "
                  f"(draft {row['predicted_on_s_as_first_drafted']:+.4f})", flush=True)
        del cells, data

    def cell(readout, T):
        return sorted((r for r in rows if r["readout"] == readout and r["timesteps"] == T),
                      key=lambda r: r["seed"])

    excludes_zero = lambda ci: ci[0] > 0 or ci[1] < 0  # noqa: E731
    mean_delta = {T: st.mean(r["delta"] for r in cell("count", T)) for T in COUNT_T}
    t2 = cell("count", 2)
    p12 = all(r["delta_ci"][0] > 0 for r in t2)
    p13 = all(r["predicted_within_on_s_interval"] for r in t2)
    ordering = (mean_delta[2] > mean_delta[4] > mean_delta[6]
                and mean_delta[8] < mean_delta[6] and mean_delta[10] < mean_delta[6])
    high_t = [r for T in (8, 10) for r in cell("count", T)]
    p14 = bool(ordering and not any(excludes_zero(r["delta_ci"]) for r in high_t))
    accum = [r for T in ACCUM_T for r in cell("accumulator", T)]
    p15 = not any(excludes_zero(r["delta_ci"]) for r in accum)
    verdicts = {
        "12": {"statement": "count, T = 2: delta positive with its interval excluding zero in every seed",
               "delta": [r["delta"] for r in t2], "delta_ci": [r["delta_ci"] for r in t2], "held": bool(p12)},
        "13": {"statement": "count, T = 2: 0.5 P(S)(a_A,S - a_B,S) lies within the interval of the "
                            "on-S part of delta in every seed",
               "delta_on_s": [r["delta_on_s"] for r in t2], "delta_on_s_ci": [r["delta_on_s_ci"] for r in t2],
               "predicted": [r["predicted_on_s"] for r in t2],
               "as_first_drafted": [r["predicted_on_s_as_first_drafted"] for r in t2],
               "drafted_within_interval": [r["drafted_within_on_s_interval"] for r in t2],
               "r1_gives_a_the_larger_scale": [r["r1_gives_a_the_larger_scale"] for r in t2],
               "held": bool(p13)},
        "14": {"statement": "count: mean delta strictly decreasing over T = 2, 4, 6, with T = 8 and 10 "
                            "below T = 6, and the interval including zero at T = 8 and 10 in every seed",
               "mean_delta_by_t": {str(T): mean_delta[T] for T in COUNT_T}, "ordering_held": bool(ordering),
               "intervals_excluding_zero_at_8_and_10": sum(excludes_zero(r["delta_ci"]) for r in high_t),
               "held": p14},
        "15": {"statement": "accumulator: the interval on delta includes zero in all nine runs",
               "intervals_excluding_zero": sum(excludes_zero(r["delta_ci"]) for r in accum),
               "runs": len(accum), "held": bool(p15)},
    }
    print("\n  predictions, against the conditions registered in NOTES.md:")
    for k in ("12", "13", "14", "15"):
        print(f"    {k}  {'HELD' if verdicts[k]['held'] else 'FALSIFIED':<10} {verdicts[k]['statement']}")

    path = os.path.join(args.out, "fusion_experiment.json")
    with open(path, "w") as fh:
        json.dump({"dataset": DATASET, "n_bins": args.bins, "pairings": args.pairings,
                   "registered": "NOTES.md, predictions 12 to 15, 2026-10-14",
                   "dropout": {k: rate_info[k] for k in ("rate", "clean_calibration_accuracy",
                                                         "calibration_accuracy_at_rate", "target_drop")},
                   "rows": rows, "predictions": verdicts}, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
