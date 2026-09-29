#!/usr/bin/env python3
"""Read every number the paper quotes from results/, and write it as a LaTeX macro.

main.tex never types a measured value. It uses the macros defined in numbers.tex, which this
script regenerates from the artifacts, and each macro's definition carries a comment naming
the file it was read from. Derived arithmetic (the bound at a given T and C) is computed here
too and labelled [DERIVED] rather than attributed to a result.

    python paper/make_numbers.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "numbers.tex")
DERIVED = "[DERIVED] arithmetic in paper/make_numbers.py"

macros: list[tuple[str, str, str]] = []


def put(name: str, value: str, source: str) -> None:
    assert name.isalpha(), name
    assert name not in {m[0] for m in macros}, name
    macros.append((name, value, source))


def f(x: float, d: int = 4, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s.replace("-", "$-$")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def load(name: str):
    return json.load(open(os.path.join(R, name)))


def ceiling(T: float, C: int) -> float:
    return math.exp(T) / (math.exp(T) + C - 1)


def mean(rows, key):
    return st.mean(r[key] for r in rows)


# ------------------------------------------------------------------ the bound [DERIVED]
put("ceilTenTwo", f(ceiling(2, 10)), DERIVED)
put("ceilHundredOneTwo", f(ceiling(2, 101)), DERIVED)
put("floorTenTwo", f(-math.log(ceiling(2, 10))), DERIVED)
put("floorTenSix", f(-math.log(ceiling(6, 10))), DERIVED)
put("floorTenTen", f(-math.log(ceiling(10, 10))), DERIVED)
tmin = lambda C, a: math.log((C - 1) * a / (1 - a))
put("tminTenExact", f(tmin(10, 0.95), 3), DERIVED)
put("tminTen", str(math.ceil(tmin(10, 0.95))), DERIVED)
put("tminHundredOne", str(math.ceil(tmin(101, 0.95))), DERIVED)
c80 = np.exp(np.float64(80)) / (np.exp(np.float64(80)) + np.float64(9))
assert c80 == 1.0, c80
put("ceilEightyIsOne", "1.0", DERIVED + " (float64 e^80/(e^80+9) == 1.0)")
put("ceilTwentyTen", f"{ceiling(20, 10):.8f}", DERIVED)

# ------------------------------------------------------------------ splits and models
for key, fn in (("nm", "nmnist_T2_seed0_split.npz"), ("nc", "ncaltech101_T2_seed0_split.npz")):
    d = np.load(os.path.join(R, fn), allow_pickle=True)
    sizes = {k: int(d[k].size) for k in ("train", "val", "cal", "test")}
    for k, v in sizes.items():
        put(f"{key}Split{k.capitalize()}", thousands(v), f"results/{fn}")
    if key == "nc":
        put(f"{key}SplitTotal", thousands(sum(sizes.values())), f"results/{fn} (sum)")

weights = glob.glob(os.path.join(R, "*_weights.pt"))
put("modelsTotal", str(len(weights)), "results/*_weights.pt (file count)")
put("modelsNm", str(sum("nmnist" in w for w in weights)), "results/nmnist_*_weights.pt (file count)")
put("modelsNc", str(sum("ncaltech101" in w for w in weights)), "results/ncaltech101_*_weights.pt (file count)")

rec = load("nmnist_T2_seed0_record.json")
mc = rec["model_config"]
put("hiddenUnits", str(mc["hidden"]), "results/nmnist_T2_seed0_record.json model_config")
put("lifTau", f"{mc['tau']:.1f}", "results/nmnist_T2_seed0_record.json model_config")
put("lifThreshold", f"{mc['v_threshold']:.1f}", "results/nmnist_T2_seed0_record.json model_config")
put("surrogateAlpha", f"{mc['surrogate_alpha']:.1f}", "results/nmnist_T2_seed0_record.json model_config")
put("learningRate", f"{rec['lr']:g}".replace("0.001", "10^{-3}"), "results/nmnist_T2_seed0_record.json lr")
put("batchSize", str(rec["batch_size"]), "results/nmnist_T2_seed0_record.json batch_size")

records = [json.load(open(p)) for p in glob.glob(os.path.join(R, "*_record.json"))]
gaps = {r["epochs_run"] - r["best_epoch"] for r in records if r.get("early_stopped")}
assert len(gaps) == 1, gaps
gap = gaps.pop()
# best_epoch is 0-indexed and epochs_run is a count (spikecalib/train.py), so a constant gap
# of `gap` means every run stopped after gap - 1 epochs without improvement.
put("patienceEpochs", str(gap - 1),
    "results/*_record.json epochs_run - best_epoch, less one for the 0-indexed best_epoch in spikecalib/train.py")
assert rec["model_config"]["channels"] == [32, 64]
nc_rec = json.load(open(os.path.join(R, "ncaltech101_T2_seed0_record.json")))
assert nc_rec["model_config"]["channels"] == [32, 64, 128, 128], nc_rec["model_config"]
# "Every run stopped early" in the Method section rests on this.
assert all(r.get("early_stopped") for r in records), [r.get("early_stopped") for r in records]

sweep = list(csv.DictReader(open(os.path.join(R, "sweep_nmnist.csv"))))
put("epochCap", sweep[0]["epochs_requested"], "results/sweep_nmnist.csv epochs_requested")
er = [int(r["epochs_run"]) for r in sweep]
put("epochsMin", str(min(er)), "results/sweep_nmnist.csv epochs_run")
put("epochsMax", str(max(er)), "results/sweep_nmnist.csv epochs_run")
assert {r["hit_epoch_cap"] for r in sweep} == {"False"}
put("nBins", sweep[0]["n_bins"], "results/sweep_nmnist.csv n_bins")
put("nBootstrap", thousands(int(sweep[0]["n_bootstrap"])), "results/sweep_nmnist.csv n_bootstrap")
enc = [float(r["encoder_energy_fraction"]) for r in sweep]
put("encoderMin", f"{100 * min(enc):.0f}", "results/sweep_nmnist.csv encoder_energy_fraction")
put("encoderMax", f"{100 * max(enc):.0f}", "results/sweep_nmnist.csv encoder_energy_fraction")

arch = load("arch_matched.json")["rows"]
for ds, key in (("nmnist", "Nm"), ("ncaltech101", "Nc")):
    for b, bk in ((2, "Two"), (4, "Four")):
        r = next(x for x in arch if x["dataset"] == ds and x["n_blocks"] == b)
        put(f"params{key}{bk}", thousands(int(r["n_parameters"])), "results/arch_matched.json n_parameters")

# ------------------------------------------------------------------ 5.1 the bound holds
cells = {}
for r in load("readout_experiment.json")["rows"]:
    if r["readout"] == "count":
        cells[("readout", r["timesteps"], r["seed"])] = (r["max_confidence_observed"], r["ceiling_derived"], 101 if False else 10, r["timesteps"])
for r in load("caltech_experiment.json")["rows"]:
    if r["readout"] == "count":
        cells[("caltech", r["timesteps"], r["seed"])] = (r["max_confidence_observed"], r["ceiling_derived"], 101, r["timesteps"])
for r in arch:
    if r["readout"] == "count":
        cells[("arch", r["dataset"], r["n_blocks"], r["timesteps"], r["seed"])] = (
            r["max_confidence_observed"], r["ceiling_derived"], r["n_classes"], r["timesteps"])
src_union = "results/readout_experiment.json + caltech_experiment.json + arch_matched.json (count readout)"
violations = sum(1 for o, c, *_ in cells.values() if o > c + 1e-12)
assert violations == 0
slack = [(c - o, C, T) for o, c, C, T in cells.values() if abs(c - o) > 1e-9]
put("cellsUnion", str(len(cells)), src_union)
put("cellsSaturated", str(len(cells) - len(slack)), src_union + " |max - ceiling| < 1e-9")
put("cellsSlack", str(len(slack)), src_union)
assert all(C == 101 and T == 10 for _, C, T in slack), slack
put("slackMeanGap", f"{st.mean(g for g, *_ in slack):.3e}".replace("e-03", r"\times 10^{-3}"), src_union)
arch_count = [r for r in arch if r["readout"] == "count"]
put("cellsMatched", str(len(arch_count)), "results/arch_matched.json (count readout)")
put("matchedMaxDev", f"{max(abs(r['max_confidence_observed'] - r['ceiling_derived']) for r in arch_count):.2e}"
    .replace("e-13", r"\times 10^{-13}"), "results/arch_matched.json")

# ------------------------------------------------------------------ 5.2 shares
cond = load("conditional_ceiling.json")["rows"]
def share(ds, T, key="share_explained_theoretical", rows=cond):
    return 100 * st.mean(r[key] for r in rows if r["dataset"] == ds and r["timesteps"] == T)
put("shareTenTwo", f"{share('nmnist', 2):.1f}", "results/conditional_ceiling.json share_explained_theoretical, mean of seeds")
put("shareTenFour", f"{share('nmnist', 4):.1f}", "results/conditional_ceiling.json")
put("shareTenSix", f"{share('nmnist', 6):.1f}", "results/conditional_ceiling.json")
put("shareHundredOneTwo", f"{share('ncaltech101', 2):.1f}", "results/conditional_ceiling.json")
for T in (8, 10):
    assert share("nmnist", T) == 0.0
for T in (6, 10):
    assert share("ncaltech101", T) == 0.0
def ashare(ds, b, T):
    rs = [r for r in arch if r["dataset"] == ds and r["n_blocks"] == b and r["readout"] == "count" and r["timesteps"] == T]
    return 100 * mean(rs, "share_explained_theoretical")
put("shareTenTwoFour", f"{ashare('nmnist', 4, 2):.1f}", "results/arch_matched.json share_explained_theoretical")
put("shareHundredOneTwoTwo", f"{ashare('ncaltech101', 2, 2):.1f}", "results/arch_matched.json share_explained_theoretical")
# ECE equals the gap a - mean confidence exactly when no bin is overconfident. Where the cap
# binds, that holds in every T=2 count cell of the matched grid, under both estimators.
t2 = [r for r in arch if r["readout"] == "count" and r["timesteps"] == 2]
dev = max(abs(r[k] - r["residual"]) for r in t2 for k in ("ece_equal_width", "ece_equal_mass"))
assert dev < 1e-4, dev
put("eceGapMaxDev", f"{dev:.0e}".replace("e-05", r"\times 10^{-5}"),
    "results/arch_matched.json max |ece - residual|, count readout, T=2")
def cmean(ds, T, key):
    return st.mean(r[key] for r in cond if r["dataset"] == ds and r["timesteps"] == T)
put("residTenTwo", f(cmean("nmnist", 2, "residual")), "results/conditional_ceiling.json residual")
put("residHundredOneTwo", f(cmean("ncaltech101", 2, "residual")), "results/conditional_ceiling.json residual")
gap10 = {T: cmean("nmnist", T, "ceiling_gap_theo_minus_cond") for T in (2, 4, 6, 8, 10)}
put("condGapTenMax", f(max(gap10.values())), "results/conditional_ceiling.json ceiling_gap_theo_minus_cond, seed means")
put("condGapTenWorstSeed", f(max(r["ceiling_gap_theo_minus_cond"] for r in cond if r["n_classes"] == 10)),
    "results/conditional_ceiling.json ceiling_gap_theo_minus_cond, single seed")
put("condGapHundredOneSix", f(cmean("ncaltech101", 6, "ceiling_gap_theo_minus_cond"), 3), "results/conditional_ceiling.json")
put("condCeilHundredOneSix", f(cmean("ncaltech101", 6, "conditional_ceiling_mean")), "results/conditional_ceiling.json conditional_ceiling_mean")
put("accHundredOneSix", f(cmean("ncaltech101", 6, "test_accuracy")), "results/conditional_ceiling.json test_accuracy")

# ------------------------------------------------------------------ 5.3 readouts
ro = load("readout_experiment.json")["rows"]
def rmean(name, T, key):
    return st.mean(r[key] for r in ro if r["readout"] == name and r["timesteps"] == T)
NAMES = {"count": "Count", "weighted-count": "Wc", "membrane": "Mem", "accumulator": "Acc"}
for name, k in NAMES.items():
    put(f"ro{k}ConfAcc", f(rmean(name, 2, "mean_conf_minus_acc"), sign=True), "results/readout_experiment.json T=2 mean_conf_minus_acc")
    put(f"ro{k}EceW", f(rmean(name, 2, "ece_equal_width")), "results/readout_experiment.json T=2 ece_equal_width")
    put(f"ro{k}EceM", f(rmean(name, 2, "ece_equal_mass")), "results/readout_experiment.json T=2 ece_equal_mass")
    put(f"ro{k}Levels", thousands(round(rmean(name, 2, "n_distinct"))), "results/readout_experiment.json T=2 n_distinct")
    put(f"ro{k}Ops", thousands(round(rmean(name, 2, "n_achievable_coverage_levels"))), "results/readout_experiment.json T=2 n_achievable_coverage_levels")
    put(f"ro{k}Acc", f(rmean(name, 2, "test_accuracy")), "results/readout_experiment.json T=2 test_accuracy")
put("roAccConfAccSix", f(rmean("accumulator", 6, "mean_conf_minus_acc"), sign=True), "results/readout_experiment.json")
put("roAccConfAccTen", f(rmean("accumulator", 10, "mean_conf_minus_acc"), sign=True), "results/readout_experiment.json")
put("roCountOpsExact", f"{rmean('count', 2, 'n_achievable_coverage_levels'):.1f}", "results/readout_experiment.json")
# leak: share of membrane's AUROC deficit against count that the accumulator closes
for T, k in ((6, "Six"), (10, "Ten")):
    c, m_, a = (rmean(n, T, "auroc") for n in ("count", "membrane", "accumulator"))
    put(f"leakClosed{k}", f"{100 * (1 - (c - a) / (c - m_)):.0f}", "results/readout_experiment.json auroc")

cal = load("caltech_experiment.json")["rows"]
def cm(name, T, key):
    return st.mean(r[key] for r in cal if r["readout"] == name and r["timesteps"] == T)
for T, k in ((2, "Two"), (6, "Six"), (10, "Ten")):
    put(f"ncCount{k}ConfAcc", f(cm("count", T, "mean_conf_minus_acc"), sign=True), "results/caltech_experiment.json mean_conf_minus_acc")
    put(f"ncAcc{k}ConfAcc", f(cm("accumulator", T, "mean_conf_minus_acc"), sign=True), "results/caltech_experiment.json mean_conf_minus_acc")
    put(f"ncCount{k}EceW", f(cm("count", T, "ece_equal_width")), "results/caltech_experiment.json")
    put(f"ncCount{k}EceM", f(cm("count", T, "ece_equal_mass")), "results/caltech_experiment.json")
    put(f"ncAcc{k}EceW", f(cm("accumulator", T, "ece_equal_width")), "results/caltech_experiment.json")
    put(f"ncAcc{k}EceM", f(cm("accumulator", T, "ece_equal_mass")), "results/caltech_experiment.json")
    put(f"ncCount{k}Acc", f(cm("count", T, "test_accuracy")), "results/caltech_experiment.json test_accuracy")
    put(f"ncCount{k}Levels", thousands(round(cm("count", T, "n_distinct"))), "results/caltech_experiment.json n_distinct")
    put(f"ncAcc{k}Levels", thousands(round(cm("accumulator", T, "n_distinct"))), "results/caltech_experiment.json n_distinct")
nc_acc = [cm("count", T, "test_accuracy") for T in (2, 6, 10)] + [cm("accumulator", T, "test_accuracy") for T in (2, 6, 10)]
put("ncAccMin", f(min(nc_acc), 2), "results/caltech_experiment.json test_accuracy")
put("ncAccMax", f(max(nc_acc), 2), "results/caltech_experiment.json test_accuracy")

# accuracy scope, matched grid
def gain(ds, b, T):
    c = [r["test_accuracy"] for r in arch if r["dataset"] == ds and r["n_blocks"] == b and r["timesteps"] == T and r["readout"] == "count"]
    a = [r["test_accuracy"] for r in arch if r["dataset"] == ds and r["n_blocks"] == b and r["timesteps"] == T and r["readout"] == "accumulator"]
    return st.mean(c), st.mean(a), 100 * (st.mean(a) - st.mean(c))
for ds, dk in (("ncaltech101", "Nc"), ("nmnist", "Nm")):
    for b, bk in ((4, "Four"), (2, "Two")):
        for T, tk in ((2, "Two"), (6, "Six")):
            c, a, g = gain(ds, b, T)
            put(f"gain{dk}{bk}{tk}", f(g, 2, sign=True), "results/arch_matched.json test_accuracy (accumulator - count)")
            put(f"accCount{dk}{bk}{tk}", f(c), "results/arch_matched.json test_accuracy")
            put(f"accAccum{dk}{bk}{tk}", f(a), "results/arch_matched.json test_accuracy")
put("gainNcFourTwoShort", f(gain("ncaltech101", 4, 2)[2], 1, sign=True), "results/arch_matched.json")
put("gainNcTwoTwoShort", f(gain("ncaltech101", 2, 2)[2], 1, sign=True), "results/arch_matched.json")
put("residNcFourSix", f(st.mean(r["residual"] for r in arch if r["dataset"] == "ncaltech101" and r["n_blocks"] == 4 and r["timesteps"] == 6 and r["readout"] == "count")), "results/arch_matched.json residual")
put("residNcTwoSix", f(st.mean(r["residual"] for r in arch if r["dataset"] == "ncaltech101" and r["n_blocks"] == 2 and r["timesteps"] == 6 and r["readout"] == "count")), "results/arch_matched.json residual")

# ------------------------------------------------------------------ 5.4 calibrators
def smean(T, arm, key):
    return st.mean(float(r[key]) for r in sweep if int(r["timesteps"]) == T and r["arm"] == arm)
ARMS = {"uncalibrated": "Unc", "temperature": "Temp", "isotonic-top-label": "Iso"}
TK = {2: "Two", 4: "Four", 6: "Six", 8: "Eight", 10: "Ten"}
for T, tk in TK.items():
    for arm, ak in ARMS.items():
        put(f"sw{ak}{tk}W", f(smean(T, arm, "ece_equal_width")), "results/sweep_nmnist.csv ece_equal_width")
        put(f"sw{ak}{tk}M", f(smean(T, arm, "ece_equal_mass")), "results/sweep_nmnist.csv ece_equal_mass")
    put(f"swAcc{tk}", f(smean(T, "uncalibrated", "test_accuracy")), "results/sweep_nmnist.csv test_accuracy")
runs = {}
for r in sweep:
    runs.setdefault((int(r["timesteps"]), int(r["seed"])), r)
ew = {k for k, r in runs.items() if r["iso_minus_temp_resolvable"] == "True"}
em = {k for k, r in runs.items() if r["iso_minus_temp_em_resolvable"] == "True"}
put("runsTotal", str(len(runs)), "results/sweep_nmnist.csv")
put("straddleW", str(len(runs) - len(ew)), "results/sweep_nmnist.csv iso_minus_temp_resolvable")
put("straddleM", str(len(runs) - len(em)), "results/sweep_nmnist.csv iso_minus_temp_em_resolvable")
put("resolveEither", str(len(ew | em)), "results/sweep_nmnist.csv (union of resolvable runs)")
put("resolveBoth", str(len(ew & em)), "results/sweep_nmnist.csv (intersection of resolvable runs)")
assert not (ew & em)
def ci(k, est):
    r = runs[k]
    if est == "w":
        return float(r["iso_minus_temp_ece_ew"]), float(r["iso_minus_temp_ci_low"]), float(r["iso_minus_temp_ci_high"])
    return float(r["iso_minus_temp_ece_em"]), float(r["iso_minus_temp_em_ci_low"]), float(r["iso_minus_temp_em_ci_high"])
for (T, s), est, name in (((2, 1), "w", "WTwoOne"), ((6, 0), "w", "WSixZero"),
                          ((4, 1), "m", "MFourOne"), ((4, 2), "m", "MFourTwo"), ((8, 0), "m", "MEightZero")):
    d, lo, hi = ci((T, s), est)
    put(f"res{name}", f(d, sign=True), "results/sweep_nmnist.csv paired difference")
    put(f"res{name}Lo", f(lo, sign=True), "results/sweep_nmnist.csv interval")
    put(f"res{name}Hi", f(hi, sign=True), "results/sweep_nmnist.csv interval")
for k in ew | em:
    assert k in {(2, 1), (6, 0), (4, 1), (4, 2), (8, 0)}, k
put("isoMinusTempTwoW", f(smean(2, "isotonic-top-label", "ece_equal_width") - smean(2, "temperature", "ece_equal_width"), sign=True), "results/sweep_nmnist.csv")
put("isoMinusTempTwoM", f(smean(2, "isotonic-top-label", "ece_equal_mass") - smean(2, "temperature", "ece_equal_mass"), sign=True), "results/sweep_nmnist.csv")

put("cellsReadout", str(sum(1 for k in cells if k[0] == "readout")), src_union)
put("cellsCaltech", str(sum(1 for k in cells if k[0] == "caltech")), src_union)
put("expectedFalse", f"{len(runs) * 0.05:.2f}", "[DERIVED] run count in results/sweep_nmnist.csv x 0.05")
put("nSeeds", str(len({r['seed'] for r in sweep})), "results/sweep_nmnist.csv seed")
fp0 = {r["split_fingerprint"] for r in sweep if r["seed"] == "0"}
assert len(fp0) == 1, fp0
put("splitFingerprint", fp0.pop(), "results/sweep_nmnist.csv split_fingerprint, seed 0")
for T, k in ((2, "Two"), (6, "Six"), (10, "Ten")):
    put(f"ncAccum{k}Acc", f(cm("accumulator", T, "test_accuracy")), "results/caltech_experiment.json test_accuracy")

# ------------------------------------------------------------------ write
with open(OUT, "w") as fh:
    fh.write("% Generated by paper/make_numbers.py from results/ -- do not edit by hand.\n")
    fh.write("% Each definition names the artifact it was read from.\n")
    for name, value, source in macros:
        fh.write(f"\\newcommand{{\\{name}}}{{{value}}}% {source}\n")
print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(macros)} macros")
for name, value, source in macros:
    print(f"  \\{name:<24} {value}")
