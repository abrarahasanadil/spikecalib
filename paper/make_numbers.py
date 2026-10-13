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
put("ceilTenOne", f(ceiling(1, 10)), DERIVED)
put("ceilTenThree", f(ceiling(3, 10)), DERIVED)
# the rate convention: logits c/T lie in [0, 1] whatever T is, so the width is 1
put("rateCeilTen", f(ceiling(1, 10)), DERIVED + " (width 1)")
put("rateCeilHundredOne", f(ceiling(1, 101)), DERIVED + " (width 1)")
levels = lambda T, C: math.comb(T + C - 1, C - 1)
put("levelsTenTwo", str(levels(2, 10)), DERIVED + " binomial(T+C-1, C-1)")
put("levelsTenFour", str(levels(4, 10)), DERIVED + " binomial(T+C-1, C-1)")
put("levelsTenSix", thousands(levels(6, 10)), DERIVED + " binomial(T+C-1, C-1)")
put("levelsHundredOneTwo", thousands(levels(2, 101)), DERIVED + " binomial(T+C-1, C-1)")

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
nc_batches = {json.load(open(p))["batch_size"] for p in glob.glob(os.path.join(R, "ncaltech101_*_record.json"))}
nm_batches = {json.load(open(p))["batch_size"] for p in glob.glob(os.path.join(R, "nmnist_*_record.json"))}
assert nm_batches == {rec["batch_size"]} and len(nc_batches) == 1, (nm_batches, nc_batches)
put("batchSizeNc", str(nc_batches.pop()), "results/ncaltech101_*_record.json batch_size (one value across all runs)")

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
# A cell is one (dataset, architecture, T, seed). The three artifacts overlap: the matched
# grid reuses the two-block N-MNIST and four-block N-Caltech101 models at T in {2, 6} that the
# readout and C-axis experiments already measured. Keying each row by its artifact counted
# those models twice (15 + 9 + 24 = 48); keying by the cell counts each once. The depth of
# the readout and C-axis rows is read from the run record, which those rows do not carry.
def n_blocks(stem: str) -> int:
    return len(load(f"{stem}_record.json")["model_config"]["channels"])


cells = {}
per_artifact = {"readout": 0, "caltech": 0, "arch": 0}
shared = 0


def add_cell(artifact, dataset, blocks, r, n_classes):
    global shared
    key = (dataset, blocks, r["timesteps"], r["seed"])
    val = (r["max_confidence_observed"], r["ceiling_derived"], n_classes, r["timesteps"])
    per_artifact[artifact] += 1
    if key in cells:
        # the same trained model measured in two experiments must give the same numbers
        assert cells[key] == val, (key, cells[key], val)
        shared += 1
    cells[key] = val


for r in load("readout_experiment.json")["rows"]:
    if r["readout"] == "count":
        add_cell("readout", "nmnist", n_blocks(f"nmnist_T{r['timesteps']}_seed{r['seed']}"), r, 10)
for r in load("caltech_experiment.json")["rows"]:
    if r["readout"] == "count":
        add_cell("caltech", "ncaltech101",
                 n_blocks(f"ncaltech101_T{r['timesteps']}_seed{r['seed']}"), r, r["n_classes"])
for r in arch:
    if r["readout"] == "count":
        add_cell("arch", r["dataset"], r["n_blocks"], r, r["n_classes"])
assert len(cells) == sum(per_artifact.values()) - shared
# every distinct cell is a count-readout model on disk, and there are no others
n_count_weights = sum(1 for p in glob.glob(os.path.join(R, "*_weights.pt"))
                      if not any(t in os.path.basename(p)
                                 for t in ("membrane", "weighted-count", "accumulator", "scaled-count")))
assert n_count_weights in (0, len(cells)), (n_count_weights, len(cells))  # 0: weights absent in a clone
src_union = "results/readout_experiment.json + caltech_experiment.json + arch_matched.json (count readout, distinct cells)"
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
put("ncAccMin", f(min(nc_acc)), "results/caltech_experiment.json test_accuracy")
put("ncAccMax", f(max(nc_acc)), "results/caltech_experiment.json test_accuracy")

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

put("cellsReadout", str(per_artifact["readout"]), "results/readout_experiment.json (count readout)")
put("cellsCaltech", str(per_artifact["caltech"]), "results/caltech_experiment.json (count readout)")
put("cellsShared", str(shared), src_union + ", rows of arch_matched.json that repeat a cell of the other two")
put("expectedFalse", f"{len(runs) * 0.05:.2f}", "[DERIVED] run count in results/sweep_nmnist.csv x 0.05")
put("nSeeds", str(len({r['seed'] for r in sweep})), "results/sweep_nmnist.csv seed")
fp0 = {r["split_fingerprint"] for r in sweep if r["seed"] == "0"}
assert len(fp0) == 1, fp0
put("splitFingerprint", fp0.pop(), "results/sweep_nmnist.csv split_fingerprint, seed 0")
for T, k in ((2, "Two"), (6, "Six"), (10, "Ten")):
    put(f"ncAccum{k}Acc", f(cm("accumulator", T, "test_accuracy")), "results/caltech_experiment.json test_accuracy")

# ------------------------------------------------------------------ the learned-scale arm
# Predictions 6a, 6b, 7 and 8 (NOTES.md, recorded 2026-10-13). The verdicts are read from the
# block the experiment script wrote against the registered conditions, and re-derived here
# from the rows so that a verdict and the numbers beside it cannot disagree.
sc = load("scaled_count_experiment.json")
sc_rows, sc_pred = sc["rows"], sc["predictions"]
assert len(sc_rows) == 9 and {r["seed"] for r in sc_rows} == {0, 1, 2}
SRC_SC = "results/scaled_count_experiment.json"


def sc_cell(ds, T):
    rs = [r for r in sc_rows if r["dataset"] == ds and r["timesteps"] == T]
    assert len(rs) == 3
    return rs


for ds, T, key in (("nmnist", 2, "ScNm"), ("ncaltech101", 2, "ScNcTwo"), ("ncaltech101", 6, "ScNcSix")):
    rs = sc_cell(ds, T)
    put(f"gap{key}", f(mean(rs, "mean_conf_minus_acc"), sign=True), SRC_SC + " mean_conf_minus_acc, mean of seeds")
    put(f"acc{key}", f(mean(rs, "test_accuracy")), SRC_SC + " test_accuracy")
    put(f"eceW{key}", f(mean(rs, "ece_equal_width")), SRC_SC + " ece_equal_width")
    put(f"eceM{key}", f(mean(rs, "ece_equal_mass")), SRC_SC + " ece_equal_mass")
    put(f"scaleMin{key}", f"{min(r['logit_scale'] for r in rs):.2f}", SRC_SC + " logit_scale")
    put(f"scaleMax{key}", f"{max(r['logit_scale'] for r in rs):.2f}", SRC_SC + " logit_scale")
    put(f"sminMin{key}", f"{min(r['s_min'] for r in rs):.2f}", SRC_SC + " s_min")
    put(f"sminMax{key}", f"{max(r['s_min'] for r in rs):.2f}", SRC_SC + " s_min")
    put(f"accCountRef{key}", f(mean(rs, "count_test_accuracy")), SRC_SC + " count_test_accuracy (the existing count runs)")
    put(f"accAccumRef{key}", f(mean(rs, "accumulator_test_accuracy")), SRC_SC + " accumulator_test_accuracy")
    put(f"gapCountRef{key}", f(mean(rs, "count_mean_conf_minus_acc"), sign=True), SRC_SC + " count_mean_conf_minus_acc")
    put(f"gapAccumRef{key}", f(mean(rs, "accumulator_mean_conf_minus_acc"), sign=True), SRC_SC + " accumulator_mean_conf_minus_acc")
    put(f"accDelta{key}", f(100 * (mean(rs, "test_accuracy") - mean(rs, "count_test_accuracy")), 2, sign=True),
        SRC_SC + " test_accuracy minus count_test_accuracy, in points")
    put(f"levelsMin{key}", thousands(min(r["n_distinct"] for r in rs)), SRC_SC + " n_distinct")
    put(f"levelsMax{key}", thousands(max(r["n_distinct"] for r in rs)), SRC_SC + " n_distinct")

nm, c2, c6 = sc_cell("nmnist", 2), sc_cell("ncaltech101", 2), sc_cell("ncaltech101", 6)
assert sc_pred["6a"]["held"] == (abs(mean(nm, "mean_conf_minus_acc")) < 0.05)
assert sc_pred["6b"]["held"] == (mean(c2, "mean_conf_minus_acc") > 0)
assert sc_pred["7"]["held"] == (mean(c2, "test_accuracy") >= 0.5184)
assert sc_pred["8"]["held"] == (mean(c6, "mean_conf_minus_acc") > -0.05)
verdict = lambda held: "held" if held else "falsified"
put("verdictSixA", verdict(sc_pred["6a"]["held"]), SRC_SC + " predictions.6a")
put("verdictSixB", verdict(sc_pred["6b"]["held"]), SRC_SC + " predictions.6b")
put("verdictSeven", verdict(sc_pred["7"]["held"]), SRC_SC + " predictions.7")
put("verdictEight", verdict(sc_pred["8"]["held"]), SRC_SC + " predictions.8")
put("shareRecovered", f"{100 * sc_pred['7']['share_of_gap_recovered']:.0f}", SRC_SC + " predictions.7 share_of_gap_recovered")
put("thresholdSeven", f(0.5184), "NOTES.md, prediction 7 (0.4543 + 0.1282 / 2)")
put("scRuns", str(len(sc_rows)), SRC_SC)
put("scRunsAboveSmin", str(sum(r["scale_exceeds_s_min"] for r in sc_rows)), SRC_SC + " scale_exceeds_s_min")
put("scRunsPositiveGapNc", str(sum(r["mean_conf_minus_acc"] > 0 for r in c2 + c6)), SRC_SC + " N-Caltech101 runs with a positive gap")
put("scRunsNc", str(len(c2 + c6)), SRC_SC)
# The scale was still climbing when early stopping ended each run: last epoch against best epoch.
rising = sum(r["logit_scale_last_epoch"] > r["logit_scale"] for r in sc_rows)
put("scRunsStillRising", str(rising), SRC_SC + " logit_scale_last_epoch > logit_scale")
# every learned-scale run stays within the cap at its own scale, to within the float32 the
# logits are computed in
assert max(r["ceiling_signed_excess"] for r in sc_rows) < 1e-9
assert all(r["n_distinct"] <= r["max_distinct_bound"] for r in sc_rows)

# ------------------------------------------------------------------ resolution and the bound check
rs_all = load("resolution_experiment.json")
SRC_RES = "results/resolution_experiment.json"
chk = rs_all["check"]
assert chk["violations"] == 0
put("checkCells", str(chk["cells"]), SRC_RES + " check.cells (count and scaled-count cells)")
put("checkInformative", str(chk["cells_where_bound_is_below_n_test"]), SRC_RES + " check.cells_where_bound_is_below_n_test")
nm_t2_levels = [c["n_distinct_confidences"] for c in chk["rows"]
                if c["dataset"] == "nmnist" and c["timesteps"] == 2]
put("levelsObsMinNmTwo", str(min(nm_t2_levels)), SRC_RES + " check rows, N-MNIST T=2, count and scaled-count, both depths")
put("levelsObsMaxNmTwo", str(max(nm_t2_levels)), SRC_RES + " check rows, N-MNIST T=2")
put("shareOfBoundMax", f"{100 * max(c['share_of_bound_used'] for c in chk['rows'] if c['bound_below_n_test']):.0f}",
    SRC_RES + " check rows where the bound is below n_test: largest observed/bound, per cent")


def res_rows(readout, state, est, ds="nmnist", blocks=2, T=2):
    out = [r for r in rs_all["rows"] if r["dataset"] == ds and r["n_blocks"] == blocks and r["timesteps"] == T
           and r["readout"] == readout and r["state"] == state and r["estimator"] == est]
    assert len(out) == 3, (readout, state, est, ds, blocks, T, len(out))
    return out


p9 = rs_all["prediction_9"]
assert p9["complete"]
for c in p9["comparisons"]:
    a = mean(res_rows(c["readout"], c["state"], c["estimator"]), "resolution")
    b = mean(res_rows("accumulator", c["state"], c["estimator"]), "resolution")
    assert c["count_below_accumulator"] == (a < b)
put("verdictNine", verdict(p9["held"]), SRC_RES + " prediction_9")
put("nineBelow", str(sum(c["count_below_accumulator"] for c in p9["comparisons"])), SRC_RES + " prediction_9.comparisons")
put("nineTotal", str(len(p9["comparisons"])), SRC_RES + " prediction_9.comparisons")
put("nineBelowCount", str(sum(c["count_below_accumulator"] for c in p9["comparisons"] if c["readout"] == "count")), SRC_RES)
put("nineBelowDegenerate", str(sum(c["count_below_accumulator"] for c in p9["comparisons"]
                                    if c["readout"] == "count" and c["estimator"] == "equal_mass")),
    SRC_RES + " prediction_9.comparisons: count readout, equal-mass, where it forms one bin")
held = [sc_pred[k]["held"] for k in ("6a", "6b", "7", "8")] + [p9["held"]]
put("nineBelowScaled", str(sum(c["count_below_accumulator"] for c in p9["comparisons"] if c["readout"] == "scaled-count")), SRC_RES)
for readout, key in (("count", "Count"), ("accumulator", "Accum"), ("scaled-count", "Scaled")):
    for est, ek in (("equal_width", "W"), ("equal_mass", "M")):
        rr = res_rows(readout, "uncalibrated", est)
        put(f"res{key}{ek}", f(mean(rr, "resolution")), SRC_RES + f" resolution, N-MNIST T=2, {readout}, {est}")
        put(f"resRatio{key}{ek}", f"{mean(rr, 'res_over_unc'):.2f}", SRC_RES + f" res_over_unc, {readout}, {est}")
    rr = res_rows(readout, "uncalibrated", "equal_width")
    put(f"unc{key}", f(mean(rr, "uncertainty")), SRC_RES + f" uncertainty, {readout}")
    put(f"auroc{key}NmTwo", f(mean(rr, "auroc")), SRC_RES + f" auroc, {readout}, uncalibrated")
# the count readout collapses to one equal-mass bin in every seed, which is why its resolution there is zero
assert all(r["n_groups"] == 1 for r in res_rows("count", "uncalibrated", "equal_mass"))
# exact resolution of the count readout: unchanged by temperature scaling, not raised by isotonic
ex_u, ex_t, ex_i = (res_rows("count", st_, "exact") for st_ in ("uncalibrated", "temperature", "isotonic-top-label"))
by_seed = lambda rows: {r["seed"]: r for r in rows}
for seed in (0, 1, 2):
    assert abs(by_seed(ex_u)[seed]["resolution"] - by_seed(ex_t)[seed]["resolution"]) < 1e-12
    assert by_seed(ex_u)[seed]["n_groups"] == by_seed(ex_t)[seed]["n_groups"]
    assert by_seed(ex_i)[seed]["resolution"] <= by_seed(ex_u)[seed]["resolution"] + 1e-12
    assert by_seed(ex_i)[seed]["n_groups"] <= by_seed(ex_u)[seed]["n_groups"]
put("resCountExact", f(mean(ex_u, "resolution")), SRC_RES + " resolution, exact, count, uncalibrated and after temperature (equal per seed)")
put("resCountExactIso", f(mean(ex_i, "resolution")), SRC_RES + " resolution, exact, count, after top-label isotonic")
put("groupsCountExactMin", str(min(r["n_groups"] for r in ex_u)), SRC_RES + " n_groups, exact, count")
put("groupsCountExactMax", str(max(r["n_groups"] for r in ex_u)), SRC_RES + " n_groups, exact, count")
put("groupsCountIsoMin", str(min(r["n_groups"] for r in ex_i)), SRC_RES + " n_groups, exact, count after isotonic")
put("groupsCountIsoMax", str(max(r["n_groups"] for r in ex_i)), SRC_RES + " n_groups, exact, count after isotonic")
# temperature scaling changes AUROC of the count readout: the groups are kept and reordered
au_u, au_t = (res_rows("count", st_, "equal_width") for st_ in ("uncalibrated", "temperature"))
changed = sum(abs(by_seed(au_u)[k]["auroc"] - by_seed(au_t)[k]["auroc"]) > 1e-9 for k in (0, 1, 2))
put("aurocChangedSeeds", str(changed), SRC_RES + " auroc of the count readout before and after temperature scaling, N-MNIST T=2")
all_count = [r for r in rs_all["rows"] if r["readout"] == "count" and r["estimator"] == "equal_width"]
cells_u = {(r["dataset"], r["n_blocks"], r["timesteps"], r["seed"]): r["auroc"] for r in all_count if r["state"] == "uncalibrated"}
cells_t = {(r["dataset"], r["n_blocks"], r["timesteps"], r["seed"]): r["auroc"] for r in all_count if r["state"] == "temperature"}
moved = [abs(cells_u[k] - cells_t[k]) for k in cells_u]
put("aurocMovedCells", str(sum(m > 1e-9 for m in moved)), SRC_RES + " count cells whose AUROC differs after temperature scaling")
put("aurocCells", str(len(moved)), SRC_RES + " count cells")
put("aurocMovedMax", f(max(moved)), SRC_RES + " largest |change in AUROC| after temperature scaling, count cells")
# N-Caltech101, descriptive only
for readout, key in (("count", "Count"), ("accumulator", "Accum"), ("scaled-count", "Scaled")):
    for est, ek in (("equal_width", "W"), ("equal_mass", "M")):
        rr = res_rows(readout, "uncalibrated", est, ds="ncaltech101", blocks=4)
        put(f"ncResRatio{key}{ek}", f"{mean(rr, 'res_over_unc'):.2f}", SRC_RES + f" res_over_unc, N-Caltech101 T=2, {readout}, {est}")

# ------------------------------------------------------------------ calibrators on N-Caltech101
cc = load("caltech_calibrators.json")
SRC_CC = "results/caltech_calibrators.json"
p10 = cc["prediction_10"]
cc_count = [p_ for p_ in cc["paired"] if p_["readout"] == "count"]
assert len(cc_count) == 9 and p10["ties"] == sum(p_["tie"] for p_ in cc_count)
put("verdictTen", verdict(p10["held"]), SRC_CC + " prediction_10")
put("ncTies", str(p10["ties"]), SRC_CC + " prediction_10.ties")
put("ncDiffer", str(9 - p10["ties"]), SRC_CC + " prediction_10")
assert all(p_["favours"] == "isotonic" for p_ in cc_count if p_["differ"])
for T, k in ((2, "Two"), (6, "Six"), (10, "Ten")):
    put(f"ncDiffer{k}", str(sum(p_["differ"] for p_ in cc_count if p_["timesteps"] == T)), SRC_CC + f" paired, count, T={T}")
    for arm, ak in (("uncalibrated", "Unc"), ("temperature", "Temp"), ("isotonic-top-label", "Iso")):
        rr = [r for r in cc["rows"] if r["readout"] == "count" and r["timesteps"] == T and r["arm"] == arm]
        assert len(rr) == 3
        put(f"cc{ak}{k}W", f(mean(rr, "ece_equal_width")), SRC_CC + f" ece_equal_width, count, {arm}, T={T}")
        put(f"cc{ak}{k}M", f(mean(rr, "ece_equal_mass")), SRC_CC + f" ece_equal_mass, count, {arm}, T={T}")
        if arm == "temperature":
            put(f"ccTemperature{k}", f"{mean(rr, 'temperature'):.2f}", SRC_CC + f" temperature, count, T={T}")
        if arm == "uncalibrated":
            put(f"ccAcc{k}", f(mean(rr, "test_accuracy")), SRC_CC + f" test_accuracy, count, T={T}")
put("ccCalSize", thousands(cc["rows"][0]["n_cal"]), SRC_CC + " n_cal")
put("ccTestSize", thousands(cc["rows"][0]["n_test"]), SRC_CC + " n_test")
acc_pairs = [p_ for p_ in cc["paired"] if p_["readout"] == "accumulator"]
assert all(p_["favours"] == "isotonic" for p_ in acc_pairs if p_["differ"])
put("ccAccumDiffer", str(sum(p_["differ"] for p_ in acc_pairs)), SRC_CC + " paired, accumulator")
put("ccAccumRuns", str(len(acc_pairs)), SRC_CC + " paired, accumulator")
sc_pairs = [p_ for p_ in cc["paired"] if p_["readout"] == "scaled-count"]
put("ccScaledRuns", str(len(sc_pairs)), SRC_CC + " paired, scaled-count")
put("ccScaledIso", str(sum(p_["favours"] == "isotonic" for p_ in sc_pairs)), SRC_CC + " paired, scaled-count")
put("ccScaledTemp", str(sum(p_["favours"] == "temperature" for p_ in sc_pairs)), SRC_CC + " paired, scaled-count")
put("ccScaledTie", str(sum(p_["tie"] for p_ in sc_pairs)), SRC_CC + " paired, scaled-count")
# the residual the accumulator and the learned scale leave after temperature scaling, T = 2
for readout, key in (("accumulator", "Accum"), ("scaled-count", "Scaled")):
    for arm, ak in (("temperature", "Temp"), ("isotonic-top-label", "Iso")):
        rr = [r for r in cc["rows"] if r["readout"] == readout and r["timesteps"] == 2 and r["arm"] == arm]
        put(f"cc{key}{ak}TwoW", f(mean(rr, "ece_equal_width")), SRC_CC + f" ece_equal_width, {readout}, {arm}, T=2")
        put(f"cc{key}{ak}TwoM", f(mean(rr, "ece_equal_mass")), SRC_CC + f" ece_equal_mass, {readout}, {arm}, T=2")

# ------------------------------------------------------------------ calibration-set size
cs = load("calibration_size_experiment.json")
SRC_CS = "results/calibration_size_experiment.json"
p11 = cs["prediction_11"]
put("verdictEleven", verdict(p11["held"]), SRC_CS + " prediction_11")
put("csDraws", str(cs["draws"]), SRC_CS + " draws")
for n, k in ((100, "Hundred"), (1000, "Thousand"), (5000, "Full")):
    cell = [x for x in cs["summary"] if x["n_cal"] == n]
    assert len(cell) == 3
    total = sum(x["draws"] for x in cell)
    put(f"csTempWins{k}", str(sum(x["temperature_wins"] for x in cell)), SRC_CS + f" summary, n_cal={n}")
    put(f"csIsoWins{k}", str(sum(x["isotonic_wins"] for x in cell)), SRC_CS + f" summary, n_cal={n}")
    put(f"csTempWinsMax{k}", str(max(x["temperature_wins"] for x in cell)), SRC_CS + f" summary, n_cal={n}, largest per-seed count")
    put(f"csDifferMax{k}", str(max(x["differ"] for x in cell)), SRC_CS + f" summary, n_cal={n}, largest per-seed count")
    rr = [r for r in cs["rows"] if r["n_cal"] == n]
    assert len(rr) == total == 60
    for col, ck in (("temp_ece_ew", "TempW"), ("temp_ece_em", "TempM"), ("iso_ece_ew", "IsoW"), ("iso_ece_em", "IsoM")):
        put(f"cs{ck}{k}", f(mean(rr, col)), SRC_CS + f" {col}, n_cal={n}, mean over seeds and draws")
put("csDrawsTotal", "60", SRC_CS + " 3 seeds x 20 draws")
assert p11["seeds_where_temperature_wins_at_100"] == sum(x["temperature_wins"] >= 10 for x in cs["summary"] if x["n_cal"] == 100)
held += [p10["held"], p11["held"]]
put("predictionsNew", str(len(held)), "NOTES.md predictions 6a, 6b, 7, 8, 9, 10, 11")
put("predictionsHeld", str(sum(held)), "verdict blocks of the four experiment files")
put("predictionsFalsified", str(len(held) - sum(held)), "verdict blocks of the four experiment files")
put("csSeedsCounting", str(p11["seeds_where_temperature_wins_at_100"]), SRC_CS + " prediction_11")

# ------------------------------------------------------------------ fusion (predictions 12 to 15)
fu = load("fusion_experiment.json")
SRC_FU = "results/fusion_experiment.json"
fu_rows, fu_pred = fu["rows"], fu["predictions"]


def fu_cell(readout, T):
    rs = sorted((r for r in fu_rows if r["readout"] == readout and r["timesteps"] == T), key=lambda r: r["seed"])
    assert len(rs) == 3, (readout, T, len(rs))
    return rs


TKEY = {2: "Two", 4: "Four", 6: "Six", 8: "Eight", 10: "Ten"}
put("fuRate", f"{fu['dropout']['rate']:.2f}", SRC_FU + " dropout.rate (results/fusion_dropout_rate.json)")
put("fuRatePercent", f"{100 * fu['dropout']['rate']:.0f}", SRC_FU + " dropout.rate")
put("fuCalClean", f(fu["dropout"]["clean_calibration_accuracy"]), SRC_FU + " dropout.clean_calibration_accuracy")
put("fuCalDropped", f(fu["dropout"]["calibration_accuracy_at_rate"]), SRC_FU + " dropout.calibration_accuracy_at_rate")
put("fuPairings", str(fu["pairings"]), SRC_FU + " pairings")
for T in (2, 4, 6, 8, 10):
    rs = fu_cell("count", T)
    k = "fuCount" + TKEY[T]
    put(k + "AccA", f(mean(rs, "accuracy_a")), SRC_FU + f" accuracy_a, count, T={T}")
    put(k + "AccB", f(mean(rs, "accuracy_b")), SRC_FU + f" accuracy_b, count, T={T}")
    put(k + "RZero", f(mean(rs, "acc_r0")), SRC_FU + f" acc_r0, count, T={T}")
    put(k + "ROne", f(mean(rs, "acc_r1")), SRC_FU + f" acc_r1, count, T={T}")
    put(k + "RTwo", f(mean(rs, "acc_r2")), SRC_FU + f" acc_r2, count, T={T}")
    put(k + "PS", f"{mean(rs, 'p_s'):.3f}", SRC_FU + f" p_s, count, T={T}")
    put(k + "Delta", f(mean(rs, "delta"), sign=True), SRC_FU + f" delta, count, T={T}, mean of seeds")
    put(k + "DeltaOnS", f(mean(rs, "delta_on_s"), sign=True), SRC_FU + f" delta_on_s, count, T={T}")
    put(k + "DeltaOffS", f(mean(rs, "delta_off_s"), sign=True), SRC_FU + f" delta_off_s, count, T={T}")
    put(k + "Predicted", f(mean(rs, "predicted_on_s"), sign=True), SRC_FU + f" predicted_on_s, count, T={T}")
    put(k + "Excl", str(sum(r["delta_ci"][0] > 0 or r["delta_ci"][1] < 0 for r in rs)),
        SRC_FU + f" seeds whose delta interval excludes zero, count, T={T}")
for T in (2, 6, 10):
    rs = fu_cell("accumulator", T)
    k = "fuAccum" + TKEY[T]
    put(k + "AccA", f(mean(rs, "accuracy_a")), SRC_FU + f" accuracy_a, accumulator, T={T}")
    put(k + "AccB", f(mean(rs, "accuracy_b")), SRC_FU + f" accuracy_b, accumulator, T={T}")
    put(k + "RZero", f(mean(rs, "acc_r0")), SRC_FU + f" acc_r0, accumulator, T={T}")
    put(k + "ROne", f(mean(rs, "acc_r1")), SRC_FU + f" acc_r1, accumulator, T={T}")
    put(k + "RTwo", f(mean(rs, "acc_r2")), SRC_FU + f" acc_r2, accumulator, T={T}")
    put(k + "Delta", f(mean(rs, "delta"), sign=True), SRC_FU + f" delta, accumulator, T={T}")
    put(k + "Excl", str(sum(r["delta_ci"][0] > 0 or r["delta_ci"][1] < 0 for r in rs)),
        SRC_FU + f" seeds whose delta interval excludes zero, accumulator, T={T}")
t2 = fu_cell("count", 2)
excl_ = lambda ci: ci[0] > 0 or ci[1] < 0  # noqa: E731
put("fuTwoDeltaMin", f(min(r["delta"] for r in t2), sign=True), SRC_FU + " delta, count, T=2, smallest seed")
put("fuTwoDeltaMax", f(max(r["delta"] for r in t2), sign=True), SRC_FU + " delta, count, T=2, largest seed")
put("fuTwoCiLowMin", f(min(r["delta_ci"][0] for r in t2), sign=True), SRC_FU + " delta_ci lower end, count, T=2, smallest")
put("fuTwoAccAOnS", f(mean(t2, "accuracy_a_on_s")), SRC_FU + " accuracy_a_on_s, count, T=2")
put("fuTwoAccBOnS", f(mean(t2, "accuracy_b_on_s")), SRC_FU + " accuracy_b_on_s, count, T=2")
put("fuTwoDrafted", f(mean(t2, "predicted_on_s_as_first_drafted"), sign=True), SRC_FU + " predicted_on_s_as_first_drafted, count, T=2")
put("fuTwoPDisagreeGivenS", f"{st.mean(r['p_s_and_disagree'] / r['p_s'] for r in t2):.3f}", SRC_FU + " p_s_and_disagree / p_s, count, T=2")
put("fuTwoTempA", f"{mean(t2, 'temperature_a'):.2f}", SRC_FU + " temperature_a, count, T=2")
put("fuTwoTempB", f"{mean(t2, 'temperature_b'):.2f}", SRC_FU + " temperature_b, count, T=2")
put("fuTwoShareOnS", f"{100 * mean(t2, 'delta_on_s') / mean(t2, 'delta'):.0f}", SRC_FU + " delta_on_s / delta, count, T=2, per cent")
put("fuTwoPSMin", f"{min(r['p_s'] for r in t2):.3f}", SRC_FU + " p_s, count, T=2, smallest seed")
put("fuTwoPSMax", f"{max(r['p_s'] for r in t2):.3f}", SRC_FU + " p_s, count, T=2, largest seed")
err0, err1 = 1 - mean(t2, "acc_r0"), 1 - mean(t2, "acc_r1")
put("fuTwoErrZero", f"{100 * err0:.2f}", SRC_FU + " 1 - acc_r0, count, T=2, per cent")
put("fuTwoErrOne", f"{100 * err1:.2f}", SRC_FU + " 1 - acc_r1, count, T=2, per cent")
put("fuTwoErrCut", f"{100 * (1 - err1 / err0):.0f}", SRC_FU + " share of R0's errors that R1 removes, count, T=2, per cent")
put("fuTwoDeltaPoints", f"{100 * mean(t2, 'delta'):.2f}", SRC_FU + " delta, count, T=2, percentage points")
put("fuTwoDeltaOffS", f(mean(t2, "delta_off_s"), sign=True), SRC_FU + " delta_off_s, count, T=2")
put("fuTwoRZeroMinusRTwo", f(-mean(t2, "r2_minus_r0")), SRC_FU + " -r2_minus_r0, count, T=2")
assert mean(t2, "r2_minus_r0") < 0
for rule, rk in (("r0", "RZero"), ("r1", "ROne"), ("r2", "RTwo")):
    put(f"fuTwoEce{rk}W", f(mean(t2, f"ece_{rule}_ew")), SRC_FU + f" ece_{rule}_ew, count, T=2")
    put(f"fuTwoEce{rk}M", f(mean(t2, f"ece_{rule}_em")), SRC_FU + f" ece_{rule}_em, count, T=2")
put("fuThirteenWithin", str(sum(r["predicted_within_on_s_interval"] for r in t2)), SRC_FU + " predicted_within_on_s_interval, count, T=2")
fu_count_all = [r for r in fu_rows if r["readout"] == "count"]
put("fuCountScaleOrderRuns", str(sum(r["r1_gives_a_the_larger_scale"] for r in fu_count_all)), SRC_FU + " r1_gives_a_the_larger_scale, count")
put("fuCountRuns", str(len(fu_count_all)), SRC_FU + " count rows")
put("fuTwoScaleOrderRuns", str(sum(r["r1_gives_a_the_larger_scale"] for r in t2)), SRC_FU + " r1_gives_a_the_larger_scale, count, T=2")
acc_all = [r for r in fu_rows if r["readout"] == "accumulator"]
put("fuAccumDeltaMin", f(min(r["delta"] for r in acc_all), d=5, sign=True), SRC_FU + " delta, accumulator, smallest run")
put("fuAccumDeltaMax", f(max(r["delta"] for r in acc_all), d=5, sign=True), SRC_FU + " delta, accumulator, largest run")
high = [r for r in fu_count_all if r["timesteps"] >= 4]
put("fuHighDeltaMin", f(min(r["delta"] for r in high), sign=True), SRC_FU + " delta, count, T>=4, smallest run")
put("fuHighDeltaMax", f(max(r["delta"] for r in high), sign=True), SRC_FU + " delta, count, T>=4, largest run")
put("fuHighRuns", str(len(high)), SRC_FU + " count rows at T>=4")
put("fuHighExclAll", str(sum(excl_(r["delta_ci"]) for r in high)), SRC_FU + " count rows at T>=4 whose delta interval excludes zero")
for T in (4, 6, 8, 10):
    put(f"fuCount{TKEY[T]}DeltaFine", f(fu_pred["14"]["mean_delta_by_t"][str(T)], d=5, sign=True),
        SRC_FU + f" predictions.14 mean_delta_by_t, T={T}, five decimals")
    assert abs(fu_pred["14"]["mean_delta_by_t"][str(T)] - mean(fu_cell("count", T), "delta")) < 1e-15
acc_excl = [r for r in acc_all if excl_(r["delta_ci"])]
assert len(acc_excl) == fu_pred["15"]["intervals_excluding_zero"]
if len(acc_excl) == 1:
    put("fuAccumExclT", str(acc_excl[0]["timesteps"]), SRC_FU + " the accumulator run whose delta interval excludes zero")
    put("fuAccumExclDelta", f(acc_excl[0]["delta"], d=5, sign=True), SRC_FU + " delta of that run")
    put("fuAccumExclLo", f(acc_excl[0]["delta_ci"][0], d=5, sign=True), SRC_FU + " delta_ci of that run")
    put("fuAccumExclHi", f(acc_excl[0]["delta_ci"][1], d=5, sign=True), SRC_FU + " delta_ci of that run")
put("fuPredWithinAll", str(sum(r["predicted_within_on_s_interval"] for r in fu_count_all)),
    SRC_FU + " predicted_within_on_s_interval, all count rows")
missed = [r for r in fu_count_all if not r["predicted_within_on_s_interval"]]
put("fuPredMissAll", str(len(missed)), SRC_FU + " count rows whose identity lies outside the on-S interval")
# what the paper says of those runs: T >= 6, B given the larger scale, a handful of pairs
assert all(r["timesteps"] >= 6 and not r["r1_gives_a_the_larger_scale"]
           and r["p_s_and_disagree"] * r["n_test"] * r["pairings"] < 10 for r in missed)
assert all(r["n_test"] * r["pairings"] == 200000 for r in fu_rows)
late = [r for r in fu_count_all if r["timesteps"] >= 6]
put("fuLateRuns", str(len(late)), SRC_FU + " count rows at T>=6")
put("fuLateScaleB", str(sum(not r["r1_gives_a_the_larger_scale"] for r in late)),
    SRC_FU + " count rows at T>=6 where temperature scaling gave B the larger scale")
put("fuTwoPSDisagree", f(mean(t2, "p_s_and_disagree")), SRC_FU + " p_s_and_disagree, count, T=2")
put("fuFourPSDisagree", f(mean(fu_cell("count", 4), "p_s_and_disagree"), d=5), SRC_FU + " p_s_and_disagree, count, T=4")
put("fuTwoCiLowMax", f(max(r["delta_ci"][0] for r in t2), sign=True), SRC_FU + " delta_ci lower end, count, T=2, largest")
for r in fu_rows:
    assert abs(r["delta"] - (r["delta_on_s"] + r["delta_off_s"])) < 1e-12
excl = lambda ci: ci[0] > 0 or ci[1] < 0
assert fu_pred["12"]["held"] == all(r["delta_ci"][0] > 0 for r in t2)
assert fu_pred["13"]["held"] == all(r["delta_on_s_ci"][0] <= r["predicted_on_s"] <= r["delta_on_s_ci"][1] for r in t2)
assert fu_pred["15"]["held"] == (not any(excl(r["delta_ci"]) for T in (2, 6, 10) for r in fu_cell("accumulator", T)))
for key, name in (("12", "Twelve"), ("13", "Thirteen"), ("14", "Fourteen"), ("15", "Fifteen")):
    put("verdict" + name, verdict(fu_pred[key]["held"]), SRC_FU + f" predictions.{key}")
put("fuThirteenDraftWithin", str(sum(fu_pred["13"]["drafted_within_interval"])), SRC_FU + " predictions.13 drafted_within_interval")
put("fuHighTExcl", str(fu_pred["14"]["intervals_excluding_zero_at_8_and_10"]), SRC_FU + " predictions.14")
put("fuAccumExcl", str(fu_pred["15"]["intervals_excluding_zero"]), SRC_FU + " predictions.15")
put("fuAccumRuns", str(fu_pred["15"]["runs"]), SRC_FU + " predictions.15")
put("fuScaleOrderRuns", str(sum(r["r1_gives_a_the_larger_scale"] for r in fu_rows)), SRC_FU + " r1_gives_a_the_larger_scale")
put("fuRuns", str(len(fu_rows)), SRC_FU)
put("probVectorsTenTwo", thousands((2 + 1) ** 10 - 2 ** 10), DERIVED + " (T+1)^C - T^C")
held += [fu_pred[k_]["held"] for k_ in ("12", "13", "14", "15")]
put("predictionsAll", str(len(held)), "NOTES.md predictions 6a to 15")
put("predictionsAllHeld", str(sum(held)), "verdict blocks of the five experiment files")
put("predictionsAllFalsified", str(len(held) - sum(held)), "verdict blocks of the five experiment files")

# ------------------------------------------------------------------ write
with open(OUT, "w") as fh:
    fh.write("% Generated by paper/make_numbers.py from results/ -- do not edit by hand.\n")
    fh.write("% Each definition names the artifact it was read from.\n")
    for name, value, source in macros:
        fh.write(f"\\newcommand{{\\{name}}}{{{value}}}% {source}\n")
print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(macros)} macros")
for name, value, source in macros:
    print(f"  \\{name:<24} {value}")
