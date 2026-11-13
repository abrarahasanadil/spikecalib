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
OUT = os.environ.get("SPIKECALIB_NUMBERS_OUT", os.path.join(ROOT, "paper", "numbers.tex"))   # tests write elsewhere
DERIVED = "[DERIVED] arithmetic in paper/make_numbers.py"

macros: list[tuple[str, str, str]] = []


def put(name: str, value: str, source: str) -> None:
    assert name.isalpha(), name
    assert name not in {m[0] for m in macros}, name
    macros.append((name, value, source))


def f(x: float, d: int = 4, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0.0:   # a value that rounds to zero is printed without a sign
        return f"{0.0:.{d}f}"
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
put("ceilThreeTwoShort", f(ceiling(2, 3), 2), DERIVED + ": e^2/(e^2+2), the cap of a softmax over two votes and three classes")
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

# The post-review A7 checkpoints (*_converged_*_weights.pt) are not among the study's trained
# models; they are counted separately in the A7 section.
weights = [w for w in glob.glob(os.path.join(R, "*_weights.pt")) if "_converged_" not in w]
assert weights, "no trained weights in results/: unpack the release asset into results/ first"
assert all(json.load(open(w.replace("_weights.pt", "_record.json"))) for w in weights)
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
all_records = [json.load(open(p_)) for p_ in sorted(glob.glob(os.path.join(R, "*_record.json")))]
assert all(r_["early_stopped"] for r_ in all_records), "a training record did not stop early"
assert all(r_["epochs_run"] < r_["epochs_requested"] for r_ in all_records), "a training record reached its epoch cap"
put("recordsAll", str(len(all_records)), "results/*_record.json, every training record")
put("epochsAllMin", str(min(r_["epochs_run"] for r_ in all_records)), "results/*_record.json epochs_run, smallest")
put("epochsAllMax", str(max(r_["epochs_run"] for r_ in all_records)), "results/*_record.json epochs_run, largest")
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
n_count_weights = sum(1 for p in glob.glob(os.path.join(R, "*_weights.pt")) if "_converged_" not in p
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
    cost = 100 * (mean(rs, "count_test_accuracy") - mean(rs, "test_accuracy"))
    if cost > 0:      # a cost only where the learned scale lowered accuracy
        put(f"accCost{key}", f"{cost:.1f}",
            SRC_SC + " count_test_accuracy minus test_accuracy, in points, one decimal")
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
assert all(r["mean_conf_minus_acc"] > 0 for r in c2 + c6)   # "positive in all ... runs"
assert all(r["test_accuracy"] < r["count_test_accuracy"] for r in c6)   # "less accurate than the count readout in every seed" at T = 6
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
_lv = lambda ro: [r["n_distinct_confidences"] for r in load("resolution_experiment.json")["check"]["rows"]
                  if r["dataset"] == "nmnist" and r["n_blocks"] == 2 and r["timesteps"] == 2 and r["readout"] == ro]
put("levelsScMinNmTwo", str(min(_lv("scaled-count"))), SRC_RES + " check rows, N-MNIST T=2, two blocks, scaled count: fewest distinct values in a seed")
put("levelsScMaxNmTwo", str(max(_lv("scaled-count"))), SRC_RES + " check rows, N-MNIST T=2, two blocks, scaled count: most distinct values in a seed")
put("levelsCtMinNmTwo", str(min(_lv("count"))), SRC_RES + " check rows, N-MNIST T=2, two blocks, count: fewest distinct values in a seed")
put("levelsCtMaxNmTwo", str(max(_lv("count"))), SRC_RES + " check rows, N-MNIST T=2, two blocks, count: most distinct values in a seed")
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
assert not any(p_["differ"] for p_ in cc_count if p_["timesteps"] == 10)   # "all at T = 2 and T = 6"
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
_tt = [r["temperature"] for r in cc["rows"] if r["readout"] == "count" and r["timesteps"] == 2 and r["arm"] == "temperature"]
put("capAtTempNcTwo", f(st.mean(math.exp(2 / t) / (math.exp(2 / t) + 100) for t in _tt), 2),
    DERIVED + ": cap e^(2/t) / (e^(2/t) + 100) at each seed's fitted temperature, N-Caltech101 T=2 count, seed mean (" + SRC_CC + ")")
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
put("ccScaledDiffer", str(sum(p_["differ"] for p_ in sc_pairs)), SRC_CC + " paired, scaled-count: runs resolved under the registered rule")
# main.tex names the resolved scaled-count runs
assert sorted((p_["timesteps"], p_["seed"], p_["favours"]) for p_ in sc_pairs if p_["differ"]) == \
    [(2, 1, "temperature"), (2, 2, "isotonic"), (6, 2, "temperature")]
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
    put(f"csDifferTotal{k}", str(sum(x["differ"] for x in cell)), SRC_CS + f" summary, n_cal={n}, draws that differ, all seeds")
    if n == 5000:
        assert sum(x["differ"] for x in cell) == 0   # main.tex Table A2: "in none at 5,000"
    if n == 1000:
        assert max(x["differ"] for x in cell) < 10   # a tie as registered: fewer than 10 of 20 in every seed
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
_ppr = {r["n_test"] * fu["pairings"] for r in fu["rows"]}
assert len(_ppr) == 1, _ppr
put("fuPairsPerRun", thousands(_ppr.pop()), SRC_FU + " rows n_test x pairings: fused pairs per run")
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
assert sum(excl_(r["delta_ci"]) for r in high) == 0   # "the interval includes zero in all ... runs"
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
assert all(not r["r1_gives_a_the_larger_scale"] for r in late)   # fusion_appendix.tex: "in every one of the ... count runs"
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

# ------------------------------------------------------------------ predictions 1 to 5, with their parts
# The same computation the dashboard's ledger uses: each registered condition re-applied to the
# artifacts, and the export stops if a verdict disagrees with the one NOTES.md records.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dashboard_summary import early_predictions  # noqa: E402

with open(os.path.join(R, "sweep_nmnist.csv")) as fh:
    sweep_text = fh.read()
early = early_predictions(load("readout_experiment.json")["rows"], load("conditional_ceiling.json")["rows"],
                          load("arch_matched.json")["rows"], sweep_text)
EP = {r["id"]: r for r in early}
SRC_EP = "scripts/dashboard_summary.py early_predictions"
V = lambda pid: EP[pid]["values"]  # noqa: E731
put("epOneIso", str(V("1")["ew_iso"]), SRC_EP + " 1, results/sweep_nmnist.csv iso_minus_temp_ci_high")
put("epOneRuns", str(V("1")["runs"]), SRC_EP + " 1")
put("epOneIsoT", str(V("1")["iso_t"]), SRC_EP + " 1")
put("epOneIsoSeed", str(V("1")["iso_seed"]), SRC_EP + " 1")
put("epOneEm", str(V("1")["em_res"]), SRC_EP + " 1, iso_minus_temp_em_resolvable")
put("epOneBoth", str(V("1")["both"]), SRC_EP + " 1")
put("epTwoMemEce", f(V("2.1")["mem"]), SRC_EP + " 2.1, results/readout_experiment.json ece_equal_width, membrane, T=2")
put("epTwoCountEce", f(V("2.1")["count"]), SRC_EP + " 2.1, ece_equal_width, count, T=2")
put("epTwoMemAcc", f(V("2.1")["acc_mem"]), SRC_EP + " 2.1, test_accuracy, membrane, T=2")
put("epTwoCountAcc", f(V("2.1")["acc_count"]), SRC_EP + " 2.1, test_accuracy, count, T=2")
put("epTwoWcEce", f(V("2.2")["wc"]), SRC_EP + " 2.2, ece_equal_width, weighted-count, T=2")
put("epTwoLGap", f(-V("2.L")["gap2"]), SRC_EP + " 2.L, accuracy minus mean confidence, membrane, T=2")
put("epTwoLUnder", str(V("2.L")["under2"]), SRC_EP + " 2.L")
put("epTwoLSeeds", str(V("2.L")["seeds"]), SRC_EP + " 2.L")
put("epThreeDefSix", f(V("3.1")["d6"]), SRC_EP + " 3.1, AUROC count minus accumulator, T=6")
put("epThreeDefTen", f(V("3.1")["d10"]), SRC_EP + " 3.1, AUROC count minus accumulator, T=10")
put("epThreeEce", f(V("3.1")["ece"]), SRC_EP + " 3.1, ece_equal_width, accumulator, T=2")
put("epThreeWins", str(V("3.1")["wins"]), SRC_EP + " 3.1, seeds where the accumulator matches or beats count")
put("epThreeBeats", str(V("3.2")["beats"]), SRC_EP + " 3.2")
put("epThreeAuAcc", f(V("3.3")["au_acc"]), SRC_EP + " 3.3, auroc, accumulator, T=2")
put("epThreeAuMem", f(V("3.3")["au_mem"]), SRC_EP + " 3.3, auroc, membrane, T=2")
put("epFourShare", f"{100 * V('4.1')['share']:.0f}", SRC_EP + " 4.1, results/conditional_ceiling.json share_explained_conditional")
put("epFourCond", f(V("4.1")["cond"]), SRC_EP + " 4.1, conditional_ceiling_mean, N-Caltech101, T=6")
put("epFourAcc", f(V("4.1")["acc"]), SRC_EP + " 4.1, test_accuracy, N-Caltech101, T=6")
put("epFourGapTen", f(V("4.2")["gap"]), SRC_EP + " 4.2, ceiling_gap_theo_minus_cond, N-MNIST, T=6")
put("epFourGapHundred", f(V("4.3")["gap"]), SRC_EP + " 4.3, ceiling_gap_theo_minus_cond, N-Caltech101, T=6")
mant, ex = f"{V('5.1')['worst']:.1e}".split("e")
put("epFiveWorst", f"${mant}\\times10^{{{int(ex)}}}$", SRC_EP + " 5.1, results/arch_matched.json ceiling_signed_excess, largest")
put("epFiveRuns", str(V("5.1")["runs"]), SRC_EP + " 5.1, count rows")
put("epFiveGainFour", f(V("5.2")["gain4"], sign=True), SRC_EP + " 5.2, four blocks")
put("epFiveGainTwo", f(V("5.2")["gain2"], sign=True), SRC_EP + " 5.2, two blocks")
put("epFiveResFour", f(V("5.3")["res4"]), SRC_EP + " 5.3, residual, four blocks")
put("epFiveResTwo", f(V("5.3")["res2"]), SRC_EP + " 5.3, residual, two blocks")
put("epFiveShareFour", f"{100 * V('5.4')['share4']:.0f}", SRC_EP + " 5.4, four blocks")
put("epFiveShareTwo", f"{100 * V('5.4')['share2']:.0f}", SRC_EP + " 5.4, two blocks")
VNAME = {"1": "One", "2.1": "TwoOne", "2.2": "TwoTwo", "2.L": "TwoL", "3.1": "ThreeOne", "3.2": "ThreeTwo",
         "3.3": "ThreeThree", "4.1": "FourOne", "4.2": "FourTwo", "4.3": "FourThree",
         "5.1": "FiveOne", "5.2": "FiveTwo", "5.3": "FiveThree", "5.4": "FiveFour"}
assert set(VNAME) == set(EP)
for pid, name in VNAME.items():
    put("verdictEp" + name, verdict(EP[pid]["held"]), SRC_EP + f" {pid}, checked against NOTES.md")
eh = sum(r["held"] for r in early)
put("predictionsEarly", str(len(early)), "NOTES.md predictions 1 to 5 and their parts")
put("predictionsEarlyHeld", str(eh), SRC_EP)
put("predictionsEarlyFalsified", str(len(early) - eh), SRC_EP)
put("predictionsEverything", str(len(early) + len(held)), "NOTES.md, every prediction and part, 1 to 15")
put("predictionsEverythingHeld", str(eh + sum(held)), "verdict blocks and " + SRC_EP)
put("predictionsEverythingFalsified", str(len(early) - eh + len(held) - sum(held)), "verdict blocks and " + SRC_EP)


# ================================================================== post-review analyses
# Added after review, not pre-registered. Thresholds fixed in NOTES.md, "Post-review
# analyses, thresholds fixed 2026-11-12 before computation", before any of these was computed.
WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 8: "Eight", 10: "Ten"}


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}"


# ------------------------------------------------------------------ A1, A2: the floor on ECE
fl = load("ece_floor.json")
SRC_FL = "results/ece_floor.json"
FL_CELLS = {"NmTwo": ("nmnist", "32-64", 2), "NmFour": ("nmnist", "32-64", 4), "NmSix": ("nmnist", "32-64", 6),
            "NmDeepTwo": ("nmnist", "32-64-128-128", 2), "NmDeepSix": ("nmnist", "32-64-128-128", 6),
            "NcTwo": ("ncaltech101", "32-64-128-128", 2), "NcShallowTwo": ("ncaltech101", "32-64", 2)}
EST = {"W": "equal_width", "M": "equal_mass", "D": "distinct"}


# S3 (second internal review): N-MNIST seeds share one test set, so intervals on N-MNIST seed
# means come from one shared resample per draw (results/shared_resample_intervals.json); the
# independent-resample intervals stored with A1, A4 and A8 are reproduced there and replaced here.
sh_ = load("shared_resample_intervals.json")
SRC_SH = "results/shared_resample_intervals.json"
SH_FLOOR = {(c["channels"], c["timesteps"]): c["estimators"] for c in sh_["floor_ratio"]}
SH_ECE = {(e["timesteps"], e["arm"], e["estimator"]): e for e in sh_["ece"]}
_shw_all = ([v for c in sh_["floor_ratio"] for v in c["estimators"].values()] + sh_["ece"]
            + list(sh_["auroc"].values()) + list(sh_["same_weights"].values()))
_ratio = [(v["shared"][1] - v["shared"][0]) / (v["independent"][1] - v["independent"][0]) for v in _shw_all]
put("shWidthMin", f(min(_ratio), 2), SRC_SH + ": smallest ratio of shared to independent interval width")
put("shWidthMax", f(max(_ratio), 2), SRC_SH + ": largest ratio of shared to independent interval width")


def fl_cell(ds, ch, T):
    c = [c for c in fl["cells"] if c["dataset"] == ds and c["channels"] == ch and c["timesteps"] == T]
    assert len(c) == 1 and c[0]["binds_in"] == 3, (ds, ch, T)
    return c[0]


for key, (ds, ch, T) in FL_CELLS.items():
    c = fl_cell(ds, ch, T)
    src = SRC_FL + f" cells, {ds} {ch} T={T}"
    put("floor" + key, f(c["floor"]["mean"]), src + " floor a - p*, seed mean")
    for e, est in EST.items():
        fr = c["schemes"][est]["floor_ratio"]
        lo, hi = c["schemes"][est]["floor_ratio_ci95"]
        if ds == "nmnist":
            assert max(abs(u - v) for u, v in zip(SH_FLOOR[(ch, T)][est]["independent"], (lo, hi))) < 1e-12
            lo, hi = SH_FLOOR[(ch, T)][est]["shared"]
        assert fr["max"] <= 1.0 + 1e-12  # the floor never exceeds the ECE
        put(f"fr{key}{e}", pct(fr["mean"]), src + f" floor_ratio {est}, mean over seeds, per cent")
        put(f"fr{key}{e}Sd", pct(fr["sd"]), src + f" floor_ratio {est}, standard deviation over seeds, percentage points")
        put(f"fr{key}{e}Min", pct(fr["min"]), src + f" floor_ratio {est}, smallest seed")
        put(f"fr{key}{e}Max", pct(fr["max"]), src + f" floor_ratio {est}, largest seed")
        isrc = (SRC_SH + f" floor_ratio, {ch} T={T} {est}, shared resample") if ds == "nmnist" else (src + f" floor_ratio_ci95 {est}")
        put(f"fr{key}{e}Lo", pct(lo), isrc + ", lower")
        put(f"fr{key}{e}Hi", pct(hi), isrc + ", upper")
        put(f"ece{key}{e}", f(c["schemes"][est]["ece"]["mean"]), src + f" ece {est}, seed mean")
        put(f"ece{key}{e}Sd", f(c["schemes"][est]["ece"]["sd"]), src + f" ece {est}, sd over seeds")
# overconfident bins where the cap binds hardest
nm2, nc2 = fl_cell("nmnist", "32-64", 2), fl_cell("ncaltech101", "32-64-128-128", 2)
for est in ("equal_width", "equal_mass"):
    assert max(nc2["schemes"][est]["n_overconfident_bins"]) == 0
put("overNmTwoMaxBins", str(max(nm2["schemes"]["equal_width"]["n_overconfident_bins"]
                                + nm2["schemes"]["equal_mass"]["n_overconfident_bins"])),
    SRC_FL + " N-MNIST T=2: most overconfident bins in any seed, either binned estimator")
put("overNmTwoShare", pct(max(nm2["schemes"]["equal_width"]["share_in_overconfident_bins"]["max"],
                              nm2["schemes"]["equal_mass"]["share_in_overconfident_bins"]["max"])),
    SRC_FL + " N-MNIST T=2: largest share of test samples in overconfident bins, per cent")
dn = nc2["schemes"]["distinct"]
put("overNcTwoDistinctMin", str(min(dn["n_overconfident_bins"])), SRC_FL + " N-Caltech101 T=2, distinct values: fewest overconfident groups in a seed")
put("overNcTwoDistinctMax", str(max(dn["n_overconfident_bins"])), SRC_FL + " N-Caltech101 T=2, distinct values: most overconfident groups in a seed")
put("overNcTwoDistinctShare", pct(dn["share_in_overconfident_bins"]["max"], 1), SRC_FL + " N-Caltech101 T=2, distinct values: largest share in overconfident groups, per cent")
put("overNcTwoDistinctGap", f(dn["largest_overconfident_gap"]), SRC_FL + " N-Caltech101 T=2, distinct values: largest overconfident gap")
# round-1 checks: where the ratio's shortfall from one comes from. ECE - (a - p*) = (p* - pbar) + (ECE - (a - pbar))
put("pbarNmTwo", f(nm2["mean_confidence"]["mean"]), SRC_FL + " cells, N-MNIST two blocks T=2: mean confidence, seed mean")
put("pbarNcTwo", f(nc2["mean_confidence"]["mean"]), SRC_FL + " cells, N-Caltech101 four blocks T=2: mean confidence, seed mean")
_over_share = [(r["schemes"][e]["ece"] - (r["accuracy"] - r["mean_confidence"])) / r["schemes"][e]["ece"]
               for r in fl["runs"] if r["binds"] for e in ("equal_width", "equal_mass")]
_short_share = [(r["cap"] - r["mean_confidence"]) / r["schemes"][e]["ece"]
                for r in fl["runs"] if r["binds"] for e in ("equal_width", "equal_mass")]
# in every binding run and binned estimator the shortfall p* - pbar exceeds the overconfidence term
assert all(o < sh for o, sh in zip(_over_share, _short_share))
put("overShareMax", pct(max(_over_share), 1), SRC_FL + " runs where the cap binds, both binned estimators: largest (ECE - (a - pbar)) / ECE, per cent")
# A8: are those groups more than sampling noise?
_nc2_runs = [r for r in fl["runs"] if r["dataset"] == "ncaltech101" and r["channels"] == "32-64-128-128"
             and r["timesteps"] == 2]
assert len(_nc2_runs) == 3
_dr = [r["schemes"]["distinct"] for r in _nc2_runs]
assert sum(d["overconfident_significant_005"] for d in _dr) == 0          # "none is significantly overconfident"
assert all(d["n_overconfident_bins"] < d["expected_negative_gap_groups_if_calibrated"] for d in _dr)  # "fewer in every seed"
assert len({d["overconfident_median_size"] for d in _dr}) == 1
put("overNcTwoMedianSize", f"{_dr[0]['overconfident_median_size']:.0f}", SRC_FL + " runs, N-Caltech101 T=2, distinct: median size of the overconfident groups, every seed (A8)")
put("overNcTwoMinP", f"{min(d['overconfident_min_p'] for d in _dr):.2f}", SRC_FL + " runs, N-Caltech101 T=2, distinct: smallest one-sided binomial p-value of an overconfident group (A8)")
put("overNcTwoExpMin", f"{min(d['expected_negative_gap_groups_if_calibrated'] for d in _dr):.0f}", SRC_FL + " runs, N-Caltech101 T=2: fewest negative-gap groups expected under calibration, per seed (A8)")
put("overNcTwoExpMax", f"{max(d['expected_negative_gap_groups_if_calibrated'] for d in _dr):.0f}", SRC_FL + " runs, N-Caltech101 T=2: most negative-gap groups expected under calibration, per seed (A8)")
# second internal review: the binomial test has no power here. Largest overconfident group, and the
# smallest group in which even zero correct predictions would be significant at 5% at the cap.
_over_n = [b["n"] for d in _dr for b in d["bins"] if b["gap"] < 0]
put("overNcTwoMaxSize", str(max(_over_n)), SRC_FL + " runs, N-Caltech101 four blocks T=2, distinct: largest overconfident group, samples")
_cap101 = ceiling(2, 101)
_nsig = math.floor(math.log(0.05) / math.log(1 - _cap101)) + 1
assert (1 - _cap101) ** _nsig < 0.05 <= (1 - _cap101) ** (_nsig - 1) and max(_over_n) < _nsig
put("overNcTwoSigN", str(_nsig), DERIVED + ": smallest n with (1 - p*)^n < 0.05 at p* = p*(2, 101)")
put("groupsNcTwoMin", str(min(dn["occupied_bins"]["values"])), SRC_FL + " N-Caltech101 T=2: fewest distinct values in a seed")
put("groupsNcTwoMax", str(max(dn["occupied_bins"]["values"])), SRC_FL + " N-Caltech101 T=2: most distinct values in a seed")
# A2: equal-mass occupancy
assert nm2["schemes"]["equal_mass"]["occupied_bins"]["values"] == [1, 1, 1]
put("emBinsNmTwo", "1", SRC_FL + " N-MNIST T=2: effective equal-mass bins, every seed (asserted)")
put("emOccNmTwo", thousands(nm2["schemes"]["equal_mass"]["max_occupancy"]), SRC_FL + " N-MNIST T=2: the one bin's occupancy")
nm4 = fl_cell("nmnist", "32-64", 4)
assert set(nm4["schemes"]["equal_mass"]["occupied_bins"]["values"]) == {2}
put("emBinsNmFour", "2", SRC_FL + " N-MNIST T=4: effective equal-mass bins, every seed (asserted)")
put("emOccMinNmFour", thousands(nm4["schemes"]["equal_mass"]["min_occupancy"]), SRC_FL + " N-MNIST T=4: smallest equal-mass bin")
_nm4_runs = [r for r in fl["runs"] if r["dataset"] == "nmnist" and r["channels"] == "32-64" and r["timesteps"] == 4]
put("emOccMinMaxNmFour", thousands(max(r["schemes"]["equal_mass"]["min_occupancy"] for r in _nm4_runs)), SRC_FL + " runs, N-MNIST T=4: largest smaller-bin size over seeds")
put("emOccMaxNmFour", thousands(nm4["schemes"]["equal_mass"]["max_occupancy"]), SRC_FL + " N-MNIST T=4: largest equal-mass bin")
put("emBinsMinNcTwo", str(int(min(nc2["schemes"]["equal_mass"]["occupied_bins"]["values"]))), SRC_FL + " N-Caltech101 T=2: fewest effective equal-mass bins")
put("emBinsMaxNcTwo", str(int(max(nc2["schemes"]["equal_mass"]["occupied_bins"]["values"]))), SRC_FL + " N-Caltech101 T=2: most effective equal-mass bins")
# main.tex says "every seed has \emBinsMinNcTwo{} equal-mass bins"; true only while min == max.
assert min(nc2["schemes"]["equal_mass"]["occupied_bins"]["values"]) == max(nc2["schemes"]["equal_mass"]["occupied_bins"]["values"])
put("emOccMinNcTwo", str(nc2["schemes"]["equal_mass"]["min_occupancy"]), SRC_FL + " N-Caltech101 T=2: smallest equal-mass bin")
put("emOccMaxNcTwo", str(nc2["schemes"]["equal_mass"]["max_occupancy"]), SRC_FL + " N-Caltech101 T=2: largest equal-mass bin")
put("emTarget", str(fl["n_bins"]), "results/ece_floor.json n_bins")
put("ewBinsNcTwo", str(int(max(nc2["schemes"]["equal_width"]["occupied_bins"]["values"]))), SRC_FL + " N-Caltech101 T=2: occupied equal-width bins")
all_em = [r["schemes"]["equal_mass"]["occupied_bins"] for r in fl["runs"]]
put("emBinsMin", str(min(all_em)), SRC_FL + " runs: fewest effective equal-mass bins over the 36 count test sets")
put("countTestSets", str(len(fl["runs"])), SRC_FL + " runs")
_slack = [c for c in fl["cells"] if c["binds_in"] == 0]
_slack_under = [c for c in _slack if c["mean_confidence"]["mean"] < c["accuracy"]["mean"]]
_slack_over = [c for c in _slack if c["mean_confidence"]["mean"] > c["accuracy"]["mean"]]
assert [(c["dataset"], c["timesteps"], c["channels"]) for c in _slack_over] == [("ncaltech101", 10, "32-64-128-128")]
assert all(c["binds_in"] in (0, 3) for c in fl["cells"])
put("slackSettings", str(len(_slack)), SRC_FL + " cells where the cap is slack in every seed")
put("slackUnder", str(len(_slack_under)), SRC_FL + " cells where the cap is slack and the count readout is underconfident (seed means)")
put("bindsRuns", str(sum(r["binds"] for r in fl["runs"])), SRC_FL + " runs where accuracy exceeds the cap")

# ------------------------------------------------------------------ A3: equivalence tests
cm_ = load("calibration_metrics.json")
SRC_CM = "results/calibration_metrics.json"
DELTA = {"Strict": 0.005, "Mid": 0.01, "Wide": 0.02}
assert cm_["deltas"] == list(DELTA.values())
for name, d in DELTA.items():
    put("delta" + name, f"{d:g}", "NOTES.md, Post-review analyses, A3; " + SRC_CM + " deltas")
GROUPS = {"Nm": "nmnist count", "NcCount": "ncaltech101 count", "NcAccum": "ncaltech101 accumulator",
          "NcScaled": "ncaltech101 scaled-count"}
for g, label in GROUPS.items():
    srow = next(x for x in cm_["tost_summary"] if x["group"] == label)
    put(f"tost{g}Runs", str(srow["runs"]), SRC_CM + f" tost_summary, {label}, runs")
    put(f"tost{g}Differ", str(srow["differ_registered_rule"]), SRC_CM + f" tost_summary, {label}, differ under the registered rule")
    for name, d in DELTA.items():
        for e, est in (("W", "equal_width"), ("M", "equal_mass"), ("Both", "both")):
            put(f"tost{g}{e}{name}", str(srow[f"equivalent_{est}_{d}"]),
                SRC_CM + f" tost_summary, {label}, equivalent at {d} under {est}")
tn = [t for t in cm_["tost"] if t["dataset"] == "nmnist"]
put("tostNmWidest", f(max(max(abs(t["ci90"][0]), abs(t["ci90"][1])) for t in tn)),
    SRC_CM + " tost, N-MNIST: largest |endpoint| of any 90% interval")
put("tostNmWidestBasic", f(max(max(abs(t["ci90_basic"][0]), abs(t["ci90_basic"][1])) for t in tn)),
    SRC_CM + " tost, N-MNIST: largest |endpoint| of any basic 90% interval (A8)")
_em4 = [t["difference"] for t in tn if t["estimator"] == "equal_mass" and t["timesteps"] == 4]
assert len(_em4) == 3 and all(d < 0 for d in _em4)   # isotonic ahead in all three seeds
put("emFourIsoMin", f(min(abs(d) for d in _em4)), SRC_CM + " tost, N-MNIST T=4 equal-mass: smallest isotonic advantage")
put("emFourIsoMax", f(max(abs(d) for d in _em4)), SRC_CM + " tost, N-MNIST T=4 equal-mass: largest isotonic advantage")
_fail_strict = sorted((t["timesteps"], t["seed"]) for t in tn if t["estimator"] == "equal_mass" and not t["equivalent_0.005"])
assert _fail_strict == [(4, 0), (4, 1), (4, 2)], _fail_strict   # the runs not equivalent within 0.005 are the T = 4 equal-mass runs

put("tostTests", str(len(cm_["tost"]) * len(DELTA)), SRC_CM + " tost entries x margins")
put("tostRunsAll", str(len({t["name"] for t in cm_["tost"]})), SRC_CM + " tost runs")
# the published paired verdicts are reproduced (asserted in scripts/calibration_metrics.py)
srow = next(x for x in cm_["tost_summary"] if x["group"] == "ncaltech101 count")
assert srow["differ_registered_rule"] == 5
# A8: the same tests with basic (reverse-percentile) intervals, and the bootstrap bias
for g, label in GROUPS.items():
    srow = next(x for x in cm_["tost_summary"] if x["group"] == label)
    put(f"tost{g}DifferBasic", str(srow["differ_registered_rule_basic"]), SRC_CM + f" tost_summary, {label}, differ under the registered rule with basic intervals (A8)")
    put(f"tost{g}EitherBasic", str(srow["resolve_either_basic"]), SRC_CM + f" tost_summary, {label}, resolve under at least one estimator with basic intervals (A8)")
    put(f"tost{g}Either", str(srow["resolve_either"]), SRC_CM + f" tost_summary, {label}, resolve under at least one estimator, percentile intervals")
    for name, d in DELTA.items():
        put(f"tost{g}BothBasic{name}", str(srow[f"equivalent_both_basic_{d}"]), SRC_CM + f" tost_summary, {label}, equivalent at {d} under both, basic intervals (A8)")
    put(f"bias{g}Max", f(srow["largest_abs_bootstrap_bias"]), SRC_CM + f" tost_summary, {label}, largest |mean of draws - point| (A8)")
# second internal review: how often the bias exceeds the difference at N-Caltech101 T = 10 (count),
# the lenient margin in binomial standard errors, and the largest absolute N-MNIST difference
_t10 = [e for e in cm_["tost"] if e["dataset"] == "ncaltech101" and e["readout"] == "count" and e["timesteps"] == 10]
put("biasExceedsTen", str(sum(abs(e["bootstrap_bias"]) > abs(e["difference"]) for e in _t10)), SRC_CM + " tost, ncaltech101 count T=10: tests whose |bias| exceeds |difference|")
put("biasTestsTen", str(len(_t10)), SRC_CM + " tost, ncaltech101 count T=10: tests")
_nct = int(np.load(os.path.join(R, "ncaltech101_T2_seed0_split.npz"))["test"].size)
put("wideMarginSe", f"{DELTA['Wide'] / math.sqrt(0.25 / _nct):.1f}", DERIVED + ": lenient margin / sqrt(0.25 / N-Caltech101 test size), binomial standard errors")
_nmabs = [abs(e["difference"]) for e in cm_["tost"] if e["dataset"] == "nmnist"]
put("nmAbsDiffMax", f(max(_nmabs)), SRC_CM + " tost, N-MNIST: largest |isotonic - temperature| ECE difference, either estimator")
_nm_s = next(x for x in cm_["tost_summary"] if x["group"] == "nmnist count")
assert _nm_s["resolve_either"] == int(dict((m[0], m[1]) for m in macros)["resolveEither"])
# the two N-MNIST runs that the basic intervals resolve under both estimators, same direction
_res = []
for n in sorted({t["name"] for t in tn}):
    ew = next(t for t in tn if t["name"] == n and t["estimator"] == "equal_width")
    em = next(t for t in tn if t["name"] == n and t["estimator"] == "equal_mass")
    if ew["excludes_zero_95_basic"] and em["excludes_zero_95_basic"] and np.sign(ew["difference"]) == np.sign(em["difference"]):
        _res.append((n, "temperature" if ew["difference"] > 0 else "isotonic"))
assert len(_res) == _nm_s["differ_registered_rule_basic"]
assert sorted(_res) == [("nmnist_T2_seed2", "temperature"), ("nmnist_T6_seed0", "isotonic")], _res
put("penalisedTop", str(cm_["clipped"]["penalised_top_label"]), SRC_CM + " clipped.penalised_top_label (A8)")
# round-1 checks: margins against the calibrated N-MNIST ECEs; exclusions per estimator; tiny bins
_cal_nm = [r[k] for r in cm_["rows"] if r["dataset"] == "nmnist" and r["n_blocks"] == 2 and r["readout"] == "count"
           and r["arm"] != "uncalibrated" for k in ("ece_equal_width", "ece_equal_mass")]
assert max(_cal_nm) < DELTA["Mid"]   # main.tex: every calibrated N-MNIST ECE is below the middle margin
put("maxCalEceNm", f(max(_cal_nm)), SRC_CM + " rows, nmnist two blocks count, calibrated arms: largest ECE under either estimator")
_ex = {e: [t for t in tn if t["estimator"] == e and t["excludes_zero_95"]] for e in ("equal_width", "equal_mass")}
put("ewExclNm", str(len(_ex["equal_width"])), SRC_CM + " tost, N-MNIST: equal-width 95% intervals excluding zero")
put("emExclNm", str(len(_ex["equal_mass"])), SRC_CM + " tost, N-MNIST: equal-mass 95% intervals excluding zero")
from scipy.stats import binom as _binom
_nr = len({e["name"] for e in cm_["tost"] if e["dataset"] == "nmnist"})
put("emExclTail", f"{_binom.sf(len(_ex['equal_mass']) - 1, _nr, 0.05):.2f}",
    "[DERIVED] P(at least that many of the N-MNIST runs' 95% intervals exclude zero by chance), binomial with p = 0.05, independent exact intervals (" + SRC_CM + " tost)")
assert all(t["difference"] < 0 for t in _ex["equal_mass"])   # "all favour isotonic regression"
put("cmRows", str(len(cm_["rows"])), SRC_CM + " rows (run and arm)")
assert all(r["equal_mass_bins"] == 1 for r in cm_["rows"] if r["dataset"] == "nmnist" and r["n_blocks"] == 2
           and r["readout"] == "count" and r["timesteps"] == 2)   # main.tex: one equal-mass bin for every arm at N-MNIST T = 2
_all_records = [p_ for p_ in glob.glob(os.path.join(R, "*_record.json")) if "_converged_" not in p_]
_missing = len(_all_records) - len({r["name"] for r in cm_["rows"]})
assert _missing == len([p_ for p_ in _all_records if "_membrane_" in p_ or "_weighted-count_" in p_])
put("rlMissingRuns", str(_missing), "results/*_record.json against results/calibration_metrics.json rows: runs without stored outputs (membrane and weighted count)")
_sct = [r["temperature"] for r in cm_["rows"] if r["readout"] == "scaled-count" and r["dataset"] == "ncaltech101" and r["arm"] == "temperature"]
assert len(_sct) == 6 and min(_sct) > 1.0   # "divides the learned logits further in every run"
put("scTempNcMin", f(min(_sct), 2), SRC_CM + " rows, ncaltech101 scaled-count, temperature arm: smallest fitted temperature")
put("scTempNcMax", f(max(_sct), 2), SRC_CM + " rows, ncaltech101 scaled-count, temperature arm: largest fitted temperature")
# proper scores, temperature against isotonic, count readout, T = 2
def _arm(name, arm):
    return next(r for r in cm_["rows"] if r["name"] == name and r["arm"] == arm)
for key, ds in (("Nm", "nmnist"), ("Nc", "ncaltech101")):
    names = [f"{ds}_T2_seed{sd}" for sd in (0, 1, 2)]
    tb = [_arm(n, "temperature")["brier_top_label"] for n in names]
    ib = [_arm(n, "isotonic-top-label")["brier_top_label"] for n in names]
    tl = [_arm(n, "temperature")["log_loss_top_label"] for n in names]
    il = [_arm(n, "isotonic-top-label")["log_loss_top_label"] for n in names]
    put(f"brier{key}TwoTemp", f(st.mean(tb)), SRC_CM + f" rows, {ds} T=2 count, temperature: top-label Brier, seed mean")
    put(f"brier{key}TwoIso", f(st.mean(ib)), SRC_CM + f" rows, {ds} T=2 count, isotonic: top-label Brier, seed mean")
    put(f"ll{key}TwoTemp", f(st.mean(tl)), SRC_CM + f" rows, {ds} T=2 count, temperature: top-label log loss, seed mean")
    put(f"ll{key}TwoIso", f(st.mean(il)), SRC_CM + f" rows, {ds} T=2 count, isotonic: top-label log loss, seed mean")
    # main.tex: isotonic regression has the lower Brier score in every seed at T = 2 (log loss is not used to rank)
    assert all(i < t for i, t in zip(ib, tb)), key
# N-MNIST at T = 4 to 10: how often the top-label Brier score favours temperature scaling
_bd = [_arm(f"nmnist_T{T}_seed{sd}", "isotonic-top-label")["brier_top_label"] - _arm(f"nmnist_T{T}_seed{sd}", "temperature")["brier_top_label"]
       for T in (4, 6, 8, 10) for sd in (0, 1, 2)]
put("brierTempBetterNmHigh", str(sum(d > 0 for d in _bd)), SRC_CM + " rows, nmnist two blocks count, T=4 to 10: runs where temperature scaling has the lower top-label Brier score")
put("brierRunsNmHigh", str(len(_bd)), SRC_CM + " rows, nmnist two blocks count, T=4 to 10: runs compared")
put("brierDiffNmHighMax", f(max(abs(d) for d in _bd)), SRC_CM + " rows, nmnist two blocks count, T=4 to 10: largest |isotonic - temperature| top-label Brier")
put("emSmallBinRows", str(sum(r["equal_mass_min_occupancy"] < 30 for r in cm_["rows"])), SRC_CM + " rows with an equal-mass bin of fewer than 30 samples")
put("emOneSampleRows", str(sum(r["equal_mass_min_occupancy"] == 1 for r in cm_["rows"])), SRC_CM + " rows with an equal-mass bin of one sample")
assert all(r["n_penalised_top_label"] == 0 for r in cm_["rows"] if r["arm"] != "isotonic-top-label")
# A8: share of the accumulator's confidences in the top equal-width bin, N-MNIST T=2
_tb = [next(r for r in cm_["rows"] if r["name"] == f"nmnist_T2_seed{sd}_accumulator" and r["arm"] == "uncalibrated")
       ["share_top_equal_width_bin"] for sd in (0, 1, 2)]
put("topBinAccNmTwoMin", pct(min(_tb), 1), SRC_CM + " rows, nmnist T=2 accumulator uncalibrated: share_top_equal_width_bin, smallest seed, per cent (A8)")
put("topBinAccNmTwoMax", pct(max(_tb), 1), SRC_CM + " rows, nmnist T=2 accumulator uncalibrated: share_top_equal_width_bin, largest seed, per cent (A8)")
# AUROC count against accumulator at T = 6 and 10 (two blocks, uncalibrated), per seed
_above = [next(r for r in cm_["rows"] if r["name"] == f"nmnist_T{T}_seed{sd}" and r["arm"] == "uncalibrated")["auroc"]
          > next(r for r in cm_["rows"] if r["name"] == f"nmnist_T{T}_seed{sd}_accumulator" and r["arm"] == "uncalibrated")["auroc"]
          for T in (6, 10) for sd in (0, 1, 2)]
put("aurocCountAboveSixTen", str(sum(_above)), SRC_CM + " rows, nmnist T=6 and 10: seeds where the count readout's AUROC exceeds the accumulator's")
put("aurocSeedsSixTen", str(len(_above)), SRC_CM + " rows, nmnist T=6 and 10: seeds compared")
from scipy.stats import t as _tdist
for T, w in ((6, "Six"), (10, "Ten")):
    _dT = [next(r for r in cm_["rows"] if r["name"] == f"nmnist_T{T}_seed{sd}" and r["arm"] == "uncalibrated")["auroc"]
           - next(r for r in cm_["rows"] if r["name"] == f"nmnist_T{T}_seed{sd}_accumulator" and r["arm"] == "uncalibrated")["auroc"]
           for sd in (0, 1, 2)]
    _hw = _tdist.ppf(0.975, 2) * st.stdev(_dT) / 3 ** 0.5
    put(f"aurocDiff{w}", f(st.mean(_dT), sign=True), SRC_CM + f" rows, nmnist two blocks T={T}: count minus accumulator AUROC, seed mean")
    put(f"aurocDiff{w}TLo", f(st.mean(_dT) - _hw, sign=True), SRC_CM + f" rows, nmnist two blocks T={T}: t-interval over 3 seeds, lower")
    put(f"aurocDiff{w}THi", f(st.mean(_dT) + _hw, sign=True), SRC_CM + f" rows, nmnist two blocks T={T}: t-interval over 3 seeds, upper")
assert all(_above)   # main.tex: "the higher in all ... seeds"

# ------------------------------------------------------------------ A4: headline intervals and scores
hd = cm_["headline"]
ARMK = {"uncalibrated": "Unc", "temperature": "Temp", "isotonic-top-label": "Iso"}
DSK = {"nmnist": "Nm", "ncaltech101": "Nc"}
for e in hd["ece"]:
    if e["timesteps"] not in (2, 6, 10):
        continue
    k = f"hd{DSK[e['dataset']]}{WORD[e['timesteps']]}{ARMK[e['arm']]}{'W' if e['estimator'] == 'equal_width' else 'M'}"
    src = SRC_CM + f" headline.ece, {e['dataset']} T={e['timesteps']} {e['arm']} {e['estimator']}"
    put(k, f(e["mean"]), src + ", seed mean")
    put(k + "Sd", f(e["sd"]), src + ", sd over seeds")
    lo_, hi_ = e["ci95_seed_mean"]
    if e["dataset"] == "nmnist":
        sh = SH_ECE[(e["timesteps"], e["arm"], e["estimator"])]
        assert max(abs(u - v) for u, v in zip(sh["independent"], (lo_, hi_))) < 1e-12
        lo_, hi_ = sh["shared"]
        src = SRC_SH + f" ece, nmnist T={e['timesteps']} {e['arm']} {e['estimator']}, shared resample"
    put(k + "Lo", f(lo_), src + ", 95% bootstrap interval of the seed mean")
    put(k + "Hi", f(hi_), src + ", 95% bootstrap interval of the seed mean")
au = hd["auroc"]
for k, name in (("Count", "count"), ("Accum", "accumulator"), ("Diff", "count_minus_accumulator")):
    a = au[name]
    sgn = name == "count_minus_accumulator"
    put(f"hdAuroc{k}", f(a["mean"], sign=sgn), SRC_CM + f" headline.auroc.{name}, seed mean")
    put(f"hdAuroc{k}Sd", f(a["sd"]), SRC_CM + f" headline.auroc.{name}, sd over seeds")
    shA = sh_["auroc"][name]
    assert max(abs(u - v) for u, v in zip(shA["independent"], a["ci95_seed_mean"])) < 1e-12
    put(f"hdAuroc{k}Lo", f(shA["shared"][0], sign=sgn), SRC_SH + f" auroc.{name}, 95% interval, shared resample")
    put(f"hdAuroc{k}Hi", f(shA["shared"][1], sign=sgn), SRC_SH + f" auroc.{name}, 95% interval, shared resample")
assert abs(au["count"]["mean"] - float(dict((m[0], m[1]) for m in macros)["aurocCountNmTwo"])) < 5e-5
# A8: t-interval over the three seeds for the AUROC difference
from scipy.stats import t as _student_t
_d = au["count_minus_accumulator"]
_half = _student_t.ppf(0.975, 2) * _d["sd"] / 3 ** 0.5
put("hdAurocDiffTLo", f(_d["mean"] - _half, sign=True), SRC_CM + " headline.auroc.count_minus_accumulator, t-interval over 3 seeds, lower (A8)")
put("hdAurocDiffTHi", f(_d["mean"] + _half, sign=True), SRC_CM + " headline.auroc.count_minus_accumulator, t-interval over 3 seeds, upper (A8)")
assert _d["mean"] + _half < 0
put("clipTop", thousands(cm_["clipped"]["top_label"]), SRC_CM + " clipped.top_label")
# N-Caltech101 AUROC at T = 2 (four blocks, uncalibrated), seed means, for the README
for k, suffix in (("Count", ""), ("Accum", "_accumulator")):
    _au = [next(r for r in cm_["rows"] if r["name"] == f"ncaltech101_T2_seed{sd}{suffix}" and r["arm"] == "uncalibrated")["auroc"]
           for sd in (0, 1, 2)]
    put(f"hdNcAuroc{k}", f(st.mean(_au)), SRC_CM + f" rows, ncaltech101 four blocks T=2 {'count' if not suffix else 'accumulator'}, uncalibrated: AUROC, seed mean")
assert all(r["n_clipped_top_label"] == 0 for r in cm_["rows"] if r["arm"] != "isotonic-top-label")
assert all(r["n_clipped_top_label"] == r["n_clipped_top_label_exact_zero_or_one"] for r in cm_["rows"])
put("clipTopTotal", thousands(cm_["clipped"]["scores_computed_top_label"]), SRC_CM + " clipped.scores_computed_top_label")
put("clipMulti", str(cm_["clipped"]["multiclass"]), SRC_CM + " clipped.multiclass")
put("refitRuns", str(len({r["name"] for r in cm_["rows"]})), SRC_CM + " rows, runs refitted")

# run-level table rows (appendix): one macro per row, generated, never typed
RDK = {"count": "Count", "accumulator": "Accum", "scaled-count": "Scaled"}
CFG_ORDER = []
for r in cm_["rows"]:
    cfg = (r["dataset"], r["n_blocks"], r["readout"], r["timesteps"])
    if cfg not in CFG_ORDER:
        CFG_ORDER.append(cfg)
CFG_ORDER.sort(key=lambda c: (c[0] != "nmnist", c[2] != "count", c[2] != "accumulator", c[1], c[3]))
by = {(r["dataset"], r["n_blocks"], r["readout"], r["timesteps"], r["seed"], r["arm"]): r for r in cm_["rows"]}
rl_names = []
for (ds, b, ro, T) in CFG_ORDER:
    for seed in (0, 1, 2):
        u, t_, i_ = (by[(ds, b, ro, T, seed, a)] for a in ("uncalibrated", "temperature", "isotonic-top-label"))
        key = f"rl{DSK[ds]}{RDK[ro]}{WORD[b]}{WORD[T]}{WORD[seed]}"
        src = SRC_CM + f" rows, {ds} {b} blocks {ro} T={T} seed {seed}"
        cells_a = [f(u["accuracy"])] + [f(x[k_]) for x in (u, t_, i_) for k_ in ("ece_equal_width", "ece_equal_mass")]
        put(key + "A", " & ".join(cells_a), src + ": accuracy; ECE w, m for uncalibrated, temperature, isotonic")
        cells_b = ([f(x["brier_top_label"]) for x in (u, t_, i_)] + [f(x["log_loss_top_label"]) for x in (u, t_, i_)]
                   + [f(u["auroc"])] + [f(u["nll_multiclass"]), f(t_["nll_multiclass"]),
                                        f(u["brier_multiclass"]), f(t_["brier_multiclass"])])
        put(key + "B", " & ".join(cells_b), src + ": top-label Brier and log loss per arm; AUROC; multiclass NLL and Brier")
        rl_names.append((ds, b, ro, T, seed, key))
put("rlRows", str(len(rl_names)), SRC_CM + " rows / 3 arms")


# ------------------------------------------------------------------ A5: spiking activity
act = load("spiking_activity.json")
SRC_ACT = "results/spiking_activity.json"
assert act["device"] == "mps"
for p_ in act["output_patterns_T2"]:
    k = DSK[p_["dataset"]] + ("Deep" if (p_["dataset"] == "nmnist" and p_["n_blocks"] == 4) else
                              "Shallow" if (p_["dataset"] == "ncaltech101" and p_["n_blocks"] == 2) else "")
    src = SRC_ACT + f" output_patterns_T2, {p_['dataset']} {p_['n_blocks']} blocks"
    put(f"sat{k}Two", pct(p_["saturated"]["mean"], 1), src + " saturated (top class fires at both steps, every other class silent), per cent, seed mean")
    put(f"sat{k}TwoSd", pct(p_["saturated"]["sd"], 1), src + " saturated, sd over seeds, percentage points")
    put(f"tie{k}Two", pct(p_["tie_at_top"]["mean"], 1), src + " tie for the top count, per cent, seed mean")
    put(f"tie{k}TwoSd", pct(p_["tie_at_top"]["sd"], 1), src + " tie for the top count, sd over seeds")
    put(f"topEvery{k}Two", pct(p_["top_fires_every_step"]["mean"], 1), src + " top count equals T, per cent, seed mean")
n_silent_out = [round(r["all_silent"] * (10000 if r["dataset"] == "nmnist" else 1742))
                for r in act["output_patterns_T2_runs"]]
put("silentOutMax", str(max(n_silent_out)), SRC_ACT + " output_patterns_T2_runs: most test samples with every output silent at T=2, any run")
assert sum(n > 0 for n in n_silent_out) == 1   # main.tex: "in one run"
put("silentOutRuns", str(sum(n > 0 for n in n_silent_out)), SRC_ACT + " output_patterns_T2_runs: runs with any such sample")
LAYER_ORDER = ["conv1", "conv2", "conv3", "conv4", "hidden", "output"]
act_rows = 0
for c in act["cells"]:
    for lname in LAYER_ORDER:
        if lname not in c["layers"]:
            continue
        L = c["layers"][lname]
        key = f"act{DSK[c['dataset']]}{WORD[c['n_blocks']]}{WORD[c['timesteps']]}{lname.capitalize().replace('1','One').replace('2','Two').replace('3','Three').replace('4','Four')}"
        put(key, f"{thousands(L['n_neurons'])} & {f(L['rate']['mean'], 3)} & {f(L['rate']['sd'], 3)} & "
                 f"{pct(L['silent_share']['mean'], 1)} & {pct(L['silent_share']['sd'], 1)}",
            SRC_ACT + f" cells, {c['dataset']} {c['n_blocks']} blocks T={c['timesteps']} {lname}: neurons; rate mean, sd; silent share mean, sd (per cent)")
        act_rows += 1
put("actRows", str(act_rows), SRC_ACT + " cells x layers")
hid_rates = [c["layers"]["hidden"]["rate"]["mean"] for c in act["cells"]]
conv_rates = [c["layers"][n]["rate"]["mean"] for c in act["cells"] for n in ("conv1", "conv2", "conv3", "conv4") if n in c["layers"]]
put("rateConvMin", f(min(conv_rates), 3), SRC_ACT + " cells: smallest convolutional-layer rate, seed mean")
put("rateConvMax", f(max(conv_rates), 3), SRC_ACT + " cells: largest convolutional-layer rate, seed mean")
put("rateHiddenMin", f(min(hid_rates), 3), SRC_ACT + " cells: smallest hidden-layer rate")
put("rateHiddenMax", f(max(hid_rates), 3), SRC_ACT + " cells: largest hidden-layer rate")
silent_first = [c["layers"]["conv1"]["silent_share"]["mean"] for c in act["cells"]]
put("silentFirstMin", pct(min(silent_first), 0), SRC_ACT + " cells: smallest share of first-layer neurons never firing, per cent")
put("silentFirstMax", pct(max(silent_first), 0), SRC_ACT + " cells: largest share of first-layer neurons never firing, per cent")
put("silentHiddenMax", pct(max(c["layers"]["hidden"]["silent_share"]["max"] for c in act["cells"]), 1),
    SRC_ACT + " cells: largest share of hidden neurons never firing, any seed, per cent")
# main.tex says "no hidden-layer neuron is silent in any seed"; true only while this is exactly zero.
assert max(c["layers"]["hidden"]["silent_share"]["max"] for c in act["cells"]) == 0.0
_rn = {len(L["neurons"]) for r in act["rasters"] for n, L in r["layers"].items() if n != "output"}
assert len(_rn) == 1, _rn
put("rasterNeurons", str(_rn.pop()), SRC_ACT + " rasters, neurons shown per hidden layer")

# ------------------------------------------------------------------ A6: the same weights, read differently
sw = load("same_weights_readouts.json")["rows"]
SRC_SW = "results/same_weights_readouts.json"
TRAINED_ACC = {}
for r in load("readout_experiment.json")["rows"]:
    if r["readout"] == "accumulator":
        TRAINED_ACC.setdefault(("nmnist", 2, r["timesteps"]), []).append(r["test_accuracy"])
for r in load("caltech_experiment.json")["rows"]:
    if r["readout"] == "accumulator":
        TRAINED_ACC.setdefault(("ncaltech101", 4, r["timesteps"]), []).append(r["test_accuracy"])
for r in load("arch_matched.json")["rows"]:
    if r["readout"] == "accumulator":
        k_ = (r["dataset"], r["n_blocks"], r["timesteps"])
        if k_ not in (("nmnist", 2, 2), ("nmnist", 2, 6), ("ncaltech101", 4, 2), ("ncaltech101", 4, 6)):
            TRAINED_ACC.setdefault(k_, []).append(r["test_accuracy"])
SW_ORDER = sorted({(r["dataset"], r["n_blocks"], r["timesteps"]) for r in sw},
                  key=lambda c: (c[0] != "nmnist", c[1] != (2 if c[0] == "nmnist" else 4), c[2]))
sw_cell = {}
for c in SW_ORDER:
    rs = [r for r in sw if (r["dataset"], r["n_blocks"], r["timesteps"]) == c]
    assert len(rs) == 3
    m_ = lambda path: st.mean(eval("r" + path, {"r": r}) for r in rs)  # noqa: E731
    vals = {}
    for ro in ("count", "accumulator_same_weights", "membrane_same_weights"):
        vals[ro] = {"acc": st.mean(r[ro]["uncalibrated"]["accuracy"] for r in rs),
                    "gap": st.mean(r[ro]["uncalibrated"]["mean_conf_minus_acc"] for r in rs),
                    "eceW": st.mean(r[ro]["uncalibrated"]["ece_equal_width"] for r in rs),
                    "eceM": st.mean(r[ro]["uncalibrated"]["ece_equal_mass"] for r in rs),
                    "tW": st.mean(r[ro]["temperature"]["ece_equal_width"] for r in rs),
                    "tM": st.mean(r[ro]["temperature"]["ece_equal_mass"] for r in rs)}
    sw_cell[c] = vals
    trained = TRAINED_ACC.get(c)
    tr = f(st.mean(trained)) if trained and len(trained) == 3 else "n/a"
    key = f"sw{DSK[c[0]]}{WORD[c[1]]}{WORD[c[2]]}"
    cells_ = []
    for ro in ("count", "accumulator_same_weights", "membrane_same_weights"):
        v = vals[ro]
        cells_ += [f(v["acc"]), f(v["gap"], sign=True), f(v["eceW"]), f(v["tW"])]
    put(key, " & ".join(cells_ + [tr]), SRC_SW + f" rows, {c[0]} {c[1]} blocks T={c[2]}, seed means: per readout accuracy, conf-acc, ECE w before and after temperature scaling; trained accumulator accuracy from its own experiment")
put("swRows", str(len(SW_ORDER)), SRC_SW + " cells")
nc2s = sw_cell[("ncaltech101", 4, 2)]
nc6s = sw_cell[("ncaltech101", 4, 6)]
tr2 = st.mean(TRAINED_ACC[("ncaltech101", 4, 2)])
tr6 = st.mean(TRAINED_ACC[("ncaltech101", 4, 6)])
put("swNcTwoCountAcc", f(nc2s["count"]["acc"]), SRC_SW + " N-Caltech101 T=2 count accuracy")
put("swNcTwoAccumAcc", f(nc2s["accumulator_same_weights"]["acc"]), SRC_SW + " N-Caltech101 T=2 accumulator on count weights, accuracy")
put("swNcTwoMemAcc", f(nc2s["membrane_same_weights"]["acc"]), SRC_SW + " N-Caltech101 T=2 membrane on count weights, accuracy")
put("swNcTwoTrainedAcc", f(tr2), "results/caltech_experiment.json accumulator T=2, test_accuracy, seed mean")
nm2s = sw_cell[("nmnist", 2, 2)]
put("swNmTwoCountAcc", f(nm2s["count"]["acc"]), SRC_SW + " N-MNIST two blocks T=2 count accuracy")
put("swNmTwoAccumAcc", f(nm2s["accumulator_same_weights"]["acc"]), SRC_SW + " N-MNIST two blocks T=2 accumulator on count weights, accuracy")
put("swNmTwoTrainedAcc", f(st.mean(TRAINED_ACC[("nmnist", 2, 2)])), "results/readout_experiment.json accumulator T=2, test_accuracy, seed mean")
put("swNcTwoShare", f"{100 * (nc2s['accumulator_same_weights']['acc'] - nc2s['count']['acc']) / (tr2 - nc2s['count']['acc']):.0f}",
    DERIVED + ": (same-weights accumulator - count) / (trained accumulator - count), N-Caltech101 T=2, per cent")
put("swNcSixCountAcc", f(nc6s["count"]["acc"]), SRC_SW + " N-Caltech101 T=6 count accuracy")
put("swNcSixAccumAcc", f(nc6s["accumulator_same_weights"]["acc"]), SRC_SW + " N-Caltech101 T=6 accumulator on count weights")
put("swNcSixTrainedAcc", f(tr6), "results/caltech_experiment.json accumulator T=6, test_accuracy, seed mean")
put("swNcTwoAccumGap", f(nc2s["accumulator_same_weights"]["gap"], sign=True), SRC_SW + " N-Caltech101 T=2 accumulator on count weights, conf-acc")
put("swNcSixAccumGap", f(nc6s["accumulator_same_weights"]["gap"], sign=True), SRC_SW + " N-Caltech101 T=6 accumulator on count weights, conf-acc")
put("swNcTwoMemGap", f(nc2s["membrane_same_weights"]["gap"], sign=True), SRC_SW + " N-Caltech101 T=2 membrane on count weights, conf-acc")
nm_diffs = [abs(v["accumulator_same_weights"]["acc"] - v["count"]["acc"]) for c, v in sw_cell.items() if c[0] == "nmnist"]
put("swNmMaxDiff", f(100 * max(nm_diffs), 2), SRC_SW + " N-MNIST cells: largest |same-weights accumulator - count| accuracy, points")
assert all(v["accumulator_same_weights"]["gap"] > 0 for c, v in sw_cell.items() if c[0] == "ncaltech101")
put("swNcAccumOverAll", str(sum(1 for c, v in sw_cell.items() if c[0] == "ncaltech101")), SRC_SW + " N-Caltech101 cells, all with the same-weights accumulator overconfident (asserted)")
# the membrane reading of the same weights (round-1 checks: quantifiers)
nm_mem = [v["count"]["acc"] - v["membrane_same_weights"]["acc"] for c, v in sw_cell.items() if c[0] == "nmnist"]
put("swNmMemMaxDiff", f(100 * max(nm_mem), 2), SRC_SW + " N-MNIST settings: largest (count - same-weights membrane) accuracy, seed means, points")
_nc_mem_runs = [r for r in sw if r["dataset"] == "ncaltech101"]
assert all(r["membrane_same_weights"]["uncalibrated"]["mean_conf_minus_acc"] < 0 for r in _nc_mem_runs)
put("swNcMemUnderRuns", str(len(_nc_mem_runs)), SRC_SW + " N-Caltech101 runs, every same-weights membrane reading underconfident (asserted)")
# where the count readout of the same weights is underconfident: T = 2 and 6 only
assert all(v["count"]["gap"] < 0 for c, v in sw_cell.items() if c[0] == "ncaltech101" and c[2] in (2, 6))
assert all(v["count"]["gap"] > 0 for c, v in sw_cell.items() if c[0] == "ncaltech101" and c[2] == 10)

# A8: ranking with the weights held fixed (results/same_weights_ranking.json)
swr = load("same_weights_ranking.json")
SRC_SWR = "results/same_weights_ranking.json"
_sm = swr["summary"]
assert abs(_sm["count"]["mean"] - cm_["headline"]["auroc"]["count"]["mean"]) < 1e-12   # same count readout as A4
assert abs(_sm["accumulator_trained"]["mean"] - cm_["headline"]["auroc"]["accumulator"]["mean"]) < 1e-12
for r in swr["rows"]:   # the same-weights accuracies are those of A6
    a6 = next(x for x in sw if x["name"] == r["name"])
    assert abs(r["accuracy"]["accumulator_same_weights"] - a6["accumulator_same_weights"]["uncalibrated"]["accuracy"]) < 1e-12
    assert r["pairs"]["count"]["reversed"] < r["pairs"]["accumulator_same_weights"]["reversed"]   # "fewer pairs in the wrong order"
    assert r["pairs"]["accumulator_same_weights"]["tied"] < 1e-3 and r["pairs"]["accumulator_trained"]["tied"] < 1e-3
put("swrAurocAccum", f(_sm["accumulator_same_weights"]["mean"]), SRC_SWR + " summary: AUROC of the accumulator reading of the count-trained weights, seed mean")
put("swrAurocMem", f(_sm["membrane_same_weights"]["mean"]), SRC_SWR + " summary: AUROC of the membrane reading of the count-trained weights, seed mean")
_dd = _sm["count_minus_accumulator_same_weights"]
put("swrAurocDiff", f(_dd["mean"], sign=True), SRC_SWR + " summary: count minus accumulator reading, seed mean")
_shw = sh_["same_weights"]["count_minus_accumulator_same_weights"]
assert max(abs(u - v) for u, v in zip(_shw["independent"], _dd["ci95_seed_mean"])) < 1e-12
put("swrAurocDiffLo", f(_shw["shared"][0], sign=True), SRC_SH + " same_weights: 95% test-set bootstrap interval, shared resample, lower")
put("swrAurocDiffHi", f(_shw["shared"][1], sign=True), SRC_SH + " same_weights: 95% test-set bootstrap interval, shared resample, upper")
assert _shw["shared"][1] < 0
put("pairsCountTied", pct(_sm["pairs_count_tied"]["mean"], 1), SRC_SWR + " summary: share of correct-wrong pairs tied by the count readout, seed mean, per cent")
put("pairsCountReversed", pct(_sm["pairs_count_reversed"]["mean"], 1), SRC_SWR + " summary: share ordered wrongly by the count readout, seed mean, per cent")
put("pairsAccSwReversed", pct(_sm["pairs_accumulator_same_weights_reversed"]["mean"], 1), SRC_SWR + " summary: share ordered wrongly by the accumulator reading, seed mean, per cent")
put("pairsAccTrReversed", pct(_sm["pairs_accumulator_trained_reversed"]["mean"], 1), SRC_SWR + " summary: share ordered wrongly by the trained accumulator, seed mean, per cent")

# training length per readout, from the run records
def best_epochs(ds, T, readout, blocks_suffix=""):
    out = []
    for seed in (0, 1, 2):
        suffix = "" if readout == "count" else f"_{readout}"
        with open(os.path.join(R, f"{ds}_T{T}_seed{seed}{suffix}{blocks_suffix}_record.json")) as fh:
            out.append(json.load(fh)["best_epoch"] + 1)
    return out
for ds, T, readouts in (("nmnist", 2, ("count", "weighted-count", "membrane", "accumulator", "scaled-count")),
                        ("ncaltech101", 2, ("count", "accumulator", "scaled-count"))):
    for ro in readouts:
        b = best_epochs(ds, T, ro)
        k = f"best{DSK[ds]}Two{''.join(w.capitalize() for w in ro.split('-'))}"
        put(k, f"{st.mean(b):.1f}", f"results/{ds}_T{T}_seed*{'' if ro == 'count' else '_' + ro}_record.json best_epoch + 1, mean over seeds")


# ------------------------------------------------------------------ abstract-level rounding of the floor
put("frNmTwoInt", f"{100 * fl_cell('nmnist', '32-64', 2)['schemes']['equal_width']['floor_ratio']['mean']:.0f}",
    SRC_FL + " N-MNIST T=2 floor_ratio equal_width, seed mean, per cent, whole number")
put("frNcTwoInt", f"{100 * fl_cell('ncaltech101', '32-64-128-128', 2)['schemes']['equal_width']['floor_ratio']['mean']:.0f}",
    SRC_FL + " N-Caltech101 T=2 floor_ratio equal_width, seed mean, per cent, whole number")
for k_, (ds_, ch_) in {"Nm": ("nmnist", "32-64"), "Nc": ("ncaltech101", "32-64-128-128")}.items():
    c_ = fl_cell(ds_, ch_, 2)["schemes"]
    assert abs(c_["equal_width"]["floor_ratio"]["mean"] - c_["equal_mass"]["floor_ratio"]["mean"]) < 5e-4

# ------------------------------------------------------------------ compute: recorded training time
recs_all = sorted(glob.glob(os.path.join(R, "*_record.json")))
secs = [json.load(open(p_))["seconds"] for p_ in recs_all]
put("trainHoursAll", f"{sum(secs) / 3600:.1f}", "results/*_record.json seconds, summed over the " + str(len(secs)) + " records, hours")
put("trainRecords", str(len(secs)), "results/*_record.json, count")

# ------------------------------------------------------------------ architecture table [DERIVED], checked against the records
SPATIAL = {"nmnist": (34, 34), "ncaltech101": (45, 60)}
NCLS = {"nmnist": 10, "ncaltech101": 101}
arch_rows = load("arch_matched.json")["rows"]
for ds_ in ("nmnist", "ncaltech101"):
    for chans in ((32, 64), (32, 64, 128, 128)):
        h, w = SPATIAL[ds_]
        cin, total, cells_ = 2, 0, []
        cells_.append(f"$2\\times{h}\\times{w}$")
        for k_ in range(4):
            if k_ < len(chans):
                c_ = chans[k_]
                pr = 9 * cin * c_ + 2 * c_          # 3x3 convolution without bias, batch-norm scale and shift
                h, w = h // 2, w // 2
                total += pr
                cells_.append(f"${c_}\\times{h}\\times{w}$; {thousands(pr)}")
                cin = c_
            else:
                cells_.append("n/a")
        flat = cin * h * w
        ph = flat * 256 + 2 * 256
        po = 256 * NCLS[ds_] + 2 * NCLS[ds_]
        total += ph + po
        cells_.append(f"{thousands(flat)} $\\to$ 256; {thousands(ph)}")
        cells_.append(f"256 $\\to$ {NCLS[ds_]}; {thousands(po)}")
        cells_.append(thousands(total))
        rec_n = {r["n_parameters"] for r in arch_rows if r["dataset"] == ds_ and r["n_blocks"] == len(chans)}
        assert rec_n == {total}, (ds_, chans, rec_n, total)
        key = f"arch{DSK[ds_]}{WORD[len(chans)]}"
        for i_, name_ in enumerate(("In", "BOne", "BTwo", "BThree", "BFour", "Hid", "Out", "Total")):
            put(key + name_, cells_[i_], DERIVED + f": {ds_}, {len(chans)} blocks; total asserted equal to results/arch_matched.json n_parameters")


# ------------------------------------------------------------------ mean confidence minus accuracy, per calibrator (A4 rows)
for ds_, b_, Ts_ in (("nmnist", 2, (2, 4, 6, 8, 10)), ("ncaltech101", 4, (2, 6, 10))):
    for T_ in Ts_:
        for arm_, ak in ARMK.items():
            rs_ = [r for r in cm_["rows"] if r["dataset"] == ds_ and r["n_blocks"] == b_ and r["readout"] == "count"
                   and r["timesteps"] == T_ and r["arm"] == arm_]
            assert len(rs_) == 3
            put(f"gapCal{DSK[ds_]}{WORD[T_]}{ak}", f(st.mean(r["mean_confidence"] - r["accuracy"] for r in rs_), sign=True),
                SRC_CM + f" rows, {ds_} count T={T_} {arm_}: mean confidence minus accuracy, seed mean")
sw_t = {c: st.mean(r["accumulator_same_weights"]["temperature"]["mean_conf_minus_acc"]
                   for r in sw if (r["dataset"], r["n_blocks"], r["timesteps"]) == c) for c in SW_ORDER}
put("swNcTwoAccumGapTemp", f(sw_t[("ncaltech101", 4, 2)], sign=True), SRC_SW + " N-Caltech101 T=2 accumulator on count weights after temperature scaling, conf-acc")
gNc = [r for r in cm_["rows"] if r["dataset"] == "ncaltech101" and r["n_blocks"] == 4 and r["readout"] == "count" and r["timesteps"] == 2]
u_ = st.mean(r["mean_confidence"] - r["accuracy"] for r in gNc if r["arm"] == "uncalibrated")
t_ = st.mean(r["mean_confidence"] - r["accuracy"] for r in gNc if r["arm"] == "temperature")
put("tempRemovesNcTwo", f"{100 * (1 - t_ / u_):.0f}", DERIVED + ": share of the N-Caltech101 T=2 underconfidence removed by temperature scaling, per cent")


# ------------------------------------------------------------------ A7: the learned scale, trained to convergence
SRC_A7 = "results/scale_convergence.json"
a7 = load("scale_convergence.json") if os.path.exists(os.path.join(R, "scale_convergence.json")) else {}
if a7.get("training_finished_at"):
    rows7 = a7["rows"]
    assert a7["thresholds"]["max_epochs"] == 150 and a7["thresholds"]["theta_tolerance"] == 0.01
    put("aSevenRuns", str(len(rows7)), SRC_A7 + " rows")
    put("aSevenSkipped", str(len(a7.get("skipped", []))), SRC_A7 + " skipped")
    put("aSevenConverged", str(sum(r["stopped_by"] == "convergence" for r in rows7)), SRC_A7 + " rows stopped_by convergence")
    put("aSevenCapped", str(sum(r["stopped_by"] == "cap" for r in rows7)), SRC_A7 + " rows stopped_by cap")
    put("aSevenEpochCap", str(a7["thresholds"]["max_epochs"]), SRC_A7 + " thresholds.max_epochs")
    put("aSevenHours", f"{sum(r['seconds'] for r in rows7) / 3600:.1f}", SRC_A7 + " rows seconds, summed, hours")
    put("aSevenProjected", f"{a7['projected_worst_case_seconds'] / 3600:.1f}", SRC_A7 + " projected_worst_case_seconds, hours")
    A7CELLS = {"NmTwo": ("nmnist", 2), "NcTwo": ("ncaltech101", 2), "NcSix": ("ncaltech101", 6)}
    for key, (ds_, T_) in A7CELLS.items():
        rs7 = sorted((r for r in rows7 if r["dataset"] == ds_ and r["timesteps"] == T_), key=lambda r: r["seed"])
        if not rs7:
            continue
        src = SRC_A7 + f" rows, {ds_} T={T_}"
        put(f"aSeven{key}N", str(len(rs7)), src + ", runs")
        put(f"aSeven{key}Conv", str(sum(r["stopped_by"] == "convergence" for r in rs7)), src + ", converged")
        put(f"aSeven{key}EpMin", str(min(r["epochs_run"] for r in rs7)), src + ", fewest epochs")
        put(f"aSeven{key}EpMax", str(max(r["epochs_run"] for r in rs7)), src + ", most epochs")
        for which, wk in (("original_rule", "Orig"), ("end_point", "End")):
            ev = [r[which] for r in rs7]
            put(f"aSeven{key}{wk}SMin", f(min(e["logit_scale"] for e in ev), 2), src + f" {which}.logit_scale, smallest")
            put(f"aSeven{key}{wk}SMax", f(max(e["logit_scale"] for e in ev), 2), src + f" {which}.logit_scale, largest")
            put(f"aSeven{key}{wk}SminMin", f(min(e["s_min"] for e in ev), 2), src + f" {which}.s_min, smallest")
            put(f"aSeven{key}{wk}SminMax", f(max(e["s_min"] for e in ev), 2), src + f" {which}.s_min, largest")
            put(f"aSeven{key}{wk}Acc", f(st.mean(e["test_accuracy"] for e in ev)), src + f" {which}.test_accuracy, mean")
            put(f"aSeven{key}{wk}Gap", f(st.mean(e["conf_minus_acc"] for e in ev), sign=True), src + f" {which}.conf_minus_acc, mean")
            put(f"aSeven{key}{wk}EceW", f(st.mean(e["ece_equal_width"] for e in ev)), src + f" {which}.ece_equal_width, mean")
            put(f"aSeven{key}{wk}EceM", f(st.mean(e["ece_equal_mass"] for e in ev)), src + f" {which}.ece_equal_mass, mean")
            put(f"aSeven{key}{wk}Floor", f(st.mean(e["floor"] for e in ev)), src + f" {which}.floor, mean")
            put(f"aSeven{key}{wk}AboveSmin", str(sum(e["logit_scale"] > e["s_min"] for e in ev)), src + f" {which}: runs with s above s_min")
    # the share of the count-to-accumulator accuracy gap recovered, at both checkpoints of the retrained runs
    sc_rows = load("scaled_count_experiment.json")["rows"]
    nc2 = [r for r in sc_rows if r["dataset"] == "ncaltech101" and r["timesteps"] == 2]
    cnt = st.mean(r["count_test_accuracy"] for r in nc2)
    acc_ = st.mean(r["accumulator_test_accuracy"] for r in nc2)
    rs7 = [r for r in rows7 if r["dataset"] == "ncaltech101" and r["timesteps"] == 2]
    for which, wk in (("original_rule", "Orig"), ("end_point", "End")):
        sc_acc = st.mean(r[which]["test_accuracy"] for r in rs7)
        put(f"aSevenNcTwo{wk}Recovered", f"{100 * (sc_acc - cnt) / (acc_ - cnt):.0f}",
            DERIVED + f": ({which} accuracy - count accuracy) / (accumulator - count), N-Caltech101 T=2, per cent")
    put("aSevenNmGapMax", f(max(abs(r["end_point"]["conf_minus_acc"]) for r in rows7 if r["dataset"] == "nmnist")),
        SRC_A7 + " N-MNIST end points: largest |conf - acc|")
    th_span = []
    for r in rows7:
        if r["stopped_by"] == "cap":
            th = [h["theta"] for h in r["history"]]
            th_span.append(max(th[-6:]) - min(th[-6:]))
    put("aSevenCapSpanMin", f(min(th_span), 3), SRC_A7 + " capped runs: smallest span of theta over the last six epochs")
    put("aSevenCapSpanMax", f(max(th_span), 3), SRC_A7 + " capped runs: largest span of theta over the last six epochs")
    a7rows = 0
    for r in sorted(rows7, key=lambda r: (r["dataset"] != "nmnist", r["timesteps"], r["seed"])):
        k_ = f"aSevenRow{DSK[r['dataset']]}{WORD[r['timesteps']]}{WORD[r['seed']]}"
        o, e = r["original_rule"], r["end_point"]
        put(k_, " & ".join([str(r["epochs_run"]), r["stopped_by"].replace("convergence", "settled").replace("cap", "limit"),
                            f(o["logit_scale"], 2), f(o["test_accuracy"]), f(o["conf_minus_acc"], sign=True), f(o["ece_equal_width"]),
                            f(e["logit_scale"], 2), f(e["s_min"], 2), f(e["test_accuracy"]), f(e["conf_minus_acc"], sign=True), f(e["ece_equal_width"])]),
            SRC_A7 + f" rows, {r['dataset']} T={r['timesteps']} seed {r['seed']}: epochs, stop; original-rule s, acc, gap, ECE w; end s, s_min, acc, gap, ECE w")
        a7rows += 1
    put("aSevenTableRows", str(a7rows), SRC_A7 + " rows")


# ------------------------------------------------------------------ S1, S2: N-Caltech101 calibrators refitted to the test class distribution
rw_ = load("reweighted_calibration.json")
SRC_RW = "results/reweighted_calibration.json"
_rws = {e["readout"]: e for e in rw_["summary"]}
_rw4 = [r for r in rw_["runs"] if r["n_blocks"] == 4]
for k, ro in (("Count", "count"), ("Accum", "accumulator"), ("Scaled", "scaled-count")):
    e = _rws[ro]
    # the unweighted refit reproduces the published verdict counts
    assert e["unweighted"]["favour_isotonic"] == {"count": 5, "accumulator": 6, "scaled-count": 1}[ro], ro
    put(f"rw{k}Runs", str(e["runs"]), SRC_RW + f" summary, {ro}: runs")
    put(f"rw{k}Iso", str(e["reweighted"]["favour_isotonic"]), SRC_RW + f" summary, {ro}: reweighted, isotonic better under the registered rule")
    put(f"rw{k}Temp", str(e["reweighted"]["favour_temperature"]), SRC_RW + f" summary, {ro}: reweighted, temperature scaling better under the registered rule")
put("rwUnderUnw", str(sum(e["unweighted"]["underconfident_calibrated"] for e in rw_["summary"])), SRC_RW + " summary: calibrated four-block runs underconfident on the test set, unweighted fits")
put("rwUnderRw", str(sum(e["reweighted"]["underconfident_calibrated"] for e in rw_["summary"])), SRC_RW + " summary: calibrated four-block runs underconfident on the test set, reweighted fits")
put("rwCalibrated", str(sum(e["reweighted"]["calibrated_runs"] for e in rw_["summary"])), SRC_RW + " summary: calibrated four-block runs (two calibrators per run)")
assert _rws["count"]["reweighted"]["favour_temperature"] == 0   # main.tex: "never resolved as worse" after reweighting
assert [(r["timesteps"], r["seed"]) for r in _rw4 if r["readout"] == "scaled-count"
        and r["paired_unweighted"]["favours"] != r["paired_reweighted"]["favours"]
        and r["paired_unweighted"]["differ"] and r["paired_reweighted"]["differ"]] == [(2, 1)]   # main.tex: the one reversal
put("rwUnderCountRw", str(_rws["count"]["reweighted"]["underconfident_calibrated"]), SRC_RW + " summary, count: reweighted calibrated runs underconfident on the test set")
put("rwCountCalibrated", str(_rws["count"]["reweighted"]["calibrated_runs"]), SRC_RW + " summary, count: calibrated runs")
_c2 = [r for r in _rw4 if r["readout"] == "count" and r["timesteps"] == 2]
assert len(_c2) == 3
for arm, ak in (("temperature", "Temp"), ("isotonic-top-label", "Iso")):
    put(f"rwNcTwo{ak}W", f(st.mean(r["reweighted"][arm]["ece_equal_width"] for r in _c2)), SRC_RW + f" runs, count T=2 reweighted {arm}: equal-width ECE, seed mean")
    put(f"rwNcTwo{ak}M", f(st.mean(r["reweighted"][arm]["ece_equal_mass"] for r in _c2)), SRC_RW + f" runs, count T=2 reweighted {arm}: equal-mass ECE, seed mean")
    put(f"rwNcTwo{ak}Gap", f(st.mean(r["reweighted"][arm]["mean_conf_minus_acc"] for r in _c2), sign=True), SRC_RW + f" runs, count T=2 reweighted {arm}: mean confidence minus accuracy, seed mean")
assert all(r["paired_reweighted"]["favours"] == "isotonic" for r in _c2)   # "in every seed at T = 2"
_u = st.mean(r["unweighted"]["uncalibrated"]["mean_conf_minus_acc"] for r in _c2)
_t = st.mean(r["reweighted"]["temperature"]["mean_conf_minus_acc"] for r in _c2)
_t0 = st.mean(r["unweighted"]["temperature"]["mean_conf_minus_acc"] for r in _c2)
assert f"{100 * (1 - _t0 / _u):.0f}" == dict((m[0], m[1]) for m in macros)["tempRemovesNcTwo"]
put("rwTempRemovesNcTwo", f"{100 * (1 - _t / _u):.0f}", DERIVED + ": share of the N-Caltech101 T=2 underconfidence removed by the reweighted temperature, per cent (" + SRC_RW + ")")
put("rwTempNcTwoMin", f(min(r["temperature_reweighted"] for r in _c2), 2), SRC_RW + " runs, count T=2: smallest reweighted temperature")
put("rwTempNcTwoMax", f(max(r["temperature_reweighted"] for r in _c2), 2), SRC_RW + " runs, count T=2: largest reweighted temperature")
put("rwWeightMin", f(min(r["weights"]["min"] for r in _rw4), 2), SRC_RW + " runs: smallest calibration weight")
put("rwWeightMax", f(max(r["weights"]["max"] for r in _rw4), 2), SRC_RW + " runs: largest calibration weight")
put("calAccNcTwo", f(st.mean(r["cal_accuracy"] for r in _c2)), SRC_RW + " runs, count T=2: calibration-set accuracy, seed mean")
put("calAccRwNcTwo", f(st.mean(r["cal_accuracy_reweighted"] for r in _c2)), SRC_RW + " runs, count T=2: reweighted calibration-set accuracy, seed mean")
_sc = [r for r in _rw4 if r["readout"] == "scaled-count"]
assert len(_sc) == 6 and min(r["temperature_reweighted"] for r in _sc) > 1.0
put("rwScTempMin", f(min(r["temperature_reweighted"] for r in _sc), 2), SRC_RW + " runs, scaled-count: smallest reweighted temperature")
put("rwScTempMax", f(max(r["temperature_reweighted"] for r in _sc), 2), SRC_RW + " runs, scaled-count: largest reweighted temperature")
# S2: the single-temperature scan on the test set, count readout at T = 2
put("scanNcTwoLowW", f"{math.floor(1000 * min(r['scan']['equal_width']['min_ece'] for r in _c2)) / 1000:.3f}", SRC_RW + " runs, count T=2: smallest equal-width ECE over the temperature grid, lowest seed, rounded down to three decimals")
put("scanNcTwoLowM", f"{math.floor(1000 * min(r['scan']['equal_mass']['min_ece'] for r in _c2)) / 1000:.3f}", SRC_RW + " runs, count T=2: smallest equal-mass ECE over the temperature grid, lowest seed")
put("scanNcTwoMatchLo", f(min(r["scan"]["mean_matched"]["ece_equal_width"] for r in _c2)), SRC_RW + " runs, count T=2: equal-width ECE where mean confidence equals accuracy, smallest seed")
put("scanNcTwoMatchHi", f(max(r["scan"]["mean_matched"]["ece_equal_width"] for r in _c2)), SRC_RW + " runs, count T=2: equal-width ECE where mean confidence equals accuracy, largest seed")

# second internal review: N-Caltech101 validation accuracy against test accuracy, and the training
# loss of the unbounded readouts at the kept checkpoint against their test NLL (run records)
_vg, _tl, _nll = [], [], []
for _f in sorted(glob.glob(os.path.join(R, "ncaltech101_*_record.json"))):
    if "_converged_" in _f:
        continue
    _r = json.load(open(_f))
    _nm = os.path.basename(_f)[: -len("_record.json")]
    _h = next(h for h in _r["history"] if h["epoch"] == _r["best_epoch"])
    _row = next(x for x in cm_["rows"] if x["name"] == _nm and x["arm"] == "uncalibrated")
    _vg.append(_row["accuracy"] - _h["val_accuracy"])
    if _row["readout"] in ("accumulator", "scaled-count"):
        _tl.append(_h["train_loss"])
        _nll.append(_row["nll_multiclass"])
assert len(_vg) == 36 and min(_vg) > 0
put("valGapNcRuns", str(len(_vg)), "results/ncaltech101_*_record.json: runs compared")
put("valGapNcMin", f(100 * min(_vg), 1), "results/ncaltech101_*_record.json history at best_epoch; " + SRC_CM + ": smallest (test - validation) accuracy, points")
put("valGapNcMax", f(100 * max(_vg), 1), "results/ncaltech101_*_record.json history at best_epoch; " + SRC_CM + ": largest (test - validation) accuracy, points")
assert max(_tl) < min(_nll)
put("ovTrainLossMin", f(min(_tl), 2), "results/ncaltech101_*_record.json: accumulator and scaled-count, train_loss at best_epoch, smallest")
put("ovTrainLossMax", f(max(_tl), 2), "results/ncaltech101_*_record.json: accumulator and scaled-count, train_loss at best_epoch, largest")
put("ovNllMin", f(min(_nll), 2), SRC_CM + " rows, ncaltech101 accumulator and scaled-count, uncalibrated: multiclass NLL, smallest")
put("ovNllMax", f(max(_nll), 2), SRC_CM + " rows, ncaltech101 accumulator and scaled-count, uncalibrated: multiclass NLL, largest")
_smt = json.load(open(os.path.join(R, "same_weights_ranking.json")))["summary"]
assert _smt["pairs_accumulator_same_weights_tied"]["max"] < 1e-5   # main.tex: "leaves almost no pair unordered"

# ------------------------------------------------------------------ the literature search (added after review)
ls_ = load("literature_search.json")
SRC_LS = "results/literature_search.json"
assert ls_["papers_read"] == ls_["papers_read_full_text"] + ls_["papers_read_abstract_only"]
put("lsQueries", str(len(ls_["required_queries"])), SRC_LS + " required_queries")
put("lsQueryRows", str(ls_["query_rows"]), SRC_LS + " query_rows")
put("lsScreenings", thousands(ls_["record_screenings"]), SRC_LS + " record_screenings")
put("lsRead", str(ls_["papers_read"]), SRC_LS + " papers_read")
put("lsReadFull", str(ls_["papers_read_full_text"]), SRC_LS + " papers_read_full_text")
put("lsReadAbstract", str(ls_["papers_read_abstract_only"]), SRC_LS + " papers_read_abstract_only")
put("lsLow", str(ls_["low_relevance_records"]), SRC_LS + " low_relevance_records")
put("lsSeeds", str(len(ls_["forward_citation_seeds"])), SRC_LS + " forward_citation_seeds")
put("lsSourcesDirect", str(len(ls_["sources_queried_directly"])), SRC_LS + " sources_queried_directly")

# ------------------------------------------------------------------ write
with open(OUT, "w") as fh:
    fh.write("% Generated by paper/make_numbers.py from results/ -- do not edit by hand.\n")
    fh.write("% Each definition names the artifact it was read from.\n")
    for name, value, source in macros:
        fh.write(f"\\newcommand{{\\{name}}}{{{value}}}% {source}\n")
print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(macros)} macros")
for name, value, source in macros:
    print(f"  \\{name:<24} {value}")
