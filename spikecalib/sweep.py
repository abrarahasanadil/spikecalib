"""Experiment orchestration: the T x seed x dataset sweep and the results table.

Runs ``T`` in {2, 4, 6, 8, 10} with 3 seeds on every dataset in the study -- 15 runs per
dataset per calibration arm. The sweep is what tests the hypothesis: the isotonic
advantage over temperature scaling is predicted to track the samples-per-distinct-level
ratio ``n_cal / n_distinct``, for which ``T`` is a proxy. If the advantage is flat in
``T``, or tracks ``T`` but not the ratio when ``n_cal`` is varied at fixed ``T``, the
capacity mechanism is wrong and the study reports that.

**Two preconditions are enforced before any training starts**, because both failures are
expensive to discover late and cheap to detect early.

1. *Every frame cache must exist and be complete.* A sweep that dies four hours in because
   ``T = 8`` was never built has wasted four hours. :func:`assert_caches_ready` checks all
   of them up front and names the exact command to build any that are missing.

2. *Every run must be allowed to converge.* Epochs are capped, and a run that stops because
   it hit the cap rather than because validation stopped improving may be undertrained.
   Comparing a converged low-``T`` model against an undertrained high-``T`` one would
   confound the entire sweep -- the calibration difference would partly be a training
   difference. So ``epochs_run`` and ``early_stopped`` are recorded on **every row**, and
   :func:`convergence_audit` reports any run that hit the cap before the results are read.

Owns the reporting contract that makes results readable rather than merely correct.
Every row of the results table carries, alongside the calibration metrics:

- **both** ECE estimators (correctness rule 2) -- never one without the other;
- **bootstrap confidence intervals** on both, because no ECE difference can be called real
  without them;
- the **tie rate**, a descriptive measure of how discrete the output is (it is not
  itself the mechanism -- neither calibrator breaks ties; see NOTES.md);
- the number of **distinct confidence levels** and the resulting
  **samples-per-level ratio** ``n_cal / n_distinct``, which is the quantity the
  hypothesis names;
- the **realised isotonic block count** after monotonicity pooling, because the
  synthetic pre-test showed ``n_distinct`` is a very loose bound on the capacity
  actually used;
- the **uncalibrated ECE**, so the isotonic advantage can be compared at matched
  miscalibration severity -- otherwise a trend across ``T`` cannot be separated from
  a trend in how much error there was to remove;
- **``epochs_run`` and ``early_stopped``**, per the precondition above;
- the **calibration-set size**, so a reader can judge whether an isotonic win on a small
  dataset is real or is small-sample overfitting;
- the **seed** and the **split fingerprint**, per correctness rule 4.

Owns the multi-dataset reporting rule from ``NOTES.md``: a trend that holds on one
dataset and not another is a finding to report, not a failure to hide. Every dataset that
is run appears in the output.

Calls into ``data.assert_disjoint`` before any calibrator is fitted -- via
:func:`spikecalib.data.make_splits`, which asserts internally -- so correctness rule 1 is
enforced on the live experiment path and not only in tests.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .calibration import (
    IsotonicOneVsRest,
    discrimination,
    IsotonicTopLabel,
    TemperatureScaling,
    Uncalibrated,
    bootstrap_ece_difference,
    compute_metrics,
    model_outputs,
    tie_rate,
)
from .data import load_dataset, make_splits, training_labels
from .datasets import DatasetSpec, get_spec
from .models import ModelConfig, build_model
from .train import RunRecord, TrainConfig, predict_counts, resolve_device, train

__all__ = [
    "CacheStatus",
    "CacheMissingError",
    "verify_caches",
    "assert_caches_ready",
    "SweepConfig",
    "run_paths",
    "reuse_decision",
    "run_one",
    "run_sweep",
    "convergence_audit",
    "val_accuracy_audit",
    "VAL_OUTLIER_TOLERANCE",
    "ARMS",
]

VAL_OUTLIER_TOLERANCE = 0.02  # 2 accuracy points
# "More than 2 points below" must not fire on a run that is exactly 2 points below.
# Accuracies are ratios of integers, so a gap that is mathematically 0.02 can come back
# as 0.020000000000000018 and clear a bare > comparison. The slack is far below any
# difference worth flagging and far above float64 representation error.
_VAL_TOLERANCE_SLACK = 1e-9

ARMS = (
    ("uncalibrated", Uncalibrated),
    ("temperature", TemperatureScaling),
    ("isotonic-top-label", IsotonicTopLabel),
    ("isotonic-one-vs-rest", IsotonicOneVsRest),
)


class CacheMissingError(FileNotFoundError):
    """Raised when a frame cache the sweep needs is absent or incomplete."""


@dataclass(frozen=True)
class CacheStatus:
    timesteps: int
    path: str
    exists: bool
    n_found: int
    n_expected: Optional[int]

    @property
    def complete(self) -> bool:
        if not self.exists:
            return False
        if self.n_expected is None:
            return self.n_found > 0
        return self.n_found == self.n_expected

    @property
    def verdict(self) -> str:
        if not self.exists:
            return "MISSING"
        if not self.complete:
            return f"INCOMPLETE ({self.n_found:,} of {self.n_expected:,})"
        return f"ok ({self.n_found:,})"


def verify_caches(
    spec: DatasetSpec, timesteps_list: Sequence[int], data_root: str
) -> list[CacheStatus]:
    """Check every frame cache the sweep will need, without touching any of them.

    Counting files rather than trusting the directory's existence is the same defence
    ``scripts/fetch_data.py`` applies: SpikingJelly checks only that a cache directory is
    present, so an interrupted build leaves a stump that looks fine.
    """
    statuses = []
    root = spec.root(data_root)
    for timesteps in timesteps_list:
        path = os.path.join(root, spec.frames_dir(timesteps))
        exists = os.path.isdir(path)
        n_found = 0
        if exists:
            for _dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                n_found += sum(1 for f in filenames if f.endswith(".npz"))
        statuses.append(
            CacheStatus(
                timesteps=timesteps,
                path=path,
                exists=exists,
                n_found=n_found,
                n_expected=spec.n_total,
            )
        )
    return statuses


def assert_caches_ready(
    spec: DatasetSpec, timesteps_list: Sequence[int], data_root: str
) -> list[CacheStatus]:
    """Fail before any training starts if a cache the sweep needs is not ready.

    The message names every missing cache and the exact command that builds it, because
    the useful thing to hand someone whose sweep just refused to start is the fix.
    """
    statuses = verify_caches(spec, timesteps_list, data_root)
    bad = [s for s in statuses if not s.complete]
    if bad:
        lines = [
            f"{len(bad)} of {len(statuses)} frame caches for {spec.display_name} are not "
            f"ready, so the sweep will not start:",
        ]
        for s in bad:
            lines.append(f"  T={s.timesteps:<3} {s.verdict:<28} {s.path}")
        lines.append("")
        lines.append("Build them with:")
        for s in bad:
            lines.append(
                f"  python scripts/fetch_data.py --dataset {spec.key} "
                f"--timesteps {s.timesteps}"
            )
        raise CacheMissingError("\n".join(lines))
    return statuses


@dataclass(frozen=True)
class SweepConfig:
    """Settings shared by every run in a sweep.

    ``epochs`` is a cap and ``patience`` is the stopping rule. The defaults are chosen so
    the **stopping rule decides, not the cap**: 30 epochs is well beyond where N-MNIST
    converges, and patience 5 ends a run once validation accuracy has been flat for five
    epochs. A run that still reports ``early_stopped=False`` hit the cap and is suspect --
    see :func:`convergence_audit`.
    """

    epochs: int = 30
    patience: int = 5
    lr: float = 1e-3
    batch_size: int = 128
    num_workers: int = 6
    eval_seed: int = 0
    n_bins: int = 15
    n_bootstrap: int = 1000
    channels: tuple[int, ...] = (32, 64)
    hidden: int = 256
    log_every: int = 0

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            patience=self.patience,
            eval_seed=self.eval_seed,
            log_every=self.log_every,
        )

    def model_config(self) -> ModelConfig:
        return ModelConfig(channels=tuple(self.channels), hidden=self.hidden)


def run_paths(out_dir: str, spec: DatasetSpec, timesteps: int, seed: int) -> tuple[str, str]:
    """Weights and record paths for one run, matching ``scripts/train_one.py``'s naming.

    Deliberately the same convention, so a model trained by either entry point is visible
    to the other.
    """
    stem = f"{spec.key}_T{timesteps}_seed{seed}"
    return (
        os.path.join(out_dir, f"{stem}_weights.pt"),
        os.path.join(out_dir, f"{stem}_record.json"),
    )


def reuse_decision(
    record: dict,
    spec: DatasetSpec,
    timesteps: int,
    seed: int,
    cfg: "SweepConfig",
    split_fingerprint: str,
) -> tuple[bool, str]:
    """Decide whether a saved run can stand in for retraining, and say why or why not.

    Identity has to match exactly: dataset, timesteps, seed, model configuration and split
    fingerprint. Any difference means the saved weights answer a different question.

    There is one further condition beyond identity, and it exists because of a hazard this
    project already walked into. A saved run that **hit its epoch cap** was still improving
    when it stopped, so it is only equivalent to what this sweep would produce if the cap
    was at least as generous. Reusing a run capped at 15 epochs inside a sweep that allows
    50 would silently mix an undertrained model in with converged ones -- exactly the
    confound the convergence audit exists to prevent. A run that stopped on the patience
    rule converged on its own terms and is reusable whatever cap it had.
    """
    if record.get("dataset") != spec.key:
        return False, f"dataset {record.get('dataset')!r} != {spec.key!r}"
    if record.get("timesteps") != timesteps:
        return False, f"timesteps {record.get('timesteps')} != {timesteps}"
    if record.get("seed") != seed:
        return False, f"seed {record.get('seed')} != {seed}"
    if record.get("split_fingerprint") != split_fingerprint:
        return False, (
            f"split fingerprint {record.get('split_fingerprint')} != {split_fingerprint}"
        )

    saved_model = record.get("model_config") or {}
    wanted = cfg.model_config()
    if list(saved_model.get("channels", [])) != list(wanted.channels):
        return False, f"channels {saved_model.get('channels')} != {list(wanted.channels)}"
    for field in ("hidden", "tau", "v_threshold", "surrogate_alpha"):
        if saved_model.get(field) != getattr(wanted, field):
            return False, f"{field} {saved_model.get(field)} != {getattr(wanted, field)}"

    if not record.get("early_stopped", False):
        saved_cap = record.get("epochs_requested", 0)
        if saved_cap < cfg.epochs:
            return False, (
                f"saved run hit its {saved_cap}-epoch cap and this sweep allows "
                f"{cfg.epochs}; it may be undertrained"
            )
    return True, "identity and convergence match"


def _load_saved_run(
    out_dir: str, spec: DatasetSpec, timesteps: int, seed: int,
    cfg: "SweepConfig", split_fingerprint: str,
) -> tuple[Optional[dict], Optional[dict], str]:
    """Return (state_dict, record_dict, reason) for a reusable saved run, else (None, None, reason)."""
    weights_path, record_path = run_paths(out_dir, spec, timesteps, seed)
    if not (os.path.exists(weights_path) and os.path.exists(record_path)):
        return None, None, "no saved run"
    try:
        with open(record_path) as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, None, f"record unreadable ({exc})"
    ok, reason = reuse_decision(record, spec, timesteps, seed, cfg, split_fingerprint)
    if not ok:
        return None, None, reason
    try:
        state = torch.load(weights_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must not kill the sweep
        return None, None, f"weights unreadable ({exc})"
    return state, record, reason


def _save_run(
    out_dir: str, spec: DatasetSpec, timesteps: int, seed: int,
    cfg: "SweepConfig", state: dict, record: RunRecord,
) -> None:
    """Persist weights and record after every run, so a crash at run 12 keeps runs 1-11."""
    os.makedirs(out_dir, exist_ok=True)
    weights_path, record_path = run_paths(out_dir, spec, timesteps, seed)
    torch.save(state, weights_path)
    payload = record.as_dict()
    mc = cfg.model_config()
    payload["model_config"] = {
        "channels": list(mc.channels), "hidden": mc.hidden, "tau": mc.tau,
        "v_threshold": mc.v_threshold, "surrogate_alpha": mc.surrogate_alpha,
    }
    tmp = record_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, record_path)  # atomic: never leave a half-written record


def run_one(
    spec: DatasetSpec,
    timesteps: int,
    seed: int,
    cfg: SweepConfig,
    data_root: str,
    device: Optional[torch.device] = None,
    out_dir: Optional[str] = None,
    allow_resume: bool = True,
) -> list[dict]:
    """Train one (dataset, T, seed) and return one results row per calibration arm.

    Every row carries the training provenance -- ``epochs_run``, ``early_stopped``,
    ``best_val_accuracy`` -- so a reader comparing calibration across ``T`` can see at a
    glance whether they are comparing equally converged models.
    """
    device = device or resolve_device()
    started = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        train_set, test_set = load_dataset(spec.key, timesteps, data_root)
    labels = training_labels(train_set)
    # make_splits asserts disjointness internally (correctness rule 1).
    splits = make_splits(labels, spec, seed=seed, n_test=len(test_set))

    def loader(dataset, idx, shuffle=False):
        return DataLoader(
            Subset(dataset, idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )

    model = build_model(spec, timesteps, cfg.model_config())

    saved_state = saved_record = None
    reuse_reason = "resume disabled"
    if allow_resume and out_dir:
        saved_state, saved_record, reuse_reason = _load_saved_run(
            out_dir, spec, timesteps, seed, cfg, splits.fingerprint
        )

    if saved_state is not None:
        model.load_state_dict(saved_state)
        record = RunRecord(
            seed=saved_record["seed"],
            eval_seed=saved_record.get("eval_seed", cfg.eval_seed),
            dataset=saved_record["dataset"],
            timesteps=saved_record["timesteps"],
            split_fingerprint=saved_record["split_fingerprint"],
            epochs_requested=saved_record["epochs_requested"],
            epochs_run=saved_record["epochs_run"],
            best_val_accuracy=saved_record["best_val_accuracy"],
            final_val_accuracy=saved_record["final_val_accuracy"],
            final_val_tie_rate=saved_record.get("final_val_tie_rate", float("nan")),
            best_epoch=saved_record["best_epoch"],
            early_stopped=saved_record["early_stopped"],
            lr=saved_record["lr"],
            batch_size=saved_record["batch_size"],
            device=saved_record["device"],
            n_parameters=saved_record["n_parameters"],
            seconds=saved_record["seconds"],
            mps_nondeterministic=saved_record["mps_nondeterministic"],
            history=saved_record.get("history", []),
        )
        resumed = True
    else:
        state, record = train(
            model,
            {"train": loader(train_set, splits.train, shuffle=True),
             "val": loader(train_set, splits.val)},
            cfg.train_config(),
            seed=seed,
            dataset=spec.key,
            timesteps=timesteps,
            split_fingerprint=splits.fingerprint,
        )
        model.load_state_dict(state)
        resumed = False
        if out_dir:
            _save_run(out_dir, spec, timesteps, seed, cfg, state, record)

    cal_counts, cal_labels, _ = predict_counts(
        model, loader(train_set, splits.cal), device, count_synops=False
    )
    test_counts, test_labels, synops = predict_counts(
        model, loader(test_set, splits.test), device
    )

    cal = model_outputs(cal_counts, cal_labels, np.random.default_rng(seed))
    test = model_outputs(test_counts, test_labels, np.random.default_rng(seed + 1))
    n_distinct = int(np.unique(cal.conf).size)
    # AUROC on the RAW readout, before any calibrator touches it. A monotone calibrator
    # cannot reorder samples, so this is the ceiling on what any calibrated confidence in
    # this run can express -- and the per-arm value below shows whether a calibrator threw
    # some of it away by collapsing distinct confidences into ties.
    raw_test = discrimination(test.conf, test.correct)
    raw_cal = discrimination(cal.conf, cal.correct)

    fitted: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    iso_blocks = None
    temperature = None

    for name, cls in ARMS:
        calibrator = cls().fit(cal.conf, cal.correct, outputs=cal)
        calibrated = calibrator.transform(test.conf, outputs=test)
        fitted[name] = calibrated
        metrics = compute_metrics(
            calibrated, test.correct, n_bins=cfg.n_bins,
            n_bootstrap=cfg.n_bootstrap, bootstrap_seed=seed,
        )
        if name == "isotonic-top-label":
            iso_blocks = calibrator.n_levels_
        if name == "temperature":
            temperature = calibrator.temperature_
        arm_disc = discrimination(calibrated, test.correct)

        rows.append({
            "dataset": spec.key,
            "timesteps": timesteps,
            "seed": seed,
            "arm": name,
            # --- training provenance: is this run comparable to the others? ---
            "epochs_requested": record.epochs_requested,
            "epochs_run": record.epochs_run,
            "early_stopped": record.early_stopped,
            "hit_epoch_cap": not record.early_stopped,
            "best_epoch": record.best_epoch,
            "best_val_accuracy": record.best_val_accuracy,
            "final_val_tie_rate": record.final_val_tie_rate,
            "mps_nondeterministic": record.mps_nondeterministic,
            # --- readout ---
            "test_accuracy": test.accuracy,
            "test_tie_rate": tie_rate(test_counts),
            "n_cal": cal.n,
            "n_test": test.n,
            "n_distinct_cal": n_distinct,
            "samples_per_level": cal.n / max(n_distinct, 1),
            "isotonic_blocks": iso_blocks,
            "temperature": temperature,
            # --- can confidence separate right from wrong at all? ---
            "auroc_raw_test": raw_test["auroc"],
            "auroc_raw_cal": raw_cal["auroc"],
            "auroc": arm_disc["auroc"],
            "auroc_lost_to_calibration": raw_test["auroc"] - arm_disc["auroc"],
            "mean_conf_correct": arm_disc["mean_conf_correct"],
            "mean_conf_incorrect": arm_disc["mean_conf_incorrect"],
            "conf_gap_correct_minus_incorrect": arm_disc["gap"],
            "conf_std": arm_disc["std"],
            "conf_std_over_mean": arm_disc["cv"],
            "raw_conf_std_over_mean": raw_test["cv"],
            # --- calibration, both estimators, both intervals ---
            **metrics.as_dict(),
            # --- energy proxy ---
            "energy_joules_proxy": synops.energy_joules,
            "energy_per_sample_joules_proxy": synops.energy_per_sample_joules,
            "encoder_energy_fraction": synops.encoder_energy_fraction,
            # --- provenance ---
            "split_fingerprint": splits.fingerprint,
            "resumed": resumed,
            "resume_reason": reuse_reason,
            "seconds": time.time() - started,
        })

    # The headline quantity is a paired difference, not two separate intervals.
    diff = bootstrap_ece_difference(
        fitted["isotonic-top-label"], fitted["temperature"], test.correct,
        n_bins=cfg.n_bins, n_bootstrap=cfg.n_bootstrap, seed=seed,
    )
    for row in rows:
        row["iso_minus_temp_ece_ew"] = diff["difference"]
        row["iso_minus_temp_ci_low"] = diff["ci"][0]
        row["iso_minus_temp_ci_high"] = diff["ci"][1]
        row["iso_minus_temp_resolvable"] = diff["resolvable"]
    return rows


def run_sweep(
    dataset: str,
    timesteps_list: Sequence[int],
    seeds: Sequence[int],
    cfg: SweepConfig,
    data_root: str,
    on_run_complete=None,
    out_dir: Optional[str] = None,
    allow_resume: bool = True,
) -> list[dict]:
    """Run the full sweep, refusing to start if any frame cache is missing."""
    spec = get_spec(dataset)
    assert_caches_ready(spec, timesteps_list, data_root)

    device = resolve_device()
    rows: list[dict] = []
    for timesteps in timesteps_list:
        for seed in seeds:
            run_rows = run_one(spec, timesteps, seed, cfg, data_root, device,
                               out_dir=out_dir, allow_resume=allow_resume)
            rows.extend(run_rows)
            if on_run_complete is not None:
                on_run_complete(run_rows)
    return rows


def convergence_audit(rows: Iterable[dict]) -> dict:
    """Which runs stopped because they converged, and which because they ran out of epochs.

    This is read *before* the calibration numbers. A run that hit the epoch cap may be
    undertrained, and an undertrained model has a different readout -- fewer distinct
    confidence levels, a higher tie rate, worse accuracy -- than a converged one. Comparing
    a converged low-``T`` model against an undertrained high-``T`` one would attribute a
    training difference to a calibration mechanism.
    """
    per_run: dict[tuple, dict] = {}
    for row in rows:
        key = (row["dataset"], row["timesteps"], row["seed"])
        per_run[key] = {
            "dataset": row["dataset"],
            "timesteps": row["timesteps"],
            "seed": row["seed"],
            "epochs_run": row["epochs_run"],
            "epochs_requested": row["epochs_requested"],
            "early_stopped": row["early_stopped"],
            "hit_cap": row["hit_epoch_cap"],
            "best_val_accuracy": row["best_val_accuracy"],
        }
    runs = sorted(per_run.values(), key=lambda r: (r["timesteps"], r["seed"]))
    capped = [r for r in runs if r["hit_cap"]]
    return {
        "runs": runs,
        "n_runs": len(runs),
        "n_capped": len(capped),
        "capped": capped,
        "all_converged": not capped,
    }


def val_accuracy_audit(rows: Iterable[dict], tolerance: float = VAL_OUTLIER_TOLERANCE) -> dict:
    """Flag any run whose validation accuracy sits well below the best at the same ``T``.

    Seeds at a fixed ``T`` differ only by initialisation, shuffling and the split draw, so
    they should land close together. One seed trailing the others by more than a couple of
    accuracy points is not seed noise -- it is a run that failed to train properly, and its
    readout will differ from its siblings in exactly the ways that matter here: fewer
    distinct confidence levels, a higher tie rate, a different confidence distribution.
    Averaging it into a per-``T`` calibration result would smear a training failure across
    the trend the study is measuring.

    The comparison is within a ``T``, not across, because accuracy legitimately varies with
    ``T`` and a cross-``T`` threshold would flag the low-``T`` runs for being low-``T``.
    """
    per_run: dict[tuple, dict] = {}
    for row in rows:
        key = (row["dataset"], row["timesteps"], row["seed"])
        per_run[key] = {
            "dataset": row["dataset"],
            "timesteps": row["timesteps"],
            "seed": row["seed"],
            "best_val_accuracy": row["best_val_accuracy"],
            "test_accuracy": row["test_accuracy"],
        }
    runs = sorted(per_run.values(), key=lambda r: (r["timesteps"], r["seed"]))

    best_at_t: dict[int, float] = {}
    for r in runs:
        t = r["timesteps"]
        best_at_t[t] = max(best_at_t.get(t, float("-inf")), r["best_val_accuracy"])

    outliers = []
    for r in runs:
        r["best_at_timesteps"] = best_at_t[r["timesteps"]]
        r["below_best"] = best_at_t[r["timesteps"]] - r["best_val_accuracy"]
        r["val_outlier"] = r["below_best"] > tolerance + _VAL_TOLERANCE_SLACK
        if r["val_outlier"]:
            outliers.append(r)
    return {
        "runs": runs,
        "best_at_timesteps": best_at_t,
        "tolerance": tolerance,
        "outliers": outliers,
        "n_outliers": len(outliers),
        "all_consistent": not outliers,
    }
