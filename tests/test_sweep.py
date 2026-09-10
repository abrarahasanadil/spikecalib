"""Tests for the sweep's two preconditions.

Both exist to catch failures that are cheap to detect up front and expensive to discover
late: a frame cache that was never built, and a training run that stopped because it ran
out of epochs rather than because it converged. The second is the subtler one -- an
undertrained model produces a perfectly plausible readout, and comparing it against a
converged one would attribute a training difference to a calibration mechanism.
"""

from __future__ import annotations

import os

import pytest

from spikecalib.datasets import DatasetSpec
from spikecalib.sweep import (
    CacheMissingError,
    SweepConfig,
    assert_caches_ready,
    convergence_audit,
    verify_caches,
)


def toy_spec(n_train=6, n_test=4) -> DatasetSpec:
    return DatasetSpec(
        key="toy", display_name="Toy", module="builtins", class_name="object",
        n_classes=2, spatial=(8, 8), n_train=n_train, n_test=n_test,
        n_val_per_class=1, n_cal_per_class=1,
        sj_downloadable=False, implemented=False,
    )


def make_cache(root, spec, timesteps, n_files):
    path = os.path.join(spec.root(str(root)), spec.frames_dir(timesteps))
    os.makedirs(path, exist_ok=True)
    for i in range(n_files):
        open(os.path.join(path, f"{i}.npz"), "wb").close()
    return path


# ---------------------------------------------------------------------------------------
# Precondition 1: frame caches
# ---------------------------------------------------------------------------------------


def test_complete_cache_is_recognised(tmp_path):
    spec = toy_spec()
    make_cache(tmp_path, spec, 2, spec.n_total)
    (status,) = verify_caches(spec, [2], str(tmp_path))
    assert status.exists and status.complete
    assert status.n_found == spec.n_total


def test_missing_cache_is_reported_not_assumed(tmp_path):
    spec = toy_spec()
    (status,) = verify_caches(spec, [4], str(tmp_path))
    assert not status.exists and not status.complete
    assert status.verdict == "MISSING"


def test_stump_cache_is_detected_by_counting_not_by_existence(tmp_path):
    """A directory that exists but is short is exactly what an interrupted build leaves."""
    spec = toy_spec()
    make_cache(tmp_path, spec, 6, spec.n_total - 1)
    (status,) = verify_caches(spec, [6], str(tmp_path))
    assert status.exists
    assert not status.complete
    assert "INCOMPLETE" in status.verdict


def test_assert_caches_ready_raises_naming_every_gap_and_the_fix(tmp_path):
    spec = toy_spec()
    make_cache(tmp_path, spec, 2, spec.n_total)      # fine
    make_cache(tmp_path, spec, 6, 1)                 # stump
    # T=8 absent entirely

    with pytest.raises(CacheMissingError) as excinfo:
        assert_caches_ready(spec, [2, 6, 8], str(tmp_path))

    message = str(excinfo.value)
    assert "T=6" in message and "T=8" in message
    assert "T=2" not in message.split("Build them with:")[0].split("\n", 1)[1], \
        "the complete cache should not be listed as a problem"
    assert message.count("scripts/fetch_data.py") == 2, "one build command per missing cache"
    assert "--timesteps 6" in message and "--timesteps 8" in message


def test_assert_caches_ready_passes_and_returns_statuses_when_all_present(tmp_path):
    spec = toy_spec()
    for t in (2, 4):
        make_cache(tmp_path, spec, t, spec.n_total)
    statuses = assert_caches_ready(spec, [2, 4], str(tmp_path))
    assert len(statuses) == 2 and all(s.complete for s in statuses)


def test_hidden_files_do_not_count_towards_a_cache(tmp_path):
    """macOS metadata must not make a stump look complete."""
    spec = toy_spec()
    path = make_cache(tmp_path, spec, 2, spec.n_total - 1)
    open(os.path.join(path, ".DS_Store"), "wb").close()
    (status,) = verify_caches(spec, [2], str(tmp_path))
    assert not status.complete


# ---------------------------------------------------------------------------------------
# Precondition 2: early stopping must decide, not the epoch cap
# ---------------------------------------------------------------------------------------


def row(timesteps, seed, epochs_run, early_stopped, epochs_requested=30, val=0.99):
    return {
        "dataset": "nmnist", "timesteps": timesteps, "seed": seed,
        "epochs_run": epochs_run, "epochs_requested": epochs_requested,
        "early_stopped": early_stopped, "hit_epoch_cap": not early_stopped,
        "best_val_accuracy": val,
    }


def test_sweep_defaults_let_the_stopping_rule_decide():
    cfg = SweepConfig()
    assert cfg.epochs == 30
    assert cfg.patience == 5
    assert cfg.train_config().epochs == 30
    assert cfg.train_config().patience == 5


def test_audit_reports_all_converged_when_every_run_stopped_early():
    rows = [row(t, s, 12, True) for t in (2, 4) for s in (0, 1)]
    audit = convergence_audit(rows)
    assert audit["n_runs"] == 4
    assert audit["n_capped"] == 0
    assert audit["all_converged"] is True


def test_audit_flags_runs_that_hit_the_epoch_cap():
    rows = [
        row(2, 0, 11, True),
        row(4, 0, 30, False),   # ran out of epochs
        row(8, 1, 30, False),
    ]
    audit = convergence_audit(rows)
    assert audit["all_converged"] is False
    assert audit["n_capped"] == 2
    capped = {(r["timesteps"], r["seed"]) for r in audit["capped"]}
    assert capped == {(4, 0), (8, 1)}


def test_audit_deduplicates_the_four_arm_rows_of_one_run():
    """Every arm of a run repeats its training provenance; the audit counts runs, not rows."""
    rows = [
        {**row(2, 0, 30, False), "arm": arm}
        for arm in ("uncalibrated", "temperature", "isotonic-top-label", "isotonic-one-vs-rest")
    ]
    audit = convergence_audit(rows)
    assert audit["n_runs"] == 1
    assert audit["n_capped"] == 1


def test_audit_orders_runs_by_timesteps_then_seed():
    rows = [row(8, 1, 5, True), row(2, 0, 5, True), row(2, 1, 5, True), row(4, 0, 5, True)]
    audit = convergence_audit(rows)
    assert [(r["timesteps"], r["seed"]) for r in audit["runs"]] == [(2, 0), (2, 1), (4, 0), (8, 1)]
