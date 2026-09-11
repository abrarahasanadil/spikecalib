"""Tests for the split machinery -- correctness rule 1 and the rule 4 reproducibility half.

These tests deliberately weight failure paths over happy ones. A split bug does not
announce itself: an overlapping calibration set produces an *excellent*-looking
calibration error, because the number being reported is training error. So the tests that
matter most are the ones asserting that bad input is rejected loudly -- that
``assert_disjoint`` actually raises, that an undersized dataset fails instead of silently
returning short splits, and that the error messages carry enough detail to act on.

Everything here runs on small synthetic label arrays. No N-MNIST, no disk, no torch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from spikecalib.data import (
    make_train_test_split,
    make_dataset_splits,
    CARVED_SPLITS,
    InsufficientSamplesError,
    SplitOverlapError,
    assert_disjoint,
    load_splits,
    make_splits,
    save_splits,
    summarise,
)
from spikecalib.datasets import DATASETS, DatasetSpec


def toy_spec(
    n_classes: int = 4,
    n_val_per_class: int = 5,
    n_cal_per_class: int = 7,
    n_test: int | None = 40,
) -> DatasetSpec:
    """A synthetic spec, so the tests do not depend on any real dataset's numbers."""
    return DatasetSpec(
        key="toy",
        display_name="Toy",
        module="builtins",
        class_name="object",
        n_classes=n_classes,
        spatial=(8, 8),
        n_train=None,
        n_test=n_test,
        n_val_per_class=n_val_per_class,
        n_cal_per_class=n_cal_per_class,
        sj_downloadable=False,
        implemented=False,
    )


def toy_labels(n_classes: int = 4, per_class: int = 30) -> np.ndarray:
    """Balanced synthetic labels, shuffled so ordering cannot flatter the splitter."""
    labels = np.repeat(np.arange(n_classes), per_class)
    return np.random.default_rng(12345).permutation(labels)


# ---------------------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------------------


def test_splits_are_disjoint_on_a_normal_construction():
    spec, labels = toy_spec(), toy_labels()
    splits = make_splits(labels, spec, seed=0)

    for a, b in [("train", "val"), ("train", "cal"), ("val", "cal")]:
        overlap = np.intersect1d(getattr(splits, a), getattr(splits, b))
        assert overlap.size == 0, f"{a} and {b} overlap on {overlap.size} indices"

    # And the explicit check agrees.
    assert_disjoint(splits)


def test_carved_splits_partition_the_training_set_exactly():
    """Every training index lands in exactly one of train/val/cal -- none lost, none duplicated."""
    spec, labels = toy_spec(), toy_labels()
    splits = make_splits(labels, spec, seed=0)

    union = np.concatenate([getattr(splits, s) for s in CARVED_SPLITS])
    assert union.size == labels.size
    assert np.array_equal(np.sort(union), np.arange(labels.size))


def test_test_split_is_the_test_sets_own_range_and_is_never_carved():
    """test lives in a different index space; it is the test object's full range."""
    spec, labels = toy_spec(n_test=40), toy_labels()
    splits = make_splits(labels, spec, seed=0)
    assert np.array_equal(splits.test, np.arange(40))
    # Nothing was taken out of the training set to build it.
    assert sum(getattr(splits, s).size for s in CARVED_SPLITS) == labels.size


# ---------------------------------------------------------------------------------------
# Failure paths -- the ones that actually protect the study
# ---------------------------------------------------------------------------------------


def test_assert_disjoint_raises_split_overlap_error_naming_both_splits_and_the_size():
    """A deliberately overlapped split must raise, not warn, and must say what collided."""
    spec, labels = toy_spec(), toy_labels()
    splits = make_splits(labels, spec, seed=0)

    # Force a three-index overlap between cal and val.
    poisoned = dataclasses.replace(
        splits, cal=np.sort(np.concatenate([splits.cal[:-3], splits.val[:3]]))
    )

    with pytest.raises(SplitOverlapError) as excinfo:
        assert_disjoint(poisoned)

    message = str(excinfo.value)
    assert "'val'" in message and "'cal'" in message, f"message names neither split: {message}"
    assert "3 shared" in message, f"message does not report the overlap size: {message}"


def test_assert_disjoint_raises_on_a_single_shared_index():
    """One shared sample is still a violation; there is no tolerated overlap."""
    spec, labels = toy_spec(), toy_labels()
    splits = make_splits(labels, spec, seed=0)
    poisoned = dataclasses.replace(
        splits, train=np.sort(np.concatenate([splits.train, splits.cal[:1]]))
    )
    with pytest.raises(SplitOverlapError) as excinfo:
        assert_disjoint(poisoned)
    assert "1 shared index" in str(excinfo.value)


def test_make_splits_calls_assert_disjoint_itself():
    """A caller must not be able to obtain an unchecked split."""
    import spikecalib.data as data_mod

    spec, labels = toy_spec(), toy_labels()
    calls: list[object] = []
    original = data_mod.assert_disjoint
    data_mod.assert_disjoint = lambda s: (calls.append(s), original(s))[1]
    try:
        make_splits(labels, spec, seed=0)
    finally:
        data_mod.assert_disjoint = original
    assert len(calls) == 1, "make_splits returned without asserting disjointness"


def test_dataset_too_small_fails_loudly_instead_of_returning_short_splits():
    """The DVS-Gesture hazard, in miniature: ask for more than exists."""
    spec = toy_spec(n_classes=3, n_val_per_class=20, n_cal_per_class=20)
    labels = toy_labels(n_classes=3, per_class=25)  # 25 < 20 + 20 + 1

    with pytest.raises(InsufficientSamplesError) as excinfo:
        make_splits(labels, spec, seed=0)

    message = str(excinfo.value)
    assert "class 0" in message, f"message does not name the class: {message}"
    assert "25" in message, f"message does not state what is available: {message}"
    assert "41" in message, f"message does not state what was requested: {message}"


def test_a_single_short_class_fails_even_when_the_dataset_is_large_overall():
    """Imbalance must not hide behind the total; the check is per class."""
    spec = toy_spec(n_classes=3, n_val_per_class=5, n_cal_per_class=5)
    labels = np.concatenate([np.zeros(100), np.ones(100), np.full(8, 2)]).astype(np.int64)

    with pytest.raises(InsufficientSamplesError) as excinfo:
        make_splits(labels, spec, seed=0)
    assert "class 2" in str(excinfo.value)


def test_missing_class_is_reported_as_zero_available():
    spec = toy_spec(n_classes=4, n_val_per_class=2, n_cal_per_class=2)
    labels = toy_labels(n_classes=3, per_class=30)  # class 3 absent entirely

    with pytest.raises(InsufficientSamplesError) as excinfo:
        make_splits(labels, spec, seed=0)
    message = str(excinfo.value)
    assert "class 3" in message and "0 training samples" in message


def test_missing_test_count_is_an_error_not_a_silent_empty_test_set():
    """CIFAR10-DVS has no official test set; that must be raised, not defaulted away."""
    spec = toy_spec(n_test=None)
    labels = toy_labels()
    with pytest.raises(ValueError, match="no official test set"):
        make_splits(labels, spec, seed=0)


# ---------------------------------------------------------------------------------------
# Reproducibility -- correctness rule 4
# ---------------------------------------------------------------------------------------


def test_same_seed_gives_identical_indices():
    spec, labels = toy_spec(), toy_labels()
    a = make_splits(labels, spec, seed=7)
    b = make_splits(labels, spec, seed=7)
    for s in (*CARVED_SPLITS, "test"):
        assert np.array_equal(getattr(a, s), getattr(b, s)), f"{s} differs at a fixed seed"
    assert a.fingerprint == b.fingerprint


def test_different_seeds_give_different_indices():
    spec, labels = toy_spec(), toy_labels()
    a = make_splits(labels, spec, seed=0)
    b = make_splits(labels, spec, seed=1)
    assert not np.array_equal(a.cal, b.cal), "the calibration split did not move with the seed"
    assert a.fingerprint != b.fingerprint


def test_splits_do_not_touch_the_global_numpy_rng():
    """default_rng only: drawing a split must not perturb anyone else's stream."""
    spec, labels = toy_spec(), toy_labels()
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    make_splits(labels, spec, seed=3)
    after = np.random.random()
    assert before == after


# ---------------------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------------------


def test_every_class_gets_exactly_the_requested_val_and_cal_counts():
    spec = toy_spec(n_classes=4, n_val_per_class=5, n_cal_per_class=7)
    labels = toy_labels(n_classes=4, per_class=30)
    splits = make_splits(labels, spec, seed=0)

    for cls in range(spec.n_classes):
        assert int((labels[splits.val] == cls).sum()) == 5
        assert int((labels[splits.cal] == cls).sum()) == 7


def test_within_class_selection_is_shuffled_not_positional():
    """Selecting the first N of each class would couple the split to file ordering.

    SpikingJelly builds its index by walking class directories, so positional selection
    would systematically pick the same recordings into every calibration set -- correlated
    with whatever the filesystem ordering happens to encode, and identical across seeds.
    """
    spec = toy_spec(n_classes=2, n_val_per_class=5, n_cal_per_class=5)
    labels = np.repeat(np.arange(2), 40)  # deliberately NOT shuffled

    splits = make_splits(labels, spec, seed=0)
    for cls in range(2):
        cls_idx = np.flatnonzero(labels == cls)
        positional = cls_idx[: spec.n_val_per_class]
        chosen = np.intersect1d(splits.val, cls_idx)
        assert not np.array_equal(chosen, positional), (
            f"class {cls}: val is the first {spec.n_val_per_class} indices by position, "
            "so the within-class shuffle is not happening"
        )


def test_stratification_holds_on_an_imbalanced_training_set():
    """val and cal stay balanced even when train does not."""
    spec = toy_spec(n_classes=3, n_val_per_class=4, n_cal_per_class=4)
    labels = np.concatenate([np.zeros(200), np.ones(50), np.full(20, 2)]).astype(np.int64)
    splits = make_splits(labels, spec, seed=0)

    val_counts = np.bincount(labels[splits.val], minlength=3)
    cal_counts = np.bincount(labels[splits.cal], minlength=3)
    assert val_counts.tolist() == [4, 4, 4]
    assert cal_counts.tolist() == [4, 4, 4]
    # train keeps the imbalance rather than having it quietly equalised
    train_counts = np.bincount(labels[splits.train], minlength=3)
    assert train_counts.tolist() == [192, 42, 12]


# ---------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------


def test_save_load_round_trips_to_identical_indices(tmp_path):
    spec, labels = toy_spec(), toy_labels()
    original = make_splits(labels, spec, seed=11)

    path = save_splits(original, str(tmp_path / "split.npz"))
    restored = load_splits(path)

    for s in (*CARVED_SPLITS, "test"):
        assert np.array_equal(getattr(original, s), getattr(restored, s))
    assert restored.seed == original.seed
    assert restored.dataset == original.dataset
    assert restored.n_val_per_class == original.n_val_per_class
    assert restored.n_cal_per_class == original.n_cal_per_class
    assert restored.fingerprint == original.fingerprint


def test_saved_split_still_passes_the_disjointness_assertion(tmp_path):
    spec, labels = toy_spec(), toy_labels()
    path = save_splits(make_splits(labels, spec, seed=2), str(tmp_path / "s.npz"))
    assert_disjoint(load_splits(path))


def test_corrupted_saved_split_is_detected_on_load(tmp_path):
    """The stored fingerprint is a checksum, not decoration."""
    import json

    spec, labels = toy_spec(), toy_labels()
    original = make_splits(labels, spec, seed=4)
    path = save_splits(original, str(tmp_path / "s.npz"))

    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    tampered_cal = arrays["cal"].copy()
    tampered_cal[0] = arrays["train"][0]  # smuggle a training index into cal
    np.savez_compressed(
        path,
        train=arrays["train"],
        val=arrays["val"],
        cal=tampered_cal,
        test=arrays["test"],
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="corrupt"):
        load_splits(path)


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------


def test_summarise_reports_sizes_and_per_class_counts():
    spec = toy_spec(n_classes=3, n_val_per_class=4, n_cal_per_class=6)
    labels = toy_labels(n_classes=3, per_class=40)
    splits = make_splits(labels, spec, seed=0)

    text = summarise(splits, labels, spec=spec)
    assert "train" in text and "val" in text and "cal" in text
    assert "never carved" in text
    assert str(splits.fingerprint) in text
    # one row per class
    for cls in range(3):
        assert f"    {cls:>7} " in text


def test_summarise_makes_imbalance_visible():
    spec = toy_spec(n_classes=2, n_val_per_class=2, n_cal_per_class=2)
    labels = np.concatenate([np.zeros(100), np.ones(20)]).astype(np.int64)
    splits = make_splits(labels, spec, seed=0)
    text = summarise(splits, labels, spec=spec)
    assert "imbalance" in text
    assert "6.000" in text  # 96 / 16


# ---------------------------------------------------------------------------------------
# The registry the real runs use
# ---------------------------------------------------------------------------------------


def test_registry_split_sizes_fit_every_dataset_that_declares_counts():
    """A spec whose split sizes cannot fit its own dataset is a bug in the registry."""
    for key, spec in DATASETS.items():
        if spec.n_train is None:
            continue
        per_class = spec.n_train / spec.n_classes
        needed = spec.n_val_per_class + spec.n_cal_per_class + 1
        assert needed < per_class, (
            f"{key}: split wants {needed} per class but only ~{per_class:.0f} are available"
        )


def test_nmnist_registry_split_sizes_are_the_documented_ones():
    spec = DATASETS["nmnist"]
    assert (spec.n_val_per_class, spec.n_cal_per_class) == (500, 500)
    assert spec.n_val == 5_000 and spec.n_cal == 5_000


# ---------------------------------------------------------------------------------------
# Datasets that must define their own train/test split
# ---------------------------------------------------------------------------------------


def caltech_like_labels():
    """Class sizes with Caltech101's shape: a few huge classes, a long tail of tiny ones.

    The real distribution runs from about 31 samples to about 800. What matters for the
    split code is that both extremes are present, because a scheme tuned for the mean
    breaks at one end or the other.
    """
    sizes = [800, 467, 435] + [90] * 40 + [50] * 40 + [31] * 18
    labels = np.concatenate([np.full(n, c) for c, n in enumerate(sizes)])
    return labels, sizes


def caltech_like_spec(**overrides):
    base = dict(
        key="fake101", display_name="Fake-101", module="m", class_name="C",
        n_classes=101, spatial=(180, 240), n_train=None, n_test=None,
        n_val_per_class=3, n_cal_per_class=10, sj_downloadable=False, implemented=True,
        test_fraction=0.20, test_floor=6,
    )
    base.update(overrides)
    return DatasetSpec(**base)


def test_proportional_test_split_scales_with_class_size_and_respects_the_floor():
    """The reason the split is proportional: fixed counts cannot serve 31 and 800 at once."""
    labels, sizes = caltech_like_labels()
    spec = caltech_like_spec()
    train_pool, test = make_train_test_split(labels, spec, seed=0)

    per_class = np.bincount(labels[test], minlength=spec.n_classes)
    assert per_class[0] == 160, "the 800-sample class should give 20%"
    assert per_class[-1] == 6, "the 31-sample class should hit the floor, not 20%"
    # Monotone in class size: a bigger class never contributes fewer test samples.
    order = np.argsort(sizes)[::-1]
    assert list(per_class[order]) == sorted(per_class[order], reverse=True)


def test_every_class_survives_the_test_split_with_enough_left_for_val_cal_and_train():
    labels, _ = caltech_like_labels()
    spec = caltech_like_spec()
    train_pool, test = make_train_test_split(labels, spec, seed=0)
    remaining = np.bincount(labels[train_pool], minlength=spec.n_classes)
    assert remaining.min() >= spec.n_val_per_class + spec.n_cal_per_class + 1
    # And make_splits must then actually succeed on that pool.
    splits = make_splits(labels[train_pool], spec, seed=0, n_test=test.size)
    assert splits.val.size == spec.n_val
    assert splits.cal.size == spec.n_cal


def test_train_test_split_partitions_the_dataset_exactly():
    labels, _ = caltech_like_labels()
    train_pool, test = make_train_test_split(labels, caltech_like_spec(), seed=0)
    assert np.intersect1d(train_pool, test).size == 0
    assert train_pool.size + test.size == labels.size
    assert np.array_equal(np.union1d(train_pool, test), np.arange(labels.size))


def test_train_test_split_is_reproducible_and_seed_dependent():
    labels, _ = caltech_like_labels()
    spec = caltech_like_spec()
    a = make_train_test_split(labels, spec, seed=0)
    b = make_train_test_split(labels, spec, seed=0)
    c = make_train_test_split(labels, spec, seed=1)
    assert np.array_equal(a[1], b[1]), "same seed must give byte-identical indices"
    assert not np.array_equal(a[1], c[1]), "a different seed must give a different split"
    # Sizes are seed-independent even though membership is not.
    assert a[1].size == c[1].size


def test_a_class_too_small_to_survive_the_split_is_refused_loudly():
    """Silently dropping a class would corrupt the calibration set without a symptom."""
    labels = np.concatenate([np.full(100, c) for c in range(100)] + [np.full(9, 100)])
    spec = caltech_like_spec()
    with pytest.raises(InsufficientSamplesError, match="class 100"):
        make_train_test_split(labels, spec, seed=0)


def test_carving_a_test_set_from_a_dataset_that_ships_one_is_refused():
    labels = np.concatenate([np.full(50, c) for c in range(10)])
    with pytest.raises(ValueError, match="official train/test split"):
        make_train_test_split(labels, DATASETS["nmnist"], seed=0)


def test_ncaltech101_registry_entry_matches_the_documented_design():
    spec = DATASETS["ncaltech101"]
    assert spec.n_classes == 101, "C = 101 is the entire point of this dataset"
    assert spec.spatial == (180, 240)
    assert spec.defines_own_test_split
    assert (spec.test_fraction, spec.test_floor) == (0.20, 6)
    assert (spec.n_val_per_class, spec.n_cal_per_class) == (3, 10)
    assert spec.n_train is None and spec.n_test is None, "no official split to verify against"
    assert spec.mirror is not None and not spec.sj_downloadable


def test_ceiling_at_101_classes_is_far_more_binding_than_at_10():
    """The measurement this dataset exists to make."""
    from spikecalib.models import readout_ceiling

    assert readout_ceiling("count", 2, 101) == pytest.approx(0.068806, abs=1e-6)
    assert readout_ceiling("count", 2, 10) == pytest.approx(0.450853, abs=1e-6)
    # More classes always lowers the ceiling at fixed T.
    for T in (2, 6, 10):
        assert readout_ceiling("count", T, 101) < readout_ceiling("count", T, 10)


def test_flat_dataset_splits_all_index_one_space_and_never_overlap():
    """The leak this guards against is invisible to the ordinary disjointness check.

    For a dataset with an official split, ``test`` addresses a different object, so train
    and test sharing an integer is meaningless. For a flat dataset they address the same
    rows, and a collision would evaluate on training data while every existing assertion
    stayed green.
    """
    labels, _ = caltech_like_labels()
    spec = caltech_like_spec()
    splits = make_dataset_splits(labels, spec, seed=0)

    pools = [splits.train, splits.val, splits.cal, splits.test]
    everything = np.concatenate(pools)
    assert everything.size == labels.size, "the four splits must partition the dataset"
    assert np.array_equal(np.sort(everything), np.arange(labels.size)), "no gaps, no repeats"
    for i, a in enumerate(pools):
        for b in pools[i + 1:]:
            assert np.intersect1d(a, b).size == 0

    # Stratification survives the round trip through the flat index space.
    assert np.bincount(labels[splits.val], minlength=101).min() == spec.n_val_per_class
    assert np.bincount(labels[splits.cal], minlength=101).min() == spec.n_cal_per_class


def test_make_dataset_splits_delegates_for_datasets_that_ship_a_split():
    labels = np.concatenate([np.full(1200, c) for c in range(10)])
    spec = DATASETS["nmnist"]
    a = make_dataset_splits(labels, spec, seed=0, n_test=10_000)
    b = make_splits(labels, spec, seed=0, n_test=10_000)
    assert a.fingerprint == b.fingerprint


def test_flat_dataset_splits_are_reproducible():
    labels, _ = caltech_like_labels()
    spec = caltech_like_spec()
    assert (make_dataset_splits(labels, spec, seed=0).fingerprint
            == make_dataset_splits(labels, spec, seed=0).fingerprint)
    assert (make_dataset_splits(labels, spec, seed=0).fingerprint
            != make_dataset_splits(labels, spec, seed=1).fingerprint)
