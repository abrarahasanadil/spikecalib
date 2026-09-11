"""Dataset loading and the four-way train / val / calibration / test split.

Owns correctness rule 1: **the calibration split is disjoint from both the test split
and the validation split, asserted on index sets.**

Why an assertion and not a construction that looks right. A calibrator is a fitted
model. Isotonic regression with enough steps can fit a calibration set almost perfectly.
If the data it is fitted on overlaps the data it is evaluated on, the reported
calibration error is training error -- and training error looks *excellent*, which is
precisely why this failure is never caught by reading a results table. It does not
produce an implausible number; it produces the number you were hoping for. So the
disjointness is checked on the index sets themselves, inside :func:`make_splits`, where
a caller cannot forget to call it.

**Two index spaces.** ``train``, ``val`` and ``cal`` are positions in the *training*
dataset; ``test`` is positions in the dataset's own *test* object. They are different
Python objects and different index spaces, so the test set is disjoint from the other
three structurally rather than numerically -- it is never carved. Comparing a train-space
index against a test-space index would be meaningless, so :func:`assert_disjoint` checks
the three that share a space and the test set's separateness is maintained by never
slicing it. :func:`summarise` states this on every report so it cannot be misread.

Owns correctness rule 4 as it applies to splitting: **splits use CPU RNG only.** MPS
float ops are not bitwise deterministic, so anything that must be reproducible
bit-for-bit is kept off the accelerator. Splitting uses ``numpy.random.default_rng(seed)``
and nothing else -- no torch, no global seeding, no MPS. Split index sets are therefore
reproducible exactly, even though trained weights are not, which is why the study reports
over 3 seeds.

Split sizes are per-dataset configuration read from :mod:`spikecalib.datasets`, never a
global constant: DVS-Gesture is too small for the 500-per-class split N-MNIST affords.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from .datasets import DatasetSpec, get_spec

__all__ = [
    "SplitIndices",
    "SplitOverlapError",
    "InsufficientSamplesError",
    "load_dataset",
    "training_labels",
    "make_splits",
    "make_train_test_split",
    "make_dataset_splits",
    "assert_disjoint",
    "save_splits",
    "load_splits",
    "summarise",
]

# The three splits carved out of the training set. They share an index space, so they are
# the ones that can actually collide. ``test`` is a different object entirely.
CARVED_SPLITS = ("train", "val", "cal")

SPLIT_FORMAT_VERSION = 1


class SplitOverlapError(Exception):
    """Raised when two splits that must be disjoint share at least one index.

    A distinct exception type rather than a bare ``AssertionError`` for two reasons:
    assertions vanish under ``python -O``, and this condition invalidates every number in
    the paper, so it deserves a name a caller can catch and a test can assert on.
    """


class InsufficientSamplesError(ValueError):
    """Raised when a dataset cannot supply the per-class split sizes requested."""


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays for one seeded split of one dataset.

    Arrays, not ``Subset`` objects, so they can be asserted on, saved, reloaded and
    compared between runs. A ``Subset`` hides its indices behind a dataset wrapper and
    cannot be diffed against another run's.
    """

    dataset: str
    seed: int
    n_val_per_class: int
    n_cal_per_class: int
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> dict[str, int]:
        return {name: int(getattr(self, name).size) for name in (*CARVED_SPLITS, "test")}

    @property
    def fingerprint(self) -> str:
        """A short hash of the exact indices, for citing a split next to a result.

        Two runs that report the same fingerprint used byte-identical splits. This is
        what lets a results row prove which split produced it rather than assert it.
        """
        h = hashlib.sha256()
        h.update(f"{SPLIT_FORMAT_VERSION}|{self.dataset}|{self.seed}".encode())
        for name in (*CARVED_SPLITS, "test"):
            arr = np.asarray(getattr(self, name), dtype=np.int64)
            h.update(name.encode())
            h.update(arr.tobytes())
        return h.hexdigest()[:16]


def load_dataset(name: str, timesteps: int, root: str):
    """Load the train and test halves of a dataset from its built frame cache.

    Paths come from the registry rather than being re-derived by hand, so this cannot
    drift from what ``scripts/fetch_data.py`` built.
    """
    spec = get_spec(name)
    dataset_root = spec.root(root)
    frames = os.path.join(dataset_root, spec.frames_dir(timesteps))
    if not os.path.isdir(frames):
        raise FileNotFoundError(
            f"no frame cache for {spec.display_name} at T={timesteps}: {frames} does not "
            f"exist. Build it with:\n"
            f"    python scripts/fetch_data.py --dataset {spec.key} --timesteps {timesteps}"
        )
    sj_class = spec.load_class()
    common = dict(root=dataset_root, data_type="frame", frames_number=timesteps, split_by="number")
    if spec.defines_own_test_split:
        # No official split, so SpikingJelly's class takes no ``train`` argument and there
        # is exactly one object. Both halves are that same object; which samples belong to
        # which split is carried entirely by the index arrays from
        # :func:`make_dataset_splits`, whose ``test`` indexes this same flat space.
        # Returning it twice keeps every caller's ``(train_set, test_set)`` shape working.
        flat = sj_class(**common)
        return flat, flat
    return sj_class(train=True, **common), sj_class(train=False, **common)


def training_labels(train_set) -> np.ndarray:
    """Class labels of the training set, without loading a single frame.

    SpikingJelly's dataset is a torchvision ``DatasetFolder`` underneath, which records
    ``targets`` when it indexes the directory tree. Reading that avoids decoding 60,000
    npz files just to learn which class each one belongs to.
    """
    targets = getattr(train_set, "targets", None)
    if targets is None:
        raise AttributeError(
            "dataset exposes no .targets; cannot read labels without loading every sample"
        )
    return np.asarray(targets, dtype=np.int64)


def make_train_test_split(
    labels: Sequence[int] | np.ndarray, spec: DatasetSpec, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Define a train/test split for a dataset that has no official one.

    Returns ``(train_pool, test)`` as index arrays into the single flat dataset. The
    train pool is what :func:`make_splits` then carves val and cal out of; the test set is
    never touched again.

    **Proportional per class, with a floor**, rather than a fixed count per class. The
    fixed-count scheme the other datasets use assumes classes of comparable size. Caltech101
    does not oblige: its smallest class holds about 31 samples and its largest about 800, so
    a per-class holdout big enough to be useful on the large classes would consume the small
    ones entirely, and one small enough for the small classes would waste most of the large
    ones. Each class therefore contributes ``max(test_floor, round(test_fraction * n))``.

    The floor exists because a proportional split alone would give the smallest class six
    test samples at 20% and fewer still if a smaller class appeared; below that a per-class
    accuracy is not a measurement at all.

    Reproducibility: ``numpy.random.default_rng(seed)`` only, like :func:`make_splits`, and
    the class loop runs in ascending class order so the result depends on the seed and
    nothing else.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if not spec.defines_own_test_split:
        raise ValueError(
            f"{spec.display_name} ships an official train/test split, so carving one "
            f"would discard it. Use the dataset's own train/test objects instead."
        )

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for cls in range(spec.n_classes):
        cls_idx = np.flatnonzero(labels == cls)
        available = cls_idx.size
        if available == 0:
            raise InsufficientSamplesError(
                f"{spec.display_name}: class {cls} has no samples at all. The registry "
                f"says this dataset has {spec.n_classes} classes; the cache disagrees."
            )
        n_test = max(int(spec.test_floor), int(round(spec.test_fraction * available)))
        # Every class must keep enough to supply val, cal and at least one training sample
        # once make_splits runs on what is left.
        needed_after = spec.n_val_per_class + spec.n_cal_per_class + 1
        if available - n_test < needed_after:
            raise InsufficientSamplesError(
                f"{spec.display_name}: class {cls} has {available:,} samples; holding out "
                f"{n_test:,} for test leaves {available - n_test:,}, but val + cal + at "
                f"least one training sample needs {needed_after:,}. Lower test_fraction, "
                f"test_floor, n_val_per_class or n_cal_per_class for this dataset in "
                f"spikecalib/datasets.py."
            )
        shuffled = rng.permutation(cls_idx)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])

    train_pool = np.sort(np.concatenate(train_parts)).astype(np.int64)
    test = np.sort(np.concatenate(test_parts)).astype(np.int64)
    if np.intersect1d(train_pool, test).size:
        raise SplitOverlapError("train pool and test set overlap")
    if train_pool.size + test.size != labels.size:
        raise SplitOverlapError(
            f"split lost samples: {train_pool.size:,} + {test.size:,} != {labels.size:,}"
        )
    return train_pool, test


def make_splits(
    labels: Sequence[int] | np.ndarray,
    spec: DatasetSpec,
    seed: int,
    n_test: Optional[int] = None,
) -> SplitIndices:
    """Carve stratified train / val / calibration splits out of a training set.

    ``labels`` are the training set's class labels. ``test`` is the dataset's own test
    set and is never carved from these -- it is returned as the full index range of that
    separate object.

    Stratified by class, so every class contributes exactly ``spec.n_val_per_class`` to
    val and ``spec.n_cal_per_class`` to cal. An unstratified split would let a rare class
    fall out of the calibration set entirely, and a calibrator fitted without it would be
    extrapolating on that class at test time.

    Reproducibility: ``numpy.random.default_rng(seed)`` only. No torch, no global RNG, no
    MPS. Same seed, same machine or not, same indices.

    Disjointness is asserted here, before returning, so no caller can skip it.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

    if n_test is None:
        n_test = spec.n_test
    if n_test is None:
        raise ValueError(
            f"{spec.display_name} has no official test set, so n_test cannot be inferred "
            f"from the registry. Define, seed and save the study's own train/test split "
            f"first (see NOTES.md), then pass n_test explicitly."
        )

    rng = np.random.default_rng(seed)
    n_val_pc = spec.n_val_per_class
    n_cal_pc = spec.n_cal_per_class

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []

    for cls in range(spec.n_classes):
        cls_idx = np.flatnonzero(labels == cls)
        available = cls_idx.size
        # One training sample is the minimum that still makes the split meaningful; a
        # class with zero training samples cannot be learned at all.
        needed = n_val_pc + n_cal_pc + 1
        if available < needed:
            raise InsufficientSamplesError(
                f"{spec.display_name}: class {cls} has {available:,} training samples but "
                f"the split requests {n_val_pc:,} for val + {n_cal_pc:,} for cal + at "
                f"least 1 for train = {needed:,}. Lower n_val_per_class/n_cal_per_class "
                f"for this dataset in spikecalib/datasets.py; split sizes are per-dataset "
                f"configuration for exactly this reason."
            )
        shuffled = rng.permutation(cls_idx)
        val_parts.append(shuffled[:n_val_pc])
        cal_parts.append(shuffled[n_val_pc : n_val_pc + n_cal_pc])
        train_parts.append(shuffled[n_val_pc + n_cal_pc :])

    # Sorted so that two runs of the same seed produce byte-identical arrays regardless of
    # the order classes happened to be concatenated in, which makes the fingerprint and
    # any array comparison meaningful.
    splits = SplitIndices(
        dataset=spec.key,
        seed=seed,
        n_val_per_class=n_val_pc,
        n_cal_per_class=n_cal_pc,
        train=np.sort(np.concatenate(train_parts)).astype(np.int64),
        val=np.sort(np.concatenate(val_parts)).astype(np.int64),
        cal=np.sort(np.concatenate(cal_parts)).astype(np.int64),
        test=np.arange(n_test, dtype=np.int64),
    )
    assert_disjoint(splits)
    return splits


def make_dataset_splits(
    labels: Sequence[int] | np.ndarray,
    spec: DatasetSpec,
    seed: int,
    n_test: Optional[int] = None,
) -> SplitIndices:
    """The one entry point for splits, whatever shape the dataset comes in.

    Datasets with an official train/test split get :func:`make_splits` directly: ``labels``
    are the training set's and ``test`` indexes the separate test object.

    Datasets without one get a study-defined split first. ``labels`` are then the *whole*
    dataset's, :func:`make_train_test_split` holds out a test set, val and cal are carved
    from what remains, and **every returned array indexes the same flat dataset** -- which
    is the object :func:`load_dataset` returns for both halves in that case.

    Using this rather than calling :func:`make_splits` directly is what stops a caller from
    silently treating a flat dataset's ``test`` indices as if they addressed a separate
    object, which would sample the wrong rows without raising anything.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if not spec.defines_own_test_split:
        return make_splits(labels, spec, seed=seed, n_test=n_test)

    train_pool, test = make_train_test_split(labels, spec, seed)
    # Carve val/cal out of the training pool, then lift the positions back into the flat
    # index space. make_splits works in the pool's own 0..n-1 coordinates.
    carved = make_splits(labels[train_pool], spec, seed=seed, n_test=test.size)
    splits = SplitIndices(
        dataset=spec.key,
        seed=seed,
        n_val_per_class=spec.n_val_per_class,
        n_cal_per_class=spec.n_cal_per_class,
        train=np.sort(train_pool[carved.train]).astype(np.int64),
        val=np.sort(train_pool[carved.val]).astype(np.int64),
        cal=np.sort(train_pool[carved.cal]).astype(np.int64),
        test=test,
    )
    assert_disjoint(splits)
    # The test set shares an index space with the carved splits here, which it does not for
    # datasets that ship their own, so the overlap check above cannot see it. Check it.
    for name in CARVED_SPLITS:
        overlap = np.intersect1d(getattr(splits, name), splits.test)
        if overlap.size:
            raise SplitOverlapError(
                f"{spec.display_name}: {overlap.size:,} samples appear in both {name} and "
                f"test. For a dataset with no official split these share an index space, "
                f"so this is a real leak, not a false alarm."
            )
    return splits


def assert_disjoint(splits: SplitIndices) -> None:
    """Correctness rule 1, checked pairwise on the index sets themselves.

    Only ``train``/``val``/``cal`` are compared: they index the training set and can
    genuinely collide. ``test`` indexes a different dataset object and is never carved
    from the training set, so it is disjoint by construction -- comparing its integers
    against training-set integers would compare unrelated things and produce a false
    alarm on every well-formed split.

    Raises :class:`SplitOverlapError` naming both splits and the size of the overlap.
    """
    for a, b in combinations(CARVED_SPLITS, 2):
        left = np.asarray(getattr(splits, a), dtype=np.int64)
        right = np.asarray(getattr(splits, b), dtype=np.int64)
        overlap = np.intersect1d(left, right, assume_unique=False)
        if overlap.size:
            preview = ", ".join(str(int(i)) for i in overlap[:5])
            more = "" if overlap.size <= 5 else f", ... ({overlap.size:,} total)"
            raise SplitOverlapError(
                f"splits '{a}' and '{b}' overlap: {overlap.size:,} shared index"
                f"{'' if overlap.size == 1 else 'es'} [{preview}{more}]. "
                f"A calibrator fitted on data it is evaluated on reports training error, "
                f"which invalidates every calibration number derived from this split."
            )


def save_splits(splits: SplitIndices, path: str) -> str:
    """Persist exact indices plus the provenance needed to prove which split ran.

    Stored as compressed npz: the arrays keep their exact dtype and values, and the
    metadata rides alongside as JSON so the file is self-describing without this module.
    """
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    meta = {
        "format_version": SPLIT_FORMAT_VERSION,
        "dataset": splits.dataset,
        "seed": splits.seed,
        "n_val_per_class": splits.n_val_per_class,
        "n_cal_per_class": splits.n_cal_per_class,
        "sizes": splits.sizes,
        "fingerprint": splits.fingerprint,
    }
    np.savez_compressed(
        path,
        train=splits.train,
        val=splits.val,
        cal=splits.cal,
        test=splits.test,
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    return path


def load_splits(path: str) -> SplitIndices:
    """Reload a saved split, verifying it still hashes to what was written."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        splits = SplitIndices(
            dataset=meta["dataset"],
            seed=int(meta["seed"]),
            n_val_per_class=int(meta["n_val_per_class"]),
            n_cal_per_class=int(meta["n_cal_per_class"]),
            train=z["train"].astype(np.int64),
            val=z["val"].astype(np.int64),
            cal=z["cal"].astype(np.int64),
            test=z["test"].astype(np.int64),
        )
    stored = meta.get("fingerprint")
    if stored is not None and stored != splits.fingerprint:
        raise ValueError(
            f"{path} is corrupt: stored fingerprint {stored} but the loaded indices hash "
            f"to {splits.fingerprint}"
        )
    return splits


def summarise(
    splits: SplitIndices,
    labels: Sequence[int] | np.ndarray,
    spec: Optional[DatasetSpec] = None,
) -> str:
    """A printable report of split sizes and per-class counts.

    Per-class counts are printed rather than summarised so class imbalance is *visible*.
    A calibration set that is balanced by construction and a training set that is not is
    a fact about the experiment, and it should be readable rather than assumed away.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if spec is None:
        try:
            spec = get_spec(splits.dataset)
        except KeyError:
            spec = None
    n_classes = spec.n_classes if spec is not None else int(labels.max()) + 1
    name = spec.display_name if spec is not None else splits.dataset

    lines: list[str] = []
    lines.append(f"Split summary: {name}  seed={splits.seed}  fingerprint={splits.fingerprint}")
    lines.append("")
    lines.append("  Sizes")
    sizes = splits.sizes
    carved_total = sum(sizes[s] for s in CARVED_SPLITS)
    for s in CARVED_SPLITS:
        pct = 100.0 * sizes[s] / carved_total if carved_total else 0.0
        lines.append(f"    {s:<6} {sizes[s]:>8,}  ({pct:5.1f}% of the training set)")
    lines.append(f"    {'test':<6} {sizes['test']:>8,}  (the dataset's own test set, never carved)")
    lines.append(f"    {'':<6} {'':>8}  train+val+cal = {carved_total:,} of {labels.size:,} training samples")
    lines.append("")
    lines.append("  Per-class counts (training-set index space)")
    lines.append(f"    {'class':>7} {'train':>9} {'val':>7} {'cal':>7}")
    per_class = {}
    for s in CARVED_SPLITS:
        idx = np.asarray(getattr(splits, s), dtype=np.int64)
        per_class[s] = np.bincount(labels[idx], minlength=n_classes)
    for cls in range(n_classes):
        lines.append(
            f"    {cls:>7} {per_class['train'][cls]:>9,} "
            f"{per_class['val'][cls]:>7,} {per_class['cal'][cls]:>7,}"
        )
    tr = per_class["train"]
    if tr.min() > 0:
        lines.append("")
        lines.append(
            f"    train imbalance: min {tr.min():,} / max {tr.max():,} "
            f"= ratio {tr.max() / tr.min():.3f}"
        )
    lines.append("")
    lines.append(
        "  val and cal are balanced by construction "
        f"({splits.n_val_per_class} and {splits.n_cal_per_class} per class); "
        "train carries whatever imbalance the dataset has."
    )
    return "\n".join(lines)
