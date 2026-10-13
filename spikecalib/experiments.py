"""Shared plumbing for experiments that train one arm or re-read models already trained.

The scale and resolution experiments (``NOTES.md``, predictions 6 to 11) all need the same
four things: the dataset held in memory, the split a run was trained on, a trained model
loaded under the filename its original experiment gave it, and that model's outputs on the
calibration and test splits. Each earlier script carried its own copy of that code; this
module is the one copy the new scripts share, so they cannot disagree about which split or
which tie-break seed a number came from.

Three rules are enforced here rather than left to each caller:

- **Zero DataLoader workers.** Multi-worker loading has deadlocked on this machine. Data is
  read into RAM once, single-process, and served from there.
- **The split must be the one the model was trained on.** A model's record carries the
  fingerprint of its split; :func:`load_trained` refuses a model whose fingerprint does not
  match the split rebuilt from the seed. Fitting a calibrator on samples a model was trained
  on would report training error as calibration error (correctness rule 1).
- **Tie-breaks are seeded as the main sweep seeds them**: the calibration split with
  ``default_rng(seed)`` and the test split with ``default_rng(seed + 1)``, so a number
  computed here is comparable with one in ``results/sweep_nmnist.csv``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .calibration import Outputs, model_outputs
from .data import (
    SplitIndices, load_dataset, make_dataset_splits, materialise, training_labels,
)
from .datasets import DatasetSpec, get_spec
from .models import SCALED_READOUTS, ModelConfig, SpikingClassifier, build_model
from .train import predict_outputs

__all__ = [
    "DEFAULT_CHANNELS", "BATCH_SIZE", "HIDDEN", "stem", "Data", "load_data", "all_cached",
    "splits_for", "stratified_subsample", "subset",
    "loader", "load_trained", "CellOutputs", "cell_outputs",
]

#: The architecture each dataset's main experiments used. A model at these channels has no
#: architecture tag in its filename; any other depth carries one.
DEFAULT_CHANNELS = {"nmnist": (32, 64), "ncaltech101": (32, 64, 128, 128)}
#: The batch size each dataset's existing runs used, read from their run records.
BATCH_SIZE = {"nmnist": 128, "ncaltech101": 32}
HIDDEN = 256


def stem(key: str, timesteps: int, seed: int, readout: str = "count",
         channels: Optional[tuple] = None) -> str:
    """Filename stem of a run, matching the names the earlier experiments wrote."""
    channels = tuple(channels) if channels is not None else DEFAULT_CHANNELS[key]
    rt = "" if readout == "count" else f"_{readout}"
    at = "" if channels == DEFAULT_CHANNELS[key] else "_arch" + "-".join(str(c) for c in channels)
    return f"{key}_T{timesteps}_seed{seed}{rt}{at}"


@dataclass
class Data:
    """One dataset at one ``T``, in memory. ``train`` and ``test`` are the same object for a
    dataset that defines its own split, and different objects for one that ships with one."""

    spec: DatasetSpec
    timesteps: int
    train: object
    test: object
    labels: np.ndarray


def load_data(dataset: str, timesteps: int, data_root: str, *, in_memory: bool = True,
              verbose: bool = True) -> Data:
    """The dataset at one ``T``. In memory by default, which every training run wants.

    ``in_memory=False`` leaves the frames on disk and reads only the labels. That is enough
    to rebuild a split, and it is all an analysis needs once every model's outputs are
    cached; a model without a cache is then read from disk, slowly and correctly.
    """
    spec = get_spec(dataset)
    with contextlib.redirect_stdout(io.StringIO()):
        disk_train, disk_test = load_dataset(spec.key, timesteps, data_root)
    if not in_memory:
        test = disk_train if spec.defines_own_test_split else disk_test
        return Data(spec, timesteps, disk_train, test, training_labels(disk_train))
    label = f"{spec.display_name} T={timesteps}" if verbose else ""
    train = materialise(disk_train, num_workers=0, label=label and label + " train")
    # float16 storage must not have changed a single frame: the claim is that event-count
    # frames are exactly representable, and a silent rounding here would alter every model's
    # input relative to the runs these experiments are compared with.
    probe = np.linspace(0, len(disk_train) - 1, 16).astype(int)
    for i in probe:
        a = torch.as_tensor(np.asarray(disk_train[int(i)][0])).float()
        if not torch.equal(a, train[int(i)][0].float()):
            raise RuntimeError(f"in-memory frame {i} differs from the frame on disk")
    if spec.defines_own_test_split:
        test = train
    else:
        test = materialise(disk_test, num_workers=0, label=label and label + " test")
    return Data(spec, timesteps, train, test, training_labels(train))


def all_cached(results: str, names) -> bool:
    """True when every named run already has its outputs cached."""
    return all(os.path.exists(os.path.join(results, n + "_outputs.npz")) for n in names)


def splits_for(data: Data, seed: int) -> SplitIndices:
    n_test = None if data.spec.defines_own_test_split else len(data.test)
    return make_dataset_splits(data.labels, data.spec, seed=seed, n_test=n_test)


def loader(dataset, idx: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(Subset(dataset, np.asarray(idx).tolist()), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def load_trained(results: str, data: Data, seed: int, readout: str = "count",
                 channels: Optional[tuple] = None,
                 splits: Optional[SplitIndices] = None) -> tuple[SpikingClassifier, dict, str]:
    """Load a trained model, its run record and its filename stem.

    Raises if the weights are missing, or if the record's split fingerprint is not the
    fingerprint of the split rebuilt from ``seed``.
    """
    key = data.spec.key
    channels = tuple(channels) if channels is not None else DEFAULT_CHANNELS[key]
    name = stem(key, data.timesteps, seed, readout, channels)
    wpath, rpath = (os.path.join(results, name + ext) for ext in ("_weights.pt", "_record.json"))
    if not (os.path.exists(wpath) and os.path.exists(rpath)):
        raise FileNotFoundError(f"no trained model {name}; expected {wpath} and {rpath}")
    with open(rpath) as fh:
        record = json.load(fh)
    splits = splits or splits_for(data, seed)
    recorded = record.get("split_fingerprint")
    if recorded is not None and recorded != splits.fingerprint:
        raise RuntimeError(
            f"{name} was trained on split {recorded} but seed {seed} rebuilds "
            f"{splits.fingerprint}; refusing to calibrate or evaluate on a different split"
        )
    mc = record.get("model_config", {})
    cfg = ModelConfig(channels=tuple(mc.get("channels", channels)), hidden=mc.get("hidden", HIDDEN),
                      tau=mc.get("tau", 2.0), v_threshold=mc.get("v_threshold", 1.0),
                      surrogate_alpha=mc.get("surrogate_alpha", 2.0), readout=readout)
    if tuple(cfg.channels) != channels:
        raise RuntimeError(f"{name} records channels {cfg.channels}, expected {channels}")
    model = build_model(data.spec, data.timesteps, cfg)
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    return model, record, name


@dataclass
class CellOutputs:
    """A trained model's outputs on its calibration and test splits.

    ``cal`` and ``test`` are built from the logits the model emits, which for a
    scaled-count model are ``s * counts``. ``test_counts`` holds the underlying integer
    counts for the spiking readouts and ``None`` for the potential readouts, which have
    none.
    """

    name: str
    readout: str
    scale: float
    cal: Outputs
    test: Outputs
    test_counts: Optional[np.ndarray]
    split_fingerprint: str
    record: dict


def _integer_counts(logits: np.ndarray, scale: float, timesteps: int, name: str) -> np.ndarray:
    counts = logits / scale
    rounded = np.rint(counts)
    if np.abs(counts - rounded).max() > 1e-3 or rounded.min() < 0 or rounded.max() > timesteps:
        raise RuntimeError(f"{name}: logits / scale are not integer counts in [0, {timesteps}]")
    return rounded.astype(np.int64)


def cell_outputs(results: str, data: Data, seed: int, device, readout: str = "count",
                 channels: Optional[tuple] = None, *, cache: bool = True) -> CellOutputs:
    """Outputs of one trained model on its calibration and test splits, cached on disk.

    The cache is ``results/<stem>_outputs.npz``, which ``.gitignore`` already excludes with
    the other ``.npz`` files: it is derived from the weights and regenerates from them.
    """
    splits = splits_for(data, seed)
    model, record, name = load_trained(results, data, seed, readout, channels, splits)
    scale = model.scale
    path = os.path.join(results, name + "_outputs.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        ok = (str(z["fingerprint"]) == splits.fingerprint and float(z["scale"]) == scale)
        if ok:
            cal_scores, cal_labels, test_scores, test_labels = (
                z["cal_scores"], z["cal_labels"], z["test_scores"], z["test_labels"])
        else:
            os.remove(path)
    if not (cache and os.path.exists(path)):
        bs = BATCH_SIZE[data.spec.key]
        cal_scores, cal_labels, _ = predict_outputs(
            model, loader(data.train, splits.cal, bs), device, count_synops=False)
        test_scores, test_labels, _ = predict_outputs(
            model, loader(data.test, splits.test, bs), device, count_synops=False)
        if cache:
            np.savez_compressed(path, cal_scores=cal_scores, cal_labels=cal_labels,
                                test_scores=test_scores, test_labels=test_labels,
                                fingerprint=splits.fingerprint, scale=scale)
    test_counts = None
    if readout == "count" or readout in SCALED_READOUTS:
        test_counts = _integer_counts(test_scores, scale, data.timesteps, name)
        _integer_counts(cal_scores, scale, data.timesteps, name)
    return CellOutputs(
        name=name, readout=readout, scale=scale,
        cal=model_outputs(cal_scores, cal_labels, np.random.default_rng(seed)),
        test=model_outputs(test_scores, test_labels, np.random.default_rng(seed + 1)),
        test_counts=test_counts, split_fingerprint=splits.fingerprint, record=record,
    )


def stratified_subsample(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """``size`` indices without replacement, the same number from every class."""
    classes = np.unique(labels)
    per_class, extra = divmod(size, classes.size)
    if extra:
        raise ValueError(f"size {size} is not a multiple of the {classes.size} classes")
    picks = []
    for c in classes:
        members = np.flatnonzero(labels == c)
        if members.size < per_class:
            raise ValueError(f"class {c} has {members.size} calibration samples, need {per_class}")
        picks.append(rng.choice(members, size=per_class, replace=False))
    return np.sort(np.concatenate(picks))


def subset(outputs, idx: np.ndarray):
    """The same ``Outputs`` restricted to ``idx``, tie-breaks and all."""
    return dataclasses.replace(outputs, **{
        f.name: getattr(outputs, f.name)[idx] for f in dataclasses.fields(outputs)})
