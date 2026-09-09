"""The dataset registry: one source of truth for per-dataset metadata.

Every part of the codebase that needs to know how big a dataset is, how many classes it
has, what spatial resolution it uses, or how large its splits should be reads it from
here. Nothing hardcodes 34x34 or 10 classes, because the study runs on three datasets
whose geometry and size differ (NOTES.md, "Per-dataset hazards").

Split sizes live here too, per dataset rather than as a global constant. DVS-Gesture has
roughly 1.3k samples in total, so the 500-per-class calibration split that N-MNIST can
afford is arithmetically impossible there; making the sizes configuration is what stops
that from being discovered halfway through a sweep.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Mirror:
    """A third-party source for archives SpikingJelly will not fetch itself."""

    url: str
    archive_name: str
    approx_bytes: int
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset configuration.

    ``n_train``/``n_test`` are the counts every cache is verified against -- they are the
    defence against SpikingJelly accepting a truncated cache directory. ``spatial`` and
    ``n_classes`` are what model sizing reads. ``n_val_per_class`` and ``n_cal_per_class``
    are what :func:`spikecalib.data.make_splits` carves, and they are per-dataset for the
    reason above.
    """

    key: str
    display_name: str
    module: str
    class_name: str
    n_classes: int
    spatial: tuple[int, int]
    n_train: Optional[int]
    n_test: Optional[int]
    n_val_per_class: int
    n_cal_per_class: int
    sj_downloadable: bool
    implemented: bool
    mirror: Optional[Mirror] = None
    notes: str = ""
    hazards: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_total(self) -> Optional[int]:
        if self.n_train is None or self.n_test is None:
            return None
        return self.n_train + self.n_test

    @property
    def n_val(self) -> int:
        return self.n_val_per_class * self.n_classes

    @property
    def n_cal(self) -> int:
        return self.n_cal_per_class * self.n_classes

    def load_class(self):
        """Import the SpikingJelly dataset class named by this spec."""
        return getattr(importlib.import_module(self.module), self.class_name)

    def root(self, data_root: str) -> str:
        """Where this dataset's cache lives under ``data_root``."""
        return os.path.join(data_root, self.key)

    def frames_dir(self, timesteps: int) -> str:
        """SpikingJelly's directory name for the integrated frame cache at ``T``."""
        return f"frames_number_{timesteps}_split_by_number"


DATASETS: dict[str, DatasetSpec] = {
    "nmnist": DatasetSpec(
        key="nmnist",
        display_name="N-MNIST",
        module="spikingjelly.datasets.n_mnist",
        class_name="NMNIST",
        n_classes=10,
        spatial=(34, 34),
        n_train=60_000,
        n_test=10_000,
        # 6,000 training samples per class, so 500 + 500 costs a sixth of the training
        # data and still leaves 5,000 per class to train on. A 5,000-sample calibration
        # set is comfortably large enough for isotonic regression to fit without the
        # step function chasing noise.
        n_val_per_class=500,
        n_cal_per_class=500,
        sj_downloadable=False,
        implemented=True,
        mirror=Mirror(
            # SpikingJelly's resource list gives the dataset homepage, not a file, so the
            # archives come from the author's Dropbox mirror. It resolves to N-MNIST.zip,
            # which contains Train.zip and Test.zip -- the two files SpikingJelly expects
            # in download/, and the two files its MD5s cover.
            url="https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=1",
            archive_name="N-MNIST.zip",
            approx_bytes=1_181_573_087,
            description="author's Dropbox mirror (Orchard et al.); wraps Train.zip + Test.zip",
        ),
        notes="Primary dataset. Official train/test split.",
        hazards=(
            "SpikingJelly cannot download it; archives come from the mirror and are "
            "verified against SpikingJelly's own MD5s.",
            "Archives carry macOS metadata that SpikingJelly would read as extra classes.",
            "Cache directories are checked only for existence upstream, so counts are "
            "verified here.",
        ),
    ),
    "dvsgesture": DatasetSpec(
        key="dvsgesture",
        display_name="DVS-Gesture",
        module="spikingjelly.datasets.dvs128_gesture",
        class_name="DVS128Gesture",
        n_classes=11,
        spatial=(128, 128),
        # SpikingJelly's counts, which is the convention this study uses; see NOTES.md,
        # "DVS-Gesture sample count", for why this is 1,464 and not the paper's 1,342.
        n_train=1_176,
        n_test=288,
        # About 107 training samples per class, and unevenly distributed because some
        # trials repeat a gesture. 500 per class is impossible. 10 + 30 holds out 40 of
        # roughly 107, leaving ~67 per class to train on. The calibration split is then
        # 330 samples total, which is small enough that isotonic regression can overfit
        # it -- hence cross-validated isotonic for this dataset, and hence
        # calibration-set size being reported next to every result rather than assumed.
        n_val_per_class=10,
        n_cal_per_class=30,
        sj_downloadable=False,
        implemented=False,
        mirror=None,
        notes="Second dataset. Small: ~1.5k samples total under SpikingJelly's convention.",
        hazards=(
            "Small enough that a 500-per-class calibration split is impossible; split "
            "sizes are per-dataset configuration.",
            "Isotonic regression overfits small calibration sets -- cross-validated "
            "isotonic is planned here, and calibration-set size is reported with every "
            "result.",
            "SpikingJelly cannot download it; it is gated behind an IBM Box login.",
            "SpikingJelly's train/test split is by recording trial, so subjects do not "
            "cross it. Splits carved from the training set do not inherit that property.",
        ),
    ),
    "cifar10dvs": DatasetSpec(
        key="cifar10dvs",
        display_name="CIFAR10-DVS",
        module="spikingjelly.datasets.cifar10_dvs",
        class_name="CIFAR10DVS",
        n_classes=10,
        spatial=(128, 128),
        # No official split exists, so there is no upstream train/test count to verify
        # against. The split this study uses will be defined, seeded, saved and
        # documented in NOTES.md when the dataset is brought in.
        n_train=None,
        n_test=None,
        # 10,000 samples, 1,000 per class. The planned split holds out 200 per class as
        # test, leaving 800 per class as the training pool; 100 + 100 then leaves 600 per
        # class to train on and gives a 1,000-sample calibration set -- large enough for
        # isotonic without cross-validation, unlike DVS-Gesture.
        n_val_per_class=100,
        n_cal_per_class=100,
        sj_downloadable=True,
        implemented=False,
        mirror=None,
        notes="Third dataset, if time allows. 10,000 samples, no official split.",
        hazards=(
            "No official train/test split. The split must be defined, seeded, saved to "
            "disk and documented in NOTES.md.",
            "SpikingJelly can download this one (figshare), so no mirror is needed.",
        ),
    ),
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset spec by key, with a message that lists the valid keys."""
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; registered datasets are {sorted(DATASETS)}"
        ) from None
