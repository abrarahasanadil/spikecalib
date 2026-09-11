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
    # Datasets with no official train/test split define their own. ``test_fraction`` is
    # the per-class share held out, ``test_floor`` the minimum a class contributes however
    # small it is. Both None for datasets that ship a split -- for those the test set is a
    # separate object and carving one would be wrong.
    test_fraction: Optional[float] = None
    test_floor: Optional[int] = None
    # Total sample count for datasets with no official split, where n_train/n_test cannot
    # carry it. This is what a cache is verified against -- the defence against
    # SpikingJelly accepting a truncated directory is the same either way.
    n_samples: Optional[int] = None
    notes: str = ""
    hazards: tuple[str, ...] = field(default_factory=tuple)

    @property
    def defines_own_test_split(self) -> bool:
        return self.test_fraction is not None

    @property
    def n_total(self) -> Optional[int]:
        if self.n_train is None or self.n_test is None:
            return self.n_samples
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
    "ncaltech101": DatasetSpec(
        key="ncaltech101",
        display_name="N-Caltech101",
        module="spikingjelly.datasets.n_caltech101",
        class_name="NCaltech101",
        # 101 categories: the 101 Caltech object classes with the duplicate "Faces_easy"
        # removed and the background class retained. This is the number the ceiling
        # formula takes as C, and it is the whole point of the dataset here -- every other
        # measurement in the study is at C = 10.
        n_classes=101,
        # Native ATIS resolution. 37x the pixel count of N-MNIST's 34x34, which is why
        # this dataset needs more pooling blocks than two; see hazards.
        spatial=(180, 240),
        # No official train/test split: the dataset is one flat folder of 8,709 samples.
        # n_train/n_test are therefore None, and the study defines its own split below.
        n_train=None,
        n_test=None,
        # Caltech101 is severely imbalanced -- roughly 31 samples in the smallest class
        # against 800 in the largest -- so a fixed per-class holdout large enough to be
        # useful on the big classes would exhaust the small ones. The test split is
        # therefore proportional with a floor, while val and cal stay fixed per class so
        # that every class is represented in the calibration set whatever its size.
        #
        # 3 + 10 per class costs 13 of the smallest class's ~25 post-test samples and
        # leaves it ~12 to train on. The calibration set is then ~1,010 samples: larger
        # than DVS-Gesture's planned 330, far smaller than N-MNIST's 5,000, and thin
        # enough that one-vs-rest isotonic gets about 10 positives per class. See hazards.
        n_val_per_class=3,
        n_cal_per_class=10,
        test_fraction=0.20,
        test_floor=6,
        n_samples=8_709,
        sj_downloadable=False,
        implemented=True,
        mirror=Mirror(
            # SpikingJelly's resource list points at the dataset homepage rather than a
            # file, exactly as for N-MNIST. Mendeley Data is used instead: it serves each
            # file at a stable direct URL with a published SHA-256, which is a stronger
            # check than the MD5s SpikingJelly carries. Only Caltech101.zip is needed --
            # the annotations and readmes are not used by the loader.
            url=(
                "https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/"
                "36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded"
            ),
            archive_name="Caltech101.zip",
            approx_bytes=3_989_375_649,
            description="Mendeley Data mirror (Orchard et al., doi:10.17632/cy6cvx3ryv.1)",
        ),
        notes=(
            "The C axis. Every other dataset in the study has C = 10 or 11; this one has "
            "101, where the ceiling at T = 2 is 0.0688 rather than 0.4509."
        ),
        hazards=(
            "SpikingJelly cannot download it -- downloadable() returns False and the "
            "resource list gives the homepage, not files. The Mendeley mirror is used, "
            "and SpikingJelly's own create_events_np_files does the conversion.",
            "No official train/test split. The study defines one: proportional per class "
            "at 20% with a floor of 6, seeded, saved to disk and documented in NOTES.md.",
            "Severely class-imbalanced -- about 31 samples in the smallest class against "
            "800 in the largest. Fixed per-class holdouts must be sized for the smallest "
            "class, not the average.",
            "180x240 is 37 times N-MNIST's pixel count. Two pooling blocks would leave a "
            "45x60x64 feature map and a 44M-parameter first Linear, so this dataset needs "
            "four blocks; model sizing is per-dataset configuration.",
            "The calibration split is ~1,010 samples across 101 classes, so one-vs-rest "
            "isotonic has roughly 10 positives per class and should not be trusted here.",
            "Accuracy will be far below N-MNIST's 0.99. That is the point -- it is the "
            "first genuinely uncertain regime the study has sampled -- but it also means "
            "the ceiling may not bind at every T, and whether it binds must be checked "
            "against measured accuracy rather than assumed.",
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
