#!/usr/bin/env python3
"""Dataset acquisition for SpikeCalib.

Dataset-agnostic from the start: a registry holds the per-dataset configuration, and the
acquisition path is selected from it. N-MNIST is implemented; DVS-Gesture and
CIFAR10-DVS are registered with their metadata but raise NotImplementedError on
acquisition, so the interface exists without untested download code sitting in the
repository pretending to work.

Model sizing is configuration, never hardcoded (see NOTES.md): spatial size and class
count differ across the three datasets, and this registry is where those numbers live.

Three known N-MNIST traps are handled explicitly, because each of them fails silently
rather than loudly:

(a) SpikingJelly cannot download N-MNIST. ``NMNIST.downloadable()`` returns False and the
    "url" in its resource list is the dataset homepage, not a file. The archives are
    pulled from the author's Dropbox mirror instead and verified against SpikingJelly's
    own MD5s, read from ``resource_url_md5()`` at runtime. The hashes are never hardcoded
    here -- if SpikingJelly ever revises them, this script follows.

(b) The archives carry macOS metadata entries (``.DS_Store``, ``__MACOSX``, ``._*``), and
    SpikingJelly's ``create_events_np_files`` treats *every* entry under ``extract/Train/``
    as a class directory. One ``.DS_Store`` there becomes an eleventh "class". So the
    archives are extracted here, the metadata is pruned, and only then is the tree handed
    over. Pruning once is not enough: Finder recreates ``.DS_Store`` files mid-run, so
    ``os.listdir`` is also filtered for the duration of the build, which closes the race
    rather than narrowing it.

(c) SpikingJelly only checks that a cache directory EXISTS. An interrupted run leaves a
    stump that the next run silently accepts, yielding a truncated dataset with no error
    -- and a truncated dataset produces plausible-looking calibration numbers computed on
    the wrong data. Every cache is verified by file count against the expected totals,
    rebuilt if short, and re-asserted after construction with a hard exit.

The script is idempotent: a re-run verifies the existing caches and reuses them rather
than refetching 1.1 GB.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import shutil
import sys
import time
import zipfile
from typing import NoReturn, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")

MACOS_METADATA_NAMES = {".DS_Store", "__MACOSX", ".AppleDouble", ".Spotlight-V100", ".fseventsd"}
MACOS_METADATA_PREFIX = "._"

GIB = 1024 ** 3
MIB = 1024 ** 2


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
#
# The registry itself lives in spikecalib/datasets.py so that this script and the
# experiment code cannot drift apart on what a dataset is. Dataset metadata has exactly
# one home.

sys.path.insert(0, REPO_ROOT)

from spikecalib.datasets import DATASETS, DatasetSpec, get_spec  # noqa: E402


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def human(n_bytes: float) -> str:
    if n_bytes >= GIB:
        return f"{n_bytes / GIB:,.2f} GiB"
    if n_bytes >= MIB:
        return f"{n_bytes / MIB:,.1f} MiB"
    return f"{n_bytes:,.0f} B"


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def die(msg: str, code: int = 2) -> NoReturn:
    print()
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def is_macos_metadata(name: str) -> bool:
    return name in MACOS_METADATA_NAMES or name.startswith(MACOS_METADATA_PREFIX)


def prune_macos_metadata(root: str) -> int:
    """Remove macOS metadata entries anywhere under ``root``. Returns how many went.

    This is trap (b). SpikingJelly iterates every entry under ``extract/Train/`` and
    treats it as a class directory, so a single ``.DS_Store`` becomes an eleventh class
    and the build dies partway through -- or worse, does not. Finder recreates these
    files whenever a directory is viewed, so this is called again before disk sizes are
    reported rather than once at the start.
    """
    removed = 0
    if not os.path.exists(root):
        return 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in list(dirnames):
            if is_macos_metadata(d):
                shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                dirnames.remove(d)
                removed += 1
        for f in filenames:
            if is_macos_metadata(f):
                try:
                    os.remove(os.path.join(dirpath, f))
                    removed += 1
                except OSError:
                    pass
    return removed


@contextlib.contextmanager
def macos_metadata_free_listdir():
    """Filter macOS metadata out of ``os.listdir`` for the duration of a build.

    Trap (b), in the form that survives contact with Finder. Pruning the tree before
    handing it to SpikingJelly is not sufficient on its own: macOS recreates
    ``.DS_Store`` whenever a directory is touched, including while a multi-minute build
    is running, and SpikingJelly's ``create_events_np_files`` does

        for class_name in os.listdir(source_dir):
            os.mkdir(os.path.join(target_dir, class_name))
            for bin_file in os.listdir(bin_dir):

    so a ``.DS_Store`` that appears *after* the prune becomes an eleventh class
    directory and the build dies with ``NotADirectoryError`` partway through -- leaving
    exactly the kind of stump described in trap (c). This was observed, not theorised:
    a prune at 17:46 was undone by Finder at 17:47 in the middle of the build.

    ``create_same_directory_structure`` reads ``os.listdir`` too, so the frame-cache path
    has the same exposure. Filtering the listing closes the race instead of narrowing it.
    The patch is scoped to the build and reverted on the way out.
    """
    real_listdir = os.listdir

    def filtered_listdir(path="."):
        return [name for name in real_listdir(path) if not is_macos_metadata(name)]

    os.listdir = filtered_listdir
    try:
        yield
    finally:
        os.listdir = real_listdir


def count_files(root: str, suffix: str) -> int:
    """Count files with ``suffix`` under ``root``, ignoring macOS metadata."""
    if not os.path.isdir(root):
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_macos_metadata(d)]
        total += sum(
            1 for f in filenames if f.endswith(suffix) and not is_macos_metadata(f)
        )
    return total


def dir_size(root: str) -> int:
    if not os.path.exists(root):
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def md5_of(path: str, chunk: int = 8 * MIB) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------


def download_to(url: str, dest: str, expect_bytes: Optional[int] = None) -> None:
    """Stream ``url`` to ``dest``, resuming a partial ``.part`` file where the server allows.

    Written to a ``.part`` sidecar and renamed only on success, so an interrupted download
    can never be mistaken for a finished one -- the same class of mistake as trap (c),
    one level up.
    """
    import requests

    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {}
    mode = "wb"
    if have:
        print(f"  resuming from {human(have)} already in {os.path.basename(part)}")
        headers["Range"] = f"bytes={have}-"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        if have and resp.status_code == 206:
            mode = "ab"
        elif have:
            print("  server ignored the range request; restarting the download")
            have = 0
        declared = resp.headers.get("Content-Length")
        total = (int(declared) + have) if declared else expect_bytes
        if total:
            print(f"  expecting {human(total)} total")

        written = have
        last_report = time.time()
        started = time.time()
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=4 * MIB):
                if not block:
                    continue
                fh.write(block)
                written += len(block)
                now = time.time()
                if now - last_report >= 10.0:
                    rate = (written - have) / max(now - started, 1e-6) / MIB
                    pct = f" ({100.0 * written / total:5.1f}%)" if total else ""
                    print(f"  {human(written)}{pct} at {rate:,.1f} MiB/s", flush=True)
                    last_report = now

    if expect_bytes and abs(written - expect_bytes) > 0:
        print(
            f"  note: got {human(written)}, mirror was expected to serve "
            f"{human(expect_bytes)}; MD5 verification below is the real check"
        )
    os.replace(part, dest)
    print(f"  saved {human(os.path.getsize(dest))} to {os.path.basename(dest)}")


# --------------------------------------------------------------------------------------
# N-MNIST acquisition
# --------------------------------------------------------------------------------------


def ensure_download_archives(spec: DatasetSpec, sj_class, download_root: str, keep_archive: bool) -> None:
    """Guarantee ``download/`` holds the archives SpikingJelly wants, MD5-verified.

    Trap (a). SpikingJelly's ``resource_url_md5()`` gives (filename, url, md5) where the
    url is the dataset homepage -- useless for fetching, but the md5 is authoritative, so
    it is read from there rather than copied into this file.
    """
    os.makedirs(download_root, exist_ok=True)
    resources = sj_class.resource_url_md5()

    print(f"  SpikingJelly downloadable(): {sj_class.downloadable()}")
    for file_name, url, md5 in resources:
        print(f"  requires {file_name}  md5={md5}")
        print(f"    SpikingJelly's url: {url}")

    missing = []
    for file_name, _url, md5 in resources:
        fpath = os.path.join(download_root, file_name)
        if not os.path.exists(fpath):
            missing.append(file_name)
            continue
        print(f"  verifying {file_name} ({human(os.path.getsize(fpath))}) ...", flush=True)
        actual = md5_of(fpath)
        if actual == md5:
            print(f"    md5 ok: {actual}")
        else:
            print(f"    md5 MISMATCH: got {actual}, expected {md5}; removing")
            os.remove(fpath)
            missing.append(file_name)

    if not missing:
        print("  all archives present and verified; no download needed")
        return

    if spec.mirror is None:
        die(
            f"{spec.display_name} archives are missing {missing} and no mirror is "
            f"configured. SpikingJelly cannot download this dataset."
        )

    print(f"  missing {missing}; fetching from the mirror")
    print(f"    {spec.mirror.description}")
    print(f"    {spec.mirror.url}")

    needed = spec.mirror.approx_bytes * 2
    if free_bytes(download_root) < needed:
        die(
            f"need roughly {human(needed)} free to download and unwrap the mirror "
            f"archive; only {human(free_bytes(download_root))} available"
        )

    wrapper = os.path.join(download_root, spec.mirror.archive_name)
    if not os.path.exists(wrapper):
        download_to(spec.mirror.url, wrapper, spec.mirror.approx_bytes)
    else:
        print(f"  reusing already-downloaded {spec.mirror.archive_name}")

    # The mirror serves a folder zip that wraps the two archives SpikingJelly wants.
    wanted = {name for name, _u, _m in resources}
    print(f"  unwrapping {spec.mirror.archive_name} for {sorted(wanted)}")
    with zipfile.ZipFile(wrapper) as zf:
        members = {os.path.basename(n): n for n in zf.namelist()}
        for name in sorted(wanted):
            if name not in members:
                die(
                    f"{spec.mirror.archive_name} does not contain {name}; it holds "
                    f"{sorted(members)[:10]}"
                )
            target = os.path.join(download_root, name)
            if os.path.exists(target):
                continue
            print(f"    extracting {name} ...", flush=True)
            with zf.open(members[name]) as src, open(target + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * MIB)
            os.replace(target + ".part", target)

    # Verify against SpikingJelly's own hashes. This is the gate; nothing proceeds past a
    # mismatch, because a corrupt archive would produce a short dataset that trap (c)
    # would then happily accept.
    for file_name, _url, md5 in resources:
        fpath = os.path.join(download_root, file_name)
        if not os.path.exists(fpath):
            die(f"{file_name} still missing after unwrapping the mirror archive")
        print(f"  verifying {file_name} ({human(os.path.getsize(fpath))}) ...", flush=True)
        actual = md5_of(fpath)
        if actual != md5:
            die(f"{file_name} md5 {actual} != SpikingJelly's {md5}; the mirror is not trusted")
        print(f"    md5 ok: {actual}")

    if not keep_archive and os.path.exists(wrapper):
        size = os.path.getsize(wrapper)
        os.remove(wrapper)
        print(f"  removed the {human(size)} wrapper archive (--keep-archive to retain it)")

    prune_macos_metadata(download_root)


def ensure_extracted(spec: DatasetSpec, download_root: str, extract_root: str) -> None:
    """Extract the archives ourselves and prune macOS metadata before SpikingJelly looks.

    Trap (b). SpikingJelly skips extraction entirely when ``extract/`` already exists, so
    building the tree here is the supported way to hand it a pruned one.
    """
    expected = {"Train": spec.n_train, "Test": spec.n_test}

    def extract_is_complete() -> bool:
        if not os.path.isdir(extract_root):
            return False
        for split, want in expected.items():
            sub = os.path.join(extract_root, split)
            if not os.path.isdir(sub):
                return False
            if count_files(sub, ".bin") != want:
                return False
        return True

    if extract_is_complete():
        print(f"  extract/ is complete ({spec.n_train} + {spec.n_test} .bin files); reusing")
        return

    if os.path.isdir(extract_root):
        found = {s: count_files(os.path.join(extract_root, s), ".bin") for s in expected}
        print(f"  extract/ is a stump ({found}); rebuilding it")
        shutil.rmtree(extract_root)

    os.makedirs(extract_root, exist_ok=True)
    for archive in sorted(os.listdir(download_root)):
        if not archive.endswith(".zip") or is_macos_metadata(archive):
            continue
        path = os.path.join(download_root, archive)
        print(f"  extracting {archive} ({human(os.path.getsize(path))}) ...", flush=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_root)

    removed = prune_macos_metadata(extract_root)
    print(f"  pruned {removed} macOS metadata entries from extract/")

    for split, want in expected.items():
        sub = os.path.join(extract_root, split)
        if not os.path.isdir(sub):
            die(f"expected {sub} after extraction; the archive layout is not what we assume")
        classes = sorted(d for d in os.listdir(sub) if not is_macos_metadata(d))
        got = count_files(sub, ".bin")
        print(f"  extract/{split}: {len(classes)} class dirs {classes}, {got:,} .bin files")
        if len(classes) != spec.n_classes:
            die(
                f"extract/{split} has {len(classes)} class directories, expected "
                f"{spec.n_classes}. Metadata pruning did not clean this tree."
            )
        if got != want:
            die(f"extract/{split} has {got:,} .bin files, expected {want:,}")


def ensure_cache_or_clear(path: str, expected: Optional[int], label: str) -> bool:
    """Trap (c). Verify a cache by file count; delete it if short so it gets rebuilt.

    SpikingJelly checks only that the directory exists. An interrupted run leaves a stump
    that the next run silently accepts, and the study then computes calibration numbers
    on a truncated dataset with nothing anywhere reporting a problem.
    """
    if not os.path.isdir(path):
        print(f"  {label}: absent, will be built")
        return False
    got = count_files(path, ".npz")
    if expected is None:
        print(f"  {label}: {got:,} .npz files (no expected count to check against)")
        return got > 0
    if got == expected:
        print(f"  {label}: complete ({got:,} of {expected:,} .npz); reusing")
        return True
    print(
        f"  {label}: STUMP -- {got:,} of {expected:,} .npz files. SpikingJelly would "
        f"accept this silently. Deleting so it is rebuilt."
    )
    shutil.rmtree(path)
    return False


def acquire_nmnist(spec: DatasetSpec, root: str, timesteps: int, keep_archive: bool) -> dict:
    sj_class = spec.load_class()

    download_root = os.path.join(root, "download")
    extract_root = os.path.join(root, "extract")
    events_root = os.path.join(root, "events_np")
    frames_root = os.path.join(root, spec.frames_dir(timesteps))
    build_log = os.path.join(root, "build.log")

    section("Cache state")
    # Both are evaluated deliberately -- `and` would short-circuit and leave the second
    # cache unreported, and the point of this block is to show the reader the full state.
    train_ok = ensure_cache_or_clear(
        os.path.join(events_root, "train"), spec.n_train, "events_np/train"
    )
    test_ok = ensure_cache_or_clear(
        os.path.join(events_root, "test"), spec.n_test, "events_np/test"
    )
    events_ok = train_ok and test_ok
    # events_np is built as a unit by SpikingJelly, and it refuses to build into an
    # existing directory, so a half-good events_np must go entirely.
    if not events_ok and os.path.isdir(events_root):
        print("  events_np/ is incomplete as a whole; removing it so it is rebuilt")
        shutil.rmtree(events_root)

    frames_ok = ensure_cache_or_clear(frames_root, spec.n_total, spec.frames_dir(timesteps))

    if events_ok and frames_ok:
        print()
        print("  Both caches verify complete. Nothing to download or rebuild.")
    else:
        section("Archives")
        if events_ok:
            print("  events_np/ is complete, so the raw archives are not needed for this run")
        else:
            ensure_download_archives(spec, sj_class, download_root, keep_archive)
            section("Extraction")
            ensure_extracted(spec, download_root, extract_root)

    section(f"Building the dataset (T={timesteps})")
    print(f"  SpikingJelly's per-file chatter goes to {os.path.relpath(build_log, REPO_ROOT)}")
    print("  this takes a while on a cold cache; the counts below are the real check")
    started = time.time()
    sets = {}
    # Prune right before handing over, then keep the listing filtered for the whole
    # build -- the prune alone loses the race against Finder (see the shim's docstring).
    prune_macos_metadata(root)
    with open(build_log, "a") as log:
        log.write(f"\n===== build {time.strftime('%Y-%m-%d %H:%M:%S')} T={timesteps} =====\n")
        with contextlib.redirect_stdout(log), macos_metadata_free_listdir():
            for train in (True, False):
                sets["train" if train else "test"] = sj_class(
                    root=root,
                    train=train,
                    data_type="frame",
                    frames_number=timesteps,
                    split_by="number",
                )
    print(f"  built in {time.time() - started:,.1f}s")

    # Trap (b) again: Finder recreates .DS_Store while a long build is running.
    removed = prune_macos_metadata(root)
    if removed:
        print(f"  pruned {removed} macOS metadata entries recreated during the build")

    section("Post-construction assertions")
    expected = {"train": spec.n_train, "test": spec.n_test}
    bad = []
    for split, ds in sets.items():
        got = len(ds)
        want = expected[split]
        status = "ok" if got == want else "WRONG"
        print(f"  len({split}_set) = {got:,}  expected {want:,}  [{status}]")
        if got != want:
            bad.append(f"{split}: {got:,} != {want:,}")

    sample, label = sets["train"][0]
    print(f"  sample frame shape {tuple(sample.shape)}  label {int(label)}")
    expected_shape = (timesteps, 2, spec.spatial[0], spec.spatial[1])
    if tuple(sample.shape) != expected_shape:
        bad.append(f"frame shape {tuple(sample.shape)} != {expected_shape}")
        print(f"  frame shape {tuple(sample.shape)} != expected {expected_shape}  [WRONG]")

    if bad:
        die(
            "the built dataset does not match the registry: "
            + "; ".join(bad)
            + ". This is exactly the truncation SpikingJelly does not detect. "
            "Delete the offending cache directory and re-run."
        )
    print("  all assertions passed")

    return {
        "download": download_root,
        "extract": extract_root,
        "events_np": events_root,
        spec.frames_dir(timesteps): frames_root,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def report_disk(root: str, paths: dict, spec: DatasetSpec, timesteps: int) -> None:
    section("Disk cost")
    prune_macos_metadata(root)  # Finder recreates these; prune before measuring.

    per_t_key = spec.frames_dir(timesteps)
    one_time = {k: v for k, v in paths.items() if k != per_t_key}
    per_t = {k: v for k, v in paths.items() if k == per_t_key}

    one_time_total = 0
    print("  One-time (shared by every T):")
    for name, path in one_time.items():
        size = dir_size(path)
        one_time_total += size
        note = "" if os.path.exists(path) else "  (absent)"
        print(f"    {name:<44} {human(size):>12}{note}")
    print(f"    {'subtotal':<44} {human(one_time_total):>12}")

    per_t_total = 0
    print(f"  Per-T (one such directory for each T in the sweep, this run T={timesteps}):")
    for name, path in per_t.items():
        size = dir_size(path)
        per_t_total += size
        print(f"    {name:<44} {human(size):>12}")
    print(f"    {'subtotal':<44} {human(per_t_total):>12}")

    print(f"  Total on disk now: {human(one_time_total + per_t_total)}")
    print(
        f"  Projected for the full sweep T in {{2,4,6,8,10}}: "
        f"{human(one_time_total + 5 * per_t_total)} "
        f"(one-time + 5 x per-T, assuming per-T scales with T; lower T is smaller)"
    )
    print(f"  Free after this run: {human(free_bytes(root))}")


def print_registry(spec: DatasetSpec) -> None:
    section(f"{spec.display_name} ({spec.key})")
    print(f"  SpikingJelly class    : {spec.module}.{spec.class_name}")
    print(f"  spatial               : {spec.spatial[0]} x {spec.spatial[1]}")
    print(f"  classes               : {spec.n_classes}")
    train = f"{spec.n_train:,}" if spec.n_train is not None else "n/a (no official split)"
    test = f"{spec.n_test:,}" if spec.n_test is not None else "n/a (no official split)"
    print(f"  expected train / test : {train} / {test}")
    print(f"  val / cal per class   : {spec.n_val_per_class} / {spec.n_cal_per_class} "
          f"(val {spec.n_val:,}, cal {spec.n_cal:,} total)")
    print(f"  SpikingJelly can DL   : {spec.sj_downloadable}")
    print(f"  mirror                : {spec.mirror.description if spec.mirror else 'none'}")
    print(f"  acquisition           : {'implemented' if spec.implemented else 'NOT IMPLEMENTED'}")
    if spec.notes:
        print(f"  notes                 : {spec.notes}")
    for h in spec.hazards:
        print(f"  hazard                : {h}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and build an event dataset cache for SpikeCalib.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="T for the integrated frame cache. Only T=10 is integrated for now.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the wrapper archive from the mirror after unwrapping it",
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="print the registry entry and exit without touching disk",
    )
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    root = os.path.join(args.data_root, spec.key)

    print("SpikeCalib data acquisition")
    print("=" * 27)
    print_registry(spec)

    if args.registry_only:
        return 0

    if not spec.implemented:
        section("Acquisition")
        raise NotImplementedError(
            f"Acquisition for {spec.display_name} is not implemented yet. It is "
            f"registered above with its metadata so that the rest of the codebase can be "
            f"written against the registry, but no untested download code ships for it. "
            f"N-MNIST is the implemented dataset: --dataset nmnist"
        )

    section("Paths")
    os.makedirs(root, exist_ok=True)
    print(f"  data root : {root}")
    print(f"  free disk : {human(free_bytes(root))}")

    paths = acquire_nmnist(spec, root, args.timesteps, args.keep_archive)
    report_disk(root, paths, spec, args.timesteps)

    section("Result")
    print(f"  {spec.display_name} is ready at T={args.timesteps}.")
    print("  Re-running this script verifies the caches and reuses them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
