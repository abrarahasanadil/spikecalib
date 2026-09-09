#!/usr/bin/env python3
"""Environment check for SpikeCalib.

Reports python, torch, MPS availability, the spikingjelly import and free disk, then
runs a **real 3-timestep forward pass through a spikingjelly LIFNode on MPS**.

Why the forward pass, and why zero spikes is a hard failure: a shapes-only check passes
without ever exercising the compare-and-reset path -- the membrane potential crossing
threshold, emitting a spike, and being reset. That path is the part most likely to break
under a new torch or MPS backend, and if it silently emits nothing, every downstream
spike count is zero, every confidence is a uniform softmax, and the whole calibration
study quietly measures nothing. So this script drives the neuron with an input chosen to
force a sub-threshold-to-threshold crossing and fails with a non-zero exit if no spike
comes out.

Exit codes: 0 all checks passed, 1 a hard check failed.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LIF drive. With tau=2.0, v_threshold=1.0, v_reset=0.0 and a constant input of 1.5,
# the membrane potential goes 0.75 (no spike) -> 1.125 (SPIKE, reset to 0) -> 0.75.
# That trajectory is deliberate: step 1 must not spike and step 2 must, so a neuron that
# fires unconditionally and a neuron that never fires both fail this check.
T_STEPS = 3
BATCH = 64
FEATURES = 128
X_DRIVE = 1.5
TAU = 2.0
V_THRESHOLD = 1.0

failures: list[str] = []


def _dist_version(name: str) -> str:
    """Distribution version. spikingjelly ships no ``__version__`` attribute."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def hard_fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL: {msg}")


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    print("SpikeCalib environment check")
    print("=" * 28)

    section("Interpreter")
    print(f"  python        : {platform.python_version()}  ({sys.executable})")
    print(f"  platform      : {platform.platform()}")
    print(f"  machine       : {platform.machine()}")

    section("Packages")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"  torch         : IMPORT FAILED ({exc})")
        hard_fail("torch could not be imported")
        return report()
    print(f"  torch         : {torch.__version__}")

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    print(f"  MPS available : {mps_available}   (built: {mps_built})")
    if not mps_available:
        hard_fail("MPS is not available; this study is configured for Apple silicon")

    try:
        import spikingjelly
        from spikingjelly.activation_based import functional, neuron

        sj_version = _dist_version("spikingjelly")
        print(f"  spikingjelly  : import ok  ({sj_version})")
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"  spikingjelly  : IMPORT FAILED ({exc})")
        hard_fail("spikingjelly could not be imported")
        return report()

    try:
        import sklearn

        print(f"  scikit-learn  : {sklearn.__version__}")
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"  scikit-learn  : IMPORT FAILED ({exc})")
        hard_fail("scikit-learn could not be imported (calibrator fitting depends on it)")

    section("Disk")
    usage = shutil.disk_usage(REPO_ROOT)
    gib = 1024 ** 3
    print(f"  path          : {REPO_ROOT}")
    print(f"  free          : {usage.free / gib:,.1f} GiB free of {usage.total / gib:,.1f} GiB")
    if usage.free < 25 * gib:
        hard_fail(
            f"only {usage.free / gib:,.1f} GiB free; N-MNIST acquisition and the T=10 "
            f"frame cache need roughly 25 GiB of headroom"
        )

    if not mps_available:
        print()
        print("  Skipping the LIF forward pass: no MPS device.")
        return report()

    section(f"LIFNode forward pass on MPS (T={T_STEPS})")
    device = torch.device("mps")
    lif = neuron.LIFNode(
        tau=TAU,
        v_threshold=V_THRESHOLD,
        v_reset=0.0,
        detach_reset=True,
        step_mode="m",
    ).to(device)
    functional.reset_net(lif)

    x = torch.full((T_STEPS, BATCH, FEATURES), X_DRIVE, device=device)
    with torch.no_grad():
        spikes = lif(x)

    print(f"  input         : {tuple(x.shape)} constant {X_DRIVE} on {x.device}")
    print(f"  output        : {tuple(spikes.shape)} on {spikes.device}")

    if spikes.device.type != "mps":
        hard_fail(f"output landed on {spikes.device.type}, not mps")
    if tuple(spikes.shape) != tuple(x.shape):
        hard_fail(f"output shape {tuple(spikes.shape)} != input shape {tuple(x.shape)}")

    spikes_cpu = spikes.detach().to("cpu")
    unique = spikes_cpu.unique().tolist()
    per_step = spikes_cpu.flatten(1).sum(dim=1).tolist()
    total = float(spikes_cpu.sum().item())
    elements = spikes_cpu.numel()

    print(f"  unique values : {unique}")
    print(f"  spikes / step : {[int(v) for v in per_step]}  (of {BATCH * FEATURES} neurons)")
    print(f"  total spikes  : {int(total)} / {elements}")

    # THE hard check. Zero spikes means the compare-and-reset path never ran.
    if total == 0.0:
        hard_fail(
            "the LIFNode emitted ZERO spikes -- the compare-and-reset path did not "
            "execute, so every downstream spike count would be zero"
        )
    else:
        print("  compare       : threshold crossing produced spikes (path exercised)")

    if not set(unique).issubset({0.0, 1.0}):
        hard_fail(f"spike output is not binary; got unique values {unique}")

    # Reset check. Step 1 must be silent and step 2 must fire; if step 1 already fires,
    # the threshold comparison is not doing what the drive was chosen to test.
    if per_step[0] != 0:
        hard_fail(
            f"step 1 emitted {int(per_step[0])} spikes but the chosen drive puts the "
            f"membrane potential at {X_DRIVE / TAU} < {V_THRESHOLD} -- threshold "
            f"comparison is wrong"
        )
    if per_step[1] != BATCH * FEATURES:
        hard_fail(
            f"step 2 emitted {int(per_step[1])} of {BATCH * FEATURES} expected spikes -- "
            f"threshold crossing is not firing every neuron as the drive requires"
        )

    v_final = float(lif.v.detach().to("cpu").mean().item())
    print(f"  final v       : {v_final:.4f}  (post-reset recharge, threshold {V_THRESHOLD})")
    if v_final >= V_THRESHOLD:
        hard_fail(
            f"membrane potential {v_final:.4f} is still at or above threshold after "
            f"spiking -- reset did not happen"
        )

    functional.reset_net(lif)
    return report()


def report() -> int:
    section("Result")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        print()
        print(f"  {len(failures)} hard check(s) failed.")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
