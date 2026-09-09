"""Shared fixtures and skip markers.

CI runs on CPU, so anything that genuinely needs Apple silicon is skipped rather than
failed there. Almost nothing does: the model, the SynOps counter and the split machinery
are all device-agnostic, and running them on CPU in CI is a feature, not a compromise --
CPU is bitwise deterministic where MPS is not (correctness rule 4).
"""

from __future__ import annotations

import pytest
import torch


def mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(
    not mps_available(), reason="requires Apple silicon MPS; CI runs on CPU"
)
