#!/usr/bin/env python3
"""Regenerate every figure for the site, as SVG and PNG.

Reads only cached artifacts -- ``results/sweep_nmnist.csv`` and
``results/readout_stats.json`` -- so redrawing a plot never re-runs a model. If the
readout cache is missing, build it with ``scripts/extract_readout_stats.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spikecalib.figures import ALL_FIGURES, load_readout_stats, load_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nmnist")
    parser.add_argument("--sweep", default=os.path.join(REPO_ROOT, "results", "sweep_nmnist.csv"))
    parser.add_argument("--stats", default=os.path.join(REPO_ROOT, "results", "readout_stats.json"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "figures"))
    args = parser.parse_args()

    for path, hint in (
        (args.sweep, "python scripts/run_sweep.py --epochs 50 --patience 5"),
        (args.stats, "python scripts/extract_readout_stats.py"),
    ):
        if not os.path.exists(path):
            print(f"missing {path}\n  build it with: {hint}", file=sys.stderr)
            return 1

    rows = load_sweep(args.sweep)
    stats = load_readout_stats(args.stats, args.dataset)
    print(f"  {len(rows):,} sweep rows, {len(stats)} timestep settings")

    for fn in ALL_FIGURES:
        if fn.__name__ == "figure_risk_coverage":
            cache = os.path.join(REPO_ROOT, "results", "selective_cache.npz")
            if not os.path.exists(cache):
                print(f"  {fn.__name__:<26} -> skipped (run scripts/extend_sweep_selective.py)")
                continue
            paths = fn(rows, stats, args.out, cache_path=cache)
        else:
            paths = fn(rows, stats, args.out)
        names = ", ".join(os.path.basename(p) for p in paths)
        print(f"  {fn.__name__:<26} -> {names}")
    print(f"\n  wrote {len(ALL_FIGURES) * 2} files to {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
