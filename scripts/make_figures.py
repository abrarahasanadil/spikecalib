#!/usr/bin/env python3
"""Regenerate every figure: the paper's, as vector PDF, and the site's, as SVG and PNG.

Reads only cached artifacts -- ``results/sweep_nmnist.csv``, ``results/readout_stats.json``
and, for two figures, their own caches -- so redrawing a plot never re-runs a model. Each
missing cache is reported with the command that builds it.
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

    # Two figures read a cached artifact of their own rather than the sweep; each is
    # skipped with the command that builds it rather than failing the whole run.
    extra = {
        "figure_risk_coverage": ("cache_path", "selective_cache.npz",
                                 "python scripts/extend_sweep_selective.py"),
        "figure_quantisation": ("quantisation_path", "quantisation_test.json",
                                "python scripts/quantisation_test.py"),
        "figure_fusion": ("fusion_path", "fusion_experiment.json",
                          "python scripts/fusion_experiment.py"),
    }
    drawn = 0
    for fn in ALL_FIGURES:
        if fn.__name__ in extra:
            kwarg, filename, hint = extra[fn.__name__]
            path = os.path.join(REPO_ROOT, "results", filename)
            if not os.path.exists(path):
                print(f"  {fn.__name__:<26} -> skipped (run {hint})")
                continue
            paths = fn(rows, stats, args.out, **{kwarg: path})
        else:
            paths = fn(rows, stats, args.out)
        drawn += 1
        names = ", ".join(os.path.basename(p) for p in paths)
        print(f"  {fn.__name__:<26} -> {names}")
    print(f"\n  wrote {drawn * 3} files to {os.path.relpath(args.out, REPO_ROOT)}")

    # The paper's figures: vector PDF into paper/figures/, PNG beside the site's figures.
    from spikecalib.paper_figures import PAPER_FIGURES
    pdf_dir = os.path.join(REPO_ROOT, "paper", "figures")
    for fn in PAPER_FIGURES:
        try:
            paths = fn(os.path.join(REPO_ROOT, "results"), pdf_dir, args.out)
        except FileNotFoundError as exc:
            print(f"  {fn.__name__:<26} -> skipped ({exc.filename} missing)")
            continue
        print(f"  {fn.__name__:<26} -> " + ", ".join(os.path.relpath(p, REPO_ROOT) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
