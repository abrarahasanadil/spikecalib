"""Figures for the study.

Owns correctness rule 3: **energy is the SynOps proxy** computed with 45 nm constants
``E_AC = 0.9 pJ`` and ``E_MAC = 4.6 pJ``.

Verified 2026-09-12 against the primary document. ``E_AC`` is ``[LIT]``: the 32-bit
floating-point FAdd of Horowitz, ISSCC 2014, **Figure 1.1.9**, "Rough energy costs for
various operations in 45nm 0.9V". ``E_MAC`` is ``[DERIVED]``, not ``[LIT]`` -- **the paper
states no MAC value**, and 4.6 = 0.9 + 3.7 (32-bit FMult) is the community convention of
multiply-plus-add. A caption may now cite the figure number for ``E_AC``; it must not
attribute 4.6 pJ to Horowitz.

The source calls these figures "Rough", so no result may turn on their precision. Every
appearance of an energy number -- in an axis label, a caption, a table header, or body text
-- describes it as a **design-comparison proxy**. It is never described as measured power,
and no claim about a physical device is derived from it.

Every figure is emitted as **SVG, PNG and vector PDF** so a page can prefer the vector and
fall back. Every figure also has a companion data table on the site, which is not decoration:
one of the categorical hues sits below 3:1 contrast on the light surface, so identity must
never rest on colour alone. Series are direct-labelled as well as legended.

Design constraints applied here, so they do not have to be remembered per plot:

- Categorical hues are assigned in fixed order and never cycled. The four-slot set used
  for the calibration arms was validated for colour-vision deficiency across **all** pairs,
  not merely adjacent ones, because a four-line chart invites any-pair comparison.
- **No dual-axis plots.** Where two measures of different scale belong together they get
  two panels, never two y-scales on one.
- Grid and axes are recessive; marks are thin; only endpoints carry direct labels rather
  than every point.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

__all__ = [
    "PALETTE",
    "load_sweep",
    "load_readout_stats",
    "ceiling",
    "save",
    "figure_ceiling",
    "figure_loss_floor",
    "figure_ece_by_arm",
    "figure_paired_difference",
    "figure_capacity",
    "figure_energy",
    "figure_risk_coverage",
    "figure_quantisation",
    "figure_fusion",
    "load_quantisation",
    "ALL_FIGURES",
]

# Categorical slots from the Okabe-Ito colour-blind-safe set, the palette the paper's
# figures use (spikecalib/paper_figures.py). The slot names are kept from the earlier
# palette so every figure function below is unchanged; "aqua" is Okabe-Ito's bluish green
# and "violet" its reddish purple. Every figure still ships direct labels.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "aqua": "#009E73",
    "violet": "#CC79A7",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

ARM_COLOURS = {
    "uncalibrated": PALETTE["blue"],
    "temperature": PALETTE["orange"],
    "isotonic-top-label": PALETTE["aqua"],
    "isotonic-one-vs-rest": PALETTE["violet"],
}
ARM_LABELS = {
    "uncalibrated": "uncalibrated",
    "temperature": "temperature",
    "isotonic-top-label": "isotonic (top-label)",
    "isotonic-one-vs-rest": "isotonic (one-vs-rest)",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update({
        "svg.hashsalt": "spikecalib",
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
    })
    return plt


def _halo():
    """A surface-coloured stroke behind label text.

    Point labels sit near the marks they annotate, and on a steep line the text and the
    line collide. A halo is more robust than hand-tuning each offset, which breaks the
    moment the data changes.
    """
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=3.0, foreground=SURFACE)]


def ceiling(timesteps: int, n_classes: int = 10) -> float:
    """[DERIVED] ``e^T / (e^T + C - 1)``: softmax of the extreme count vector ``[T, 0...]``."""
    return float(np.exp(timesteps) / (np.exp(timesteps) + n_classes - 1))


def load_sweep(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_readout_stats(path: str, dataset: str = "nmnist") -> dict:
    with open(path) as fh:
        return json.load(fh)[dataset]


def load_quantisation(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)["rows"]


def _by_t(rows, arm):
    out = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            out[int(r["timesteps"])].append(r)
    return dict(sorted(out.items()))


def save(fig, out_dir: str, name: str) -> list[str]:
    """Write SVG, PNG and a vector PDF with embedded TrueType fonts."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("svg", "png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        meta = {"CreationDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=160, metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------------------


def figure_ceiling(rows, stats, out_dir):
    """Attainable-confidence ceiling against T, with what the models actually reached."""
    plt = _style()
    Ts = sorted(int(t) for t in stats)
    ceil = [stats[str(t)]["ceiling_derived"] for t in Ts]
    observed = [stats[str(t)]["max_confidence_observed"] for t in Ts]
    acc = [np.mean([float(r["test_accuracy"]) for r in _by_t(rows, "uncalibrated")[t]]) for t in Ts]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    grid = np.linspace(2, 10, 240)
    curve = [ceiling(t) for t in grid]
    band = ax.fill_between(grid, curve, 1.0, color=PALETTE["blue"], alpha=0.07, zorder=1,
                           label="unreachable by any spike-count readout")
    ax.plot(grid, curve, color=PALETTE["blue"], lw=2, zorder=2,
            label="ceiling  $e^T/(e^T{+}C{-}1)$  [derived]")
    ax.plot(Ts, acc, "s--", color=PALETTE["aqua"], mec=SURFACE, mew=1.8, lw=1.6, zorder=3,
            label="test accuracy")
    # Open markers, because the observed maxima land ON the derived curve to ~1e-13 and a
    # filled marker would simply disappear into the line. That coincidence is the finding.
    ax.plot(Ts, observed, "o", color=PALETTE["orange"], mfc=SURFACE, mew=2.2, ms=9, zorder=4,
            label="observed max confidence")

    ax.set_xlabel("simulation timesteps $T$")
    ax.set_ylabel("confidence")
    ax.set_xticks(Ts)
    ax.set_xlim(1.6, 10.5)
    ax.set_ylim(0.18, 1.05)
    ax.legend(loc="lower right", fontsize=8.5, borderaxespad=0.8)
    ax.set_title("Where accuracy sits above the ceiling, correct confidence is unreachable",
                 fontsize=10.5, color=INK, pad=10)
    return save(fig, out_dir, "fig1-ceiling")


def figure_loss_floor(rows, stats, out_dir):
    """Cross-entropy floor -ln(ceiling) against the observed training-loss plateau."""
    plt = _style()
    Ts = sorted(int(t) for t in stats)
    floor = [stats[str(t)]["cross_entropy_floor"] for t in Ts]
    observed = [stats[str(t)]["train_loss_final"] for t in Ts]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(Ts, floor, "-", color=PALETTE["blue"], lw=2, zorder=2)
    ax.plot(Ts, observed, "o", color=PALETTE["orange"], mec=SURFACE, mew=2, zorder=3)
    for t, f, o in zip(Ts, floor, observed):
        ax.plot([t, t], [f, o], color=INK_SOFT, lw=0.8, alpha=0.5, zorder=1)

    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xlabel("simulation timesteps $T$")
    ax.set_ylabel("cross-entropy loss (log scale)")
    ax.annotate("floor  $-\\ln(\\mathrm{ceiling})$", xy=(8, floor[3]), xytext=(5.4, 0.0016),
                fontsize=9, color=PALETTE["blue"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["blue"], lw=1))
    ax.annotate("observed plateau", xy=(10, observed[4]), xytext=(7.2, 0.030),
                fontsize=9, color=PALETTE["orange"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["orange"], lw=1))
    ax.set_title("Training loss cannot fall below the floor, so it is not comparable across $T$",
                 fontsize=10.5, color=INK, pad=10)
    return save(fig, out_dir, "fig2-loss-floor")


def figure_ece_by_arm(rows, stats, out_dir):
    """Equal-width and equal-mass ECE for all four arms, on a log axis."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, key, title in zip(
        axes,
        ("ece_equal_width", "ece_equal_mass"),
        ("equal-width bins (Guo et al.)", "equal-mass bins"),
    ):
        for arm, colour in ARM_COLOURS.items():
            grouped = _by_t(rows, arm)
            Ts = sorted(grouped)
            means = [np.mean([float(r[key]) for r in grouped[t]]) for t in Ts]
            ax.plot(Ts, means, "-o", color=colour, mec=SURFACE, mew=1.6, label=ARM_LABELS[arm])
        ax.set_yscale("log")
        ax.set_xticks(Ts)
        ax.set_xlabel("simulation timesteps $T$")
        ax.set_title(title, fontsize=10, color=INK_SOFT, pad=8)
    axes[0].set_ylabel("expected calibration error (log scale)")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.02, 0.40), fontsize=8.5)
    fig.suptitle("Calibration removes almost all of the error; which calibrator does not matter",
                 fontsize=10.5, color=INK, y=1.0)
    return save(fig, out_dir, "fig3-ece-by-arm")


def figure_paired_difference(rows, stats, out_dir):
    """Paired isotonic-minus-temperature ECE difference, 95% bootstrap intervals, per estimator.

    The two panels share their rows, so a run that resolved because the calibrators truly
    differed would be marked in both. The title counts the runs that are.
    """
    plt = _style()
    entries = sorted(
        (r for r in rows if r["arm"] == "isotonic-top-label"),
        key=lambda r: (int(r["timesteps"]), int(r["seed"])),
    )
    estimators = (
        ("equal-width", "iso_minus_temp_ece_ew", "iso_minus_temp_ci_low",
         "iso_minus_temp_ci_high", "iso_minus_temp_resolvable"),
        ("equal-mass", "iso_minus_temp_ece_em", "iso_minus_temp_em_ci_low",
         "iso_minus_temp_em_ci_high", "iso_minus_temp_em_resolvable"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.8), sharey=True)
    for ax, (name, key, lo_key, hi_key, res_key) in zip(axes, estimators):
        ax.axvline(0.0, color=INK_SOFT, lw=1.2, zorder=1)
        for i, r in enumerate(entries):
            d = float(r[key])
            lo, hi = float(r[lo_key]), float(r[hi_key])
            resolvable = r[res_key] == "True"
            colour = PALETTE["orange"] if resolvable else PALETTE["blue"]
            ax.plot([lo, hi], [i, i], color=colour, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([d], [i], "o", color=colour, mec=SURFACE, mew=1.6, zorder=3)
            if resolvable:
                # Label on the side facing zero's far half, so it stays inside the panel.
                isotonic = hi < 0
                ax.annotate("isotonic better" if isotonic else "temperature better",
                            xy=(hi if isotonic else lo, i), xytext=(4 if isotonic else -4, 0),
                            textcoords="offset points", ha="left" if isotonic else "right",
                            fontsize=7.5, color=PALETTE["orange"], va="center",
                            path_effects=_halo(), zorder=4)
        n_straddle = sum(r[res_key] != "True" for r in entries)
        ax.set_title(f"{name}: {n_straddle} of {len(entries)} straddle zero",
                     fontsize=9.5, color=INK, pad=6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(range(len(entries)))
    axes[0].set_yticklabels([f"T={r['timesteps']}  seed {r['seed']}" for r in entries],
                            fontsize=8.5)
    axes[0].invert_yaxis()
    fig.supxlabel("isotonic ECE − temperature ECE   (negative = isotonic better)",
                  fontsize=10, y=-0.02)
    n_both = sum(all(r[e[4]] == "True" for e in estimators) for r in entries)
    both = "No run" if n_both == 0 else f"{n_both} of {len(entries)} runs"
    fig.suptitle(f"{both} resolves under both estimators", fontsize=10.5, color=INK, y=0.99)
    return save(fig, out_dir, "fig4-paired-difference")


def figure_capacity(rows, stats, out_dir):
    """Distinct confidence levels and samples per level -- two panels, never two y-axes."""
    plt = _style()
    grouped = _by_t(rows, "uncalibrated")
    Ts = sorted(grouped)
    levels = [np.mean([float(r["n_distinct_cal"]) for r in grouped[t]]) for t in Ts]
    per_level = [np.mean([float(r["samples_per_level"]) for r in grouped[t]]) for t in Ts]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    fig.subplots_adjust(wspace=0.32)
    axes[0].plot(Ts, levels, "-o", color=PALETTE["blue"], mec=SURFACE, mew=1.6)
    axes[0].set_ylabel("distinct confidence levels")
    axes[1].plot(Ts, per_level, "-o", color=PALETTE["orange"], mec=SURFACE, mew=1.6)
    axes[1].set_ylabel("calibration samples per level")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(Ts)
        ax.set_xlabel("simulation timesteps $T$")
    for ax, values in zip(axes, (levels, per_level)):
        for t, v in zip(Ts, values):
            ax.annotate(f"{v:.0f}", xy=(t, v), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=8, color=INK_SOFT, path_effects=_halo())
    fig.suptitle("The capacity statistic moves 32-fold across the sweep; the advantage does not",
                 fontsize=10.5, color=INK, y=1.02)
    return save(fig, out_dir, "fig5-capacity")


def figure_energy(rows, stats, out_dir):
    """Energy proxy split between the input encoder and the spiking layers."""
    plt = _style()
    Ts = sorted(int(t) for t in stats)
    encoder = np.array([stats[str(t)]["encoder_energy_fraction"] for t in Ts]) * 100
    spiking = 100 - encoder
    per_sample = [stats[str(t)]["energy_per_sample_joules_proxy"] * 1e6 for t in Ts]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    fig.subplots_adjust(wspace=0.30)
    width = 1.1
    axes[0].bar(Ts, encoder, width=width, color=PALETTE["blue"], label="input encoder (MAC)")
    axes[0].bar(Ts, spiking, width=width, bottom=encoder + 0.6, color=PALETTE["orange"],
                label="spiking layers (AC)")
    for t, e in zip(Ts, encoder):
        axes[0].annotate(f"{e:.0f}%", xy=(t, e / 2), ha="center", va="center",
                         fontsize=8.5, color=SURFACE, weight="bold")
    axes[0].set_ylabel("share of energy proxy (%)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8.5, ncol=2)
    axes[0].grid(axis="x", visible=False)

    axes[1].plot(Ts, per_sample, "-o", color=PALETTE["aqua"], mec=SURFACE, mew=1.6)
    axes[1].set_ylabel("energy proxy per sample (µJ)")
    for t, v in zip(Ts, per_sample):
        axes[1].annotate(f"{v:.1f}", xy=(t, v), xytext=(0, 11), textcoords="offset points",
                         ha="center", fontsize=8, color=INK_SOFT, path_effects=_halo())
    for ax in axes:
        ax.set_xticks(Ts)
        ax.set_xlabel("simulation timesteps $T$")
    fig.suptitle("Design-comparison proxy only — not measured power", fontsize=10.5,
                 color=INK, y=1.02)
    return save(fig, out_dir, "fig6-energy")


def figure_risk_coverage(rows, stats, out_dir, cache_path=None):
    """Risk against coverage for all five T, with the quantisation left visible as steps.

    Drawn as a step function on purpose. A spike-count readout has few distinct confidence
    values, samples sharing one cannot be ordered against each other, and so coverage can
    only change at a boundary between levels. Joining the points with a smooth line would
    draw operating points that do not exist.

    Two panels because one cannot do the job: at 98.5% accuracy the risk is essentially
    zero below 90% coverage, so a single full-range panel compresses every interesting
    difference into the last tenth of the axis. The right panel is that tenth.
    """
    plt = _style()
    cache = np.load(cache_path)

    from .selective import risk_coverage

    Ts = sorted({int(k.split("|")[0]) for k in cache.files})
    shades = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"], "#8a8f94"]
    series = []
    for colour, T in zip(shades, Ts):
        key = f"{T}|0|uncalibrated"
        if key + "|conf" not in cache:
            continue
        series.append((T, colour, risk_coverage(cache[key + "|conf"], cache[key + "|correct"])))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    fig.subplots_adjust(wspace=0.26)
    for ax, (lo, hi), title in zip(
        axes, [(0.0, 1.02), (0.90, 1.005)],
        ["full range", "the last tenth, where the differences live"],
    ):
        # Drawn high-T first so the sparse low-T markers land on top and stay visible.
        for T, colour, rc in reversed(series):
            m = rc.coverage >= lo - 1e-9
            ax.step(rc.coverage[m], rc.risk[m], where="post", color=colour, lw=1.8,
                    label=f"T = {T}  ({rc.n_levels} points)" if ax is axes[0] else None)
            ax.plot(rc.coverage[m], rc.risk[m], "o", color=colour,
                    ms=6 if T == 2 else 3.2, mec=SURFACE, mew=1.1,
                    zorder=5 if T == 2 else 3)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("coverage — fraction of samples acted on")
        ax.set_title(title, fontsize=10, color=INK_SOFT, pad=8)
    axes[0].set_ylabel("risk — error rate on the selected samples")
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Name the finding rather than leaving the reader to count markers.
    t2 = next((rc for T, _c, rc in series if T == 2), None)
    if t2 is not None:
        axes[1].annotate(
            f"T = 2 offers {t2.n_levels} operating points\nacross {t2.n_samples:,} samples",
            xy=(0.915, axes[1].get_ylim()[1] * 0.82), fontsize=8.5, color=PALETTE["blue"])

    fig.suptitle("Coverage is not continuous: every marker is an achievable operating point",
                 fontsize=10.5, color=INK, y=1.02)
    return save(fig, out_dir, "fig7-risk-coverage")


def figure_quantisation(rows, stats, out_dir, quantisation_path=None):
    """Does coarsening the membrane confidence recover the ranking the count readout gets?

    Two rows, because the answer and the reason are different questions. The top row is
    the measurement: AUROC against the number of confidence levels the membrane signal is
    reduced to, one panel per ``T``, against two references -- membrane at full precision
    and the count readout it is being compared with.

    The bottom row is the mechanism. Coarsening is a monotone map, so it can never reorder
    a pair, only merge pairs into ties that AUROC scores as 0.5. It can therefore raise
    AUROC only where the structure it destroys was **worse than chance**, and the bottom
    row plots exactly that quantity against the 0.5 line. A curve sitting above 0.5 says
    the fine structure is signal and coarsening costs; a curve on 0.5 says the fine
    structure is uninformative and the level count does not matter.

    The vertical rule marks the count readout's own number of distinct levels: the
    resolution at which the hypothesis predicts membrane should catch up.
    """
    plt = _style()
    qrows = [r for r in load_quantisation(quantisation_path) if r["scheme"] == "equal-mass"]
    Ts = sorted({r["timesteps"] for r in qrows})

    # squeeze=False keeps the 2-D indexing valid when the sweep is run over a single T.
    fig, axes = plt.subplots(2, len(Ts), figsize=(3.05 * len(Ts), 6.4), sharey="row",
                             squeeze=False)
    fig.subplots_adjust(wspace=0.12, hspace=0.34)

    for col, T in enumerate(Ts):
        top, bottom = axes[0][col], axes[1][col]
        cell = [r for r in qrows if r["timesteps"] == T]
        seeds = sorted({r["seed"] for r in cell})
        # Only the resolutions every seed was evaluated at. The sweep additionally probes
        # each seed at its OWN count level count, and those differ between seeds (95, 97
        # and 101 at T = 6), so averaging at such a point would average over one seed.
        counted = {n for n in (r["count_n_distinct"] for r in cell)}
        grid = sorted({r["n_levels_requested"] for r in cell if r["n_levels_requested"]}
                      - counted)

        def stack(key):
            return np.array([[next(r[key] for r in cell
                                   if r["seed"] == s and r["n_levels_requested"] == n)
                              for n in grid] for s in seeds], dtype=float)

        auroc, within = stack("auroc"), stack("auroc_within")
        full = float(np.mean([r["auroc"] for r in cell if r["n_levels_requested"] is None]))
        count_auroc = float(np.mean([r["count_auroc"] for r in cell]))
        n_count = float(np.mean(sorted(counted)))

        for ax in (top, bottom):
            ax.set_xscale("log")
            # A band, not a rule: count's level count is a per-seed quantity.
            ax.axvspan(min(counted), max(counted), color=INK_SOFT, alpha=0.14, lw=0, zorder=1)
            ax.axvline(n_count, color=INK_SOFT, lw=1.0, ls=(0, (4, 3)), zorder=1)
        top.axhline(full, color=PALETTE["violet"], lw=1.4, ls=(0, (1, 2)), zorder=2)
        top.axhline(count_auroc, color=PALETTE["orange"], lw=1.4, ls=(0, (5, 3)), zorder=2)
        top.fill_between(grid, auroc.min(0), auroc.max(0), color=PALETTE["blue"],
                         alpha=0.16, lw=0, zorder=3)
        top.plot(grid, auroc.mean(0), "-o", color=PALETTE["blue"], ms=4.5, mec=SURFACE,
                 mew=1.2, zorder=4)

        bottom.axhline(0.5, color=INK_SOFT, lw=1.2, zorder=2)
        bottom.fill_between(grid, within.min(0), within.max(0), color=PALETTE["aqua"],
                            alpha=0.18, lw=0, zorder=3)
        bottom.plot(grid, within.mean(0), "-o", color=PALETTE["aqua"], ms=4.5, mec=SURFACE,
                    mew=1.2, zorder=4)

        top.set_title(f"$T$ = {T}", fontsize=10.5, color=INK, pad=16)
        for ax in (top, bottom):
            ax.set_xlabel("confidence levels $N$", fontsize=9)
        # The rule needs naming once per panel; it moves with T, so it cannot go in a legend.
        top.annotate(f"count: {n_count:.0f}", xy=(n_count, 1.005),
                     xycoords=("data", "axes fraction"), ha="center", fontsize=8,
                     color=INK_SOFT)

    axes[0][0].set_ylabel("AUROC of the coarsened signal")
    axes[1][0].set_ylabel("AUROC within a level")
    for col in range(1, len(Ts)):
        for row in (0, 1):
            axes[row][col].tick_params(labelleft=False)

    # Direct labels on the two references, placed once, on the panel with the most room.
    ref = axes[0][0]
    xr = ref.get_xlim()[1]
    for value, colour, text in (
        (float(np.mean([r["auroc"] for r in qrows
                        if r["timesteps"] == Ts[0] and r["n_levels_requested"] is None])),
         PALETTE["violet"], "membrane, full precision"),
        (float(np.mean([r["count_auroc"] for r in qrows if r["timesteps"] == Ts[0]])),
         PALETTE["orange"], "count readout"),
    ):
        ref.annotate(text, xy=(xr, value), xytext=(-4, 4), textcoords="offset points",
                     ha="right", fontsize=8, color=colour, path_effects=_halo(), zorder=6)
    axes[0][-1].annotate("membrane, coarsened", xy=(0.97, 0.06), xycoords="axes fraction",
                         ha="right", fontsize=8, color=PALETTE["blue"], path_effects=_halo())
    # Left edge: the within-level curve is at its highest there, so the label sits clear
    # of it at every T. On the right it lands on the curve as it approaches chance.
    axes[1][0].annotate("chance", xy=(0.03, 0.5), xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points", ha="left", fontsize=8,
                        color=INK_SOFT, path_effects=_halo())

    fig.suptitle("Coarsening the membrane signal does not recover count's ranking, and the "
                 "bottom row says why", fontsize=10.5, color=INK, y=0.98)
    return save(fig, out_dir, "fig8-quantisation")


def figure_fusion(rows, stats, out_dir, fusion_path=None):
    """What per-modality calibration buys the product rule, against T, for two readouts.

    ``delta`` is the accuracy of the product rule after per-modality temperature scaling
    minus its accuracy on the raw outputs. Each seed is drawn with its 95% bootstrap
    interval, and the seed means are joined. The open markers are not a fit: they are the
    value the proposition gives for the part of ``delta`` that comes from inputs on which
    both readouts saturate, ``0.5 P(S) (a_A,S - a_B,S)``, computed from the same runs.
    """
    plt = _style()
    with open(fusion_path) as fh:
        fusion = json.load(fh)
    series = (("count", PALETTE["blue"], "count readout", -0.18),
              ("accumulator", PALETTE["orange"], "accumulator", 0.18))
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.axhline(0.0, color=INK_SOFT, lw=1.0, zorder=1)
    for readout, colour, label, shift in series:
        by_t = defaultdict(list)
        for r in fusion["rows"]:
            if r["readout"] == readout:
                by_t[int(r["timesteps"])].append(r)
        ts = sorted(by_t)
        means = [float(np.mean([r["delta"] for r in by_t[t]])) for t in ts]
        ax.plot([t + shift for t in ts], means, "-", color=colour, lw=1.6, zorder=2, label=label)
        for t in ts:
            cell = sorted(by_t[t], key=lambda r: r["seed"])
            for k, r in enumerate(cell):
                x = t + shift + (k - (len(cell) - 1) / 2) * 0.13
                ax.plot([x, x], r["delta_ci"], color=colour, lw=1.6, solid_capstyle="round", zorder=3)
                ax.plot([x], [r["delta"]], "o", color=colour, mec=SURFACE, mew=1.2, ms=5, zorder=4)
        if readout == "count":
            predicted = [float(np.mean([r["predicted_on_s"] for r in by_t[t]])) for t in ts]
            ax.plot([t + shift for t in ts], predicted, "D", mfc=SURFACE, mec=INK, mew=1.1, ms=5.5,
                    zorder=5, ls="none", label="proposition, on-$S$ part (count)")
    ax.set_xticks(sorted({int(r["timesteps"]) for r in fusion["rows"]}))
    ax.set_xlabel("timesteps $T$")
    ax.set_ylabel("accuracy gained by calibrating\neach input before fusion")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return save(fig, out_dir, "fig9-fusion")


ALL_FIGURES = (
    figure_ceiling,
    figure_loss_floor,
    figure_ece_by_arm,
    figure_paired_difference,
    figure_capacity,
    figure_energy,
    figure_risk_coverage,
    figure_quantisation,
    figure_fusion,
)
