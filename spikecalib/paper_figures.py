"""The paper's figures, drawn at their printed size.

Every figure here is built at the width it is printed at (the text width of the paper,
6.7 in, or a stated fraction of it) and included without scaling, so the type sizes set here
are the printed sizes: 8 pt for labels, 7.5 pt for ticks and legends, and nothing below 7 pt.
Colours come from one colour-blind-safe palette (Okabe and Ito), used the same way in every
figure. Each figure is written as a vector PDF with embedded TrueType fonts for the paper and
as a PNG for the repository's front page. Every value drawn is read from ``results/``.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

__all__ = ["OKABE_ITO", "TEXT_WIDTH", "PAPER_FIGURES", "draw_all"]

# Okabe, M. and Ito, K. (2008), "Color Universal Design": the eight-colour set.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
GREY = "#6E6E6E"
LIGHT = "#D9D9D9"
TEXT_WIDTH = 6.7  # inches: US letter, 0.9 in margins
MIN_PT = 7.0

DATASET_COLOUR = {"nmnist": OKABE_ITO["blue"], "ncaltech101": OKABE_ITO["vermillion"]}
DATASET_LABEL = {"nmnist": "N-MNIST", "ncaltech101": "N-Caltech101"}
ARM_COLOUR = {"uncalibrated": OKABE_ITO["blue"], "temperature": OKABE_ITO["vermillion"],
              "isotonic-top-label": OKABE_ITO["green"]}
ARM_LABEL = {"uncalibrated": "uncalibrated", "temperature": "temperature scaling",
             "isotonic-top-label": "isotonic regression"}
READOUT_COLOUR = {"count": OKABE_ITO["blue"], "accumulator": OKABE_ITO["orange"],
                  "membrane": OKABE_ITO["green"], "weighted-count": OKABE_ITO["sky"],
                  "scaled-count": OKABE_ITO["purple"]}
READOUT_STYLE = {"count": "-", "weighted-count": (0, (4, 1.5)), "membrane": (0, (1.2, 1.2)),
                 "accumulator": (0, (6, 1.5, 1.2, 1.5)), "scaled-count": (0, (2.5, 1))}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Start from matplotlib's defaults so that no style set earlier in the process (for
    # example by spikecalib.figures, which scripts/make_figures.py runs first) leaks in.
    plt.rcdefaults()
    plt.rcParams.update({
        "svg.hashsalt": "spikecalib",
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "axes.grid": False, "figure.dpi": 100,
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


def _check_type(fig) -> None:
    """Refuse to save a figure with any text below the minimum printed size."""
    for text in fig.findobj(match=lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text")):
        if text.get_text() and text.get_visible() and text.get_fontsize() < MIN_PT - 1e-9:
            raise ValueError(f"text {text.get_text()!r} is {text.get_fontsize()} pt, below {MIN_PT} pt")


def _save(fig, name: str, pdf_dir: str, png_dir: str) -> list[str]:
    _check_type(fig)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    paths = [os.path.join(pdf_dir, name + ".pdf"), os.path.join(png_dir, name + ".png")]
    fig.savefig(paths[0], format="pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(paths[1], format="png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def _load(results: str, name: str):
    with open(os.path.join(results, name)) as fh:
        return json.load(fh)


def _cap(T, C):
    return 1.0 / (1.0 + (C - 1) * math.exp(-T))


# ---------------------------------------------------------------------------------------
# Figure 1: the cap against accuracy, and the few values a count readout emits
# ---------------------------------------------------------------------------------------
def figure_overview(results, pdf_dir, png_dir):
    plt = _plt()
    floor = _load(results, "ece_floor.json")
    cond = _load(results, "conditional_ceiling.json")["rows"]
    rel = _load(results, "reliability_diagrams.json")["panels"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.55), gridspec_kw={"width_ratios": [1.15, 1]},
                               layout="constrained")
    grid = np.linspace(1, 10.4, 300)
    dodge = {"nmnist": -0.1, "ncaltech101": 0.1}   # markers offset in T so that coincident ones stay visible
    for ds, C in (("nmnist", 10), ("ncaltech101", 101)):
        col = DATASET_COLOUR[ds]
        dx = dodge[ds]
        a.plot(grid, [_cap(t, C) for t in grid], color=col, lw=1.3)
        cells = [c for c in floor["cells"] if c["dataset"] == ds and
                 c["channels"] == ("32-64" if ds == "nmnist" else "32-64-128-128")]
        Ts = [c["timesteps"] for c in cells]
        acc = [c["accuracy"]["mean"] for c in cells]
        a.plot([t + dx for t in Ts], acc, "s", color=col, ms=4.2, mfc=col, mec="white", mew=0.6, zorder=4)
        mx = [np.mean([r["observed_max_confidence"] for r in cond
                       if r["dataset"] == ds and r["timesteps"] == t]) for t in Ts]
        a.plot([t + dx for t in Ts], mx, "o", color=col, ms=5.2, mfc="white", mew=1.0, zorder=3)
        for t, ac in zip(Ts, acc):
            cap = _cap(t, C)
            if ac > cap:
                a.annotate("", xy=(t + dx, cap), xytext=(t + dx, ac),
                           arrowprops=dict(arrowstyle="<->", color=col, lw=0.8, shrinkA=2, shrinkB=2))
    t2 = [c for c in floor["cells"] if c["dataset"] == "nmnist" and c["timesteps"] == 2 and c["channels"] == "32-64"][0]
    a.annotate("floor\n$a-p^{*}$\non ECE", xy=(1.85, (t2["accuracy"]["mean"] + t2["cap"]) / 2),
               xytext=(0.12, 0.93), fontsize=7.5, color=OKABE_ITO["blue"], va="top", ha="left",
               arrowprops=dict(arrowstyle="-", color=OKABE_ITO["blue"], lw=0.6))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DATASET_COLOUR["nmnist"], lw=1.3, label="N-MNIST, $C=10$"),
               Line2D([], [], color=DATASET_COLOUR["ncaltech101"], lw=1.3, label="N-Caltech101, $C=101$"),
               Line2D([], [], color=GREY, lw=1.3, label=r"cap $p^{*}=e^{T}/(e^{T}+C-1)$"),
               Line2D([], [], color=GREY, marker="s", ls="none", ms=4.2, mec="white", label="test accuracy"),
               Line2D([], [], color=GREY, marker="o", ls="none", ms=5.2, mfc="white", label="largest confidence")]
    a.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    a.set_xlim(0, 10.6)
    a.set_ylim(0, 1.04)
    a.set_xticks([2, 4, 6, 8, 10])
    a.set_xlabel("timesteps $T$")
    a.set_ylabel("confidence or accuracy")
    a.set_title("(a) The cap against accuracy", loc="left", fontsize=8)

    p = next(x for x in rel if x["dataset"] == "nmnist" and x["timesteps"] == 2 and x["arm"] == "uncalibrated")
    vals = np.array([v["confidence"] for v in p["per_value"]])
    ns = np.array([v["n"] for v in p["per_value"]])
    acc = p["accuracy"]
    b.vlines(vals, 0.8, ns, color=OKABE_ITO["blue"], lw=1.4)
    b.plot(vals, ns, "o", color=OKABE_ITO["blue"], ms=3.5)
    b.set_yscale("log")
    b.set_ylim(0.8, 3e4)
    cap2 = _cap(2, 10)
    b.axvline(cap2, color=GREY, lw=0.8, ls=(0, (3, 2)))
    b.axvline(acc, color=OKABE_ITO["black"], lw=0.8)
    b.text(cap2 - 0.012, 2.2e4, f"cap {cap2:.3f}", ha="right", va="top", fontsize=7.5, color=GREY)
    b.text(acc - 0.012, 2.2e4, f"accuracy {acc:.3f}", ha="right", va="top", fontsize=7.5)
    big = int(np.argmax(ns))
    b.annotate(f"{ns[big]:,} of {int(ns.sum()):,}\ntest samples", xy=(vals[big], ns[big]),
               xytext=(0.60, 1.6e3), fontsize=7.5, color=OKABE_ITO["blue"],
               arrowprops=dict(arrowstyle="-", color=OKABE_ITO["blue"], lw=0.6))
    b.set_xlim(0.05, 1.02)
    b.set_xlabel("confidence (largest softmax probability)")
    b.set_ylabel("test samples (log scale)")
    b.set_title(f"(b) The {len(vals)} confidences emitted, N-MNIST, $T=2$", loc="left", fontsize=8)
    return _save(fig, "fig1-overview", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Figure 2: the network
# ---------------------------------------------------------------------------------------
def figure_architecture(results, pdf_dir, png_dir):
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    arch = _load(results, "arch_matched.json")["rows"]
    params = {(r["dataset"], r["n_blocks"]): r["n_parameters"] for r in arch if r["readout"] == "count"}
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec=OKABE_ITO["black"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=7.5, weight="bold")
        ax.text(x + w / 2, y + h - 5.0, sub, ha="center", va="top", fontsize=7, linespacing=1.25)

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=OKABE_ITO["black"], mutation_scale=7))

    y0, h = 14, 15
    box(0.3, y0, 12.4, h, "event frames", "$T$ frames\n2 polarities\n34×34 or 45×60\nevent counts", "#F2F2F2")
    arrow(13.0, 14.0, y0 + h / 2)
    box(14.3, y0, 16.7, h, "block × $K$", "conv 3×3, pad 1\nbatch norm\nLIF\nmax-pool 2×2", "#DCEBF5")
    ax.text(22.5, y0 - 1.0, "$K=2$: 32, 64 channels\n$K=4$: 32, 64, 128, 128", ha="center", va="top", fontsize=7)
    arrow(31.3, 32.7, y0 + h / 2)
    box(33.0, y0, 12.6, h, "hidden", "flatten\nlinear → 256\nbatch norm\nLIF", "#DCEBF5")
    arrow(45.9, 47.3, y0 + h / 2)
    box(47.6, y0, 12.6, h, "output", "linear → $C$\nbatch norm\n$z_t\\in\\mathbb{R}^{C}$", "#DCEBF5")
    arrow(60.5, 62.0, y0 + h / 2)
    ax.text(81.0, 35.6, "readout: softmax over $C$ logits", ha="center", va="top", fontsize=7.5, weight="bold")
    readouts = [("count", r"LIF; $\sum_t s_t$, integers in $[0,T]$"),
                ("weighted-count", r"LIF; $\sum_t w_t s_t$, $\sum_t w_t=T$"),
                ("scaled-count", r"LIF; $e^{\theta}\sum_t s_t$"),
                ("accumulator", r"no threshold; $\sum_t z_t$"),
                ("membrane", r"no threshold; $v_T$, leak $\tau$")]
    for i, (name, eq) in enumerate(readouts):
        yy = 31.6 - i * 5.3
        col = READOUT_COLOUR[name]
        ax.add_patch(FancyBboxPatch((62.6, yy - 4.1), 36.8, 4.3, boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc="white", ec=col, lw=0.9))
        ax.add_patch(FancyBboxPatch((62.6, yy - 4.1), 1.2, 4.3, boxstyle="square,pad=0", fc=col, ec=col, lw=0))
        ax.text(64.6, yy - 1.95, name.replace("-", " "), fontsize=7.5, color=OKABE_ITO["black"], weight="bold", va="center")
        ax.text(99.0, yy - 1.95, eq, fontsize=7, va="center", ha="right")
    ax.text(0.4, 3.4, "Parameters: N-MNIST {:,} ($K=2$), {:,} ($K=4$); N-Caltech101 {:,} ($K=2$), {:,} ($K=4$)."
            .format(params[("nmnist", 2)], params[("nmnist", 4)], params[("ncaltech101", 2)],
                    params[("ncaltech101", 4)]), fontsize=7, va="bottom")
    ax.text(0.4, 0.4, "LIF: $\\tau=2$, threshold 1, hard reset to 0, ATan surrogate ($\\alpha=2$). "
            "Every layer runs for $T$ steps.", fontsize=7, va="bottom")
    return _save(fig, "fig2-architecture", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------------------
def figure_reliability(results, pdf_dir, png_dir, min_n_for_interval: int = 10, max_values: int = 50):
    plt = _plt()
    rel = _load(results, "reliability_diagrams.json")["panels"]
    rows = [("nmnist", 2), ("nmnist", 10), ("ncaltech101", 2), ("ncaltech101", 10)]
    arms = ["uncalibrated", "temperature", "isotonic-top-label"]
    fig, axes = plt.subplots(4, 3, figsize=(TEXT_WIDTH, 8.2), layout="constrained", sharex=True, sharey=True)
    for i, (ds, T) in enumerate(rows):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            p = next(x for x in rel if x["dataset"] == ds and x["timesteps"] == T and x["arm"] == arm)
            many = len(p["per_value"]) > max_values
            pv = p["equal_mass"] if many else p["per_value"]
            n_total = sum(v["n"] for v in pv)
            col = ARM_COLOUR[arm]
            ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)))
            for v in pv:
                if v["n"] >= min_n_for_interval:
                    ax.plot([v["confidence"]] * 2, [v["wilson_low"], v["wilson_high"]], color=col, lw=0.7, alpha=0.8)
            x = np.array([v["confidence"] for v in pv])
            y = np.array([v["accuracy"] for v in pv])
            s = 400.0 * np.array([v["n"] for v in pv]) / n_total
            ax.scatter(x, y, s=np.maximum(s, 1.5), color=col, alpha=0.75, lw=0, clip_on=False)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if i == 0:
                ax.set_title(ARM_LABEL[arm], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}, $T={T}$\naccuracy")
            if i == 3:
                ax.set_xlabel("confidence")
            note = (f"{len(p['per_value']):,} values;\n{len(pv)} equal-mass bins" if many
                    else f"{len(pv):,} values")
            ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", fontsize=7, color=GREY)
    return _save(fig, "fig-reliability", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Paired differences between the calibrators (reworked Figure 4)
# ---------------------------------------------------------------------------------------
def figure_paired_difference(results, pdf_dir, png_dir, margin: float = 0.01):
    plt = _plt()
    tost = _load(results, "calibration_metrics.json")["tost"]
    groups = [("nmnist", "count", "N-MNIST, count readout"), ("ncaltech101", "count", "N-Caltech101, count readout")]
    ests = [("equal_width", "equal-width bins"), ("equal_mass", "equal-mass bins")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.4), layout="constrained",
                             gridspec_kw={"height_ratios": [15, 9]})
    for i, (ds, readout, gtitle) in enumerate(groups):
        for j, (est, etitle) in enumerate(ests):
            ax = axes[i][j]
            es = sorted((e for e in tost if e["dataset"] == ds and e["readout"] == readout and e["estimator"] == est),
                        key=lambda e: (e["timesteps"], e["seed"]))
            ax.axvspan(-margin, margin, color=LIGHT, lw=0, zorder=0)
            for edge in (-margin, margin):
                ax.axvline(edge, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.axvline(0, color=OKABE_ITO["black"], lw=0.8, zorder=1)
            for k, e in enumerate(es):
                lo95, hi95 = e["ci95"]
                lo90, hi90 = e["ci90"]
                col = OKABE_ITO["vermillion"] if e["excludes_zero_95"] else OKABE_ITO["blue"]
                ax.plot([lo95, hi95], [k, k], color=col, lw=0.9, zorder=2)
                ax.plot([lo90, hi90], [k, k], color=col, lw=2.6, solid_capstyle="butt", zorder=3)
                ax.plot([e["difference"]], [k], "o", color="white", mec=col, mew=1.0, ms=3.6, zorder=4)
            ax.set_yticks(range(len(es)))
            ax.set_yticklabels([f"$T={e['timesteps']}$, seed {e['seed']}" for e in es] if j == 0 else [])
            ax.set_ylim(len(es) - 0.4, -0.6)
            ax.set_title(f"{gtitle}: {etitle}", fontsize=8, loc="left")
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(5))
            lo_, hi_ = ax.get_xlim()   # keep the margin's edges inside the panel
            ax.set_xlim(min(lo_, -1.3 * margin), max(hi_, 1.3 * margin))
            if i == 1:
                ax.set_xlabel("isotonic minus temperature ECE")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=OKABE_ITO["blue"], lw=2.6, label="90% interval (equivalence test)"),
               Line2D([], [], color=OKABE_ITO["blue"], lw=0.9, label="95% interval (registered rule)"),
               Line2D([], [], color=OKABE_ITO["vermillion"], lw=2.6, label="95% interval excludes zero"),
               Patch(color=LIGHT, label=f"margin $\\pm{margin}$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    return _save(fig, "fig-paired-difference", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Fusion pilot (reworked Figure 9)
# ---------------------------------------------------------------------------------------
def figure_fusion(results, pdf_dir, png_dir):
    plt = _plt()
    fusion = _load(results, "fusion_experiment.json")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35), layout="constrained", sharey=True)
    for ax, readout, title in ((axes[0], "count", "count readout"), (axes[1], "accumulator", "accumulator readout")):
        by_t = defaultdict(list)
        for r in fusion["rows"]:
            if r["readout"] == readout:
                by_t[int(r["timesteps"])].append(r)
        ts = sorted(by_t)
        col = READOUT_COLOUR[readout]
        ax.axhline(0, color=OKABE_ITO["black"], lw=0.8)
        means = [float(np.mean([r["delta"] for r in by_t[t]])) for t in ts]
        ax.plot(ts, means, "-", color=col, lw=0.9, alpha=0.7)
        for t in ts:
            cell = sorted(by_t[t], key=lambda r: r["seed"])
            for k, r in enumerate(cell):
                x = t + (k - 1) * 0.28
                ax.plot([x, x], r["delta_ci"], color=col, lw=1.2)
                ax.plot([x], [r["delta"]], "o", color=col, ms=3.4)
        if readout == "count":
            pred = [float(np.mean([r["predicted_on_s"] for r in by_t[t]])) for t in ts]
            ax.plot(ts, pred, "D", mfc="white", mec=OKABE_ITO["black"], mew=0.8, ms=4, ls="none",
                    label=r"predicted on-$S$ part, $\frac{1}{2}P(S)(a_{A,S}-a_{B,S})$")
            ax.legend(loc="upper right")
        ax.set_xticks(ts)
        ax.set_xlabel("timesteps $T$")
        ax.set_title(title, fontsize=8, loc="left")
    axes[0].set_ylabel("$\\delta$: fused accuracy gained by\ncalibrating each input first")
    return _save(fig, "fig-fusion", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Spike rasters
# ---------------------------------------------------------------------------------------
def figure_rasters(results, pdf_dir, png_dir):
    plt = _plt()
    act = _load(results, "spiking_activity.json")
    rasters = sorted(act["rasters"], key=lambda r: (r["dataset"] != "nmnist", r["timesteps"]))
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 6.4), layout="constrained")
    for ax, r in zip(axes.flat, rasters):
        T = r["timesteps"]
        names = list(r["layers"])
        y = 0
        yt, yl = [], []
        for name in names:
            L = r["layers"][name]
            sp = np.array(L["spikes"])  # [T, n]
            n = sp.shape[1]
            col = OKABE_ITO["vermillion"] if name == "output" else OKABE_ITO["blue"]
            ts_, ns_ = np.nonzero(sp)
            ax.scatter(ts_ + 1, y + ns_, marker="|", s=9 if n > 40 else 16, color=col, lw=0.8)
            if name == "output":
                lab = r["label"]
                ax.axhspan(y + lab - 0.5, y + lab + 0.5, color=OKABE_ITO["yellow"], alpha=0.5, lw=0, zorder=0)
            yt.append(y + n / 2)
            yl.append(f"{name}, {n} of {L['n_neurons']:,}" if name != "output" else f"output, all {n}")
            y += n + 6
            ax.axhline(y - 3, color=LIGHT, lw=0.5)
        ax.set_ylim(y - 3, -2)
        ax.set_xlim(0.4, T + 0.6)
        ax.set_xticks(range(1, T + 1) if T <= 4 else [1, 5, 10])
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("timestep")
        ax.set_title(f"{DATASET_LABEL[r['dataset']]}, $T={T}$\nclass {r['label']}; top count "
                     f"{max(r['counts']):.0f}", fontsize=8, loc="left")
    return _save(fig, "fig-rasters", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# The learned scale against epoch
# ---------------------------------------------------------------------------------------
def figure_scale(results, pdf_dir, png_dir):
    plt = _plt()
    sc = _load(results, "scaled_count_experiment.json")["rows"]
    conv_path = os.path.join(results, "scale_convergence.json")
    conv = _load(results, "scale_convergence.json")["rows"] if os.path.exists(conv_path) else []
    cells = [("nmnist", 2), ("ncaltech101", 2), ("ncaltech101", 6)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), layout="constrained")
    seed_style = {0: "-", 1: (0, (4, 1.5)), 2: (0, (1.2, 1.2))}
    for ax, (ds, T) in zip(axes, cells):
        smins = []
        for r in sorted((r for r in sc if r["dataset"] == ds and r["timesteps"] == T), key=lambda r: r["seed"]):
            s = r["logit_scale_by_epoch"]
            ax.plot(range(1, len(s) + 1), s, color=OKABE_ITO["black"], lw=1.3, ls=seed_style[r["seed"]], zorder=4)
            ax.plot([len(s)], [s[-1]], "x", color=OKABE_ITO["black"], ms=5, mew=1.2, zorder=5)
            smins.append(r["s_min"])
        for r in sorted((r for r in conv if r["dataset"] == ds and r["timesteps"] == T), key=lambda r: r["seed"]):
            s = [h["logit_scale"] for h in r["history"]]
            ax.plot(range(1, len(s) + 1), s, color=OKABE_ITO["vermillion"], lw=0.9, alpha=0.85, ls=seed_style[r["seed"]], zorder=2)
            ax.plot([r["original_best_epoch"] + 1], [s[r["original_best_epoch"]]], "o", mfc="white",
                    mec=OKABE_ITO["vermillion"], ms=3.5)
            smins.append(r["end_point"]["s_min"])
        ax.axhspan(min(smins), max(smins), color=OKABE_ITO["sky"], alpha=0.35, lw=0)
        ax.set_title(f"{DATASET_LABEL[ds]}, $T={T}$", fontsize=8, loc="left")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("learned scale $s=e^{\\theta}$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=OKABE_ITO["black"], lw=1.3, label="original runs (stopped on validation accuracy)"),
               Line2D([], [], color=OKABE_ITO["vermillion"], lw=1.0, label="retrained until $\\theta$ settles"),
               Patch(color=OKABE_ITO["sky"], alpha=0.35, label="$s_{\\min}$ range")]
    handles += [Line2D([], [], color=GREY, lw=1.0, ls=seed_style[sd], label=f"seed {sd}") for sd in (0, 1, 2)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return _save(fig, "fig-scale", pdf_dir, png_dir)


# ---------------------------------------------------------------------------------------
# Validation accuracy per readout, from the run records
# ---------------------------------------------------------------------------------------
def figure_validation_curves(results, pdf_dir, png_dir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), layout="constrained")
    specs = [("nmnist", 2, ["count", "weighted-count", "membrane", "accumulator", "scaled-count"]),
             ("ncaltech101", 2, ["count", "accumulator", "scaled-count"])]
    for ax, (ds, T, readouts) in zip(axes, specs):
        for readout in readouts:
            suffix = "" if readout == "count" else f"_{readout}"
            for seed in (0, 1, 2):
                path = os.path.join(results, f"{ds}_T{T}_seed{seed}{suffix}_record.json")
                with open(path) as fh:
                    rec = json.load(fh)
                acc = [h["val_accuracy"] for h in rec["history"]]
                ax.plot(range(1, len(acc) + 1), acc, color=READOUT_COLOUR[readout], lw=0.9, alpha=0.85,
                        ls=READOUT_STYLE[readout], label=readout.replace("-", " ") if seed == 0 else None)
                ax.plot([rec["best_epoch"] + 1], [acc[rec["best_epoch"]]], "o", color=READOUT_COLOUR[readout], ms=2.8)
        ax.set_xlabel("epoch")
        ax.set_title(f"{DATASET_LABEL[ds]}, $T={T}$", fontsize=8, loc="left")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("validation accuracy")
    return _save(fig, "fig-validation-curves", pdf_dir, png_dir)


PAPER_FIGURES = (figure_overview, figure_architecture, figure_reliability, figure_paired_difference,
                 figure_fusion, figure_rasters, figure_scale, figure_validation_curves)


def draw_all(results: str, pdf_dir: str, png_dir: str, only=None) -> list[str]:
    out = []
    for fn in PAPER_FIGURES:
        if only and fn.__name__ not in only:
            continue
        out += fn(results, pdf_dir, png_dir)
    return out
