"""Training a scaled-count model until its logit scale settles.

Added after review, not pre-registered. The thresholds used here were fixed in NOTES.md,
"Post-review analyses, thresholds fixed 2026-11-12 before computation", section A7, and
committed before any of this code ran.

The original learned-scale runs stopped on validation accuracy, and in all nine the scale
was still rising when they stopped. This module trains the same models with every setting
unchanged except the stopping rule: training continues until ``theta = ln s`` has moved by
less than ``tol`` over ``window`` consecutive epochs, or until a cap. Along the way it keeps
the checkpoint that the original rule (best validation accuracy, patience 5, at most 50
epochs) would have selected in the same run, so the two can be compared on one trajectory.

The training step is :func:`spikecalib.train.train`'s, line for line: Adam over all
parameters, cross-entropy on the readout's logits, neuron state reset before every batch,
validation accuracy from :func:`spikecalib.train._evaluate` with its fixed tie-break seed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .calibration import ece_equal_mass, ece_equal_width, model_outputs
from .train import _evaluate, predict_outputs, resolve_device, set_seed, to_time_first

__all__ = [
    "theta_converged",
    "original_rule_stop_epoch",
    "ConvergenceConfig",
    "train_to_convergence",
]


def theta_converged(thetas: list[float], window: int = 5, tol: float = 0.01) -> Optional[int]:
    """First epoch at which ``theta`` has moved by less than ``tol`` over ``window`` epochs.

    ``thetas[e]`` is ``theta`` read after epoch ``e`` (0-indexed). The run has converged at
    the first ``e >= window`` for which the ``window + 1`` values ``thetas[e - window]`` to
    ``thetas[e]`` span less than ``tol`` (largest minus smallest). Returns that ``e``, or
    ``None`` if no epoch qualifies.

    Using the span of all ``window + 1`` values, and not only the difference between the
    two ends, means a ``theta`` that rises and falls back within the window does not count
    as settled.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1; got {window}")
    if tol <= 0:
        raise ValueError(f"tol must be positive; got {tol}")
    for e in range(window, len(thetas)):
        span = thetas[e - window:e + 1]
        if max(span) - min(span) < tol:
            return e
    return None


def original_rule_stop_epoch(val_accuracy: list[float], patience: int = 5,
                             cap: int = 50, min_delta: float = 0.0) -> tuple[int, int]:
    """Where the original early-stopping rule would have stopped, and what it kept.

    Replays :func:`spikecalib.train.train`'s rule on a validation-accuracy trajectory:
    an epoch improves when its accuracy exceeds the best so far by more than ``min_delta``;
    training stops once ``patience`` epochs in a row fail to improve, or after ``cap``
    epochs. Returns ``(stop_epoch, best_epoch)``, both 0-indexed: the last epoch the rule
    would have run, and the epoch whose weights it would have returned.

    Raises if the trajectory ends before the rule would have stopped, since the answer
    would then depend on epochs that were never run.
    """
    best, best_epoch, since = -1.0, -1, 0
    for epoch, acc in enumerate(val_accuracy[:cap]):
        if acc > best + min_delta:
            best, best_epoch, since = acc, epoch, 0
        else:
            since += 1
            if since >= patience:
                return epoch, best_epoch
    if len(val_accuracy) >= cap:
        return cap - 1, best_epoch
    raise ValueError(
        f"trajectory of {len(val_accuracy)} epochs ends before the original rule stops "
        f"(patience {patience}, cap {cap})"
    )


@dataclass(frozen=True)
class ConvergenceConfig:
    """The A7 settings, as fixed in NOTES.md before any run."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    window: int = 5
    tol: float = 0.01
    max_epochs: int = 150
    original_patience: int = 5
    original_cap: int = 50
    eval_seed: int = 0
    n_bins: int = 15
    device: Optional[str] = None


@dataclass
class ConvergenceRun:
    """What one run produced: its per-epoch history and two checkpoints."""

    seed: int
    epochs_run: int
    converged_epoch: Optional[int]
    stopped_by: str
    original_stop_epoch: int
    original_best_epoch: int
    seconds: float
    device: str
    history: list[dict] = field(default_factory=list)
    original_state: dict = field(default_factory=dict, repr=False)
    end_state: dict = field(default_factory=dict, repr=False)


def _val_ece(model, val_loader, device, seed: int, n_bins: int) -> tuple[float, float]:
    """Validation ECE under both estimators, from the model's own logits."""
    scores, labels, _ = predict_outputs(model, val_loader, device, count_synops=False)
    out = model_outputs(scores, labels, np.random.default_rng(seed))
    correct = (out.predictions == out.labels)
    return (ece_equal_width(out.conf, correct, n_bins), ece_equal_mass(out.conf, correct, n_bins))


def train_to_convergence(model, loaders: dict, cfg: ConvergenceConfig, *, seed: int,
                         log=print) -> ConvergenceRun:
    """Train a scaled-count model until ``theta`` settles or ``cfg.max_epochs`` is reached.

    The per-epoch step is that of :func:`spikecalib.train.train`. After each epoch the
    function records ``theta``, ``s``, the training loss, validation accuracy (the original
    evaluation) and validation ECE under both estimators. It keeps the best-validation
    weights up to the epoch where the original rule would have stopped, and the weights at
    the end.
    """
    if not hasattr(model, "log_scale"):
        raise ValueError("train_to_convergence needs a scaled-count model")
    device = resolve_device(cfg.device)
    model = model.to(device)
    set_seed(seed)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    thetas: list[float] = []
    best_val, since, original_done = -1.0, 0, False
    original_best_epoch, original_stop_epoch = -1, -1
    original_state: dict = {}
    end_state: dict = {}
    converged: Optional[int] = None
    started = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss, seen = 0.0, 0
        t0 = time.time()
        for frames, labels in loaders["train"]:
            functional.reset_net(model)
            x = to_time_first(frames.float()).to(device)
            y = labels.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item()) * int(y.numel())
            seen += int(y.numel())

        val_accuracy, val_tie_rate = _evaluate(model, loaders["val"], device, cfg.eval_seed)
        theta = float(model.log_scale.detach().cpu().item())
        # Validation accuracy rounds the scaled logits before the tie-break. That preserves
        # every order and every tie exactly when s >= 1, which is asserted rather than assumed.
        scale = math.exp(theta)
        val_ece_w, val_ece_m = _val_ece(model, loaders["val"], device, seed, cfg.n_bins)
        record = {
            "epoch": epoch, "train_loss": epoch_loss / seen if seen else float("nan"),
            "val_accuracy": val_accuracy, "val_tie_rate": val_tie_rate,
            "theta": theta, "logit_scale": scale,
            "val_ece_equal_width": val_ece_w, "val_ece_equal_mass": val_ece_m,
            "rounding_exact": bool(scale >= 1.0),
            "seconds": time.time() - t0,
        }
        history.append(record)
        thetas.append(theta)
        log(f"    epoch {epoch:>3}  loss {record['train_loss']:.4f}  val acc {val_accuracy:.4f}  "
            f"theta {theta:.4f}  s {scale:.4f}  val ECE w {val_ece_w:.4f} m {val_ece_m:.4f}  "
            f"{record['seconds']:.1f} s")

        if not original_done:
            if val_accuracy > best_val:
                best_val, original_best_epoch, since = val_accuracy, epoch, 0
                original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            if since >= cfg.original_patience or epoch == cfg.original_cap - 1:
                original_done, original_stop_epoch = True, epoch

        if converged is None:
            converged = theta_converged(thetas, cfg.window, cfg.tol)
            if converged is not None:
                # The end point is the weights at convergence. If the original rule has not
                # stopped yet, training continues only as far as that rule needs, so that its
                # checkpoint is defined; those extra epochs do not move the end point.
                end_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if converged is not None and original_done:
            break

    if converged is None:
        end_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stopped_by = "convergence" if converged is not None else "cap"
    if not original_done:
        raise RuntimeError("the run ended before the original rule stopped")
    return ConvergenceRun(
        seed=seed, epochs_run=len(history), converged_epoch=converged, stopped_by=stopped_by,
        original_stop_epoch=original_stop_epoch, original_best_epoch=original_best_epoch,
        seconds=time.time() - started, device=str(device), history=history,
        original_state=original_state, end_state=end_state,
    )
