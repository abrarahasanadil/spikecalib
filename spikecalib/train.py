"""Training loop for the directly-trained spiking networks.

Owns correctness rule 4: **every experiment is seeded and every seed is recorded
alongside its result.** :func:`set_seed` seeds Python, NumPy and torch; the seed goes into
the :class:`RunRecord` that travels with the trained weights.

The caveat is stated rather than left for a reader to discover: **MPS floating-point
operations are not bitwise deterministic**, so seeding *bounds* run-to-run variation
rather than eliminating it. Two runs at the same seed on Apple silicon will differ
slightly in trained weights. That is why the study reports over 3 seeds, and why splits
and calibrator fitting are kept on CPU RNG and scikit-learn where exact reproducibility is
achievable (see :mod:`spikecalib.data` and :mod:`spikecalib.calibration`).

This module trains; it does not calibrate. The trained network is fixed before any
calibrator is fitted -- the study is strictly post-hoc, which is what separates it from
Bayesian treatments of SNN overconfidence.

**Why a validation split exists.** Early stopping is on validation accuracy, and that is
the whole reason ``val`` is carved out. A model that fails to converge at ``T = 2`` still
emits perfectly plausible spike counts, and those counts still produce a confident-looking
ECE. Nothing downstream would flag it. Validation accuracy is the only place that failure
becomes visible before it contaminates a calibration result.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .calibration import predict_with_ties
from .models import SpikingClassifier, SynOps, SynOpsCounter

__all__ = [
    "TrainConfig",
    "RunRecord",
    "set_seed",
    "train",
    "predict_counts",
    "predict_outputs",
    "to_time_first",
    "resolve_device",
]


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings. Every value here is a choice, not a sourced constant."""

    epochs: int = 20
    # [DESIGN] Adam's default learning rate. NOT TUNED for this model, dataset or T --
    # see SOURCES.md "To verify" item 12. Adam rather than SGD because surrogate-gradient
    # SNNs produce noisy, poorly-scaled gradients through T steps of BPTT, and per-parameter
    # adaptive steps are more forgiving of that than a single global step size. When the
    # real sweep runs, this needs a search, and the result needs recording.
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    # Early stopping: stop when validation accuracy has not improved for this many epochs.
    patience: int = 5
    min_delta: float = 0.0
    device: Optional[str] = None
    log_every: int = 0  # 0 silences per-epoch printing
    # Fixed so the validation tie-break is identical across epochs; see _evaluate.
    eval_seed: int = 0


@dataclass
class RunRecord:
    """Everything needed to say what produced a set of weights.

    The split fingerprint is what lets a calibration result prove which split it came
    from, rather than assert it (see :meth:`spikecalib.data.SplitIndices.fingerprint`).
    """

    seed: int
    eval_seed: int
    dataset: str
    timesteps: int
    split_fingerprint: Optional[str]
    epochs_requested: int
    epochs_run: int
    best_val_accuracy: float
    final_val_accuracy: float
    final_val_tie_rate: float
    best_epoch: int
    early_stopped: bool
    lr: float
    batch_size: int
    device: str
    n_parameters: int
    seconds: float
    mps_nondeterministic: bool
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> int:
    """Seed Python, NumPy and torch, and return the seed for recording.

    **This bounds variation; it does not eliminate it.** MPS floating-point kernels are not
    bitwise deterministic, so a fixed seed still leaves run-to-run differences in trained
    weights on Apple silicon. Anything that must be exactly reproducible -- data splits,
    calibrator fitting -- is deliberately kept off the accelerator and off this function's
    global state, using ``numpy.random.default_rng`` instead (correctness rule 4).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed


def to_time_first(batch: torch.Tensor) -> torch.Tensor:
    """Convert a ``[N, T, C, H, W]`` collated batch to the ``[T, N, C, H, W]`` the model wants.

    SpikingJelly's frame datasets yield ``[T, C, H, W]`` per sample, so the default collate
    puts batch first. Multi-step modules want time first. Getting this backwards produces a
    shape error rather than silent corruption only because ``T != N`` in practice, which is
    not a guarantee -- hence the explicit helper rather than an inline transpose.
    """
    if batch.dim() != 5:
        raise ValueError(f"expected [N, T, C, H, W], got shape {tuple(batch.shape)}")
    return batch.transpose(0, 1).contiguous()


def _evaluate(
    model: SpikingClassifier,
    loader: Iterable,
    device: torch.device,
    eval_seed: int = 0,
) -> tuple[float, float]:
    """Top-1 accuracy and tie rate on a loader, using the study's own tie-break.

    Returns ``(accuracy, tie_rate)``.

    **This must not use ``torch.argmax``.** A previous version did, and it was wrong in a
    way that looked like a converged model. ``argmax`` resolves ties to the lowest index,
    so every tied sample is assigned class 0; with a high tie rate on balanced classes the
    reported accuracy collapses to the class-0 fraction -- exactly ``1/C``. The smoke test
    duly printed a validation accuracy of 0.1000 at epoch 0 on ten balanced classes, which
    reads as "not learning yet" and is actually "metric is broken". Early stopping and
    best-epoch selection both consume this number, so a biased metric can halt training at
    the wrong epoch or fail to notice real improvement.

    The fix is not merely to unbias it but to use **the same tie-break as every reported
    number**: :func:`spikecalib.calibration.predict_with_ties`, seeded. ``eval_seed`` is
    fixed rather than drawn per epoch so that the validation signal is comparable across
    epochs -- a tie-break that re-rolled each epoch would add noise to the early-stopping
    criterion for no reason. The seed is recorded in the :class:`RunRecord`.

    The tie rate is returned alongside because it is the diagnostic that would have caught
    the original bug immediately: a model whose validation tie rate sits near 1.0 is not
    separating classes at all, whatever its accuracy appears to be.
    """
    model.eval()
    correct = 0
    total = 0
    tied = 0
    rng = np.random.default_rng(eval_seed)
    with torch.no_grad():
        for frames, labels in loader:
            functional.reset_net(model)
            x = to_time_first(frames.float()).to(device)
            counts = model(x).round().to(torch.int64).cpu().numpy()
            y = np.asarray(labels).astype(np.int64)
            predictions, tie_mask = predict_with_ties(counts, rng)
            correct += int((predictions == y).sum())
            tied += int(tie_mask.sum())
            total += int(y.size)
    if not total:
        return 0.0, 0.0
    return correct / total, tied / total


def train(
    model: SpikingClassifier,
    loaders: dict,
    cfg: TrainConfig,
    *,
    seed: int,
    dataset: str,
    timesteps: int,
    split_fingerprint: Optional[str] = None,
) -> tuple[dict, RunRecord]:
    """Train with early stopping on validation accuracy.

    Returns ``(best_state_dict, RunRecord)``. The weights returned are the ones from the
    best validation epoch, not the last -- otherwise early stopping would detect
    overfitting and then hand back the overfitted model anyway.

    Loss is cross-entropy on the spike-count logits. The counts are integers, but
    cross-entropy needs no more than a real-valued score, and using the counts directly is
    what keeps the training objective consistent with the readout the calibrators see.
    """
    device = resolve_device(cfg.device)
    model = model.to(device)
    set_seed(seed)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = -1
    best_state: dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    since_improved = 0
    history: list[dict] = []
    early_stopped = False
    started = time.time()
    epochs_run = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for frames, labels in loaders["train"]:
            functional.reset_net(model)
            x = to_time_first(frames.float()).to(device)
            y = labels.to(device)

            counts = model(x)
            loss = criterion(counts, y)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            epoch_loss += float(loss.item()) * int(y.numel())
            seen += int(y.numel())

        val_accuracy, val_tie_rate = _evaluate(model, loaders["val"], device, cfg.eval_seed)
        epochs_run = epoch + 1
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen if seen else float("nan"),
            "val_accuracy": val_accuracy,
            "val_tie_rate": val_tie_rate,
        }
        history.append(record)
        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"    epoch {epoch:>3}  loss {record['train_loss']:.4f}  "
                f"val acc {val_accuracy:.4f}  val tie rate {val_tie_rate:.4f}"
            )

        if val_accuracy > best_val + cfg.min_delta:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                early_stopped = True
                break

    final_val = history[-1]["val_accuracy"] if history else 0.0
    final_tie = history[-1]["val_tie_rate"] if history else 0.0
    return best_state, RunRecord(
        seed=seed,
        eval_seed=cfg.eval_seed,
        dataset=dataset,
        timesteps=timesteps,
        split_fingerprint=split_fingerprint,
        epochs_requested=cfg.epochs,
        epochs_run=epochs_run,
        best_val_accuracy=best_val,
        final_val_accuracy=final_val,
        final_val_tie_rate=final_tie,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        device=str(device),
        n_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.time() - started,
        mps_nondeterministic=(device.type == "mps"),
        history=history,
    )


@torch.no_grad()
def predict_outputs(
    model: SpikingClassifier,
    loader: Iterable,
    device: Optional[torch.device] = None,
    *,
    count_synops: bool = True,
) -> tuple[np.ndarray, np.ndarray, SynOps]:
    """Raw readout scores for a split, as float64, plus the SynOps from those passes.

    The general form of :func:`predict_counts`. Only the count readout produces integers;
    weighted-count and membrane are real-valued, and rounding them would destroy exactly
    the property the readout experiment is measuring.

    Neuron state is reset before every batch, for the reason given in
    :func:`predict_counts`.
    """
    device = device or resolve_device()
    model = model.to(device)
    model.eval()

    counter = SynOpsCounter(model) if count_synops else None
    if counter is not None:
        counter.attach()

    scores: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    try:
        for frames, labels in loader:
            functional.reset_net(model)
            x = to_time_first(frames.float()).to(device)
            # Move to CPU before widening: MPS has no float64, so casting on device
            # raises rather than silently degrading.
            scores.append(model(x).cpu().numpy().astype(np.float64))
            labels_out.append(np.asarray(labels).astype(np.int64))
            if counter is not None:
                counter.add_samples(int(np.asarray(labels).size))
    finally:
        if counter is not None:
            counter.detach()

    return (
        np.concatenate(scores, axis=0),
        np.concatenate(labels_out, axis=0),
        counter.totals if counter is not None else SynOps(),
    )


@torch.no_grad()
def predict_counts(
    model: SpikingClassifier,
    loader: Iterable,
    device: Optional[torch.device] = None,
    *,
    count_synops: bool = True,
) -> tuple[np.ndarray, np.ndarray, SynOps]:
    """Integer spike counts for a split, plus the SynOps from those same forward passes.

    Returns ``(counts, labels, synops)`` with ``counts`` of dtype ``int64`` and shape
    ``[N, n_classes]``.

    **Neuron state is reset before every batch.** This is load-bearing, not hygiene: LIF
    membrane potential persists across a forward pass, so without a reset the second batch
    starts from wherever the first one left off. The counts would still be integers in
    ``[0, T]``, still look entirely plausible, and be silently wrong -- and the corruption
    would depend on batch order, so it would not even be reproducible. Two tests cover
    this: one asserts that scoring the same batch twice through this function gives
    identical counts, and one demonstrates that consecutive forwards *without* a reset do
    not.

    Runs under ``torch.no_grad`` and in eval mode. Eval mode matters for more than speed
    here: the model has BatchNorm, so train mode would make each sample's count depend on
    the other samples in its batch.
    """
    device = device or resolve_device()
    model = model.to(device)
    model.eval()

    counter = SynOpsCounter(model) if count_synops else None
    if counter is not None:
        counter.attach()

    all_counts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    try:
        for frames, labels in loader:
            functional.reset_net(model)  # <- the load-bearing line
            x = to_time_first(frames.float()).to(device)
            counts = model(x)
            all_counts.append(counts.round().to(torch.int64).cpu().numpy())
            all_labels.append(np.asarray(labels).astype(np.int64))
            if counter is not None:
                counter.add_samples(int(np.asarray(labels).size))
    finally:
        if counter is not None:
            counter.detach()

    return (
        np.concatenate(all_counts, axis=0),
        np.concatenate(all_labels, axis=0),
        counter.totals if counter is not None else SynOps(),
    )
