"""Spiking activity of a trained network: firing statistics, rasters, and its pre-threshold output.

Added after review, not pre-registered (NOTES.md, "Post-review analyses, thresholds fixed
2026-11-12 before computation", A5 and A6).

Every LIF layer of :class:`spikecalib.models.SpikingClassifier` runs in multi-step mode, so its
output is a ``[T, N, ...]`` tensor of 0/1 spikes. A forward hook on each LIF records, per
layer, the total number of spikes and which neurons fired at least once. The output of the
final BatchNorm1d, ``z_t``, is the input current of the output layer; aggregating it without a
threshold gives the accumulator (``sum_t z_t``) and the leaky membrane
(``v_t = v_(t-1) + (z_t - v_(t-1)) / tau``) readouts on the count-trained weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron

from .train import to_time_first

__all__ = ["LayerStats", "lif_layers", "ActivityRecorder", "aggregate_potential", "layer_names"]


def lif_layers(model) -> list[tuple[int, neuron.LIFNode]]:
    """``(index in model.net, module)`` for every LIF layer, in order."""
    return [(i, m) for i, m in enumerate(model.net) if isinstance(m, neuron.LIFNode)]


def layer_names(model) -> list[str]:
    """Readable names: conv1..convK for the block LIFs, hidden, output."""
    idx = lif_layers(model)
    n_conv = len(model.cfg.channels)
    names = [f"conv{k + 1}" for k in range(n_conv)] + ["hidden"]
    if model.readout not in ("membrane", "accumulator"):
        names.append("output")
    if len(names) != len(idx):
        raise RuntimeError(f"expected {len(names)} LIF layers, found {len(idx)}")
    return names


@dataclass
class LayerStats:
    """Running totals for one LIF layer."""

    n_neurons: int
    spikes: float = 0.0
    sample_steps: int = 0  # samples x timesteps seen
    ever_fired: np.ndarray = field(default=None, repr=False)

    def update(self, s: torch.Tensor) -> None:
        # s: [T, N, ...] of 0/1
        t, n = s.shape[0], s.shape[1]
        flat = s.reshape(t, n, -1)
        self.spikes += float(flat.sum().item())
        self.sample_steps += t * n
        fired = (flat.sum(dim=(0, 1)) > 0).cpu().numpy()
        self.ever_fired = fired if self.ever_fired is None else (self.ever_fired | fired)

    @property
    def rate(self) -> float:
        """Mean spikes per neuron per timestep."""
        return self.spikes / (self.sample_steps * self.n_neurons)

    @property
    def silent_share(self) -> float:
        """Share of neurons that never fired on any sample at any timestep."""
        return float(1.0 - self.ever_fired.mean())


class ActivityRecorder:
    """Forward hooks on every LIF layer, plus the input of the output layer.

    Use as a context manager around inference. ``keep_sample`` (an index into the next batch)
    stores that sample's full spike trains per layer for a raster.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.names = layer_names(model)
        self.layers = lif_layers(model)
        self.stats: dict[str, LayerStats] = {}
        self.handles = []
        self.keep_sample = None
        self.kept: dict[str, np.ndarray] = {}

    def __enter__(self):
        for name, (_, mod) in zip(self.names, self.layers):
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, name):
        def fn(_module, _inputs, out):
            s = out.detach()
            if name not in self.stats:
                self.stats[name] = LayerStats(n_neurons=int(np.prod(s.shape[2:])))
            self.stats[name].update(s)
            if self.keep_sample is not None:
                self.kept[name] = s[:, self.keep_sample].reshape(s.shape[0], -1).cpu().numpy()
        return fn


def aggregate_potential(z: torch.Tensor, tau: float, leak: bool) -> torch.Tensor:
    """The membrane (``leak=True``) or accumulator readout of ``z`` (``[T, N, C]``).

    The same arithmetic, in the same order and precision, as
    :meth:`spikecalib.models.SpikingClassifier.potential_trace`.
    """
    v = torch.zeros_like(z[0])
    for t in range(z.shape[0]):
        v = v + ((z[t] - v) / tau if leak else z[t])
    return v


@torch.no_grad()
def run_split(model, loader, device, recorder: ActivityRecorder = None, keep_first: bool = False):
    """Count logits, accumulator and membrane scores, and labels for one split.

    The count model's trunk (``net[:-1]``) gives ``z_t``; its output LIF gives the counts.
    Neuron state is reset before every batch.
    """
    model = model.to(device).eval()
    counts, acc, mem, labels = [], [], [], []
    first = True
    for frames, y in loader:
        functional.reset_net(model)
        x = to_time_first(frames.float()).to(device)
        if recorder is not None:
            recorder.keep_sample = 0 if (keep_first and first) else None
        z = model.net[:-1](x)
        spikes = model.net[-1](z)
        counts.append(spikes.sum(dim=0).cpu().numpy().astype(np.float64))
        acc.append(aggregate_potential(z, model.cfg.tau, leak=False).cpu().numpy().astype(np.float64))
        mem.append(aggregate_potential(z, model.cfg.tau, leak=True).cpu().numpy().astype(np.float64))
        labels.append(np.asarray(y).astype(np.int64))
        first = False
    if recorder is not None:
        recorder.keep_sample = None
    return (np.concatenate(counts), np.concatenate(acc), np.concatenate(mem), np.concatenate(labels))
