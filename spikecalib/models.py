"""Spiking network definitions and the SynOps instrumentation they carry.

Owns the heterogeneous-geometry hazard from ``NOTES.md``: **model sizing is
configuration, never hardcoded.** Spatial dimensions, class count and input channels all
come from the :class:`~spikecalib.datasets.DatasetSpec`, because the three datasets differ
(N-MNIST 34x34 / 10 classes, DVS-Gesture 128x128 / 11, CIFAR10-DVS 128x128 / 10). A
hardcoded 34x34 assumption surfaces as a shape error at best and a silently wrong model at
worst once the second dataset is added.

**The readout is an integer spike count.** The final layer is a spiking layer whose output
is summed over the ``T`` simulation steps, so each class output is an integer in
``[0, T]`` -- the logit the whole study is defined on (``NOTES.md``, resolved decisions).

One unavoidable subtlety, stated rather than hidden: :meth:`SpikingClassifier.forward`
returns a **float tensor whose values are integers**, not an integer tensor. Autograd
cannot flow through an integer dtype, so training requires float. The values are exactly
integral and bounded by ``T``, which is what the discreteness argument depends on, and
:meth:`SpikingClassifier.count_spikes` returns a true ``int64`` tensor for inference.
Both properties are asserted in the tests, because a model that silently returned
non-integral values would break every discreteness claim downstream without erroring.

Owns the counting half of correctness rule 3: :class:`SynOpsCounter` accumulates
accumulate (AC) and multiply-accumulate (MAC) operations separately via forward hooks that
only read, never write, so instrumentation cannot alter the computation.

Provenance of the energy constants: see ``SOURCES.md``. They are attributed to Horowitz,
ISSCC 2014, but the specific values are ``[UNVERIFIED]`` and must not be cited with a
figure number until checked against the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from .datasets import DatasetSpec

__all__ = [
    "ModelConfig",
    "SpikingClassifier",
    "build_model",
    "SynOpsCounter",
    "SynOps",
    "LayerSynOps",
    "E_AC_JOULES",
    "E_MAC_JOULES",
    "EVENT_POLARITY_CHANNELS",
]

# [LIT] Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014,
# 45 nm. [UNVERIFIED] the exact values and the figure/table they come from have NOT been
# checked against the paper: 0.9 pJ is believed to be the 32-bit floating-point ADD, and
# 4.6 pJ is either stated directly or is the community convention of FP multiply (believed
# 3.7 pJ) plus FP add. See SOURCES.md "To verify" item 1. Do not cite a figure number.
E_AC_JOULES = 0.9e-12  # J per accumulate
E_MAC_JOULES = 4.6e-12  # J per multiply-accumulate

# [DERIVED] Event cameras emit ON and OFF events, so an integrated event frame has two
# polarity channels. Confirmed by the frame shape (T, 2, H, W) that SpikingJelly produces;
# see the [MEASURED] N-MNIST frame shape in SOURCES.md.
EVENT_POLARITY_CHANNELS = 2


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and neuron settings. Every field is a choice, not a sourced constant.

    ``channels`` gives one Conv-BN-LIF-Pool block per entry. Two blocks suit N-MNIST's
    34x34; larger inputs want more, which is why this is configuration.
    """

    channels: tuple[int, ...] = (32, 64)
    hidden: int = 256
    # [DESIGN] Conventional LIF defaults, not tuned. tau=2.0 gives a membrane decay slow
    # enough to integrate over a few steps but fast enough not to saturate; v_threshold=1.0
    # is the SpikingJelly default. Neither is sourced; see SOURCES.md.
    tau: float = 2.0
    v_threshold: float = 1.0
    # [DESIGN] Threshold for the OUTPUT layer specifically. None means "same as
    # v_threshold". This exists because the readout layer has a job the hidden layers do
    # not: it must spread counts across [0, T] so the argmax is informative. With
    # BatchNorm normalising pre-activations to roughly unit variance, a threshold of 1.0
    # fires only the top few percent per step, so most classes stay at 0, most samples are
    # fully tied, and predictions are close to random. Measured, not assumed -- see LOG.md.
    output_v_threshold: Optional[float] = None
    # [DESIGN] ATan surrogate gradient. The spike function's true derivative is zero almost
    # everywhere and undefined at threshold, so a surrogate is required to train directly.
    # SpikingJelly implements g(x) = (1/pi) arctan((pi/2) alpha x) + 1/2, whose derivative
    # is bounded and smooth -- chosen over a boxcar/rectangular surrogate because the
    # bounded tail is less prone to exploding gradients through T steps of BPTT.
    # [UNVERIFIED] the originating paper for this specific arctan form; SpikingJelly's
    # source carries the formula but no citation. See SOURCES.md "To verify" item 9.
    surrogate_alpha: float = 2.0
    detach_reset: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.channels)


@dataclass
class LayerSynOps:
    """Operation counts for one instrumented layer."""

    name: str
    kind: str  # "mac" for the real-valued input encoder, "ac" for spiking layers
    ac: float = 0.0
    mac: float = 0.0
    calls: int = 0

    @property
    def energy_joules(self) -> float:
        return self.ac * E_AC_JOULES + self.mac * E_MAC_JOULES


@dataclass
class SynOps:
    """Operation counts and the energy proxy derived from them.

    ``energy_joules`` is a **design-comparison proxy, never measured power** (correctness
    rule 3). It says what these operation counts would cost under a published per-operation
    energy model; it says nothing about any physical device, clock, memory traffic or
    static power.
    """

    ac: float = 0.0
    mac: float = 0.0
    samples: int = 0
    non_binary_ac_inputs: int = 0
    per_layer: dict = field(default_factory=dict)

    @property
    def encoder_energy_joules(self) -> float:
        """Energy attributable to the layers that consume real-valued input.

        Reported separately because it is the obvious reviewer question: an SNN energy
        argument rests on spikes being cheap, and the first layer does not receive spikes.
        It receives the event frame, so every one of its operations is a genuine
        multiply-accumulate at 4.6 pJ rather than an accumulate at 0.9 pJ. If input
        encoding dominates, the energy story is about the encoder, not the network.
        """
        return sum(l.energy_joules for l in self.per_layer.values() if l.kind == "mac")

    @property
    def spiking_energy_joules(self) -> float:
        return sum(l.energy_joules for l in self.per_layer.values() if l.kind == "ac")

    @property
    def encoder_energy_fraction(self) -> float:
        total = self.energy_joules
        return self.encoder_energy_joules / total if total else 0.0

    def layer_table(self) -> str:
        """Per-layer attribution, ordered by share of the energy proxy."""
        total = self.energy_joules
        lines = [f"  {'layer':<28}{'kind':>6}{'AC ops':>16}{'MAC ops':>16}{'% energy':>10}"]
        for name, l in sorted(
            self.per_layer.items(), key=lambda kv: -kv[1].energy_joules
        ):
            share = 100.0 * l.energy_joules / total if total else 0.0
            lines.append(
                f"  {name:<28}{l.kind:>6}{l.ac:>16,.0f}{l.mac:>16,.0f}{share:>9.1f}%"
            )
        return "\n".join(lines)

    @property
    def energy_joules(self) -> float:
        """Design-comparison proxy only. See the class docstring and correctness rule 3."""
        return self.ac * E_AC_JOULES + self.mac * E_MAC_JOULES

    @property
    def energy_per_sample_joules(self) -> float:
        return self.energy_joules / self.samples if self.samples else 0.0

    def __add__(self, other: "SynOps") -> "SynOps":
        merged: dict = {}
        for source in (self.per_layer, other.per_layer):
            for name, l in source.items():
                if name not in merged:
                    merged[name] = LayerSynOps(name=l.name, kind=l.kind)
                merged[name].ac += l.ac
                merged[name].mac += l.mac
                merged[name].calls += l.calls
        return SynOps(
            ac=self.ac + other.ac,
            mac=self.mac + other.mac,
            samples=self.samples + other.samples,
            non_binary_ac_inputs=self.non_binary_ac_inputs + other.non_binary_ac_inputs,
            per_layer=merged,
        )

    def as_dict(self) -> dict:
        return {
            "synops_ac": self.ac,
            "synops_mac": self.mac,
            "samples": self.samples,
            "energy_joules_proxy": self.energy_joules,
            "energy_per_sample_joules_proxy": self.energy_per_sample_joules,
            "encoder_energy_joules_proxy": self.encoder_energy_joules,
            "spiking_energy_joules_proxy": self.spiking_energy_joules,
            "encoder_energy_fraction": self.encoder_energy_fraction,
            "non_binary_ac_inputs": self.non_binary_ac_inputs,
        }


class SpikingClassifier(nn.Module):
    """A directly-trained convolutional SNN whose readout is a per-class spike count.

    Multi-step mode throughout: the input is ``[T, N, C, H, W]`` and every layer processes
    the whole time axis at once, which is both faster than stepping manually and less
    error-prone about neuron state.

    Directly trained, not converted from an ANN -- which is one of the things separating
    this study from Li, Jones & Furber's ANN-to-SNN setting (see ``NOTES.md``, prior work;
    that citation is ``[UNVERIFIED]``).
    """

    def __init__(self, spec: DatasetSpec, timesteps: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.spec = spec
        self.timesteps = int(timesteps)
        self.cfg = cfg
        self.n_classes = spec.n_classes
        self.in_channels = EVENT_POLARITY_CHANNELS

        height, width = spec.spatial
        blocks: list[nn.Module] = []
        in_c = self.in_channels
        for out_c in cfg.channels:
            blocks += [
                layer.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(out_c),
                self._lif(),
                layer.MaxPool2d(2, 2),
            ]
            in_c = out_c
            # [DERIVED] MaxPool2d(2, 2) with no padding gives floor((n - 2) / 2) + 1, which
            # equals n // 2 for every n >= 2. So 34 -> 17 -> 8 for N-MNIST with two blocks.
            height, width = height // 2, width // 2

        if height < 1 or width < 1:
            raise ValueError(
                f"{spec.display_name} is {spec.spatial[0]}x{spec.spatial[1]} and "
                f"{cfg.n_blocks} pooling blocks reduce it to {height}x{width}. Use fewer "
                f"blocks for this dataset -- model sizing is per-dataset configuration."
            )

        self.feature_shape = (in_c, height, width)
        flat = in_c * height * width
        # BatchNorm after BOTH linear layers, not just the convolutions. Without it the
        # fully-connected head is silent at initialisation: a Linear over sparse binary
        # spikes with default PyTorch init produces pre-activations around 1e-3, far below
        # v_threshold=1.0, so no spike ever reaches the readout and no gradient signal ever
        # returns. This was observed, not anticipated -- the first version of this model
        # emitted exactly zero output spikes for every input. Normalising the
        # pre-activations puts them on the same scale as the threshold.
        # SeqToANNContainer rather than layer.BatchNorm1d because the latter requires a
        # 4-D [T, N, C, L] input in multi-step mode and the head is flat [T, N, C].
        blocks += [
            layer.Flatten(),
            layer.Linear(flat, cfg.hidden, bias=False),
            layer.SeqToANNContainer(nn.BatchNorm1d(cfg.hidden)),
            self._lif(),
            layer.Linear(cfg.hidden, self.n_classes, bias=False),
            layer.SeqToANNContainer(nn.BatchNorm1d(self.n_classes)),
            self._lif(output_layer=True),  # its spikes ARE the logits, once summed over T
        ]
        self.net = nn.Sequential(*blocks)
        functional.set_step_mode(self, step_mode="m")

    def _lif(self, output_layer: bool = False) -> neuron.LIFNode:
        threshold = self.cfg.v_threshold
        if output_layer and self.cfg.output_v_threshold is not None:
            threshold = self.cfg.output_v_threshold
        return neuron.LIFNode(
            tau=self.cfg.tau,
            v_threshold=threshold,
            surrogate_function=surrogate.ATan(alpha=self.cfg.surrogate_alpha),
            detach_reset=self.cfg.detach_reset,
        )

    def spike_train(self, x: torch.Tensor) -> torch.Tensor:
        """Output-layer spikes, shape ``[T, N, n_classes]``, values in {0, 1}."""
        if x.dim() != 5:
            raise ValueError(f"expected [T, N, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[0] != self.timesteps:
            raise ValueError(
                f"model was built for T={self.timesteps} but input has {x.shape[0]} steps"
            )
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class spike counts, shape ``[N, n_classes]``.

        Returns a **float** tensor with exactly integral values in ``[0, T]``. Float is
        forced by autograd; the integrality is the property the study depends on and is
        asserted in the tests. Use :meth:`count_spikes` when gradients are not needed.

        Does **not** reset neuron state. That is deliberate and follows SpikingJelly's
        convention: the caller resets, once per batch. :func:`spikecalib.train.predict_counts`
        does it explicitly, and a test demonstrates that omitting it changes the counts.
        """
        return self.spike_train(x).sum(dim=0)

    @torch.no_grad()
    def count_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class spike counts as a true ``int64`` tensor. Inference only."""
        return self.forward(x).round().to(torch.int64)


def build_model(spec: DatasetSpec, timesteps: int, cfg: Optional[ModelConfig] = None) -> SpikingClassifier:
    """Build a spiking classifier sized from the dataset registry."""
    return SpikingClassifier(spec, timesteps, cfg or ModelConfig())


class SynOpsCounter:
    """Forward hooks accumulating AC and MAC counts separately.

    **The hooks only read.** They inspect the input tensor and the output shape and record
    numbers; they return ``None``, so no forward output is replaced. A test asserts the
    model produces bit-identical output with hooks attached and detached, because
    instrumentation that perturbs the thing it measures would invalidate every result it
    reports on.

    Counting model, stated plainly because it is an approximation and not a measurement:

    - For a ``Conv2d``, the dense operation count per sample per timestep is
      ``C_out * H_out * W_out * (C_in / groups) * kH * kW``. For a ``Linear`` it is
      ``in_features * out_features``. ``[DERIVED]`` -- one multiply-accumulate per weight
      per output element.
    - Operations are **gated by input sparsity**: a zero input contributes nothing to the
      sum, so the count is scaled by the fraction of non-zero input elements. This is the
      standard SynOps approximation for SNNs.
    - The **AC/MAC distinction is about the input, not the layer type.** The first layer
      receives real-valued event frames, whose non-zero entries are integer event counts,
      so each operation is a genuine multiply-accumulate. Every later layer receives binary
      spikes, so a "multiply" by 1 is just an accumulate. That is the entire source of the
      SNN energy argument.
    - What this ignores: memory traffic, static power, clock, and any real hardware
      mapping. It is a design-comparison proxy (correctness rule 3), never measured power.

    The counter also records how often a layer designated AC in fact received non-binary
    input, so a wrong assumption becomes visible rather than silently mis-costed.
    """

    COUNTED = (layer.Conv2d, layer.Linear, nn.Conv2d, nn.Linear)

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.totals = SynOps()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._counted_layers = [
            (name or type(m).__name__, m)
            for name, m in model.named_modules()
            if isinstance(m, self.COUNTED)
        ]
        if not self._counted_layers:
            raise ValueError("model has no Conv2d or Linear layers to instrument")

    def __enter__(self) -> "SynOpsCounter":
        self.attach()
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    def attach(self) -> "SynOpsCounter":
        self.detach()
        for index, (name, module) in enumerate(self._counted_layers):
            # The first counted layer sees real-valued event frames; the rest see spikes.
            kind = "mac" if index == 0 else "ac"
            self.totals.per_layer.setdefault(name, LayerSynOps(name=name, kind=kind))
            self._handles.append(
                module.register_forward_hook(self._make_hook(name, kind))
            )
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def reset(self) -> None:
        self.totals = SynOps()
        for index, (name, _module) in enumerate(self._counted_layers):
            kind = "mac" if index == 0 else "ac"
            self.totals.per_layer[name] = LayerSynOps(name=name, kind=kind)

    def add_samples(self, n: int) -> None:
        self.totals.samples += int(n)

    def _make_hook(self, name: str, kind: str):
        def hook(module, inputs, output):  # noqa: ANN001 - torch hook signature
            x = inputs[0]
            if not torch.is_tensor(x) or not torch.is_tensor(output):
                return None
            dense = self._dense_ops(module, output)
            nonzero = float(torch.count_nonzero(x).item())
            total_elems = float(x.numel())
            activity = nonzero / total_elems if total_elems else 0.0
            ops = dense * activity
            entry = self.totals.per_layer[name]
            entry.calls += 1
            if kind == "mac":
                self.totals.mac += ops
                entry.mac += ops
            else:
                self.totals.ac += ops
                entry.ac += ops
                # A layer designated AC must be receiving binary spikes. If it is not, the
                # cost model is wrong for it, and that should be visible.
                if not self._is_binary(x):
                    self.totals.non_binary_ac_inputs += 1
            return None

        return hook

    @staticmethod
    def _is_binary(x: torch.Tensor) -> bool:
        return bool(torch.all((x == 0) | (x == 1)).item())

    @staticmethod
    def _dense_ops(module: nn.Module, output: torch.Tensor) -> float:
        """Dense multiply-accumulates for one forward of this module, all steps included."""
        if isinstance(module, (layer.Conv2d, nn.Conv2d)):
            # output is [T, N, C_out, H, W] in multi-step mode or [N, C_out, H, W] otherwise;
            # numel covers whichever, so the time axis needs no special case.
            kernel = 1
            for k in module.kernel_size:
                kernel *= k
            per_output = (module.in_channels / module.groups) * kernel
            return float(output.numel()) * float(per_output)
        if isinstance(module, (layer.Linear, nn.Linear)):
            return float(output.numel()) * float(module.in_features)
        return 0.0
