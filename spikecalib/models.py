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

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from .datasets import DatasetSpec

__all__ = [
    "READOUTS",
    "POTENTIAL_READOUTS",
    "SCALED_READOUTS",
    "readout_ceiling",
    "min_timesteps",
    "readout_weights",
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
# Session 1 Plenary 1.1, Digest of Technical Papers pp. 10-14. VERIFIED 2026-09-12 against
# the paper itself: the energy table is **Figure 1.1.9**, captioned "Rough energy costs for
# various operations in 45nm 0.9V" (Digest p. 14).
#
# E_AC = 0.9 pJ is CONFIRMED as the 32-bit floating-point FAdd in that figure. (The 0.9 in
# the caption is the supply voltage; the coincidence is unfortunate and is not the source of
# the number.)
#
# E_MAC = 4.6 pJ is [DERIVED], NOT [LIT]. **The paper states no MAC value anywhere** -- the
# figure has FAdd and FMult rows and no combined one. 4.6 = 0.9 (FAdd, 32 bit) + 3.7 (FMult,
# 32 bit) is the community convention of multiply-plus-add, and the paper must describe it
# that way rather than attribute it to Horowitz as a measurement.
#
# The caption says "Rough". The source presents these as order-of-magnitude figures, so no
# result may turn on their precision -- which is why the energy number is only ever reported
# as a design-comparison proxy. See SOURCES.md, "Verified 2026-09-12", item 1.
E_AC_JOULES = 0.9e-12   # [LIT] J per accumulate: 32-bit FP add, Horowitz Fig. 1.1.9
E_MAC_JOULES = 4.6e-12  # [DERIVED] 0.9 + 3.7, the multiply-plus-add convention

# [DERIVED] Event cameras emit ON and OFF events, so an integrated event frame has two
# polarity channels. Confirmed by the frame shape (T, 2, H, W) that SpikingJelly produces;
# see the [MEASURED] N-MNIST frame shape in SOURCES.md.
EVENT_POLARITY_CHANNELS = 2


READOUTS = ("count", "weighted-count", "membrane", "accumulator", "scaled-count")

#: Readouts whose class scores are a spike count times a positive scalar. Their cap depends
#: on that scalar, so it cannot be computed from ``T`` and ``C`` alone.
SCALED_READOUTS = ("scaled-count",)

#: Readouts that integrate a potential instead of aggregating output spikes. These get no
#: output LIF, have no spike train, and are unbounded above.
POTENTIAL_READOUTS = ("membrane", "accumulator")


def readout_weights(timesteps: int) -> list[float]:
    """Weights for the weighted-count readout, ``w_t = 2(t+1)/(T+1)``.

    [DERIVED] They sum to ``2(1+...+T)/(T+1) = T``, so weighted-count has **exactly the
    same attainable maximum as count** while taking far more distinct values. That is the
    point of it: count is capped and coarse, weighted-count is capped and fine, membrane is
    neither. Comparing all three separates "confidence is bounded" from "confidence is
    coarse", which the count readout confounds.
    """
    return [2.0 * (t + 1) / (timesteps + 1) for t in range(timesteps)]


def readout_ceiling(readout: str, timesteps: int, n_classes: int,
                    scale: Optional[float] = None) -> Optional[float]:
    """Largest confidence each readout can express. ``None`` means unbounded.

    ``scaled-count`` multiplies the count by a positive ``scale``, so its logits lie in
    ``[0, scale * T]`` and its cap is ``e^(sT) / (e^(sT) + C - 1)``. It is still a capped
    readout; the cap is simply not fixed by ``T``. ``scale`` is required for it and ignored
    for the others.

    [DERIVED] For any readout whose per-class score is bounded above by ``m`` and below by
    0, the extreme score vector is ``[m, 0, ..., 0]`` and the softmax ceiling is
    ``e^m / (e^m + C - 1)``. Count and weighted-count both have ``m = T``. The membrane
    readout is a real-valued potential with no upper bound, so no finite ceiling exists and
    the supremum is 1.
    """
    if readout in POTENTIAL_READOUTS:
        return None
    if readout in SCALED_READOUTS:
        if scale is None or scale <= 0:
            raise ValueError(f"the {readout} readout needs a positive scale; got {scale}")
        # 1 / (1 + (C-1) e^-W), which does not overflow as the learned scale grows
        return float(1.0 / (1.0 + (n_classes - 1) * math.exp(-float(scale) * timesteps)))
    if readout not in ("count", "weighted-count"):
        raise ValueError(f"unknown readout {readout!r}; expected one of {READOUTS}")
    m = float(timesteps)
    return float(math.exp(m) / (math.exp(m) + n_classes - 1))


def min_timesteps(n_classes: int, target_accuracy: float) -> float:
    """Smallest ``T`` at which a model of this accuracy can express calibrated confidence.

    The ceiling is descriptive; this is the same fact made prescriptive. A model whose
    accuracy exceeds ``ceiling(T, C)`` cannot emit a confidence equal to its own
    correctness rate -- it is forced into underconfidence by the readout, whatever it
    learns. So the design question is: given ``C`` classes and a target accuracy ``a``,
    how many timesteps are needed before calibrated confidence is even *reachable*?

    [DERIVED], closed form::

        ceiling(T, C) = e^T / (e^T + C - 1) >= a
        e^T >= a (e^T + C - 1)
        e^T (1 - a) >= a (C - 1)
        T >= ln( (C - 1) a / (1 - a) )

    so ``T_min = ln((C - 1) * a / (1 - a))``.

    Returns a real number. A physical simulation needs ``ceil(T_min)`` timesteps, and the
    result can be negative when the target is easy relative to the class count, meaning
    even ``T = 1`` suffices. ``a >= 1`` has no solution: no finite ``T`` reaches confidence
    1, because the ceiling approaches it only asymptotically.
    """
    if not (0.0 < target_accuracy < 1.0):
        raise ValueError(
            f"target_accuracy must lie strictly in (0, 1); got {target_accuracy}. "
            f"No finite T reaches confidence 1 -- the ceiling approaches it asymptotically."
        )
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; got {n_classes}")
    return float(math.log((n_classes - 1) * target_accuracy / (1.0 - target_accuracy)))


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
    # [DESIGN] Which aggregation turns the output layer's activity into class scores.
    # Only this changes between readouts -- every layer up to and including the final
    # Linear -> BatchNorm1d is identical, because changing anything else would confound
    # the comparison the readout experiment exists to make.
    readout: str = "count"

    @property
    def n_blocks(self) -> int:
        return len(self.channels)

    def __post_init__(self):
        if self.readout not in READOUTS:
            raise ValueError(f"unknown readout {self.readout!r}; expected one of {READOUTS}")


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
        ]
        # Everything above is IDENTICAL for all four readouts. The stack ends at the
        # final BatchNorm1d, and only what follows differs -- for count and weighted-count
        # an output LIF whose spikes are aggregated, for membrane and accumulator a
        # non-spiking integrator that differs only in whether it leaks.
        #
        # The output neuron is appended to the SAME `net` Sequential rather than wrapped in
        # a second module, so parameter names stay `net.0…net.N` and checkpoints trained
        # before the readout option existed still load unchanged. Renaming would have
        # orphaned fifteen trained models for no benefit.
        self.readout = cfg.readout
        if cfg.readout not in POTENTIAL_READOUTS:
            blocks.append(self._lif(output_layer=True))
        self.net = nn.Sequential(*blocks)
        # [DESIGN] The learnable logit scale of the scaled-count readout, held as its log so
        # it stays positive without a constraint. Initialised at 0, i.e. s = 1, where the
        # readout is exactly the count readout. It is an ordinary parameter: the one
        # optimiser trains it at the same learning rate as every weight, with nothing tuned
        # for it (NOTES.md, "The learned-scale arm"). It lives outside `net`, so the trunk's
        # parameter names are the count readout's and a count checkpoint loads into it.
        if cfg.readout in SCALED_READOUTS:
            self.log_scale = nn.Parameter(torch.zeros(()))
        # Not persistent: the weights are derived from T, so putting them in the state dict
        # would add a key that older checkpoints lack and break strict loading.
        self.register_buffer(
            "_w", torch.tensor(readout_weights(self.timesteps), dtype=torch.float32).view(-1, 1, 1),
            persistent=False,
        )
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

    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected [T, N, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[0] != self.timesteps:
            raise ValueError(
                f"model was built for T={self.timesteps} but input has {x.shape[0]} steps"
            )

    def spike_train(self, x: torch.Tensor) -> torch.Tensor:
        """Output-layer spikes, shape ``[T, N, n_classes]``, values in {0, 1}.

        Only defined for the spiking readouts. The potential readouts have no output
        spikes by construction, and pretending otherwise would hide the difference the
        experiment is measuring.
        """
        if self.readout in POTENTIAL_READOUTS:
            raise RuntimeError(
                f"the {self.readout} readout has no output spike train; it integrates the "
                "pre-threshold potential instead"
            )
        self._check(x)
        return self.net(x)

    def potential_trace(self, x: torch.Tensor, *, leak: bool = True) -> torch.Tensor:
        """Non-spiking potential of the output layer at every step, ``[T, N, C]``.

        With ``leak=True`` this is the same charge equation as every other neuron in the
        network, ``v += (z - v) / tau``, with the threshold comparison and the reset
        removed. Nothing fires, so the potential is real-valued and unbounded -- which is
        the property being tested against the count readout's hard cap.

        **The leak is not incidental, and it is what ``leak=False`` exists to isolate.**
        Expanding the recurrence, the input ``k`` steps before the end enters the final
        potential with weight ``(1/tau)(1 - 1/tau)^k``. At the default ``tau = 2`` that is
        ``0.5 * 0.5^k``: a half-life of exactly one step, and **75% of the final potential
        comes from the last two steps at any T >= 6**. The count readout, by contrast,
        weights every step equally at ``1/T``. So the two readouts do not merely differ in
        boundedness -- at large ``T`` they integrate over very different windows, and at
        ``T = 2`` their windows coincide exactly.

        With ``leak=False`` the recurrence becomes ``v += z``, a pure accumulator: every
        step carries weight 1, the same *equal* weighting the count readout uses. That
        makes it the control that separates two confounded explanations. An accumulator and
        a count differ **only** by the output threshold; an accumulator and a leaky
        membrane differ **only** by the leak.
        """
        self._check(x)
        z = self.net(x)
        v = torch.zeros_like(z[0])
        out = []
        for t in range(z.shape[0]):
            v = v + ((z[t] - v) / self.cfg.tau if leak else z[t])
            out.append(v)
        return torch.stack(out, dim=0)

    def membrane_trace(self, x: torch.Tensor) -> torch.Tensor:
        """The leaky potential trace. Retained as the name the earlier experiments used."""
        return self.potential_trace(x, leak=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class class scores, shape ``[N, n_classes]``, aggregated per the readout.

        - ``count``: sum of output spikes over ``T``. Integer-valued in ``[0, T]``.
        - ``weighted-count``: ``Σ_t w_t s_t`` with weights summing to ``T``. Real-valued,
          same maximum as count, many more attainable levels.
        - ``membrane``: final potential of a non-spiking LIF integrator. Real-valued,
          unbounded, and **leaky** -- weighted towards the most recent steps.
        - ``accumulator``: the same integrator with the leak removed. Unbounded, and
          weights every step equally, as the count readout does.
        - ``scaled-count``: the count times a learnable positive scalar ``s``. Real-valued,
          capped at ``e^(sT)/(e^(sT) + C - 1)``, and exactly as many attainable levels as
          count: the scale moves the values and not their number.

        The docstring below describes the count readout specifically, which is the study's
        default and the only one whose output is integral.

        Returns a **float** tensor with exactly integral values in ``[0, T]``. Float is
        forced by autograd; the integrality is the property the study depends on and is
        asserted in the tests. Use :meth:`count_spikes` when gradients are not needed.

        Does **not** reset neuron state. That is deliberate and follows SpikingJelly's
        convention: the caller resets, once per batch. :func:`spikecalib.train.predict_counts`
        does it explicitly, and a test demonstrates that omitting it changes the counts.
        """
        if self.readout in POTENTIAL_READOUTS:
            return self.potential_trace(x, leak=self.readout == "membrane")[-1]
        spikes = self.spike_train(x)
        if self.readout == "weighted-count":
            return (spikes * self._w.to(spikes.dtype)).sum(dim=0)
        if self.readout in SCALED_READOUTS:
            return spikes.sum(dim=0) * self.log_scale.exp()
        return spikes.sum(dim=0)

    @torch.no_grad()
    def count_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class spike counts as a true ``int64`` tensor. Count readout only."""
        if self.readout != "count":
            raise RuntimeError(
                f"count_spikes is only meaningful for the count readout; this model uses "
                f"{self.readout!r}, whose output is real-valued"
            )
        return self.forward(x).round().to(torch.int64)

    @property
    def scale(self) -> float:
        """The logit scale: the learned value for scaled-count, 1 for every other readout."""
        if self.readout in SCALED_READOUTS:
            return float(self.log_scale.detach().exp().item())
        return 1.0

    @property
    def ceiling(self) -> Optional[float]:
        """Largest confidence this readout can express, or ``None`` if unbounded."""
        return readout_ceiling(self.readout, self.timesteps, self.n_classes, scale=self.scale)


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
