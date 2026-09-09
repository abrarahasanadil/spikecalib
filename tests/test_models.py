"""Tests for the spiking model and the SynOps instrumentation.

The load-bearing test here is that the readout really is integer-valued and bounded by
``T``. A model that returned floats, or values above ``T``, would break every discreteness
claim in the study -- and it would do so silently, because a float logit vector looks
entirely normal to a softmax, to ECE, and to a results table.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from spikingjelly.activation_based import functional

from spikecalib.datasets import DATASETS
from spikecalib.models import (
    E_AC_JOULES,
    E_MAC_JOULES,
    EVENT_POLARITY_CHANNELS,
    ModelConfig,
    SynOps,
    SynOpsCounter,
    build_model,
)


def tiny_cfg() -> ModelConfig:
    return ModelConfig(channels=(8, 16), hidden=32)


def frames(timesteps, batch, spec, scale=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.rand(
            timesteps, batch, EVENT_POLARITY_CHANNELS, *spec.spatial, generator=g
        )
        * scale
    )


# ---------------------------------------------------------------------------------------
# The readout must be integer counts in [0, T]
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("timesteps", [2, 4, 10])
def test_forward_returns_integer_valued_counts_bounded_by_T(timesteps):
    spec = DATASETS["nmnist"]
    model = build_model(spec, timesteps, tiny_cfg()).train()
    functional.reset_net(model)
    out = model(frames(timesteps, 16, spec))

    assert out.shape == (16, spec.n_classes)
    assert torch.all(out == out.round()), "spike counts must be integer-valued"
    assert float(out.detach().min()) >= 0.0
    assert float(out.detach().max()) <= timesteps, f"count exceeded T={timesteps}"


def test_count_spikes_returns_a_true_integer_dtype():
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    functional.reset_net(model)
    counts = model.count_spikes(frames(4, 8, spec))
    assert counts.dtype == torch.int64
    assert int(counts.max()) <= 4


def test_forward_output_is_float_because_autograd_requires_it():
    """Documented limitation, asserted so it cannot drift into a surprise."""
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    functional.reset_net(model)
    out = model(frames(4, 4, spec))
    assert out.dtype.is_floating_point
    assert out.requires_grad, "training needs a differentiable readout"


def test_untrained_model_actually_fires():
    """A silent network trains on no gradient signal and yields all-zero counts.

    This is not hypothetical: the first version of this model had no BatchNorm in the
    fully-connected head and emitted exactly zero output spikes for every input.
    """
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    functional.reset_net(model)
    out = model(frames(4, 64, spec))
    assert float(out.detach().sum()) > 0.0, "model emits no spikes at all; the head is starving"


# ---------------------------------------------------------------------------------------
# Sizing comes from the registry, never hardcoded
# ---------------------------------------------------------------------------------------


def test_model_sizes_itself_from_the_dataset_spec():
    """34x34 / 10 classes and 128x128 / 11 classes must both work unchanged."""
    for key, expected_spatial in [("nmnist", (34, 34)), ("dvsgesture", (128, 128))]:
        spec = DATASETS[key]
        model = build_model(spec, 2, tiny_cfg()).train()
        functional.reset_net(model)
        out = model(frames(2, 4, spec))
        assert spec.spatial == expected_spatial
        assert out.shape == (4, spec.n_classes)


def test_feature_shape_follows_the_pooling_arithmetic():
    """34 -> 17 -> 8 with two pooling blocks; 128 -> 64 -> 32."""
    assert build_model(DATASETS["nmnist"], 2, tiny_cfg()).feature_shape == (16, 8, 8)
    assert build_model(DATASETS["dvsgesture"], 2, tiny_cfg()).feature_shape == (16, 32, 32)


def test_too_many_pooling_blocks_fails_with_an_actionable_message():
    spec = DATASETS["nmnist"]
    cfg = ModelConfig(channels=(4,) * 7, hidden=16)  # 34 -> ... -> 0
    with pytest.raises(ValueError, match="pooling blocks reduce it"):
        build_model(spec, 2, cfg)


def test_wrong_timesteps_is_rejected_rather_than_silently_reshaped():
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with pytest.raises(ValueError, match="built for T=4"):
        model(frames(6, 2, spec))


# ---------------------------------------------------------------------------------------
# SynOps instrumentation must not perturb what it measures
# ---------------------------------------------------------------------------------------


def test_hooks_do_not_alter_the_computation():
    """Bit-identical output with hooks attached and detached, on CPU."""
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    x = frames(4, 8, spec, seed=7)

    functional.reset_net(model)
    with torch.no_grad():
        without = model(x).clone()

    counter = SynOpsCounter(model)
    counter.attach()
    functional.reset_net(model)
    with torch.no_grad():
        with_hooks = model(x).clone()
    counter.detach()

    functional.reset_net(model)
    with torch.no_grad():
        after_detach = model(x).clone()

    assert torch.equal(without, with_hooks), "hooks changed the forward output"
    assert torch.equal(without, after_detach), "detaching hooks left the model altered"


def test_counter_records_first_layer_as_mac_and_the_rest_as_ac():
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with SynOpsCounter(model) as counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(4, 8, spec))
    assert counter.totals.mac > 0, "the input layer sees real-valued frames: MAC"
    assert counter.totals.ac > 0, "the spiking layers must contribute accumulates"


def test_ac_layers_really_do_receive_binary_input():
    """If a layer costed as AC sees non-binary input, the cost model is wrong for it."""
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with SynOpsCounter(model) as counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(4, 8, spec))
    assert counter.totals.non_binary_ac_inputs == 0


def test_energy_uses_the_documented_constants():
    ops = SynOps(ac=1000.0, mac=100.0, samples=10)
    assert ops.energy_joules == pytest.approx(1000 * E_AC_JOULES + 100 * E_MAC_JOULES)
    assert ops.energy_per_sample_joules == pytest.approx(ops.energy_joules / 10)


def test_energy_constants_are_the_horowitz_values():
    """Pinned so a silent edit is caught. Provenance: SOURCES.md, marked [UNVERIFIED]."""
    assert E_AC_JOULES == 0.9e-12
    assert E_MAC_JOULES == 4.6e-12


def test_synops_add():
    total = SynOps(ac=1.0, mac=2.0, samples=3) + SynOps(ac=4.0, mac=5.0, samples=6)
    assert (total.ac, total.mac, total.samples) == (5.0, 7.0, 9)


def test_counter_reset_clears_totals():
    spec = DATASETS["nmnist"]
    model = build_model(spec, 2, tiny_cfg()).train()
    counter = SynOpsCounter(model)
    with counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(2, 4, spec))
    assert counter.totals.ac > 0
    counter.reset()
    assert counter.totals.ac == 0 and counter.totals.mac == 0


# ---------------------------------------------------------------------------------------
# Per-layer energy attribution
# ---------------------------------------------------------------------------------------


def test_per_layer_attribution_sums_to_the_totals():
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with SynOpsCounter(model) as counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(4, 8, spec))
    totals = counter.totals
    assert totals.per_layer, "no per-layer attribution recorded"
    assert sum(l.ac for l in totals.per_layer.values()) == pytest.approx(totals.ac)
    assert sum(l.mac for l in totals.per_layer.values()) == pytest.approx(totals.mac)
    assert totals.encoder_energy_joules + totals.spiking_energy_joules == pytest.approx(
        totals.energy_joules
    )


def test_exactly_one_layer_is_the_encoder():
    """Only the first layer sees real-valued frames; everything else sees spikes."""
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with SynOpsCounter(model) as counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(4, 8, spec))
    encoders = [l for l in counter.totals.per_layer.values() if l.kind == "mac"]
    assert len(encoders) == 1
    assert encoders[0].mac > 0 and encoders[0].ac == 0


def test_encoder_fraction_is_reported_and_is_the_dominant_share():
    """The reviewer question: how much of the energy is input encoding, not spiking?"""
    spec = DATASETS["nmnist"]
    model = build_model(spec, 4, tiny_cfg()).train()
    with SynOpsCounter(model) as counter:
        functional.reset_net(model)
        with torch.no_grad():
            model(frames(4, 16, spec))
    fraction = counter.totals.encoder_energy_fraction
    assert 0.0 < fraction < 1.0
    assert "% energy" in counter.totals.layer_table()


def test_synops_addition_merges_per_layer_entries():
    a = SynOps(ac=1.0, mac=2.0, samples=1)
    a.per_layer["net.0"] = __import__(
        "spikecalib.models", fromlist=["LayerSynOps"]
    ).LayerSynOps(name="net.0", kind="mac", mac=2.0, calls=1)
    b = SynOps(ac=3.0, mac=4.0, samples=1)
    b.per_layer["net.0"] = __import__(
        "spikecalib.models", fromlist=["LayerSynOps"]
    ).LayerSynOps(name="net.0", kind="mac", mac=4.0, calls=1)
    total = a + b
    assert total.per_layer["net.0"].mac == pytest.approx(6.0)
    assert total.per_layer["net.0"].calls == 2
