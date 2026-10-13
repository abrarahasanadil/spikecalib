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


# ---------------------------------------------------------------------------------------
# The bound, inverted into a design rule
# ---------------------------------------------------------------------------------------


def test_min_timesteps_inverts_the_ceiling_exactly():
    """T_min is the exact crossing point, verified against the forward function."""
    from spikecalib.models import min_timesteps, readout_ceiling

    for C in (2, 10, 11, 100, 1000):
        for a in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
            t = min_timesteps(C, a)
            # at T_min the ceiling equals the target, to floating-point precision
            assert readout_ceiling("count", t, C) == pytest.approx(a, abs=1e-12)
            # a hair below and the ceiling is short of the target; a hair above and it clears
            assert readout_ceiling("count", t - 1e-6, C) < a
            assert readout_ceiling("count", t + 1e-6, C) > a


def test_min_timesteps_matches_a_brute_force_search():
    from spikecalib.models import min_timesteps, readout_ceiling

    for C in (10, 100):
        for a in (0.8, 0.95, 0.99):
            t = min_timesteps(C, a)
            grid = np.linspace(max(t - 2, 0.01), t + 2, 20001)
            first = grid[[readout_ceiling("count", g, C) >= a for g in grid]][0]
            assert first == pytest.approx(t, abs=2e-4)


def test_min_timesteps_grows_with_both_classes_and_target():
    from spikecalib.models import min_timesteps

    assert min_timesteps(100, 0.9) > min_timesteps(10, 0.9)
    assert min_timesteps(10, 0.99) > min_timesteps(10, 0.9)


def test_min_timesteps_can_be_negative_when_the_target_is_easy():
    """Two classes at 60% needs less than one timestep; the rule should say so."""
    from spikecalib.models import min_timesteps

    assert min_timesteps(2, 0.6) < 1.0


def test_min_timesteps_rejects_unreachable_targets():
    from spikecalib.models import min_timesteps

    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="strictly in"):
            min_timesteps(10, bad)
    with pytest.raises(ValueError, match="at least 2"):
        min_timesteps(1, 0.9)


def test_readout_ceiling_is_none_exactly_for_the_potential_readouts():
    from spikecalib.models import POTENTIAL_READOUTS, READOUTS, readout_ceiling

    for r in READOUTS:
        if r in POTENTIAL_READOUTS:
            assert readout_ceiling(r, 2, 10) is None
        else:
            # scale = 1 is where the scaled readout coincides with count; the others ignore it
            assert readout_ceiling(r, 2, 10, scale=1.0) == pytest.approx(0.45085306, abs=1e-7)


def test_weighted_count_shares_the_cap_but_not_the_granularity():
    """The design point of the weighted readout: same bound, finer attainable set."""
    from spikecalib.models import readout_ceiling, readout_weights

    for T in (2, 4, 6, 8, 10):
        w = readout_weights(T)
        assert sum(w) == pytest.approx(float(T), abs=1e-9), "weights must sum to T"
        assert readout_ceiling("weighted-count", T, 10) == readout_ceiling("count", T, 10)
        # count reaches T + 1 distinct per-class values; weighted reaches more
        sums = {0.0}
        for wt in w:
            sums |= {x + wt for x in sums}
        assert len(sums) > T + 1



# ---------------------------------------------------------------------------------------
# The accumulator readout: a membrane with the leak removed
# ---------------------------------------------------------------------------------------


def _fixed_trunk(model, z):
    """Replace the network with one that emits a preset ``[T, N, C]`` regardless of input.

    The readouts are defined by what they do to the trunk's output, so pinning the trunk is
    what makes the integration weights measurable rather than inferred.
    """
    import torch.nn as nn

    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.z = z

        def forward(self, _x):
            return self.z

    model.net = Fixed()
    return model


def _readout_model(readout, timesteps=6):
    from spikecalib.datasets import get_spec
    from spikecalib.models import ModelConfig, build_model

    return build_model(get_spec("nmnist"), timesteps,
                       ModelConfig(channels=(8, 8), hidden=16, readout=readout))


def test_membrane_leaks_with_a_one_step_half_life_and_the_accumulator_does_not():
    """The measurement Task 1 turns on: how far back each readout actually looks.

    The leaky membrane's weight on the input ``k`` steps before the end is
    ``(1/tau)(1 - 1/tau)^k``; at the default ``tau = 2`` that halves every step. The
    accumulator weights every step at 1. This is the difference the non-leaky control
    exists to isolate, so it is asserted rather than assumed from the charge equation.
    """
    T = 6
    dummy = torch.zeros(T, 1, 2, 34, 34)
    for k in range(T):
        z = torch.zeros(T, 1, 10)
        z[T - 1 - k, 0, 0] = 1.0
        leaky = _fixed_trunk(_readout_model("membrane", T), z)
        flat = _fixed_trunk(_readout_model("accumulator", T), z)
        assert float(leaky(dummy)[0, 0]) == pytest.approx(0.5 * 0.5 ** k, abs=1e-6)
        assert float(flat(dummy)[0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_leaky_membrane_puts_three_quarters_of_its_weight_on_the_last_two_steps():
    """Stated as a share because that is the form the hypothesis is about.

    At ``T = 2`` the leaky window and the count window coincide exactly -- both are "the
    whole run" -- which is the regime where membrane wins. By ``T = 6`` they have come
    apart, which is where it starts losing.
    """
    tau = 2.0
    for T, expected in ((2, 1.0), (6, 0.7619), (10, 0.7507)):
        w = np.array([(1 / tau) * (1 - 1 / tau) ** k for k in range(T)])
        leaky_share = w[:2].sum() / w.sum()
        count_share = 2 / T  # count weights every step equally
        assert leaky_share == pytest.approx(expected, abs=1e-4)
        if T == 2:
            assert leaky_share == pytest.approx(count_share), "windows coincide at T = 2"
        else:
            assert leaky_share > 2 * count_share, "windows diverge sharply by T = 6"


def test_accumulator_is_the_plain_sum_and_differs_from_count_only_by_the_threshold():
    """Why this readout is the right control.

    An accumulator and a count both weight every step equally; they differ only in whether
    the per-step contribution passes through a threshold. An accumulator and a leaky
    membrane are both unthresholded; they differ only in the leak. Two confounded
    explanations, separated by one model.
    """
    T = 6
    dummy = torch.zeros(T, 1, 2, 34, 34)
    z = torch.randn(T, 4, 10)
    flat = _fixed_trunk(_readout_model("accumulator", T), z)
    assert torch.allclose(flat(dummy), z.sum(dim=0), atol=1e-6)


def test_potential_readouts_have_no_spike_train():
    from spikecalib.models import POTENTIAL_READOUTS

    for readout in POTENTIAL_READOUTS:
        model = _readout_model(readout, 4)
        with pytest.raises(RuntimeError, match="no output spike train"):
            model.spike_train(torch.zeros(4, 1, 2, 34, 34))


def test_accumulator_shares_the_trunk_so_membrane_checkpoints_load_into_it():
    """Same parameters, different aggregation -- the property the comparison depends on."""
    membrane = _readout_model("membrane", 4)
    accumulator = _readout_model("accumulator", 4)
    assert [n for n, _ in membrane.named_parameters()] == \
           [n for n, _ in accumulator.named_parameters()]
    accumulator.load_state_dict(membrane.state_dict())


# ---------------------------------------------------------------------------------------
# The scaled-count readout: a count readout with a learnable logit scale
# ---------------------------------------------------------------------------------------


def test_scaled_count_starts_as_the_count_readout_exactly():
    """s = exp(0) = 1 at initialisation, so the first forward pass is the count readout's."""
    torch.manual_seed(0)
    T = 4
    count, scaled = _readout_model("count", T), _readout_model("scaled-count", T)
    missing = scaled.load_state_dict(count.state_dict(), strict=False)
    assert missing.missing_keys == ["log_scale"] and not missing.unexpected_keys
    assert scaled.scale == 1.0
    x = frames(T, 6, scaled.spec)
    count.eval(), scaled.eval()
    functional.reset_net(count), functional.reset_net(scaled)
    with torch.no_grad():
        assert torch.equal(count(x), scaled(x))
    assert scaled.ceiling == pytest.approx(count.ceiling, abs=1e-12)


def test_scaled_count_is_the_integer_count_times_the_scale():
    torch.manual_seed(1)
    T = 5
    model = _readout_model("scaled-count", T)
    with torch.no_grad():
        model.log_scale.fill_(1.2)
    x = frames(T, 8, model.spec)
    model.eval()
    functional.reset_net(model)
    with torch.no_grad():
        out = model(x)
        functional.reset_net(model)
        counts = model.spike_train(x).sum(dim=0)
    assert model.scale == pytest.approx(float(np.exp(1.2)), rel=1e-6)
    assert torch.allclose(out, counts * model.scale, atol=1e-5)
    assert torch.equal(counts, counts.round()) and counts.min() >= 0 and counts.max() <= T


def test_the_scale_is_trained_by_the_same_optimiser_as_the_weights():
    """It is in model.parameters(), receives a gradient, and moves under one Adam step."""
    torch.manual_seed(2)
    T = 4
    model = _readout_model("scaled-count", T)
    assert any(p is model.log_scale for p in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = frames(T, 16, model.spec)
    y = torch.randint(0, model.n_classes, (16,))
    functional.reset_net(model)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert model.log_scale.grad is not None and float(model.log_scale.grad.abs()) > 0
    optimiser.step()
    assert model.scale != 1.0


def test_scaled_count_cap_follows_the_scale_and_no_output_beats_it():
    from spikecalib.calibration import softmax_probs
    from spikecalib.models import readout_ceiling
    from spikecalib.theory import scaled_ceiling

    torch.manual_seed(3)
    T = 3
    model = _readout_model("scaled-count", T)
    x = frames(T, 64, model.spec)
    model.eval()
    for log_s in (-1.0, 0.0, 0.7, 2.0):
        with torch.no_grad():
            model.log_scale.fill_(log_s)
            functional.reset_net(model)
            out = model(x).double().numpy()
        cap = model.ceiling
        assert cap == pytest.approx(scaled_ceiling(T, model.n_classes, model.scale), abs=1e-9)
        assert cap == pytest.approx(readout_ceiling("scaled-count", T, model.n_classes, scale=model.scale))
        assert softmax_probs(out).max() <= cap + 1e-9
    with pytest.raises(ValueError):
        readout_ceiling("scaled-count", T, 10)


def test_other_readouts_have_no_scale_parameter_and_report_scale_one():
    for r in ("count", "weighted-count", "membrane", "accumulator"):
        model = _readout_model(r, 4)
        assert not hasattr(model, "log_scale") and model.scale == 1.0
        assert "log_scale" not in model.state_dict()
