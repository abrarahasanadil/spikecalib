"""Activity recording and the same-weights potential readouts (A5, A6; added after review)."""
from __future__ import annotations

import numpy as np
import torch
from spikingjelly.activation_based import functional

from spikecalib.activity import ActivityRecorder, aggregate_potential, layer_names, run_split
from spikecalib.datasets import get_spec
from spikecalib.models import ModelConfig, build_model


def _model(readout="count"):
    torch.manual_seed(0)
    return build_model(get_spec("nmnist"), 3, ModelConfig(channels=(4, 8), hidden=16, readout=readout))


def _loader(n=6):
    g = torch.Generator().manual_seed(1)
    frames = torch.poisson(torch.full((n, 3, 2, 34, 34), 0.5), generator=g)
    labels = torch.arange(n) % 10
    return [(frames, labels)]


def test_layer_names_follow_the_architecture():
    assert layer_names(_model()) == ["conv1", "conv2", "hidden", "output"]
    assert layer_names(_model("accumulator")) == ["conv1", "conv2", "hidden"]


def test_potential_readouts_on_count_weights_equal_the_trained_classes():
    count = _model("count").eval()
    for readout, leak in (("accumulator", False), ("membrane", True)):
        other = build_model(get_spec("nmnist"), 3, ModelConfig(channels=(4, 8), hidden=16, readout=readout))
        other.load_state_dict(count.state_dict(), strict=True)
        other.eval()
        (frames, _), = _loader()
        x = frames.transpose(0, 1).contiguous()
        functional.reset_net(count)
        z = count.net[:-1](x)
        functional.reset_net(other)
        assert torch.equal(aggregate_potential(z, 2.0, leak), other(x))


def test_run_split_reproduces_the_count_readout_and_counts_spikes():
    model = _model()
    rec = ActivityRecorder(model)
    with rec:
        counts, acc, mem, labels = run_split(model, _loader(), torch.device("cpu"), rec, keep_first=True)
    functional.reset_net(model)
    (frames, _), = _loader()
    ref = model(frames.transpose(0, 1).contiguous()).detach().numpy()
    assert np.array_equal(counts, ref)
    assert counts.min() >= 0 and counts.max() <= 3
    out = rec.stats["output"]
    assert out.spikes == counts.sum()
    assert out.sample_steps == 3 * 6 and out.n_neurons == 10
    assert 0.0 <= out.rate <= 1.0 and 0.0 <= out.silent_share <= 1.0
    assert rec.kept["output"].shape == (3, 10)
    assert np.array_equal(rec.kept["output"].sum(axis=0), counts[0])
