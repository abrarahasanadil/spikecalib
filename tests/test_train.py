"""Tests for seeding, the training loop, and count extraction.

The load-bearing test is the neuron-state reset in :func:`predict_counts`. A missed reset
does not raise: LIF membrane potential simply carries into the next batch, the counts stay
integers in ``[0, T]``, and every downstream number looks plausible and is wrong. Worse,
the corruption depends on batch order, so it is not even reproducible. Two tests cover it
-- one asserting the reset works, one demonstrating the failure it prevents.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader, TensorDataset

from spikecalib.datasets import DATASETS
from spikecalib.models import EVENT_POLARITY_CHANNELS, ModelConfig, build_model
from spikecalib.train import (
    RunRecord,
    TrainConfig,
    predict_counts,
    resolve_device,
    set_seed,
    to_time_first,
    train,
)

SPEC = DATASETS["nmnist"]
T = 4


def tiny_cfg() -> ModelConfig:
    return ModelConfig(channels=(8, 16), hidden=32)


def fake_loader(n=32, batch_size=16, seed=0, timesteps=T):
    """A loader shaped exactly like SpikingJelly's: [N, T, C, H, W] per collate."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, timesteps, EVENT_POLARITY_CHANNELS, *SPEC.spatial, generator=g) * 3.0
    y = torch.randint(0, SPEC.n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def warm_batchnorm(model, loader):
    """Populate BatchNorm running statistics so eval mode is not the identity.

    Without this an untrained model in eval mode emits all-zero counts, and every test
    about counts would pass trivially on a network that does nothing.
    """
    model.train()
    with torch.no_grad():
        for frames, _ in loader:
            functional.reset_net(model)
            model(to_time_first(frames.float()))
    functional.reset_net(model)


# ---------------------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------------------


def test_set_seed_returns_the_seed_and_makes_torch_reproducible_on_cpu():
    set_seed(123)
    a = torch.rand(5)
    assert set_seed(123) == 123
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_covers_python_and_numpy_too():
    import random

    set_seed(7)
    first = (random.random(), float(np.random.random()))
    set_seed(7)
    assert (random.random(), float(np.random.random())) == first


# ---------------------------------------------------------------------------------------
# Batch layout
# ---------------------------------------------------------------------------------------


def test_to_time_first_transposes_and_stays_equivalent():
    x = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    out = to_time_first(x)
    assert out.shape == (3, 2, 1, 2, 2)
    assert torch.equal(out[1, 0], x[0, 1])


def test_to_time_first_rejects_the_wrong_rank():
    with pytest.raises(ValueError, match=r"\[N, T, C, H, W\]"):
        to_time_first(torch.zeros(3, 4, 5))


# ---------------------------------------------------------------------------------------
# predict_counts: the reset is load-bearing
# ---------------------------------------------------------------------------------------


def test_predict_counts_returns_integer_counts_of_the_right_shape():
    model = build_model(SPEC, T, tiny_cfg())
    loader = fake_loader(n=32)
    warm_batchnorm(model, loader)
    counts, labels, synops = predict_counts(model, loader, torch.device("cpu"))

    assert counts.dtype == np.int64
    assert counts.shape == (32, SPEC.n_classes)
    assert labels.shape == (32,)
    assert counts.min() >= 0 and counts.max() <= T
    assert synops.samples == 32
    assert synops.energy_joules > 0


def test_scoring_the_same_data_twice_gives_identical_counts():
    """The reset works: no state survives from one call, or one batch, to the next."""
    model = build_model(SPEC, T, tiny_cfg())
    loader = fake_loader(n=48, batch_size=16)
    warm_batchnorm(model, loader)

    first, labels_a, _ = predict_counts(model, loader, torch.device("cpu"))
    second, labels_b, _ = predict_counts(model, loader, torch.device("cpu"))

    assert np.array_equal(first, second), "counts changed between identical runs"
    assert np.array_equal(labels_a, labels_b)


def test_counts_are_independent_of_batch_size():
    """State leaking across batches would make results depend on how they were chunked."""
    model = build_model(SPEC, T, tiny_cfg())
    warm_batchnorm(model, fake_loader(n=32, batch_size=16))

    a, _, _ = predict_counts(model, fake_loader(n=32, batch_size=32), torch.device("cpu"))
    b, _, _ = predict_counts(model, fake_loader(n=32, batch_size=8), torch.device("cpu"))
    assert np.array_equal(a, b)


def test_without_a_reset_consecutive_forwards_disagree():
    """The failure predict_counts prevents, demonstrated rather than asserted.

    If this test ever starts passing trivially (both forwards equal), the reset in
    predict_counts has stopped being load-bearing and the guarantee above is vacuous.
    """
    model = build_model(SPEC, T, tiny_cfg()).train()
    x = to_time_first(next(iter(fake_loader(n=32, batch_size=32)))[0].float())

    functional.reset_net(model)
    with torch.no_grad():
        first = model(x).clone()
        second = model(x).clone()  # deliberately NO reset

    assert not torch.equal(first, second), (
        "membrane state did not carry across forwards, so the reset in predict_counts "
        "would be untested by the assertions above"
    )


# ---------------------------------------------------------------------------------------
# Training loop and the run record
# ---------------------------------------------------------------------------------------


def test_train_returns_weights_and_a_complete_run_record():
    model = build_model(SPEC, T, tiny_cfg())
    loaders = {"train": fake_loader(n=32, seed=1), "val": fake_loader(n=16, seed=2)}
    cfg = TrainConfig(epochs=2, batch_size=16, patience=5, device="cpu")

    state, record = train(
        model, loaders, cfg,
        seed=0, dataset="nmnist", timesteps=T, split_fingerprint="deadbeefcafe0000",
    )

    assert isinstance(record, RunRecord)
    assert record.seed == 0
    assert record.dataset == "nmnist"
    assert record.timesteps == T
    assert record.split_fingerprint == "deadbeefcafe0000"
    assert record.epochs_requested == 2
    assert record.epochs_run == 2
    assert 0.0 <= record.best_val_accuracy <= 1.0
    assert 0.0 <= record.final_val_accuracy <= 1.0
    assert record.n_parameters > 0
    assert len(record.history) == 2
    assert set(state) == set(model.state_dict())
    assert "seed" in record.as_dict()


def test_train_stops_early_when_validation_stops_improving():
    model = build_model(SPEC, T, tiny_cfg())
    loaders = {"train": fake_loader(n=16, seed=1), "val": fake_loader(n=16, seed=2)}
    cfg = TrainConfig(epochs=20, batch_size=16, patience=1, device="cpu")

    _, record = train(model, loaders, cfg, seed=0, dataset="nmnist", timesteps=T)
    assert record.epochs_run < 20
    assert record.early_stopped


def test_train_returns_the_best_epoch_not_the_last():
    model = build_model(SPEC, T, tiny_cfg())
    loaders = {"train": fake_loader(n=32, seed=1), "val": fake_loader(n=16, seed=2)}
    cfg = TrainConfig(epochs=3, batch_size=16, patience=10, device="cpu")
    _, record = train(model, loaders, cfg, seed=0, dataset="nmnist", timesteps=T)

    best_seen = max(h["val_accuracy"] for h in record.history)
    assert record.best_val_accuracy == pytest.approx(best_seen)
    assert record.history[record.best_epoch]["val_accuracy"] == pytest.approx(best_seen)


def test_run_record_flags_mps_nondeterminism_honestly():
    model = build_model(SPEC, T, tiny_cfg())
    loaders = {"train": fake_loader(n=16, seed=1), "val": fake_loader(n=16, seed=2)}
    cfg = TrainConfig(epochs=1, batch_size=16, device="cpu")
    _, record = train(model, loaders, cfg, seed=0, dataset="nmnist", timesteps=T)
    assert record.mps_nondeterministic is False
    assert record.device == "cpu"


def test_resolve_device_honours_an_explicit_request():
    assert resolve_device("cpu").type == "cpu"
