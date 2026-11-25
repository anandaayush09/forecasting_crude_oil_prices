"""QLIKE-consistent, bounded, zero-start corrections and the GRU (ADR 0004)."""

import numpy as np
import pytest

pytest.importorskip("torch", exc_type=ModuleNotFoundError)
import torch  # noqa: E402

from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402

SEQ, CH = 8, 2


def _reg(**kw):
    base = dict(seq_len=SEQ, n_channels=CH, epochs=30, seeds=(0,), device="cpu", mlp_hidden=16)
    return SeqResidualRegressor(**{**base, **kw})


def test_qlike_loss_is_zero_when_the_correction_equals_the_target_and_positive_otherwise():
    reg = _reg(loss="qlike", bound=0.5)
    y = torch.tensor([0.2, -0.1])
    exact = torch.atanh(y / 0.5)
    assert reg._loss(exact, y).item() == pytest.approx(0.0, abs=1e-6)
    assert reg._loss(torch.zeros(2), y).item() > 0.0


def test_qlike_corrections_are_bounded_and_start_at_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, SEQ * CH)).astype(np.float32)
    zero_target = np.zeros(400)  # nothing to correct: the zero start is already optimal
    pred = _reg(loss="qlike", bound=0.5, arch="mlp").fit(x, zero_target).predict(x)
    assert np.abs(pred).max() < 0.05
    big = _reg(loss="qlike", bound=0.5, arch="mlp").fit(x, np.full(400, 2.0)).predict(x)
    assert np.abs(big).max() <= 0.5 + 1e-6


def test_qlike_mode_learns_a_pattern_in_the_window():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1200, SEQ * CH)).astype(np.float32)
    y = 0.3 * np.sign(x[:, 0])  # the newest day of channel 0 decides the sign
    reg = _reg(loss="qlike", bound=0.5, arch="mlp", epochs=60).fit(x[:900], y[:900])
    assert np.corrcoef(reg.predict(x[900:]), y[900:])[0, 1] > 0.6


@pytest.mark.parametrize("arch", ["gru", "transformer", "mlp"])
def test_every_architecture_runs_and_is_deterministic_per_seed(arch):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, SEQ * CH)).astype(np.float32)
    y = rng.normal(0, 0.1, 200)
    kw = dict(arch=arch, loss="qlike", epochs=3, patch_len=4)
    a = _reg(**kw).fit(x, y).predict(x[:20])
    b = _reg(**kw).fit(x, y).predict(x[:20])
    assert a.shape == (20,) and np.allclose(a, b)


def test_the_default_mode_is_still_squared_error():
    reg = _reg()
    assert reg.loss == "mse"
    out = torch.tensor([0.3, -2.0])
    assert torch.equal(reg._correction(out), out)


def test_an_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="unknown loss"):
        _reg(loss="huber")
