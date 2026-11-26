"""The six architectures of ADR 0007: shapes, a zero start, determinism and that they can learn."""

import numpy as np
import pytest

pytest.importorskip("torch", exc_type=ModuleNotFoundError)
import torch  # noqa: E402
from torch import nn  # noqa: E402

from oilfc.models.dl.encoders import ARCHITECTURES, build_encoder  # noqa: E402
from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402

CH = 12


def _reg(arch, seq, **kw):
    base = dict(
        arch=arch, seq_len=seq, n_channels=CH, loss="qlike", epochs=3, seeds=(0,), device="cpu",
        arch_kwargs=dict(width=16, depth=2, heads=2, kernel=2),
    )  # fmt: skip
    return SeqResidualRegressor(**{**base, **kw})


@pytest.mark.parametrize("seq", [22, 66])
@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_every_encoder_maps_a_window_to_one_number_per_row(arch, seq):
    net = build_encoder(arch, seq, CH, width=16, depth=3, heads=2, kernel=3)
    out = net(torch.randn(5, seq, CH))
    assert out.shape == (5,) and torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_the_output_layer_is_the_last_linear_so_the_zero_start_works(arch):
    net = build_encoder(arch, 22, CH, width=16, depth=2, heads=2)
    assert [m for m in net.modules() if isinstance(m, nn.Linear)][-1] is net.head


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_an_untouched_net_corrects_nothing_and_is_deterministic_per_seed(arch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 22 * CH)).astype(np.float32)
    a = _reg(arch, 22).fit(x, np.zeros(120)).predict(x[:10])
    b = _reg(arch, 22).fit(x, np.zeros(120)).predict(x[:10])
    assert np.array_equal(a, np.zeros(10)) and np.array_equal(a, b)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_every_encoder_learns_a_simple_pattern_in_the_newest_day(arch):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1200, 16 * CH)).astype(np.float32)
    y = 0.3 * np.tanh(x[:, 0])  # channel 0 of today (lag 0) drives the target
    reg = _reg(arch, 16, epochs=60, lr=3e-3, patience=60).fit(x[:900], y[:900])
    assert np.corrcoef(reg.predict(x[900:]), y[900:])[0, 1] > 0.5


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_encoder("autoformer", 22, CH)


def test_regressor_params_pass_the_tuned_settings_through():
    from oilfc.models.dl.encoders import regressor_params

    hp = dict(
        window=44, width=32, depth=2, dropout=0.1, lr=1e-3, weight_decay=1e-2, bound=0.25, kernel=3
    )
    p = regressor_params(hp, CH)
    assert (
        p["seq_len"] == 44 and p["n_channels"] == CH and p["bound"] == 0.25 and p["loss"] == "qlike"
    )
    assert p["arch_kwargs"] == dict(width=32, depth=2, kernel=3)
    reg = SeqResidualRegressor(arch="tcn", epochs=1, seeds=(0,), device="cpu", **p)
    assert reg.n_parameters() > 0


def test_the_input_channels_exist_and_the_exogenous_ones_follow_the_six_base_channels():
    from oilfc.features.vol import CHANNELS, CHANNELS_EXT

    assert CHANNELS_EXT[: len(CHANNELS)] == CHANNELS and len(CHANNELS_EXT) == 12
    assert len(set(CHANNELS_EXT)) == 12
