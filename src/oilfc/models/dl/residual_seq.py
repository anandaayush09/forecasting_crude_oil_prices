"""Small sequence regressors (a patch transformer and an attention-free MLP control) with a
scikit-learn style fit/predict, so they plug into `models.ml.residual.residual_forecast`.

Input rows are flattened windows from `features.vol.lag_window`: shape (rows, seq_len * n_channels),
lag-major, today first. The regressors predict a scalar (the anchor residual).
"""

import numpy as np
import torch
from torch import nn

from oilfc.models.dl.encoders import ARCHITECTURES, build_encoder


class _PatchTransformer(nn.Module):
    """Patches of consecutive days -> linear embedding + learned positions -> encoder -> mean pool."""

    def __init__(self, seq_len, n_channels, patch_len, d_model, n_heads, n_layers, dropout):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be a multiple of patch_len"
        self.patch_len, self.n_patches = patch_len, seq_len // patch_len
        self.embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):  # x: (B, seq_len, C), oldest day first
        b = x.shape[0]
        patches = x.reshape(b, self.n_patches, -1)
        h = self.embed(patches) + self.pos
        return self.head(self.encoder(h).mean(dim=1)).squeeze(-1)


class _GRU(nn.Module):
    """A one-layer GRU over the window (oldest day first); the last hidden state feeds a linear head."""

    def __init__(self, n_channels, hidden, dropout):
        super().__init__()
        self.rnn = nn.GRU(n_channels, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


class _FlatMLP(nn.Module):
    """Attention-free control: the whole window flattened into a two-layer MLP."""

    def __init__(self, seq_len, n_channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SeqResidualRegressor:
    """Ensemble over seeds of a small sequence net trained with early stopping.

    arch: "transformer", "mlp", "gru" or one of encoders.ARCHITECTURES (settings in arch_kwargs). Inputs are standardised per channel and the target by its
    standard deviation, using the training rows only. The last `val_fraction` of the training rows
    (they are in time order) is held out for early stopping. predict() averages the seeds.

    loss="qlike" trains the net to output a correction c to a log-volatility forecast, in raw log
    units, with the loss exp(2 (r - c)) - 2 (r - c) - 1 where r is the target (QLIKE of the corrected
    forecast, up to a constant). The output is bounded to (-bound, bound) by `bound * tanh` and the
    last linear layer starts at zero, so an untrained net corrects nothing. The target is not
    rescaled in this mode. loss="mse" is the original squared-error mode on the standardised target.
    """

    def __init__(
        self, arch="transformer", seq_len=64, n_channels=6, d_model=32, n_heads=4, n_layers=2,
        patch_len=8, mlp_hidden=64, dropout=0.1, epochs=80, lr=3e-3, weight_decay=1e-2,
        batch_size=256, patience=15, seeds=(0, 1, 2), device="cpu", val_fraction=0.15,
        loss="mse", bound=0.5, gru_hidden=32, arch_kwargs=None,
    ):  # fmt: skip
        if loss not in ("mse", "qlike"):
            raise ValueError(f"unknown loss {loss!r}")
        self.loss, self.bound, self.gru_hidden = loss, bound, gru_hidden
        self.arch_kwargs = dict(arch_kwargs or {})  # width, depth, heads, kernel (encoders.py)
        self.arch, self.seq_len, self.n_channels = arch, seq_len, n_channels
        self.d_model, self.n_heads, self.n_layers, self.patch_len = (
            d_model,
            n_heads,
            n_layers,
            patch_len,
        )
        self.mlp_hidden, self.dropout = mlp_hidden, dropout
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.patience, self.seeds = batch_size, patience, tuple(seeds)
        self.device, self.val_fraction = device, val_fraction

    def _build(self):
        if self.arch == "transformer":
            return _PatchTransformer(
                self.seq_len, self.n_channels, self.patch_len, self.d_model, self.n_heads,
                self.n_layers, self.dropout,
            )  # fmt: skip
        if self.arch == "mlp":
            return _FlatMLP(self.seq_len, self.n_channels, self.mlp_hidden, self.dropout)
        if self.arch == "gru":
            return _GRU(self.n_channels, self.gru_hidden, self.dropout)
        if self.arch in ARCHITECTURES:
            return build_encoder(
                self.arch, self.seq_len, self.n_channels, dropout=self.dropout, **self.arch_kwargs
            )
        raise ValueError(f"unknown arch {self.arch!r}")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._build().parameters())

    def _to_tensor(self, X):
        x = np.asarray(X, dtype=np.float32).reshape(len(X), self.seq_len, self.n_channels)
        x = (x - self.mean_) / self.std_
        return torch.from_numpy(x[:, ::-1].copy())  # oldest day first

    def fit(self, X, y):
        x_raw = np.asarray(X, dtype=np.float32).reshape(len(X), self.seq_len, self.n_channels)
        self.mean_ = x_raw.mean(axis=(0, 1), keepdims=True)
        self.std_ = x_raw.std(axis=(0, 1), keepdims=True) + 1e-6
        y = np.asarray(y, dtype=np.float32)
        self.y_scale_ = float(y.std()) + 1e-8 if self.loss == "mse" else 1.0
        xt, yt = self._to_tensor(X), torch.from_numpy(y / self.y_scale_)
        n_val = max(int(len(y) * self.val_fraction), 1)
        x_fit, y_fit, x_val, y_val = xt[:-n_val], yt[:-n_val], xt[-n_val:], yt[-n_val:]
        self.nets_ = [self._train_one(seed, x_fit, y_fit, x_val, y_val) for seed in self.seeds]
        return self

    def _correction(self, out):
        """Network output to the correction in the loss: bounded in qlike mode, unchanged in mse mode."""
        return self.bound * torch.tanh(out) if self.loss == "qlike" else out

    def _loss(self, out, y):
        c = self._correction(out)
        if self.loss == "mse":
            return nn.functional.mse_loss(c, y)
        d = y - c
        return (torch.exp(2 * d) - 2 * d - 1).mean()

    def _train_one(self, seed, x_fit, y_fit, x_val, y_val):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        net = self._build().to(self.device)
        if self.loss == "qlike":
            last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_val_d, y_val_d = x_val.to(self.device), y_val.to(self.device)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(x_fit), generator=gen).split(self.batch_size):
                xb, yb = x_fit[idx].to(self.device), y_fit[idx].to(self.device)
                opt.zero_grad()
                loss = self._loss(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = self._loss(net(x_val_d), y_val_d).item()
            if val_loss < best - 1e-6:
                best, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is None:
            raise ValueError(
                "validation loss never became finite: check the inputs for NaN or infinity"
            )
        net.load_state_dict(best_state)
        return net.eval()

    def predict(self, X):
        xt = self._to_tensor(X)
        preds = []
        with torch.no_grad():
            for net in self.nets_:
                out = [
                    self._correction(net(b.to(self.device))).cpu().numpy() for b in xt.split(2048)
                ]
                preds.append(np.concatenate(out))
        return np.mean(preds, axis=0) * self.y_scale_
