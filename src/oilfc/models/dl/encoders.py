"""Compact pure-PyTorch versions of the core block of six forecasting architectures (ADR 0007).

Each encoder maps a window x of shape (batch, seq_len, n_channels), oldest day first, to one number per
row. They are not ports of the reference code: they keep the idea that defines each architecture
(multi-rate pooling, a dense residual encoder, attention across channels, dilated causal convolution,
gated variable selection with attention, sparse attention with distilling) at a size that suits a few
thousand daily rows.

Every encoder registers its output `nn.Linear` last, because `SeqResidualRegressor` zero-initialises the
last linear layer it finds, so that an untrained net outputs exactly zero.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _NHiTS(nn.Module):
    """Stacks that read max-pooled copies of the window at different rates; their features are summed."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, pools=(1, 4, 11)):
        super().__init__()
        self.pools = [min(p, seq_len) for p in pools]
        self.stacks = nn.ModuleList()
        for p in self.pools:
            n_in = math.ceil(seq_len / p) * n_channels
            layers, d = [], n_in
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
                d = width
            self.stacks.append(nn.Sequential(*layers))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        z = 0
        for p, stack in zip(self.pools, self.stacks, strict=True):
            pooled = F.max_pool1d(x.transpose(1, 2), p, p, ceil_mode=True)  # (B, C, ceil(T/p))
            z = z + stack(pooled.flatten(1))
        return self.head(z).squeeze(-1)


class _ResBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.drop, self.norm = nn.Dropout(dropout), nn.LayerNorm(width)

    def forward(self, z):
        return self.norm(z + self.drop(self.fc2(F.relu(self.fc1(z)))))


class _TiDE(nn.Module):
    """Per-day feature projection, then a dense encoder of residual MLP blocks over the flattened window."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, proj=4):
        super().__init__()
        self.project = nn.Linear(n_channels, proj)
        self.inp = nn.Linear(seq_len * proj, width)
        self.blocks = nn.Sequential(*[_ResBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        z = F.relu(self.project(x)).flatten(1)
        return self.head(self.blocks(F.relu(self.inp(z)))).squeeze(-1)


class _ITransformer(nn.Module):
    """Inverted attention: the whole window of one channel is one token, attention runs across channels."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, heads):
        super().__init__()
        self.embed = nn.Linear(seq_len, width)
        # attention and the mean over tokens ignore token order, so the net needs a learned identity per channel
        self.channel = nn.Parameter(torch.randn(1, n_channels, width) * 0.1)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        tokens = self.embed(x.transpose(1, 2)) + self.channel  # (B, C, width)
        return self.head(self.norm(self.encoder(tokens).mean(dim=1))).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, z):  # causal: pad on the left only
        h = self.drop(F.relu(self.conv1(F.pad(z, (self.pad, 0)))))
        h = self.drop(F.relu(self.conv2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.skip(z))


class _TCN(nn.Module):
    """Dilated causal convolutions with residual blocks; the last time step feeds the head."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, kernel):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                _TCNBlock(n_channels if i == 0 else width, width, kernel, 2**i, dropout)
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.blocks(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class _GRN(nn.Module):
    """Gated residual network of the TFT: two layers, a gated linear unit, a skip and a layer norm."""

    def __init__(self, n_in, hidden, n_out, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_in, hidden), nn.Linear(hidden, 2 * n_out)
        self.skip = nn.Linear(n_in, n_out) if n_in != n_out else nn.Identity()
        self.drop, self.norm = nn.Dropout(dropout), nn.LayerNorm(n_out)

    def forward(self, z):
        h = self.drop(self.fc2(F.elu(self.fc1(z))))
        return self.norm(self.skip(z) + F.glu(h, dim=-1))


class _TFT(nn.Module):
    """Gated variable selection per day, an LSTM encoder, self-attention over the window and a gate."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, heads):
        super().__init__()
        self.n_channels, self.width = n_channels, width
        self.w_in = nn.Parameter(torch.randn(n_channels, width) * 0.1)
        self.b_in = nn.Parameter(torch.zeros(n_channels, width))
        self.select = _GRN(n_channels * width, width, n_channels, dropout)
        self.lstm = nn.LSTM(width, width, num_layers=depth, batch_first=True,
                            dropout=dropout if depth > 1 else 0.0)  # fmt: skip
        self.gate1 = _GRN(width, width, width, dropout)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.gate2 = _GRN(width, width, width, dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.w_in + self.b_in  # (B, T, C, width)
        weights = torch.softmax(self.select(emb.flatten(2)), dim=-1)  # (B, T, C)
        z = (weights.unsqueeze(-1) * emb).sum(dim=2)  # (B, T, width)
        h, _ = self.lstm(z)
        h = self.gate1(h) + z
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head(self.gate2(self.norm(h + a))[:, -1]).squeeze(-1)


class _ProbSparseAttention(nn.Module):
    """Only the u most 'active' queries (largest gap between their max and mean score) attend fully; the
    other queries receive the mean of the values (Informer). The sparsity measure is computed on all keys,
    which is cheap at windows of at most 66 days."""

    def __init__(self, width, heads, dropout, factor=5):
        super().__init__()
        assert width % heads == 0, "width must be a multiple of heads"
        self.h, self.d, self.factor = heads, width // heads, factor
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = (
            z.reshape(b, t, self.h, self.d).transpose(1, 2) for z in self.qkv(x).chunk(3, -1)
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)  # (B, h, T, T)
        u = min(t, max(1, math.ceil(self.factor * math.log(t + 1))))
        sparsity = scores.max(-1).values - scores.mean(-1)
        top = sparsity.topk(u, dim=-1).indices  # (B, h, u)
        out = v.mean(dim=2, keepdim=True).expand(b, self.h, t, self.d).clone()
        full = self.drop(torch.softmax(scores, -1)) @ v  # (B, h, T, d)
        idx = top.unsqueeze(-1).expand(-1, -1, -1, self.d)
        out.scatter_(2, idx, full.gather(2, idx))
        return self.out(out.transpose(1, 2).reshape(b, t, -1))


class _InformerLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attn = _ProbSparseAttention(width, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * width, width))  # fmt: skip
        self.n1, self.n2, self.drop = nn.LayerNorm(width), nn.LayerNorm(width), nn.Dropout(dropout)

    def forward(self, z):
        z = self.n1(z + self.drop(self.attn(z)))
        return self.n2(z + self.drop(self.ff(z)))


class _Informer(nn.Module):
    """ProbSparse attention layers with a distilling convolution (halving the length) between them."""

    def __init__(self, seq_len, n_channels, width, depth, dropout, heads):
        super().__init__()
        self.embed = nn.Linear(n_channels, width)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, width))
        self.layers = nn.ModuleList([_InformerLayer(width, heads, dropout) for _ in range(depth)])
        self.distil = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ELU(),
                          nn.MaxPool1d(3, stride=2, padding=1))
            for _ in range(depth - 1)
        ])  # fmt: skip
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        z = self.embed(x) + self.pos
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.distil):
                z = self.distil[i](z.transpose(1, 2)).transpose(1, 2)
        return self.head(self.norm(z.mean(dim=1))).squeeze(-1)


ARCHITECTURES = ("nhits", "tide", "itransformer", "tcn", "tft", "informer")


def build_encoder(arch, seq_len, n_channels, width=32, depth=2, dropout=0.1, heads=4, kernel=3):
    """One of ARCHITECTURES. heads applies to itransformer, tft and informer (width must be a multiple of
    it), kernel to tcn."""
    if arch == "nhits":
        return _NHiTS(seq_len, n_channels, width, depth, dropout)
    if arch == "tide":
        return _TiDE(seq_len, n_channels, width, depth, dropout)
    if arch == "itransformer":
        return _ITransformer(seq_len, n_channels, width, depth, dropout, heads)
    if arch == "tcn":
        return _TCN(seq_len, n_channels, width, depth, dropout, kernel)
    if arch == "tft":
        return _TFT(seq_len, n_channels, width, depth, dropout, heads)
    if arch == "informer":
        return _Informer(seq_len, n_channels, width, depth, dropout, heads)
    raise ValueError(f"unknown architecture {arch!r}")


def regressor_params(hp: dict, n_channels: int) -> dict:
    """Keyword arguments of `SeqResidualRegressor` for one tuned set of hyperparameters `hp` (the keys
    window, width, depth, dropout, lr, weight_decay, bound, and heads or kernel where they apply)."""
    arch_kwargs = {k: hp[k] for k in ("width", "depth", "heads", "kernel") if k in hp}
    return dict(
        seq_len=hp["window"], n_channels=n_channels, dropout=hp["dropout"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], bound=hp["bound"], loss="qlike", patch_len=2,
        arch_kwargs=arch_kwargs,
    )  # fmt: skip
