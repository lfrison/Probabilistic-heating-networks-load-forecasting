from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class InputAttention1H(nn.Module):
    """Single-head feature attention with a residual gate."""

    def __init__(self, dimension: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(values), dim=-1)
        return values + torch.tanh(self.alpha) * values * weights


class AEDLBase(nn.Module):
    """Shared encoder-decoder backbone for all three forecast heads."""

    def __init__(
        self,
        *,
        n_past_features: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        dropout: float,
        pred_len: int,
    ):
        super().__init__()
        self.future_dim = int(future_dim)
        self.pred_len = int(pred_len)
        decoder_input_dim = max(1, self.future_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.input_attention_past = InputAttention1H(n_past_features, dropout)
        self.input_attention_decoder = InputAttention1H(decoder_input_dim, dropout)
        self.encoder = nn.LSTM(
            n_past_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.encoder_attention = nn.MultiheadAttention(
            hidden_size,
            attention_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.LSTM(
            decoder_input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def encode(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, (hidden, cell) = self.encoder(self.input_attention_past(past))
        attended, _ = self.encoder_attention(encoded, encoded, encoded, need_weights=False)
        encoded = self.encoder_norm(encoded + attended)
        hidden = hidden.clone()
        cell = cell.clone()
        hidden[-1] = encoded[:, -1]
        return hidden, cell

    def decode(
        self,
        future: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Decode the full horizon in one pass, without target feedback."""

        if self.future_dim:
            decoder_input = future
        else:
            decoder_input = future.new_zeros((future.shape[0], self.pred_len, 1))
        decoder_input = self.input_attention_decoder(decoder_input)
        decoded, _ = self.decoder(decoder_input, state)
        return decoded


class DeterministicAEDL(AEDLBase):
    def __init__(self, **kwargs):
        hidden_size = int(kwargs["hidden_size"])
        super().__init__(**kwargs)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        state = self.encode(past)
        return self.output_head(self.decode(future, state)).squeeze(-1)


class GaussianAEDL(AEDLBase):
    def __init__(self, **kwargs):
        hidden_size = int(kwargs["hidden_size"])
        super().__init__(**kwargs)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        nn.init.constant_(self.scale_head.weight, 0.0)
        nn.init.constant_(self.scale_head.bias, math.log(0.1))

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode(past)
        decoded = self.decode(future, state)
        mean = self.mean_head(decoded).squeeze(-1)
        scale = self.softplus(self.scale_head(decoded)).squeeze(-1) + 1e-4
        return mean, torch.log(scale)


class QuantileAEDL(AEDLBase):
    def __init__(self, *, quantiles: Sequence[float], **kwargs):
        hidden_size = int(kwargs["hidden_size"])
        super().__init__(**kwargs)
        taus = torch.as_tensor(sorted(quantiles), dtype=torch.float32)
        self.register_buffer("quantiles", taus)
        self.median_index = int(torch.argmin(torch.abs(taus - 0.5)).item())
        self.output_head = nn.Linear(hidden_size, len(taus))

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode(past)
        raw_quantiles = self.output_head(self.decode(future, state))
        return torch.sort(raw_quantiles, dim=-1).values


def build_model(
    model_name: str,
    *,
    n_past_features: int,
    future_dim: int,
    hidden_size: int,
    num_layers: int,
    attention_heads: int,
    dropout: float,
    pred_len: int,
    quantiles: Sequence[float],
) -> nn.Module:
    common = dict(
        n_past_features=n_past_features,
        future_dim=future_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        pred_len=pred_len,
    )
    if model_name == "deterministic":
        return DeterministicAEDL(**common)
    if model_name == "gaussian":
        return GaussianAEDL(**common)
    if model_name == "quantile":
        return QuantileAEDL(quantiles=quantiles, **common)
    raise ValueError(f"Unknown model {model_name!r}.")
