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
        output_skip: str = "none",
        target_index: int = 0,
    ):
        super().__init__()
        self.future_dim = int(future_dim)
        self.pred_len = int(pred_len)
        self.output_skip = output_skip
        self.target_index = int(target_index)
        if output_skip == "linear":
            # Linear map of the last 24 h of load added to the output (as in
            # LSTNet), initialised as the seasonal-naive forecast.
            self.skip_linear = nn.Linear(24, self.pred_len)
            with torch.no_grad():
                self.skip_linear.weight.zero_()
                self.skip_linear.bias.zero_()
                self.skip_linear.weight[:, : self.pred_len].fill_diagonal_(1.0)
        elif output_skip != "none":
            raise ValueError(f"Unknown output skip {output_skip!r}.")
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

    def skip(self, past: torch.Tensor) -> torch.Tensor | float:
        if self.output_skip == "none":
            return 0.0
        return self.skip_linear(past[:, -24:, self.target_index])


class DeterministicAEDL(AEDLBase):
    def __init__(self, **kwargs):
        hidden_size = int(kwargs["hidden_size"])
        super().__init__(**kwargs)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        state = self.encode(past)
        return self.output_head(self.decode(future, state)).squeeze(-1) + self.skip(past)


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
        detach_scale_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # detach_scale_features: the scale-head loss does not train the backbone.
        state = self.encode(past)
        decoded = self.decode(future, state)
        mean = self.mean_head(decoded).squeeze(-1) + self.skip(past)
        scale_input = decoded.detach() if detach_scale_features else decoded
        scale = self.softplus(self.scale_head(scale_input)).squeeze(-1) + 1e-4
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
        if self.output_skip != "none":
            raw_quantiles = raw_quantiles + self.skip(past).unsqueeze(-1)
        return torch.sort(raw_quantiles, dim=-1).values


class PlainLSTMBase(nn.Module):
    """LSTM baseline without decoder and attention: a linear layer maps the last
    hidden state and the flattened forecast-horizon inputs to all forecast steps."""

    def __init__(
        self,
        *,
        n_past_features: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.future_dim = int(future_dim)
        self.encoder = nn.LSTM(
            n_past_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = hidden_size + self.pred_len * self.future_dim

    def features(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(past)
        flat_future = future.reshape(future.shape[0], -1)
        return torch.cat([self.dropout(hidden[-1]), flat_future], dim=-1)


class DeterministicLSTM(PlainLSTMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_head = nn.Linear(self.feature_dim, self.pred_len)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.features(past, future))


class GaussianLSTM(PlainLSTMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_head = nn.Linear(self.feature_dim, self.pred_len)
        self.scale_head = nn.Linear(self.feature_dim, self.pred_len)
        self.softplus = nn.Softplus()
        nn.init.constant_(self.scale_head.weight, 0.0)
        nn.init.constant_(self.scale_head.bias, math.log(0.1))

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        detach_scale_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(past, future)
        mean = self.mean_head(features)
        scale_input = features.detach() if detach_scale_features else features
        scale = self.softplus(self.scale_head(scale_input)) + 1e-4
        return mean, torch.log(scale)


class QuantileLSTM(PlainLSTMBase):
    def __init__(self, *, quantiles: Sequence[float], **kwargs):
        super().__init__(**kwargs)
        taus = torch.as_tensor(sorted(quantiles), dtype=torch.float32)
        self.register_buffer("quantiles", taus)
        self.median_index = int(torch.argmin(torch.abs(taus - 0.5)).item())
        self.output_head = nn.Linear(self.feature_dim, self.pred_len * len(taus))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        raw = self.output_head(self.features(past, future))
        raw = raw.view(past.shape[0], self.pred_len, len(self.quantiles))
        return torch.sort(raw, dim=-1).values


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
    output_skip: str = "none",
    target_index: int = 0,
    architecture: str = "aedl",
) -> nn.Module:
    if architecture == "lstm":
        if output_skip != "none":
            raise ValueError("The plain LSTM baseline has no output skip.")
        plain = dict(
            n_past_features=n_past_features,
            future_dim=future_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
        )
        if model_name == "deterministic":
            return DeterministicLSTM(**plain)
        if model_name == "gaussian":
            return GaussianLSTM(**plain)
        if model_name == "quantile":
            return QuantileLSTM(quantiles=quantiles, **plain)
        raise ValueError(f"Unknown model {model_name!r}.")
    if architecture != "aedl":
        raise ValueError(f"Unknown architecture {architecture!r}.")
    common = dict(
        n_past_features=n_past_features,
        future_dim=future_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        pred_len=pred_len,
        output_skip=output_skip,
        target_index=target_index,
    )
    if model_name == "deterministic":
        return DeterministicAEDL(**common)
    if model_name == "gaussian":
        return GaussianAEDL(**common)
    if model_name == "quantile":
        return QuantileAEDL(quantiles=quantiles, **common)
    raise ValueError(f"Unknown model {model_name!r}.")
