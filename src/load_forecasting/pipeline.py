from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]


WEATHER_CASES = ("historical", "observed_future", "api_forecast")
CALENDAR_COLS = ("h_sin", "h_cos", "dow_sin", "dow_cos", "moy_sin", "moy_cos")
MAX_API_FORECAST_LEAD = 24


@dataclass(frozen=True)
class DataConfig:
    # Public interface: one hourly dataframe containing load and weather.
    data_path: Path | None = ROOT / "data" / "data_Ulm.pkl"
    weather_columns: tuple[str, ...] = ("temperature",)
    past_covariates: tuple[str, ...] = ()
    selected_weather_cases: tuple[str, ...] = ("observed_future",)
    target_col: str = "total_consumption"
    target_normalization_col: str | None = None
    start_date: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    test_fraction: float = 0.10
    val_fraction: float = 0.10
    past_len: int = 48
    pred_len: int = 24
    batch_size: int = 256

    # Legacy private-data fields. They are intentionally absent from the public
    # CLI but allow existing paper checkpoints to be reconstructed.
    consumption_path: Path | None = None
    weather_path: Path | None = None
    align_to_api_start: bool = False
    common_windows: bool = False
    origin_stride: int = 1
    interpolate_limit: int = 2

    # Decoder inputs: calendar features of the target hours, and load and
    # weather 24 h before each target hour (both known at the forecast origin).
    decoder_calendar: bool = True
    decoder_lag24: bool = True
    # Calendar features in local time (e.g. "Europe/Berlin") instead of UTC.
    calendar_timezone: str | None = None
    # Date-based validation start (paper folds); default: last val_fraction.
    validation_start: str | None = None
    # Version A: divide the load of each window by the mean load of its history.
    window_load_scaling: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_size: int = 96
    num_layers: int = 1
    attn_heads: int = 8
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 50
    es_patience: int = 20
    gaussian_warmup_epochs: int = 10
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    quantile_width_weight: float = 5e-4
    quantile_median_weight: float = 0.5
    output_skip: str = "none"  # "none" or "linear" (version B)
    architecture: str = "aedl"  # "aedl" or "lstm" (plain LSTM baseline)
    seed: int = 42


@dataclass
class SplitOrigins:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: pd.Timestamp
    effective_start: pd.Timestamp
    api_start: pd.Timestamp | None


@dataclass
class PreparedCase:
    weather_case: str
    dataframe: pd.DataFrame
    past_cols: list[str]
    future_cols: list[str]
    future_cols_by_horizon: list[list[str]]
    origins: SplitOrigins
    scaler_past: StandardScaler
    scaler_future: StandardScaler | None
    scaler_target: StandardScaler
    target_normalization_col: str | None
    train_dataset: "ForecastWindowDataset"
    validation_dataset: "ForecastWindowDataset"
    test_dataset: "ForecastWindowDataset"

    def loaders(self, *, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.train_dataset.batch_size,
                shuffle=True,
                generator=generator,
            ),
            DataLoader(
                self.validation_dataset,
                batch_size=self.validation_dataset.batch_size,
                shuffle=False,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.test_dataset.batch_size,
                shuffle=False,
            ),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def future_columns(weather_case: str, weather_columns: Sequence[str]) -> list[str]:
    if weather_case == "historical":
        return []
    if weather_case == "observed_future":
        return list(weather_columns)
    if weather_case == "api_forecast":
        return [f"{column}_forecast" for column in weather_columns]
    raise ValueError(f"Unknown weather case {weather_case!r}; choose from {WEATHER_CASES}.")


def api_forecast_columns_for_lead(
    lead_hours: int, weather_columns: Sequence[str]
) -> list[str]:
    if not 0 <= int(lead_hours) <= MAX_API_FORECAST_LEAD:
        raise ValueError(
            f"API forecast lead must be between 0 and {MAX_API_FORECAST_LEAD} hours."
        )
    return [
        f"{column}_forecast_lead_{int(lead_hours):02d}h"
        for column in weather_columns
    ]


def future_columns_by_horizon(
    weather_case: str,
    pred_len: int,
    weather_columns: Sequence[str],
) -> list[list[str]]:
    """Return physical dataframe columns for each decoder step.

    The API case uses one coherent forecast issued at the load forecast origin:
    decoder step 0 selects lead 0, ..., step 23 selects lead 23.
    """

    if pred_len < 1:
        raise ValueError("Prediction length must be positive.")
    if weather_case == "historical":
        return [[] for _ in range(pred_len)]
    if weather_case == "observed_future":
        return [list(weather_columns) for _ in range(pred_len)]
    if weather_case == "api_forecast":
        if pred_len > MAX_API_FORECAST_LEAD:
            raise ValueError(
                f"API trajectories support at most {MAX_API_FORECAST_LEAD} decoder steps."
            )
        return [
            api_forecast_columns_for_lead(step, weather_columns)
            for step in range(pred_len)
        ]
    raise ValueError(f"Unknown weather case {weather_case!r}; choose from {WEATHER_CASES}.")


def add_calendar_features(df: pd.DataFrame, timezone: str | None = None) -> pd.DataFrame:
    """Add cyclic calendar features; the index is naive UTC."""

    out = df.copy()
    dt = out.index
    if timezone:
        dt = dt.tz_localize("UTC").tz_convert(timezone)
    hour = dt.hour + dt.minute / 60.0
    out["h_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["h_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    dow = dt.dayofweek
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    month = dt.month - 1
    out["moy_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["moy_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def load_complete_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """Load data on a complete hourly index without dropping missing rows.

    Missing values remain explicit. Window selection later rejects any origin whose
    48-hour history or 24-hour target/future-covariate path contains a missing row.
    """

    if cfg.data_path is not None:
        return _load_combined_dataframe(cfg)
    return _load_legacy_dataframe(cfg)


def _normalise_hourly_index(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_localize(None)
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps found in {path}.")
    if out.empty:
        raise ValueError(f"No rows found in {path}.")
    complete_index = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(complete_index)
    out.index.name = "timestamp"
    return out


def _load_combined_dataframe(cfg: DataConfig) -> pd.DataFrame:
    path = Path(cfg.data_path)
    if not path.exists():
        hint = (
            " Run `python src/scripts/prepare_ulm_data.py` first."
            if path.name == "data_Ulm.pkl"
            else ""
        )
        raise FileNotFoundError(f"Combined data file not found: {path}.{hint}")
    df = _normalise_hourly_index(pd.read_pickle(path), path)
    required = [cfg.target_col, *cfg.weather_columns, *cfg.past_covariates]
    if "api_forecast" in cfg.selected_weather_cases:
        required.extend(
            column
            for columns in future_columns_by_horizon(
                "api_forecast", cfg.pred_len, cfg.weather_columns
            )
            for column in columns
        )
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in df.columns]
    if missing:
        case_hint = (
            " API lead columns are needed only for --weather-cases api_forecast."
            if "api_forecast" in cfg.selected_weather_cases
            else ""
        )
        raise KeyError(f"Missing columns in {path}: {missing}.{case_hint}")

    # Short gaps in observed weather may be interpolated. Load and API issue/lead
    # columns are never interpolated, so invalid forecast windows remain explicit.
    if cfg.interpolate_limit:
        df.loc[:, list(cfg.weather_columns)] = df[list(cfg.weather_columns)].interpolate(
            method="time",
            limit=cfg.interpolate_limit,
            limit_area="inside",
        )
    keep = [*required, *[column for column in df.columns if column in CALENDAR_COLS]]
    return add_calendar_features(df[list(dict.fromkeys(keep))], cfg.calendar_timezone)


def _load_legacy_dataframe(cfg: DataConfig) -> pd.DataFrame:
    # Only the private paper protocol uses separate weather and consumption
    # files. Portable combined-data protocols do not require this module.
    from data_processing.weather_processing import load_weather_dataframe

    if cfg.consumption_path is None or cfg.weather_path is None:
        raise ValueError(
            "Legacy loading requires both consumption_path and weather_path."
        )
    consumption = pd.read_pickle(cfg.consumption_path).copy()
    consumption.index = pd.to_datetime(consumption.index)
    consumption = (
        consumption.sort_index()
        .resample("60min", label="left", closed="left")
        # The source is quarter-hourly. Requiring all four samples prevents a
        # partially observed hour from being interpreted as a low-load hour.
        .sum(min_count=4)
        / 4.0
    )
    consumption.index = pd.to_datetime(consumption.index).tz_localize(None)
    if cfg.target_col not in consumption.columns:
        raise KeyError(f"Target {cfg.target_col!r} not found in {cfg.consumption_path}.")
    consumption = consumption[[cfg.target_col]]

    api_source_cols = [
        column
        for step_columns in future_columns_by_horizon(
            "api_forecast", cfg.pred_len, cfg.weather_columns
        )
        for column in step_columns
    ]
    observed_weather = load_weather_dataframe(
        cfg.weather_path,
        requested_cols=cfg.weather_columns,
        interpolate_limit=cfg.interpolate_limit,
    )
    api_weather = load_weather_dataframe(
        cfg.weather_path,
        requested_cols=api_source_cols,
        # Interpolating a fixed-lead column across issue times can use a forecast
        # retrieved after the load-forecast origin. Incomplete API trajectories
        # are therefore left missing and rejected by valid_origins().
        interpolate_limit=None,
    )
    weather = observed_weather.join(api_weather, how="outer")
    weather.index = pd.to_datetime(weather.index).tz_localize(None)

    start = min(consumption.index.min(), weather.index.min())
    end = max(consumption.index.max(), weather.index.max())
    complete_index = pd.date_range(start, end, freq="h")
    df = consumption.reindex(complete_index).join(weather.reindex(complete_index), how="left")
    df.index.name = "timestamp"
    return add_calendar_features(df, cfg.calendar_timezone)


def past_feature_columns(
    target_col: str,
    weather_columns: Sequence[str],
    past_covariates: Sequence[str] = (),
) -> list[str]:
    return [target_col, *weather_columns, *past_covariates, *CALENDAR_COLS]


def _window_start_validity(row_valid: np.ndarray, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("Window length must be positive.")
    if len(row_valid) < length:
        return np.zeros(0, dtype=bool)
    counts = np.convolve(row_valid.astype(np.int16), np.ones(length, dtype=np.int16), mode="valid")
    return counts == length


def valid_origins(
    df: pd.DataFrame,
    *,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    past_cols: Sequence[str],
    future_required_cols: Sequence[str],
    future_required_cols_by_horizon: Sequence[Sequence[str]] | None = None,
    past_len: int,
    pred_len: int,
    origin_stride: int = 1,
) -> np.ndarray:
    """Return integer forecast-origin positions for fully contiguous valid windows.

    ``future_required_cols`` must be available at every decoder step. Optional
    horizon-specific columns support coherent forecast trajectories, where each
    decoder step selects a different lead from the same forecast issue.
    """

    required_past = list(dict.fromkeys(past_cols))
    required_future = list(dict.fromkeys(future_required_cols))
    past_rows_valid = df[required_past].notna().all(axis=1).to_numpy()
    future_rows_valid = df[required_future].notna().all(axis=1).to_numpy()
    past_start_ok = _window_start_validity(past_rows_valid, past_len)
    future_start_ok = _window_start_validity(future_rows_valid, pred_len)

    candidate_positions = np.arange(past_len, len(df) - pred_len + 1, dtype=np.int64)
    candidate_times = df.index[candidate_positions]
    in_target_span = (candidate_times >= target_start) & (
        df.index[candidate_positions + pred_len - 1] <= target_end
    )
    past_ok = past_start_ok[candidate_positions - past_len]
    future_ok = future_start_ok[candidate_positions]
    if future_required_cols_by_horizon is not None:
        if len(future_required_cols_by_horizon) != pred_len:
            raise ValueError(
                "Horizon-specific future columns must have one entry per decoder step."
            )
        for step, step_columns in enumerate(future_required_cols_by_horizon):
            required_at_step = list(dict.fromkeys(step_columns))
            if required_at_step:
                row_ok = df[required_at_step].notna().all(axis=1).to_numpy()
                future_ok &= row_ok[candidate_positions + step]
    origins = candidate_positions[in_target_span & past_ok & future_ok]

    if origin_stride > 1 and len(origins):
        anchor = origins[0]
        origins = origins[(origins - anchor) % origin_stride == 0]
    return origins


def _first_api_origin(
    df: pd.DataFrame,
    *,
    past_cols: Sequence[str],
    target_col: str,
    past_len: int,
    pred_len: int,
    weather_columns: Sequence[str],
) -> pd.Timestamp:
    origins = valid_origins(
        df,
        target_start=df.index.min(),
        target_end=df.index.max(),
        past_cols=past_cols,
        future_required_cols=[target_col, *weather_columns],
        future_required_cols_by_horizon=future_columns_by_horizon(
            "api_forecast", pred_len, weather_columns
        ),
        past_len=past_len,
        pred_len=pred_len,
    )
    if not len(origins):
        raise ValueError("No complete API forecast window is available in the supplied data.")
    return pd.Timestamp(df.index[origins[0]])


def _as_naive_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _future_requirements(
    cfg: DataConfig,
    weather_case: str,
) -> tuple[list[str], list[list[str]] | None]:
    cases = cfg.selected_weather_cases if cfg.common_windows else (weather_case,)
    required = [cfg.target_col]
    if cfg.target_normalization_col:
        required.append(cfg.target_normalization_col)
    if "observed_future" in cases:
        required.extend(cfg.weather_columns)
    by_horizon = None
    if "api_forecast" in cases:
        by_horizon = future_columns_by_horizon(
            "api_forecast", cfg.pred_len, cfg.weather_columns
        )
    return list(dict.fromkeys(required)), by_horizon


def _build_portable_split_origins(
    df: pd.DataFrame,
    *,
    cfg: DataConfig,
    past_cols: Sequence[str],
    weather_case: str,
) -> SplitOrigins:
    if not 0 < cfg.val_fraction < 1 or not 0 < cfg.test_fraction < 1:
        raise ValueError("Validation and test fractions must be between zero and one.")
    if cfg.val_fraction + cfg.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one.")

    effective_start = (
        _as_naive_timestamp(cfg.start_date) if cfg.start_date else df.index.min()
    )
    target_end = _as_naive_timestamp(cfg.test_end) if cfg.test_end else df.index.max()
    future_required, future_by_horizon = _future_requirements(cfg, weather_case)
    pool = valid_origins(
        df,
        target_start=effective_start,
        target_end=target_end,
        past_cols=past_cols,
        future_required_cols=future_required,
        future_required_cols_by_horizon=future_by_horizon,
        past_len=cfg.past_len,
        pred_len=cfg.pred_len,
    )
    if len(pool) < 3:
        raise ValueError("Not enough complete windows to create train/validation/test splits.")

    if cfg.train_end:
        if not cfg.test_start:
            raise ValueError("A fixed train_end requires test_start to define validation.")
        train_end = _as_naive_timestamp(cfg.train_end)
        test_start = _as_naive_timestamp(cfg.test_start)
        validation_start = train_end + pd.Timedelta(hours=1)
        if validation_start >= test_start:
            raise ValueError("train_end must be earlier than test_start.")
        train = pool[df.index[pool + cfg.pred_len - 1] <= train_end]
        validation = pool[
            (df.index[pool] >= validation_start)
            & (df.index[pool + cfg.pred_len - 1] < test_start)
        ]
        test = pool[df.index[pool] >= test_start]
    elif cfg.test_start:
        test_start = _as_naive_timestamp(cfg.test_start)
        test = pool[df.index[pool] >= test_start]
        pre_test = pool[df.index[pool + cfg.pred_len - 1] < test_start]
        if len(pre_test) < 2:
            raise ValueError("Not enough pre-test windows to create train and validation splits.")
        validation_share = cfg.val_fraction / (1.0 - cfg.test_fraction)
        validation_idx = int((1.0 - validation_share) * len(pre_test))
        validation_idx = min(max(validation_idx, 1), len(pre_test) - 1)
        validation_start = pd.Timestamp(df.index[pre_test[validation_idx]])
    else:
        test_idx = int((1.0 - cfg.test_fraction) * len(pool))
        test_idx = min(max(test_idx, 2), len(pool) - 1)
        test_start = pd.Timestamp(df.index[pool[test_idx]])
        test = pool[pool >= pool[test_idx]]
        pre_test = pool[df.index[pool + cfg.pred_len - 1] < test_start]
        validation_idx = int(
            (1.0 - cfg.val_fraction - cfg.test_fraction) * len(pool)
        )
        validation_idx = min(max(validation_idx, 1), len(pre_test) - 1)
        validation_start = pd.Timestamp(df.index[pool[validation_idx]])

    if not cfg.train_end:
        train = pre_test[df.index[pre_test + cfg.pred_len - 1] < validation_start]
        validation = pre_test[df.index[pre_test] >= validation_start]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(
            "Empty chronological split after excluding boundary-crossing windows: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

    selected_cases = cfg.selected_weather_cases if cfg.common_windows else (weather_case,)
    api_start = None
    if "api_forecast" in selected_cases:
        api_start = _first_api_origin(
            df,
            past_cols=past_cols,
            target_col=cfg.target_col,
            past_len=cfg.past_len,
            pred_len=cfg.pred_len,
            weather_columns=cfg.weather_columns,
        )
    return SplitOrigins(
        train=train,
        validation=validation,
        test=test,
        validation_start=validation_start,
        effective_start=pd.Timestamp(effective_start),
        api_start=api_start,
    )


def build_split_origins(
    df: pd.DataFrame,
    *,
    cfg: DataConfig,
    past_cols: Sequence[str],
    weather_case: str,
) -> SplitOrigins:
    if cfg.data_path is not None:
        return _build_portable_split_origins(
            df,
            cfg=cfg,
            past_cols=past_cols,
            weather_case=weather_case,
        )
    return _build_legacy_split_origins(
        df,
        cfg=cfg,
        past_cols=past_cols,
        weather_case=weather_case,
    )


def _build_legacy_split_origins(
    df: pd.DataFrame,
    *,
    cfg: DataConfig,
    past_cols: Sequence[str],
    weather_case: str,
) -> SplitOrigins:
    api_start = _first_api_origin(
        df,
        past_cols=past_cols,
        target_col=cfg.target_col,
        past_len=cfg.past_len,
        pred_len=cfg.pred_len,
        weather_columns=cfg.weather_columns,
    )
    requested_start = (
        pd.Timestamp(cfg.start_date) if cfg.start_date else df.index.min()
    )
    effective_start = (
        max(requested_start, api_start)
        if cfg.align_to_api_start
        else requested_start
    )

    case_future = future_columns(weather_case, cfg.weather_columns)
    if cfg.common_windows:
        future_required = [cfg.target_col, *cfg.weather_columns]
        future_required_by_horizon = future_columns_by_horizon(
            "api_forecast", cfg.pred_len, cfg.weather_columns
        )
    elif weather_case == "api_forecast":
        future_required = [cfg.target_col]
        future_required_by_horizon = future_columns_by_horizon(
            weather_case, cfg.pred_len, cfg.weather_columns
        )
    else:
        future_required = [cfg.target_col, *case_future]
        future_required_by_horizon = None

    train_pool = valid_origins(
        df,
        target_start=effective_start,
        target_end=pd.Timestamp(cfg.train_end),
        past_cols=past_cols,
        future_required_cols=future_required,
        future_required_cols_by_horizon=future_required_by_horizon,
        past_len=cfg.past_len,
        pred_len=cfg.pred_len,
        origin_stride=cfg.origin_stride,
    )
    if len(train_pool) < 2:
        raise ValueError(
            "Not enough complete training windows after applying date and gap filters."
        )

    if cfg.validation_start:
        validation_start = pd.Timestamp(cfg.validation_start)
        if not (df.index[train_pool] >= validation_start).any():
            raise ValueError("No validation origins after validation_start.")
    else:
        split_idx = int((1.0 - cfg.val_fraction) * len(train_pool))
        split_idx = min(max(split_idx, 1), len(train_pool) - 1)
        validation_start = pd.Timestamp(df.index[train_pool[split_idx]])

    # Prevent overlapping train and validation targets. Past context before the
    # validation boundary remains allowed, as it would be at forecast time.
    train = train_pool[df.index[train_pool + cfg.pred_len - 1] < validation_start]
    validation = train_pool[df.index[train_pool] >= validation_start]
    test = valid_origins(
        df,
        target_start=pd.Timestamp(cfg.test_start),
        target_end=pd.Timestamp(cfg.test_end),
        past_cols=past_cols,
        future_required_cols=future_required,
        future_required_cols_by_horizon=future_required_by_horizon,
        past_len=cfg.past_len,
        pred_len=cfg.pred_len,
        origin_stride=cfg.origin_stride,
    )
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(
            "Empty split after filtering: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    return SplitOrigins(
        train=train,
        validation=validation,
        test=test,
        validation_start=validation_start,
        effective_start=effective_start,
        api_start=api_start,
    )


def _rows_used_by_origins(origins: np.ndarray, past_len: int, pred_len: int) -> np.ndarray:
    used: set[int] = set()
    for origin in origins.tolist():
        used.update(range(origin - past_len, origin + pred_len))
    return np.asarray(sorted(used), dtype=np.int64)


def _future_values_for_origins(
    df: pd.DataFrame,
    origins: np.ndarray,
    columns_by_horizon: Sequence[Sequence[str]],
) -> np.ndarray:
    """Materialize origin-specific future covariates as [origin, horizon, feature]."""

    pred_len = len(columns_by_horizon)
    feature_dims = {len(columns) for columns in columns_by_horizon}
    if len(feature_dims) != 1:
        raise ValueError("Every decoder step must use the same number of future features.")
    feature_dim = feature_dims.pop()
    values = np.empty((len(origins), pred_len, feature_dim), dtype=np.float32)
    for step, columns in enumerate(columns_by_horizon):
        if feature_dim:
            values[:, step, :] = df.iloc[origins + step][list(columns)].to_numpy(
                dtype=np.float32
            )
    return values


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        origins: np.ndarray,
        past_cols: Sequence[str],
        future_cols: Sequence[str],
        future_cols_by_horizon: Sequence[Sequence[str]],
        target_col: str,
        scaler_past: StandardScaler,
        scaler_future: StandardScaler | None,
        scaler_target: StandardScaler,
        target_normalization_col: str | None,
        past_len: int,
        pred_len: int,
        batch_size: int,
        decoder_calendar: bool = False,
        decoder_lag24_cols: Sequence[str] = (),
        window_load_scaling: bool = False,
    ):
        self.origins = np.asarray(origins, dtype=np.int64)
        self.window_load_scaling = bool(window_load_scaling)
        self.past_len = int(past_len)
        self.pred_len = int(pred_len)
        self.batch_size = int(batch_size)
        self.target_index = list(past_cols).index(target_col)

        past_raw = df[list(past_cols)].to_numpy(dtype=np.float32)
        target_raw = df[[target_col]].to_numpy(dtype=np.float32)
        if target_normalization_col:
            target_multiplier = df[[target_normalization_col]].to_numpy(dtype=np.float32)
            if np.any(target_multiplier[np.isfinite(target_multiplier)] <= 0):
                raise ValueError(
                    f"Target normalizer {target_normalization_col!r} must be positive."
                )
            target_raw = target_raw / target_multiplier
        else:
            target_multiplier = np.ones_like(target_raw)
        past_raw[:, self.target_index] = target_raw.reshape(-1)
        past = scaler_past.transform(past_raw)
        target = scaler_target.transform(target_raw).reshape(-1)
        self.past = past.astype(np.float32)
        self.target = target.astype(np.float32)
        self.target_multiplier = target_multiplier.reshape(-1).astype(np.float32)

        target_rows = self.origins[:, None] + np.arange(self.pred_len)[None, :]
        if self.window_load_scaling:
            # Per-origin windows: load divided by the mean load of the history.
            past_rows = self.origins[:, None] + np.arange(-self.past_len, 0)[None, :]
            load = target_raw.reshape(-1)
            level = load[past_rows].mean(axis=1)
            if not np.all(np.isfinite(level)) or np.any(level <= 0):
                raise ValueError("Window load scaling needs a positive, complete load history.")
            self.past_windows = self.past[past_rows].copy()
            self.past_windows[..., self.target_index] = scaler_target.transform(
                (load[past_rows] / level[:, None]).reshape(-1, 1)
            ).reshape(past_rows.shape)
            self.target_windows = scaler_target.transform(
                (load[target_rows] / level[:, None]).reshape(-1, 1)
            ).reshape(target_rows.shape).astype(np.float32)
            self.multiplier_windows = np.repeat(level[:, None], self.pred_len, axis=1).astype(
                np.float32
            )

        if len(future_cols_by_horizon) != self.pred_len:
            raise ValueError("Future source columns must have one entry per decoder step.")
        if future_cols:
            if scaler_future is None:
                raise ValueError("A future scaler is required when future columns are used.")
            if any(len(columns) != len(future_cols) for columns in future_cols_by_horizon):
                raise ValueError(
                    "Each decoder step must match the logical future feature dimension."
                )
            future_raw = _future_values_for_origins(
                df, self.origins, future_cols_by_horizon
            )
            self.future = scaler_future.transform(
                future_raw.reshape(-1, len(future_cols))
            ).reshape(future_raw.shape).astype(np.float32)
        else:
            self.future = np.zeros((len(self.origins), self.pred_len, 0), dtype=np.float32)

        # Extra decoder inputs reuse the scaled encoder features.
        extras = []
        if decoder_calendar:
            calendar_index = [list(past_cols).index(column) for column in CALENDAR_COLS]
            extras.append(self.past[target_rows][..., calendar_index])
        if decoder_lag24_cols:
            lag_index = [list(past_cols).index(column) for column in decoder_lag24_cols]
            if self.window_load_scaling:
                start = self.past_len - 24
                extras.append(
                    self.past_windows[:, start : start + self.pred_len][..., lag_index]
                )
            else:
                extras.append(self.past[target_rows - 24][..., lag_index])
        if extras:
            self.future = np.concatenate([self.future, *extras], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int):
        if self.window_load_scaling:
            return (
                torch.from_numpy(self.past_windows[item]),
                torch.from_numpy(self.future[item]),
                torch.from_numpy(self.target_windows[item]),
                torch.from_numpy(self.multiplier_windows[item]),
            )
        origin = int(self.origins[item])
        return (
            torch.from_numpy(self.past[origin - self.past_len : origin]),
            torch.from_numpy(self.future[item]),
            torch.from_numpy(self.target[origin : origin + self.pred_len]),
            torch.from_numpy(
                self.target_multiplier[origin : origin + self.pred_len]
            ),
        )


def prepare_case(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    weather_case: str,
) -> PreparedCase:
    past_cols = past_feature_columns(
        data_cfg.target_col,
        data_cfg.weather_columns,
        data_cfg.past_covariates,
    )
    future_cols = future_columns(weather_case, data_cfg.weather_columns)
    future_cols_by_horizon = future_columns_by_horizon(
        weather_case, data_cfg.pred_len, data_cfg.weather_columns
    )
    origins = build_split_origins(
        df,
        cfg=data_cfg,
        past_cols=past_cols,
        weather_case=weather_case,
    )

    train_rows = _rows_used_by_origins(origins.train, data_cfg.past_len, data_cfg.pred_len)
    target_raw = df[[data_cfg.target_col]].to_numpy(dtype=np.float32)
    if data_cfg.target_normalization_col:
        divisor = df[[data_cfg.target_normalization_col]].to_numpy(dtype=np.float32)
        if np.any(divisor[np.isfinite(divisor)] <= 0):
            raise ValueError(
                f"Target normalizer {data_cfg.target_normalization_col!r} must be positive."
            )
        target_raw = target_raw / divisor
    past_raw = df[past_cols].to_numpy(dtype=np.float32)
    past_raw[:, past_cols.index(data_cfg.target_col)] = target_raw.reshape(-1)
    past_fit_rows = train_rows[np.isfinite(past_raw[train_rows]).all(axis=1)]
    target_fit_rows = train_rows[np.isfinite(target_raw[train_rows]).reshape(-1)]
    scaler_past = StandardScaler().fit(past_raw[past_fit_rows])
    scaler_target = StandardScaler().fit(target_raw[target_fit_rows])
    if data_cfg.window_load_scaling:
        if data_cfg.target_normalization_col:
            raise ValueError("Version A cannot be combined with per-building normalization. Use version B.")
        # Target scaler on the load ratios of the training windows.
        rows = origins.train[:, None] + np.arange(-data_cfg.past_len, data_cfg.pred_len)[None, :]
        load = target_raw.reshape(-1)[rows]
        ratios = load / load[:, : data_cfg.past_len].mean(axis=1, keepdims=True)
        scaler_target = StandardScaler().fit(ratios.reshape(-1, 1))

    scaler_future: StandardScaler | None = None
    if future_cols:
        if weather_case == "api_forecast":
            future_fit_values = _future_values_for_origins(
                df, origins.train, future_cols_by_horizon
            ).reshape(-1, len(future_cols))
        else:
            future_fit_rows = train_rows[
                df.iloc[train_rows][future_cols].notna().all(axis=1).to_numpy()
            ]
            future_fit_values = df.iloc[future_fit_rows][future_cols].to_numpy(
                dtype=np.float32
            )
        scaler_future = StandardScaler().fit(future_fit_values)

    dataset_kwargs = dict(
        df=df,
        past_cols=past_cols,
        future_cols=future_cols,
        future_cols_by_horizon=future_cols_by_horizon,
        target_col=data_cfg.target_col,
        scaler_past=scaler_past,
        scaler_future=scaler_future,
        scaler_target=scaler_target,
        target_normalization_col=data_cfg.target_normalization_col,
        past_len=data_cfg.past_len,
        pred_len=data_cfg.pred_len,
        batch_size=data_cfg.batch_size,
        decoder_calendar=data_cfg.decoder_calendar,
        decoder_lag24_cols=(
            (data_cfg.target_col, *data_cfg.weather_columns)
            if data_cfg.decoder_lag24
            else ()
        ),
        window_load_scaling=data_cfg.window_load_scaling,
    )
    return PreparedCase(
        weather_case=weather_case,
        dataframe=df,
        past_cols=past_cols,
        future_cols=future_cols,
        future_cols_by_horizon=future_cols_by_horizon,
        origins=origins,
        scaler_past=scaler_past,
        scaler_future=scaler_future,
        scaler_target=scaler_target,
        target_normalization_col=data_cfg.target_normalization_col,
        train_dataset=ForecastWindowDataset(origins=origins.train, **dataset_kwargs),
        validation_dataset=ForecastWindowDataset(origins=origins.validation, **dataset_kwargs),
        test_dataset=ForecastWindowDataset(origins=origins.test, **dataset_kwargs),
    )


def assert_common_origins(cases: Iterable[PreparedCase]) -> None:
    cases = list(cases)
    if not cases:
        return
    reference = cases[0]
    for case in cases[1:]:
        for split in ("train", "validation", "test"):
            left = getattr(reference.origins, split)
            right = getattr(case.origins, split)
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Common-window mode failed: {reference.weather_case} and "
                    f"{case.weather_case} have different {split} origins."
                )


def inverse_target(
    values: np.ndarray,
    scaler: StandardScaler,
    multiplier: np.ndarray | float = 1.0,
) -> np.ndarray:
    shape = values.shape
    restored = scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)
    return restored * np.asarray(multiplier)


@torch.no_grad()
def collect_predictions(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    case: PreparedCase,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run a loader and convert every target-related output back to kW."""

    targets: list[np.ndarray] = []
    primary: list[np.ndarray] = []
    secondary: list[np.ndarray] = []
    model.eval()

    multipliers: list[np.ndarray] = []
    for past, future, target, multiplier in loader:
        past, future = past.to(device), future.to(device)
        if model_name == "deterministic":
            primary.append(model(past, future).cpu().numpy())
        elif model_name == "gaussian":
            mean, log_scale = model(past, future)
            primary.append(mean.cpu().numpy())
            secondary.append(log_scale.cpu().numpy())
        elif model_name == "quantile":
            primary.append(model(past, future).cpu().numpy())
        else:
            raise ValueError(f"Unknown model {model_name!r}.")
        targets.append(target.numpy())
        multipliers.append(multiplier.numpy())

    target_scaled = np.concatenate(targets)
    primary_scaled = np.concatenate(primary)
    target_multiplier = np.concatenate(multipliers)
    result = {
        "target_kw": inverse_target(
            target_scaled, case.scaler_target, target_multiplier
        )
    }
    if model_name == "deterministic":
        result["prediction_kw"] = inverse_target(
            primary_scaled, case.scaler_target, target_multiplier
        )
    elif model_name == "gaussian":
        result["mean_kw"] = inverse_target(
            primary_scaled, case.scaler_target, target_multiplier
        )
        result["scale_kw"] = np.exp(np.concatenate(secondary)) * float(
            case.scaler_target.scale_[0]
        ) * target_multiplier
    else:
        quantile_multiplier = np.repeat(
            target_multiplier[..., None], primary_scaled.shape[-1], axis=-1
        )
        result["quantiles_kw"] = inverse_target(
            primary_scaled.reshape(-1, 1),
            case.scaler_target,
            quantile_multiplier.reshape(-1, 1),
        ).reshape(primary_scaled.shape)
    return result


def split_summary(case: PreparedCase) -> dict[str, object]:
    df = case.dataframe
    summary: dict[str, object] = {
        "weather_case": case.weather_case,
        "past_features": len(case.past_cols),
        "future_features": list(case.future_cols),
        "target_normalization": case.target_normalization_col or "none",
        "api_availability_start": (
            str(case.origins.api_start) if case.origins.api_start is not None else None
        ),
        "effective_start": str(case.origins.effective_start),
        "validation_start": str(case.origins.validation_start),
    }
    for split in ("train", "validation", "test"):
        origins = getattr(case.origins, split)
        summary[f"{split}_origins"] = int(len(origins))
        summary[f"{split}_first_origin"] = str(df.index[origins[0]])
        summary[f"{split}_last_origin"] = str(df.index[origins[-1]])
    return summary


def serializable_scaler(scaler: StandardScaler | None) -> dict[str, object] | None:
    if scaler is None:
        return None
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }


def checkpoint_metadata(
    case: PreparedCase,
    *,
    data_cfg: DataConfig,
    experiment_cfg: ExperimentConfig,
) -> dict[str, object]:
    return {
        "weather_case": case.weather_case,
        "data_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(data_cfg).items()
        },
        "experiment_config": asdict(experiment_cfg),
        "past_columns": case.past_cols,
        "future_columns": case.future_cols,
        "future_source_columns_by_horizon": case.future_cols_by_horizon,
        "target_column": data_cfg.target_col,
        "split_summary": split_summary(case),
        "scalers": {
            "past": serializable_scaler(case.scaler_past),
            "future": serializable_scaler(case.scaler_future),
            "target": serializable_scaler(case.scaler_target),
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
