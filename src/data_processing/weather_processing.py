from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_WEATHER_TIME_COLS = {"time", "time_stamp"}


def _normalize_weather_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def _collapse_raw_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    raw = weather_df.copy()
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw["time_stamp"] = pd.to_datetime(raw["time_stamp"], utc=True)
    raw["lead_hours"] = (
        (raw["time"] - raw["time_stamp"]).dt.total_seconds() / 3600
    ).round().astype(int)

    value_cols = [
        col for col in raw.columns if col not in {"time", "time_stamp", "lead_hours"}
    ]
    if not value_cols:
        raise ValueError("Raw weather data does not contain any value columns.")

    # Prefer the closest backcast/current value for each valid timestamp.
    nonneg = (
        raw[raw["lead_hours"] >= 0]
        .sort_values(["time_stamp", "lead_hours", "time"], ascending=[True, True, True])
        .drop_duplicates("time_stamp")
    )

    # Fall back to the shortest forecast if no backcast/current value exists.
    neg = (
        raw[raw["lead_hours"] < 0]
        .sort_values(["time_stamp", "lead_hours", "time"], ascending=[True, False, False])
        .drop_duplicates("time_stamp")
    )

    collapsed = (
        nonneg.set_index("time_stamp")[value_cols]
        .combine_first(neg.set_index("time_stamp")[value_cols])
        .sort_index()
    )
    collapsed.index.name = "timestamp"
    return collapsed


def prepare_weather_dataframe(
    weather_df: pd.DataFrame,
    requested_cols: Iterable[str],
    interpolate_limit: int | None = 2,
) -> pd.DataFrame:
    requested_cols = list(dict.fromkeys(requested_cols))
    if not requested_cols:
        raise ValueError("At least one weather column must be requested.")

    if RAW_WEATHER_TIME_COLS.issubset(weather_df.columns):
        weather_df = _collapse_raw_weather(weather_df)
    else:
        weather_df = _normalize_weather_index(weather_df)

    missing_cols = [col for col in requested_cols if col not in weather_df.columns]
    if missing_cols:
        raise KeyError(f"Missing weather columns: {missing_cols}")

    weather_df = weather_df[requested_cols].sort_index()
    full_index = pd.date_range(
        weather_df.index.min(),
        weather_df.index.max(),
        freq="60min",
        tz=weather_df.index.tz,
    )
    weather_df = weather_df.reindex(full_index)
    if interpolate_limit is not None:
        weather_df = weather_df.interpolate(limit=interpolate_limit)
    return weather_df


def load_weather_dataframe(
    weather_path: str | Path,
    requested_cols: Iterable[str],
    interpolate_limit: int | None = 2,
) -> pd.DataFrame:
    return prepare_weather_dataframe(
        pd.read_pickle(weather_path),
        requested_cols=requested_cols,
        interpolate_limit=interpolate_limit,
    )
