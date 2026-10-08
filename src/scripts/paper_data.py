"""Numbers and figures of the paper's data section (Section 3).

* Missing data: share of the expected load from meters without a reading
  (each missing meter weighted by its monthly mean load) and share of training
  hours without measured weather.
* Pearson correlations of load and weather on the training data of fold 1
  (pairwise complete hours): pooled, per calendar year, and lagged by 0-48 h.
* Fig. 3: aggregated load with connected consumers and consumers with a reading.
* Fig. 4: two-week summer and winter profiles of load and weather (local time).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from load_forecasting.pipeline import _first_api_origin, _load_legacy_dataframe, past_feature_columns
from load_forecasting.protocols import PAPER_FOLDS, get_protocol

METER_PATH = ROOT / "data" / "archive" / "260508_consumption_data_preprocessing.pkl"
FIG_DIR = ROOT / "Paper" / "paper_energy_informatics" / "figures"
TRAIN_PERIOD = ("2020-04-25", "2024-12-31 23:00")  # fold 1, from the first API origin
TEST_PERIOD = ("2025-10-13", "2026-04-30 23:00")  # both test periods
SEASON_PERIODS = (("2024-07-08", "2024-07-21"), ("2025-01-06", "2025-01-19"))


def missing_meter_share(meters: pd.DataFrame) -> pd.Series:
    """Hourly share of the expected load from connected meters without a reading."""

    hourly = meters.resample("60min").mean()
    first, last = hourly.apply(pd.Series.first_valid_index), hourly.apply(pd.Series.last_valid_index)
    index = hourly.index.values[:, None]
    connected = (index >= first.values[None, :].astype(index.dtype)) & (index <= last.values[None, :].astype(index.dtype))
    expected = (
        hourly.groupby([hourly.index.year, hourly.index.month]).transform("mean")
        .fillna(hourly.groupby(hourly.index.month).transform("mean"))
        .fillna(hourly.mean())
    )
    missing = connected & hourly.isna().to_numpy()
    return expected.where(missing).sum(axis=1) / expected.where(connected).sum(axis=1)


def report_missing_data(df: pd.DataFrame, meters: pd.DataFrame) -> None:
    share = missing_meter_share(meters)
    for name, (start, end) in (("training", TRAIN_PERIOD), ("test", TEST_PERIOD)):
        print(f"Expected load from meters without a reading, {name}: {share.loc[start:end].mean():.1%}")
    print(f"Training hours without measured weather: {df.loc[slice(*TRAIN_PERIOD), 'temperature'].isna().mean():.1%}")


def report_correlations(df: pd.DataFrame, cfg) -> None:
    start = _first_api_origin(
        df, past_cols=past_feature_columns(cfg.target_col, cfg.weather_columns), target_col=cfg.target_col,
        past_len=cfg.past_len, pred_len=cfg.pred_len, weather_columns=cfg.weather_columns,
    ) - pd.Timedelta(hours=cfg.past_len)
    end = pd.Timestamp(PAPER_FOLDS[1]["validation_start"]) - pd.Timedelta(hours=1)
    train = df.loc[start:end]
    for column in cfg.weather_columns:
        pair = train[[cfg.target_col, column]].dropna()
        r = pearsonr(pair[column], pair[cfg.target_col])
        per_year = [pearsonr(*part[[column, cfg.target_col]].T.to_numpy()).statistic
                    for _, part in pair.groupby(pair.index.year)]
        lagged = [train[cfg.target_col].corr(train[column].shift(lag)) for lag in range(49)]
        best = int(np.argmax(np.abs(lagged)))
        print(f"{column}: n = {len(pair)}, r = {r.statistic:.2f} (p = {r.pvalue:.1e}), "
              f"per year {min(per_year):.2f} to {max(per_year):.2f}, "
              f"largest |r| at lag {best} h ({lagged[best]:.3f} vs. {lagged[0]:.3f} at lag 0)")


def figure_aggregated_load(target: pd.Series, meters: pd.DataFrame) -> None:
    load = target.loc[target.first_valid_index():target.last_valid_index()]
    first_valid = meters.apply(pd.Series.first_valid_index)
    days = pd.date_range(meters.index.min().floor("D"), meters.index.max().ceil("D"), freq="1D")
    connected = pd.Series([(first_valid <= day).sum() for day in days], index=days)
    reporting = meters.notna().sum(axis=1).resample("1D").mean()
    reporting = reporting[reporting.index < meters.index.max().floor("D")]  # complete days only

    fig, ax_load = plt.subplots(figsize=(7.2, 3.1), dpi=300)
    ax_consumer = ax_load.twinx()
    lines = ax_load.plot(load.index, load, color="#1f77b4", linewidth=0.4, alpha=0.85,
                         label="Aggregated thermal hourly load")
    lines += ax_consumer.step(connected.index, connected, where="post", color="#d62728", linewidth=0.75,
                              alpha=0.9, label="Connected consumers")
    lines += ax_consumer.plot(reporting.index, reporting, color="#f4a582", linewidth=0.6,
                              label="Consumers with a reading")
    ax_load.set_ylabel("Thermal load [kW]", color="#1f77b4")
    ax_consumer.set_ylabel("Number of consumers", color="#d62728")
    ax_load.tick_params(axis="y", labelcolor="#1f77b4")
    ax_consumer.tick_params(axis="y", labelcolor="#d62728")
    ax_load.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_load.xaxis.set_major_locator(mdates.YearLocator())
    ax_load.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_load.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax_load.set_xlim(load.index.min(), load.index.max())
    ax_load.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aggregated_load_revised.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_seasonal_profiles(df: pd.DataFrame, target_col: str) -> None:
    local = df.copy()
    local.index = local.index.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)
    colors = {"load": "#0072b2", "temperature": "#d55e00", "ghi": "#009e73"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    for ax, (start, end) in zip(axes, SEASON_PERIODS):
        part = local.loc[start:f"{end} 23:00"]
        load = part[target_col]
        print(f"Fig. 4, {start} to {end}: mean load {load.mean():.0f} kW, maximum {load.max():.0f} kW")
        lines = ax.plot(part.index, load, color=colors["load"], alpha=0.3, linewidth=0.7, label="Hourly load")
        lines += ax.plot(part.index, load.rolling(24, center=True).mean(), color=colors["load"], linewidth=1.8,
                         label="24 h mean load")
        ax.set_ylabel("Thermal load [kW]")
        ax.grid(True, alpha=0.3)
        ax_t = ax.twinx()
        lines += ax_t.plot(part.index, part["temperature"], color=colors["temperature"], linewidth=1.0,
                           label="Outdoor temperature")
        ax_t.set_ylabel("Temperature [°C]", color=colors["temperature"])
        ax_t.tick_params(axis="y", labelcolor=colors["temperature"])
        ax_g = ax.twinx()
        ax_g.spines["right"].set_position(("axes", 1.12))
        lines += ax_g.plot(part.index, part["ghi_backwards"], color=colors["ghi"], linewidth=0.8, label="GHI")
        ax_g.set_ylabel("GHI [W m$^{-2}$]", color=colors["ghi"])
        ax_g.tick_params(axis="y", labelcolor=colors["ghi"])
        ax.set_xlim(part.index.min(), part.index.max())
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].legend(lines, [line.get_label() for line in lines], loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summer_winter_two_weeks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = replace(get_protocol("paper").data, calendar_timezone="Europe/Berlin")
    df = _load_legacy_dataframe(cfg)
    meters = pd.read_pickle(METER_PATH)
    meters.index = pd.to_datetime(meters.index, utc=True).tz_localize(None)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": 0.7})
    report_missing_data(df, meters)
    report_correlations(df, cfg)
    figure_aggregated_load(df[cfg.target_col], meters)
    figure_seasonal_profiles(df, cfg.target_col)
    print(f"Saved Figs. 3 and 4 to {FIG_DIR}")


if __name__ == "__main__":
    main()
