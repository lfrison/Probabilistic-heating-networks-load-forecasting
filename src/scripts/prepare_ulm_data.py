"""Convert the redistributable Ulm CSV to the combined model-data format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = 70_928
EXPECTED_START = pd.Timestamp("2014-09-02 00:00:00", tz="UTC")
EXPECTED_END = pd.Timestamp("2022-10-05 07:00:00", tz="UTC")


def repair_thousandfold(values: pd.Series, threshold: float) -> tuple[pd.Series, int]:
    """Divide isolated values outside a physical threshold by 1,000."""

    corrected = values.astype(float).copy()
    mask = corrected.abs() > threshold
    corrected.loc[mask] /= 1_000.0
    return corrected, int(mask.sum())


def convert_ulm_csv(input_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(input_path)
    source = source.drop(columns=["Unnamed: 0"], errors="ignore")
    required = {
        "MESS_DATUM",
        "Load_MW",
        "Temperature",
        "Dewpoint",
        "Pressure_NN",
        "Winddirection",
        "Windspeed",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"Missing source columns in {input_path}: {missing}")

    timestamps = pd.to_datetime(source.pop("MESS_DATUM"), utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Ulm CSV contains duplicate timestamps.")

    load_mw, load_repairs = repair_thousandfold(source["Load_MW"], 100.0)
    temperature, temperature_repairs = repair_thousandfold(
        source["Temperature"], 100.0
    )
    dewpoint, dewpoint_repairs = repair_thousandfold(source["Dewpoint"], 100.0)
    wind_direction, direction_repairs = repair_thousandfold(
        source["Winddirection"], 360.0
    )
    wind_speed, speed_repairs = repair_thousandfold(source["Windspeed"], 100.0)

    radians = np.deg2rad(wind_direction)
    converted = pd.DataFrame(
        {
            "total_consumption": load_mw.to_numpy() * 1_000.0,
            "temperature": temperature.to_numpy(),
            "dewpoint": dewpoint.to_numpy(),
            "pressure": source["Pressure_NN"].astype(float).to_numpy(),
            "wind_speed": wind_speed.to_numpy(),
            "wind_direction_sin": np.sin(radians.to_numpy()),
            "wind_direction_cos": np.cos(radians.to_numpy()),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    ).sort_index()

    expected_index = pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")
    if len(converted) != EXPECTED_ROWS or not converted.index.equals(expected_index):
        raise ValueError(
            "Expected 70,928 consecutive hourly rows from 2014-09-02 00:00 UTC "
            "through 2022-10-05 07:00 UTC."
        )
    if converted.isna().any().any():
        raise ValueError("Converted Ulm data contains missing values.")
    if not converted["total_consumption"].between(300.0, 30_000.0).all():
        raise ValueError("Corrected load lies outside the expected 300-30,000 kW range.")
    if not converted["temperature"].between(-25.0, 40.0).all():
        raise ValueError("Corrected temperature lies outside the expected -25-40 °C range.")

    repairs = {
        "load": load_repairs,
        "temperature": temperature_repairs,
        "dewpoint": dewpoint_repairs,
        "wind_direction": direction_repairs,
        "wind_speed": speed_repairs,
    }
    return converted, repairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path", type=Path, default=ROOT / "data" / "data_Ulm.csv"
    )
    parser.add_argument(
        "--output-path", type=Path, default=ROOT / "data" / "data_Ulm.pkl"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe, repairs = convert_ulm_csv(args.input_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Protocol 4 remains readable by the older NumPy/Pandas versions commonly
    # used in the published training environment.
    dataframe.to_pickle(args.output_path, protocol=4)
    print(f"Wrote {len(dataframe):,} hourly rows to {args.output_path}")
    print(f"Period: {dataframe.index.min()} to {dataframe.index.max()}")
    print(
        "Ranges: "
        f"load {dataframe['total_consumption'].min():.1f}-"
        f"{dataframe['total_consumption'].max():.1f} kW; "
        f"temperature {dataframe['temperature'].min():.1f}-"
        f"{dataframe['temperature'].max():.1f} °C"
    )
    print(f"Repaired thousand-fold values: {repairs}")


if __name__ == "__main__":
    main()
