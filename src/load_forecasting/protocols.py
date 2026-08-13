"""Named, versioned experiment definitions used by the command-line runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from load_forecasting.pipeline import DataConfig, ROOT, WEATHER_CASES


MODEL_NAMES = ("deterministic", "gaussian", "quantile")


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    version: int
    description: str
    data: DataConfig
    weather_cases: tuple[str, ...]
    allowed_weather_cases: tuple[str, ...]
    models: tuple[str, ...] = MODEL_NAMES
    target_normalization_columns: tuple[str | None, ...] = (None,)
    export_forecasts: bool = False
    epochs: int = 50
    seed: int = 42

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload["data"].items()
        }
        return payload


PROTOCOLS = {
    "ulm": ExperimentProtocol(
        name="ulm",
        version=1,
        description="Public Ulm load forecasting with perfect future temperature.",
        data=DataConfig(
            data_path=ROOT / "data" / "data_Ulm.pkl",
            weather_columns=("temperature",),
            selected_weather_cases=("observed_future",),
            target_col="total_consumption",
            common_windows=False,
        ),
        weather_cases=("observed_future",),
        allowed_weather_cases=("historical", "observed_future"),
        models=MODEL_NAMES,
    ),
    "paper": ExperimentProtocol(
        name="paper",
        version=1,
        description=(
            "Manuscript experiment with temperature, irradiance, perfect weather, "
            "and coherent API forecasts on common origins."
        ),
        data=DataConfig(
            data_path=None,
            consumption_path=ROOT / "data" / "260508_consumption_data_aggregated.pkl",
            weather_path=(
                ROOT / "data" / "260509_weather_data_forecast_origin_coherent.pkl"
            ),
            weather_columns=("temperature", "ghi_backwards"),
            selected_weather_cases=WEATHER_CASES,
            align_to_api_start=True,
            common_windows=True,
            train_end="2026-01-31 23:00",
            test_start="2026-02-01 00:00",
            test_end="2026-04-30 23:00",
        ),
        weather_cases=WEATHER_CASES,
        allowed_weather_cases=WEATHER_CASES,
        models=MODEL_NAMES,
    ),
    "eurosun": ExperimentProtocol(
        name="eurosun",
        version=2,
        description=(
            "Weil am Rhein network-expansion experiment with aggregate and "
            "per-active-building demand."
        ),
        data=DataConfig(
            data_path=ROOT / "data" / "260809_eurosun_data_WeilAmRhein.pkl",
            target_col="demand",
            weather_columns=("temperature",),
            past_covariates=("active_buildings",),
            selected_weather_cases=("observed_future",),
            start_date="2020-01-01 00:00",
            train_end="2023-12-31 23:00",
            test_start="2025-01-01 00:00",
            test_end="2025-12-31 23:00",
        ),
        weather_cases=("observed_future",),
        allowed_weather_cases=("historical", "observed_future"),
        models=("deterministic",),
        target_normalization_columns=(None, "active_buildings"),
        export_forecasts=True,
    ),
    "malmoe": ExperimentProtocol(
        name="malmoe",
        version=2,
        description=(
            "Malmö network-expansion experiment with aggregate and "
            "per-active-building demand."
        ),
        data=DataConfig(
            data_path=(
                ROOT
                / "data"
                / "malmoe_residential_aggregated_2020_2025_openmeteo_temp.pkl"
            ),
            target_col="demand",
            weather_columns=("temperature",),
            past_covariates=("active_buildings",),
            selected_weather_cases=("observed_future",),
            start_date="2020-01-01 00:00",
            train_end="2023-12-31 23:00",
            test_start="2025-01-01 00:00",
            test_end="2025-12-31 23:00",
        ),
        weather_cases=("observed_future",),
        allowed_weather_cases=("historical", "observed_future"),
        models=("deterministic",),
        target_normalization_columns=(None, "active_buildings"),
        export_forecasts=True,
    ),
}


def get_protocol(name: str) -> ExperimentProtocol:
    try:
        return PROTOCOLS[name]
    except KeyError as error:
        choices = ", ".join(PROTOCOLS)
        raise ValueError(f"Unknown protocol {name!r}; choose from {choices}.") from error
