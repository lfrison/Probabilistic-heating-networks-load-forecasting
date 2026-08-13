from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_forecasting.metrics import (
    central_quantile_interval,
    expand_quantiles,
    fit_gaussian_scale_calibration,
    fit_quantile_interval_calibration,
    gaussian_quantiles,
    interpolate_quantile,
    picp,
)
from load_forecasting.models import build_model
from load_forecasting.pipeline import (
    DataConfig,
    ExperimentConfig,
    PreparedCase,
    WEATHER_CASES,
    collect_predictions,
    load_complete_dataframe,
    prepare_case,
    resolve_device,
)
from load_forecasting.protocols import PROTOCOLS, get_protocol
DEFAULT_OUTPUT_DIR = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate calibrated reliability and forecast-example figures from "
            "saved Gaussian and quantile AEDL checkpoints."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="ulm",
        help="Protocol whose result directory is used by default (default: ulm).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the Gaussian and quantile checkpoints. "
            "Default: results/<protocol>/seed_<seed>."
        ),
    )
    parser.add_argument(
        "--weather-case",
        choices=WEATHER_CASES,
        default="observed_future",
        help=(
            "Checkpoint weather case (default: observed_future, the idealized "
            "perfect-weather benchmark)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Optional combined-data path override, useful after moving a checkpoint "
            "or repository. Otherwise the checkpoint metadata is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Optional zero-based test-origin index for the forecast example. "
            "By default, a representative median-error origin is selected."
        ),
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def config_from_metadata(metadata: dict[str, Any]) -> tuple[DataConfig, ExperimentConfig]:
    raw_data = dict(metadata["data_config"])
    allowed_data = {field.name for field in fields(DataConfig)}
    data_kwargs = {key: value for key, value in raw_data.items() if key in allowed_data}
    # Checkpoints created before the public combined-data interface retain their
    # separate private load/weather paths and original two weather variables.
    if "data_path" not in raw_data:
        data_kwargs["data_path"] = None
        data_kwargs["weather_columns"] = ("temperature", "ghi_backwards")
    for path_key in ("data_path", "consumption_path", "weather_path"):
        if data_kwargs.get(path_key) is not None:
            data_kwargs[path_key] = Path(data_kwargs[path_key])
    for tuple_key in (
        "weather_columns",
        "past_covariates",
        "selected_weather_cases",
    ):
        if tuple_key in data_kwargs:
            data_kwargs[tuple_key] = tuple(data_kwargs[tuple_key])

    raw_experiment = dict(metadata["experiment_config"])
    allowed_experiment = {field.name for field in fields(ExperimentConfig)}
    experiment_kwargs = {
        key: value for key, value in raw_experiment.items() if key in allowed_experiment
    }
    experiment_kwargs["quantiles"] = tuple(experiment_kwargs["quantiles"])
    return DataConfig(**data_kwargs), ExperimentConfig(**experiment_kwargs)


def assert_matching_experiments(
    gaussian_checkpoint: dict[str, Any],
    quantile_checkpoint: dict[str, Any],
    weather_case: str,
) -> None:
    gaussian_metadata = gaussian_checkpoint["metadata"]
    quantile_metadata = quantile_checkpoint["metadata"]
    keys = (
        "weather_case",
        "data_config",
        "past_columns",
        "future_columns",
        "future_source_columns_by_horizon",
        "experiment_config",
        "split_summary",
    )
    for key in keys:
        if gaussian_metadata[key] != quantile_metadata[key]:
            raise ValueError(f"Gaussian and quantile checkpoint metadata differ for {key!r}.")
    if gaussian_metadata["weather_case"] != weather_case:
        raise ValueError(
            f"Expected {weather_case!r} checkpoints, found "
            f"{gaussian_metadata['weather_case']!r}."
        )


def build_checkpoint_model(
    checkpoint: dict[str, Any],
    case: PreparedCase,
    experiment_cfg: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    metadata = checkpoint["metadata"]
    if metadata["past_columns"] != case.past_cols:
        raise ValueError("Checkpoint past columns do not match the reconstructed pipeline.")
    if metadata["future_columns"] != case.future_cols:
        raise ValueError("Checkpoint future columns do not match the reconstructed pipeline.")
    if metadata["future_source_columns_by_horizon"] != case.future_cols_by_horizon:
        raise ValueError("Checkpoint future covariates do not match the pipeline.")

    model_name = checkpoint["model_name"]
    model = build_model(
        model_name,
        n_past_features=len(case.past_cols),
        future_dim=len(case.future_cols),
        hidden_size=experiment_cfg.hidden_size,
        num_layers=experiment_cfg.num_layers,
        attention_heads=experiment_cfg.attn_heads,
        dropout=experiment_cfg.dropout,
        pred_len=case.test_dataset.pred_len,
        quantiles=experiment_cfg.quantiles,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def reliability_from_quantiles(
    target: np.ndarray,
    values: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [np.mean(target <= values[..., index]) for index in range(len(quantiles))]
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def make_reliability_figure(
    output_dir: Path,
    quantiles: np.ndarray,
    gaussian_raw: np.ndarray,
    gaussian_calibrated: np.ndarray,
    quantile_raw: np.ndarray,
    quantile_calibrated: np.ndarray,
    stem: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="0.15", linestyle=":", linewidth=1.6, label="Ideal")
    ax.plot(
        quantiles,
        gaussian_raw,
        "o--",
        color="#D55E00",
        linewidth=1.5,
        markersize=2.5,
        label="Gaussian raw",
    )
    ax.plot(
        quantiles,
        gaussian_calibrated,
        "s-",
        color="#D55E00",
        linewidth=1.8,
        markersize=4.5,
        label="Gaussian cal.",
    )
    ax.plot(
        quantiles,
        quantile_raw,
        "o--",
        color="#0072B2",
        linewidth=1.5,
        markersize=2.5,
        label="Quantile raw",
    )
    ax.plot(
        quantiles,
        quantile_calibrated,
        "s-",
        color="#0072B2",
        linewidth=1.8,
        markersize=4.5,
        label="Quantile cal.",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"Nominal quantile level $\tau$",
        ylabel=r"Empirical frequency $P(y \leq q_\tau)$",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, linestyle=":", linewidth=0.8, color="0.82")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def representative_sample(
    target: np.ndarray,
    gaussian_mean: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
) -> int:
    quantile_median = interpolate_quantile(quantile_values, quantiles, 0.5)
    combined_mae = 0.5 * (
        np.mean(np.abs(gaussian_mean - target), axis=1)
        + np.mean(np.abs(quantile_median - target), axis=1)
    )
    return int(np.argmin(np.abs(combined_mae - np.median(combined_mae))))


def make_forecast_figure(
    output_dir: Path,
    times: pd.DatetimeIndex,
    target: np.ndarray,
    gaussian_mean: np.ndarray,
    gaussian_scale_raw: np.ndarray,
    gaussian_scale_calibrated: np.ndarray,
    quantile_values_raw: np.ndarray,
    quantile_values_calibrated: np.ndarray,
    quantiles: np.ndarray,
    stem: str,
) -> list[Path]:
    z80 = float(norm.ppf(0.90))
    gaussian_lower_raw = gaussian_mean - z80 * gaussian_scale_raw
    gaussian_upper_raw = gaussian_mean + z80 * gaussian_scale_raw
    gaussian_lower = gaussian_mean - z80 * gaussian_scale_calibrated
    gaussian_upper = gaussian_mean + z80 * gaussian_scale_calibrated
    quantile_lower_raw, quantile_upper_raw = central_quantile_interval(
        quantile_values_raw, quantiles, coverage=0.80
    )
    quantile_lower, quantile_upper = central_quantile_interval(
        quantile_values_calibrated, quantiles, coverage=0.80
    )
    quantile_median = interpolate_quantile(quantile_values_calibrated, quantiles, 0.5)
    gaussian_mae = float(np.mean(np.abs(gaussian_mean - target)))
    gaussian_rmse = float(np.sqrt(np.mean((gaussian_mean - target) ** 2)))
    gaussian_mape = float(np.mean(np.abs((gaussian_mean - target) / target)) * 100)
    quantile_mae = float(np.mean(np.abs(quantile_median - target)))
    quantile_rmse = float(np.sqrt(np.mean((quantile_median - target) ** 2)))
    quantile_mape = float(np.mean(np.abs((quantile_median - target) / target)) * 100)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, sharey=True)
    axes[0].fill_between(
        times,
        gaussian_lower,
        gaussian_upper,
        color="#D55E00",
        alpha=0.13,
        label="Calibrated 80% interval",
    )
    axes[0].fill_between(
        times,
        gaussian_lower_raw,
        gaussian_upper_raw,
        color="#D55E00",
        alpha=0.28,
        label="Raw 80% interval",
    )
    axes[0].plot(times, gaussian_mean, color="#D55E00", linewidth=1.8, label="Mean forecast")
    axes[0].plot(times, target, color="black", linewidth=1.6, label="Observed load")
    axes[0].set_title(
        f"Gaussian-AEDL — MAE: {gaussian_mae:.0f} kW; "
        f"RMSE: {gaussian_rmse:.0f} kW; MAPE: {gaussian_mape:.1f}%",
        loc="left",
        fontsize=8.5,
        fontweight="normal",
    )

    axes[1].fill_between(
        times,
        quantile_lower,
        quantile_upper,
        color="#0072B2",
        alpha=0.13,
        label="Calibrated 80% interval",
    )
    axes[1].fill_between(
        times,
        quantile_lower_raw,
        quantile_upper_raw,
        color="#0072B2",
        alpha=0.28,
        label="Raw 80% interval",
    )
    axes[1].plot(times, quantile_median, color="#0072B2", linewidth=1.8, label="Median forecast")
    axes[1].plot(times, target, color="black", linewidth=1.6, label="Observed load")
    axes[1].set_title(
        f"Quantile-AEDL — MAE: {quantile_mae:.0f} kW; "
        f"RMSE: {quantile_rmse:.0f} kW; MAPE: {quantile_mape:.1f}%",
        loc="left",
        fontsize=8.5,
        fontweight="normal",
    )

    for ax in axes:
        ax.set_ylabel("Thermal load [kW]")
        ax.grid(True, linestyle=":", linewidth=0.8, color="0.82")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].set_xlabel("Forecast valid time")
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def main() -> None:
    args = parse_args()
    protocol = get_protocol(args.protocol)
    if args.weather_case not in protocol.allowed_weather_cases:
        raise ValueError(
            f"Protocol {protocol.name!r} does not support {args.weather_case!r}. "
            f"Allowed: {protocol.allowed_weather_cases}."
        )
    default_result_group = ROOT / "results" / protocol.name
    if args.weather_case not in protocol.weather_cases:
        default_result_group = default_result_group / args.weather_case
    results_dir = args.results_dir or default_result_group / f"seed_{args.seed}"
    legacy_ulm_dir = ROOT / "results" / "observed_future" / f"seed_{args.seed}"
    if (
        args.results_dir is None
        and protocol.name == "ulm"
        and args.weather_case == "observed_future"
        and not results_dir.exists()
        and legacy_ulm_dir.exists()
    ):
        results_dir = legacy_ulm_dir
    file_prefix = "api" if args.weather_case == "api_forecast" else args.weather_case
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
        }
    )

    gaussian_path = results_dir / f"gaussian_{args.weather_case}.pt"
    quantile_path = results_dir / f"quantile_{args.weather_case}.pt"
    gaussian_checkpoint = load_checkpoint(gaussian_path)
    quantile_checkpoint = load_checkpoint(quantile_path)
    assert_matching_experiments(
        gaussian_checkpoint,
        quantile_checkpoint,
        args.weather_case,
    )

    data_cfg, experiment_cfg = config_from_metadata(gaussian_checkpoint["metadata"])
    if args.data_path is not None:
        data_cfg = replace(data_cfg, data_path=args.data_path)
    dataframe = load_complete_dataframe(data_cfg)
    case = prepare_case(
        dataframe,
        data_cfg=data_cfg,
        weather_case=args.weather_case,
    )
    device = resolve_device(args.device)
    gaussian_model = build_checkpoint_model(
        gaussian_checkpoint, case, experiment_cfg, device
    )
    quantile_model = build_checkpoint_model(
        quantile_checkpoint, case, experiment_cfg, device
    )

    _, validation_loader, test_loader = case.loaders(seed=experiment_cfg.seed)
    gaussian_validation = collect_predictions(
        "gaussian", gaussian_model, validation_loader, case, device
    )
    gaussian_test = collect_predictions("gaussian", gaussian_model, test_loader, case, device)
    quantile_validation = collect_predictions(
        "quantile", quantile_model, validation_loader, case, device
    )
    quantile_test = collect_predictions("quantile", quantile_model, test_loader, case, device)

    if not np.allclose(gaussian_test["target_kw"], quantile_test["target_kw"], atol=1e-5):
        raise RuntimeError("Gaussian and quantile test targets do not align.")
    if not np.allclose(
        gaussian_validation["target_kw"], quantile_validation["target_kw"], atol=1e-5
    ):
        raise RuntimeError("Gaussian and quantile validation targets do not align.")

    quantiles = np.asarray(experiment_cfg.quantiles, dtype=float)
    sigma_scale = fit_gaussian_scale_calibration(
        gaussian_validation["target_kw"],
        gaussian_validation["mean_kw"],
        gaussian_validation["scale_kw"],
    )
    quantile_delta = fit_quantile_interval_calibration(
        quantile_validation["target_kw"],
        quantile_validation["quantiles_kw"],
        quantiles,
    )
    saved_sigma_scale = float(gaussian_checkpoint["metrics"]["calibration"]["sigma_scale"])
    saved_quantile_delta = float(
        quantile_checkpoint["metrics"]["calibration"]["symmetric_delta_kw"]
    )
    if not np.isclose(sigma_scale, saved_sigma_scale, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            "Recomputed Gaussian calibration differs materially from the saved result: "
            f"{sigma_scale} versus {saved_sigma_scale}."
        )
    if not np.isclose(quantile_delta, saved_quantile_delta, rtol=1e-5, atol=1e-4):
        raise RuntimeError(
            "Recomputed quantile calibration differs materially from the saved result: "
            f"{quantile_delta} versus {saved_quantile_delta}."
        )

    target = gaussian_test["target_kw"]
    gaussian_mean = gaussian_test["mean_kw"]
    gaussian_scale_raw = gaussian_test["scale_kw"]
    gaussian_scale_calibrated = gaussian_scale_raw * sigma_scale
    gaussian_quantiles_raw = gaussian_quantiles(
        gaussian_mean, gaussian_scale_raw, quantiles
    )
    gaussian_quantiles_calibrated = gaussian_quantiles(
        gaussian_mean, gaussian_scale_calibrated, quantiles
    )
    quantile_values_raw = quantile_test["quantiles_kw"]
    quantile_values_calibrated = expand_quantiles(
        quantile_values_raw, quantiles, quantile_delta
    )

    reliability_paths = make_reliability_figure(
        args.output_dir,
        quantiles,
        reliability_from_quantiles(target, gaussian_quantiles_raw, quantiles),
        reliability_from_quantiles(target, gaussian_quantiles_calibrated, quantiles),
        reliability_from_quantiles(target, quantile_values_raw, quantiles),
        reliability_from_quantiles(target, quantile_values_calibrated, quantiles),
        stem=f"{file_prefix}_reliability_diagram",
    )

    sample_index = (
        representative_sample(target, gaussian_mean, quantile_values_raw, quantiles)
        if args.sample_index is None
        else args.sample_index
    )
    if not 0 <= sample_index < len(case.origins.test):
        raise IndexError(
            f"sample index {sample_index} outside test range 0..{len(case.origins.test) - 1}"
        )
    origin = int(case.origins.test[sample_index])
    times = case.dataframe.index[origin : origin + data_cfg.pred_len]
    forecast_paths = make_forecast_figure(
        args.output_dir,
        times,
        target[sample_index],
        gaussian_mean[sample_index],
        gaussian_scale_raw[sample_index],
        gaussian_scale_calibrated[sample_index],
        quantile_values_raw[sample_index],
        quantile_values_calibrated[sample_index],
        quantiles,
        stem=f"{file_prefix}_forecast_example_calibrated",
    )

    z80 = float(norm.ppf(0.90))
    gaussian_picp_raw = picp(
        target,
        gaussian_mean - z80 * gaussian_scale_raw,
        gaussian_mean + z80 * gaussian_scale_raw,
    )
    gaussian_picp_calibrated = picp(
        target,
        gaussian_mean - z80 * gaussian_scale_calibrated,
        gaussian_mean + z80 * gaussian_scale_calibrated,
    )
    quantile_lower_raw, quantile_upper_raw = central_quantile_interval(
        quantile_values_raw, quantiles
    )
    quantile_lower_calibrated, quantile_upper_calibrated = central_quantile_interval(
        quantile_values_calibrated, quantiles
    )

    provenance = {
        "protocol": protocol.name,
        "protocol_version": protocol.version,
        "weather_case": args.weather_case,
        "results_dir": str(results_dir.resolve()),
        "data_path_override": str(args.data_path.resolve()) if args.data_path else None,
        "gaussian_checkpoint": str(gaussian_path.resolve()),
        "quantile_checkpoint": str(quantile_path.resolve()),
        "seed": experiment_cfg.seed,
        "past_columns": case.past_cols,
        "test_origins": len(case.origins.test),
        "sample_index": sample_index,
        "sample_origin": str(times[0]),
        "gaussian_sigma_scale": sigma_scale,
        "quantile_symmetric_delta_kw": quantile_delta,
        "gaussian_picp80_raw": gaussian_picp_raw,
        "gaussian_picp80_calibrated": gaussian_picp_calibrated,
        "quantile_picp80_raw": picp(target, quantile_lower_raw, quantile_upper_raw),
        "quantile_picp80_calibrated": picp(
            target, quantile_lower_calibrated, quantile_upper_calibrated
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = (
        args.output_dir / f"{file_prefix}_probabilistic_figures_latest.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"Protocol: {protocol.name} v{protocol.version}")
    print(f"Weather case: {args.weather_case}")
    print(f"Results directory: {results_dir}")
    print(f"Past features: {len(case.past_cols)}")
    print(f"Gaussian sigma scale: {sigma_scale:.6f}")
    print(f"Quantile conformal delta: {quantile_delta:.6f} kW")
    print(
        f"Gaussian PICP80 raw/cal: {gaussian_picp_raw:.6f} / "
        f"{gaussian_picp_calibrated:.6f}"
    )
    print(
        "Quantile PICP80 raw/cal: "
        f"{provenance['quantile_picp80_raw']:.6f} / "
        f"{provenance['quantile_picp80_calibrated']:.6f}"
    )
    print(f"Representative sample: {sample_index} ({times[0]})")
    for path in [*reliability_paths, *forecast_paths, provenance_path]:
        print(f"Saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
