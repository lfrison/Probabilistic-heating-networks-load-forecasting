from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.stats import norm

from load_forecasting.metrics import (
    pinball_loss,
    expand_quantiles,
    fit_gaussian_scale_calibration,
    fit_quantile_interval_calibration,
    gaussian_quality_metrics,
    point_metrics,
    quantile_quality_metrics,
)
from load_forecasting.models import build_model
from load_forecasting.pipeline import (
    DataConfig,
    ExperimentConfig,
    PreparedCase,
    WEATHER_CASES,
    assert_common_origins,
    checkpoint_metadata,
    collect_predictions,
    inverse_target,
    load_complete_dataframe,
    prepare_case,
    resolve_device,
    seed_everything,
    split_summary,
    write_json,
)
from load_forecasting.protocols import (
    MODEL_NAMES,
    PAPER_FOLDS,
    PROTOCOLS,
    get_protocol,
)



def gaussian_dual_loss(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    *,
    mae_weight: float = 0.15,
    variance_weight: float = 2e-3,
) -> torch.Tensor:
    variance = torch.exp(2.0 * log_scale)
    nll = (
        0.5 * (math.log(2.0 * math.pi) + 2.0 * log_scale)
        + (target - mean) ** 2 / (2.0 * variance)
    )
    return (
        nll.mean()
        + mae_weight * torch.abs(mean - target).mean()
        + variance_weight * variance.mean()
    )


def quantile_pinball_torch(
    values: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    error = target.unsqueeze(-1) - values
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


@torch.no_grad()
def validation_scores(
    model_name: str,
    model: nn.Module,
    loader,
    case: PreparedCase,
    device: torch.device,
    quantiles: Sequence[float],
) -> dict[str, float]:
    """Validation MAE and, for probabilistic heads, raw nine-quantile CRPS [kW]."""

    model.eval()
    levels = np.asarray(quantiles, dtype=float)
    z_values = norm.ppf(levels)
    absolute_error = crps_sum = 0.0
    count = 0
    for past, future, target, multiplier in loader:
        past, future = past.to(device), future.to(device)
        multiplier = multiplier.numpy()
        target_kw = inverse_target(target.numpy(), case.scaler_target, multiplier)
        quantile_kw = None
        if model_name == "deterministic":
            prediction = model(past, future)
        elif model_name == "gaussian":
            prediction, log_scale = model(past, future)
            mean_kw = inverse_target(prediction.cpu().numpy(), case.scaler_target, multiplier)
            scale_kw = (
                np.exp(log_scale.cpu().numpy()) * float(case.scaler_target.scale_[0]) * multiplier
            )
            quantile_kw = mean_kw[..., None] + scale_kw[..., None] * z_values
        else:
            values = model(past, future)
            prediction = values[..., model.median_index]
            repeated = np.repeat(multiplier[..., None], values.shape[-1], axis=-1)
            quantile_kw = inverse_target(
                values.cpu().numpy().reshape(-1, 1), case.scaler_target, repeated.reshape(-1, 1)
            ).reshape(values.shape)
        prediction_kw = inverse_target(
            prediction.detach().cpu().numpy(), case.scaler_target, multiplier
        )
        absolute_error += float(np.abs(prediction_kw - target_kw).sum())
        if quantile_kw is not None:
            crps_sum += float(2.0 * pinball_loss(target_kw, quantile_kw, levels).mean(axis=-1).sum())
        count += int(target_kw.size)
    return {
        "mae": absolute_error / count,
        "crps": crps_sum / count if model_name != "deterministic" else float("nan"),
    }


def train_model(
    model_name: str,
    model: nn.Module,
    case: PreparedCase,
    cfg: ExperimentConfig,
    device: torch.device,
    on_epoch=None,
) -> tuple[nn.Module, dict[str, object]]:
    """Early stopping on validation MAE (deterministic head) or validation CRPS
    (probabilistic heads). ``on_epoch(epoch, mae)`` allows HPO pruning."""

    train_loader, validation_loader, _ = case.loaders(seed=cfg.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
        eta_min=cfg.lr * 0.3,
    )
    total_steps = max(1, len(train_loader) * cfg.epochs)
    quantiles_t = torch.as_tensor(cfg.quantiles, dtype=torch.float32, device=device)

    select_on_crps = model_name != "deterministic"
    best_score = float("inf")
    best_mae = best_crps = float("nan")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    global_step = 0
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.time()
        model.train()
        accumulated_loss = 0.0
        sample_count = 0
        for past, future, target, _ in train_loader:
            past = past.to(device)
            future = future.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)

            if model_name == "deterministic":
                prediction = model(past, future)
                loss = torch.abs(prediction - target).mean()
            elif model_name == "gaussian":
                warmup = epoch <= cfg.gaussian_warmup_epochs
                if warmup:
                    # Backbone and mean head learn from MAE only; the scale head
                    # learns from the Gaussian loss on detached inputs.
                    mean, log_scale = model(past, future, detach_scale_features=True)
                    loss = torch.abs(mean - target).mean() + gaussian_dual_loss(
                        mean.detach(), log_scale, target
                    )
                else:
                    mean, log_scale = model(past, future)
                    loss = gaussian_dual_loss(mean, log_scale, target)
            else:
                values = model(past, future)
                progress = min(1.0, global_step / total_steps)
                interval_width_penalty = (values[..., -1] - values[..., 0]).pow(2).mean()
                median_loss = torch.abs(values[..., model.median_index] - target).mean()
                loss = (
                    quantile_pinball_torch(values, target, quantiles_t)
                    + cfg.quantile_width_weight * progress * interval_width_penalty
                    + cfg.quantile_median_weight * median_loss
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1
            accumulated_loss += float(loss.item()) * len(past)
            sample_count += len(past)

        scheduler.step()
        scores = validation_scores(
            model_name,
            model,
            validation_loader,
            case,
            device,
            cfg.quantiles,
        )
        current_mae = scores["mae"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": accumulated_loss / max(sample_count, 1),
            "validation_mae_kw": current_mae,
            "validation_crps_kw": scores["crps"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        epoch_minutes = (time.time() - epoch_started) / 60.0
        elapsed_minutes = (time.time() - started) / 60.0
        print(
            f"{case.weather_case:>15} | {model_name:>13} | epoch {epoch:02d} | "
            f"loss {epoch_record['train_loss']:.5f} | val MAE {current_mae:.2f} kW | "
            + (f"val CRPS {scores['crps']:.2f} kW | " if select_on_crps else "")
            + f"epoch {epoch_minutes:.2f} min | elapsed {elapsed_minutes:.2f} min"
        )

        if on_epoch is not None:
            on_epoch(epoch, current_mae)
        score = scores["crps"] if select_on_crps else current_mae
        if score < best_score - 0.05:
            best_score = score
            best_mae, best_crps, best_epoch = current_mae, scores["crps"], epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.es_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_validation_mae_kw": best_mae,
        "best_validation_crps_kw": best_crps,
        "best_epoch": best_epoch,
        "selection_metric": "crps" if select_on_crps else "mae",
        "epochs_completed": len(history),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "history": history,
    }


def evaluate_model(
    model_name: str,
    model: nn.Module,
    case: PreparedCase,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    _, validation_loader, test_loader = case.loaders(seed=cfg.seed)
    validation = collect_predictions(
        model_name,
        model,
        validation_loader,
        case,
        device,
    )
    test = collect_predictions(model_name, model, test_loader, case, device)
    y = test["target_kw"]
    quantiles = np.asarray(cfg.quantiles, dtype=float)

    probabilistic: dict[str, object] = {}
    if model_name == "deterministic":
        prediction = test["prediction_kw"]
        calibration: dict[str, float] = {}
    elif model_name == "gaussian":
        prediction = test["mean_kw"]
        calibration_scale = fit_gaussian_scale_calibration(
            validation["target_kw"],
            validation["mean_kw"],
            validation["scale_kw"],
        )
        calibration = {"sigma_scale": calibration_scale}
        mean, scale = test["mean_kw"], test["scale_kw"]
        probabilistic = {
            "probabilistic_raw": gaussian_quality_metrics(y, mean, scale, quantiles),
            "probabilistic_calibrated": gaussian_quality_metrics(
                y, mean, scale * calibration_scale, quantiles
            ),
        }

    else:
        median_index = int(np.argmin(np.abs(quantiles - 0.5)))
        prediction = test["quantiles_kw"][..., median_index]
        delta = fit_quantile_interval_calibration(
            validation["target_kw"],
            validation["quantiles_kw"],
            quantiles,
        )
        calibration = {"symmetric_delta_kw": delta}
        calibrated_values = expand_quantiles(test["quantiles_kw"], quantiles, delta)
        probabilistic = {
            "probabilistic_raw": quantile_quality_metrics(y, test["quantiles_kw"], quantiles),
            "probabilistic_calibrated": quantile_quality_metrics(y, calibrated_values, quantiles),
        }

    metrics: dict[str, object] = {
        "point": point_metrics(y, prediction),
        "point_breakdown": point_metric_breakdown(y, prediction, case),
        **probabilistic,
    }
    if calibration:
        metrics["calibration"] = calibration
    return metrics, point_forecast_frame(y, prediction, case)


def point_forecast_frame(
    target: np.ndarray,
    prediction: np.ndarray,
    case: PreparedCase,
) -> pd.DataFrame:
    """Return one row per origin and horizon for downstream evaluation."""

    n_origins, pred_len = target.shape
    positions = case.origins.test[:, None] + np.arange(pred_len)[None, :]
    anchors = np.asarray(case.dataframe.index)[case.origins.test - 1]
    return pd.DataFrame(
        {
            "anchor": np.repeat(anchors, pred_len),
            "t_eval": np.asarray(case.dataframe.index)[positions].reshape(-1),
            "h": np.tile(np.arange(1, pred_len + 1), n_origins),
            "y_true": target.reshape(-1),
            "y_pred": prediction.reshape(-1),
        }
    )


def point_metric_breakdown(
    target: np.ndarray,
    prediction: np.ndarray,
    case: PreparedCase,
) -> dict[str, object]:
    """Report lead-wise, monthly, and conventional summer/winter point metrics."""

    by_horizon = {
        f"lead_{lead + 1:02d}": point_metrics(target[:, lead], prediction[:, lead])
        for lead in range(target.shape[1])
    }
    positions = case.origins.test[:, None] + np.arange(target.shape[1])[None, :]
    timestamps = np.asarray(case.dataframe.index)[positions]
    months = pd.DatetimeIndex(timestamps.reshape(-1)).month.to_numpy()
    target_flat = target.reshape(-1)
    prediction_flat = prediction.reshape(-1)
    by_month = {
        f"{month:02d}": point_metrics(
            target_flat[months == month], prediction_flat[months == month]
        )
        for month in sorted(set(months))
    }
    seasons = {
        "winter": (10, 11, 12, 1, 2, 3),
        "summer": (4, 5, 6, 7, 8, 9),
    }
    by_season = {}
    for season, season_months in seasons.items():
        selected = np.isin(months, season_months)
        if selected.any():
            by_season[season] = point_metrics(
                target_flat[selected], prediction_flat[selected]
            )
    return {
        "by_horizon": by_horizon,
        "by_month": by_month,
        "by_season": by_season,
    }


def save_run(
    *,
    output_dir: Path,
    model_name: str,
    model: nn.Module,
    case: PreparedCase,
    data_cfg: DataConfig,
    experiment_cfg: ExperimentConfig,
    training: dict[str, object],
    metrics: dict[str, object],
    protocol_metadata: dict[str, object],
    filename_suffix: str | None = None,
    forecasts: pd.DataFrame | None = None,
    forecast_case: str | None = None,
) -> None:
    stem = f"{model_name}_{case.weather_case}"
    if filename_suffix:
        stem = f"{stem}_{filename_suffix}"
    metadata = checkpoint_metadata(
        case,
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
    )
    metadata["protocol"] = protocol_metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": portable_state,
            "model_name": model_name,
            "metadata": metadata,
            "training": training,
            "metrics": metrics,
        },
        output_dir / f"{stem}.pt",
    )
    write_json(
        output_dir / f"{stem}.json",
        {
            "model_name": model_name,
            "metadata": metadata,
            "training": training,
            "metrics": metrics,
        },
    )
    if forecasts is not None and forecast_case is not None:
        forecasts.to_csv(
            output_dir / f"forecasts_{forecast_case}.csv",
            index=False,
        )


# Model choices of the command line: versions A and B and the paper baselines.
VARIANTS = {
    "A": dict(window_load_scaling=True),
    "B": dict(output_skip="linear"),
    "plain": dict(decoder_lag24=False),
    "lstm": dict(decoder_lag24=False, architecture="lstm"),
    "lstm-ws": dict(decoder_lag24=False, architecture="lstm", window_load_scaling=True),
}


def model_variant_name(data_cfg: DataConfig, variant: str) -> str:
    parts = []
    if data_cfg.start_date:
        parts.append(f"from{pd.Timestamp(data_cfg.start_date):%Y-%m-%d}")
    if data_cfg.calendar_timezone:
        parts.append("localcal")
    return "_".join([*parts, variant])


def summary_line(label: str, metrics: dict[str, object]) -> str:
    """Test MAE and MAPE, plus CRPS and coverage of the calibrated intervals."""

    point = metrics["point"]
    line = f"{label:<40} MAE {point['mae_kw']:7.1f} kW | MAPE {point['mape_percent']:5.2f} %"
    if "probabilistic_calibrated" in metrics:
        prob = metrics["probabilistic_calibrated"]
        line += f" | CRPS {prob['crps_shared_quantile_grid_kw']:7.1f} kW | PICP80 {prob['picp80']:.2f}"
    return line


def expand_selection(values: list[str], allowed: tuple[str, ...]) -> list[str]:
    if "all" in values:
        return list(allowed)
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AEDL load forecasts using a named experiment protocol."
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="ulm",
        help="Complete dataset, feature, horizon, and split definition (default: ulm).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["all", *MODEL_NAMES],
        help="Optional subset; the protocol default is used when omitted.",
    )
    parser.add_argument(
        "--weather-cases",
        nargs="+",
        default=None,
        choices=["all", *WEATHER_CASES],
        help="Optional subset of weather cases allowed by the selected protocol.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--version", choices=("A", "B"), default="A",
        help="A: load of each window divided by its 48 h mean. B: linear shortcut from the last 24 h of load.",
    )
    parser.add_argument(
        "--baseline", choices=("plain", "lstm", "lstm-ws"), default=None,
        help="Paper baselines instead of a version: plain AEDL, plain LSTM, LSTM with window scaling.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=tuple(PAPER_FOLDS),
        default=None,
        help="Paper protocol only. 1: test Oct 2025-Jan 2026, 2: test Feb-Apr 2026. "
        "Default: original manuscript split.",
    )
    parser.add_argument(
        "--start-date", default=None, help="First training origin, e.g. 2023-01-01 (default: all data)."
    )
    parser.add_argument(
        "--calendar-tz",
        default=None,
        help="Compute calendar features in this time zone, e.g. Europe/Berlin (default: UTC).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional result directory. The default is grouped by protocol and seed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = get_protocol(args.protocol)
    models = (
        list(protocol.models)
        if args.models is None
        else expand_selection(args.models, protocol.models)
    )
    invalid_models = set(models).difference(protocol.models)
    if invalid_models:
        raise ValueError(
            f"Protocol {protocol.name!r} does not support models "
            f"{sorted(invalid_models)}. Allowed: {protocol.models}."
        )
    weather_cases = (
        list(protocol.weather_cases)
        if args.weather_cases is None
        else expand_selection(args.weather_cases, protocol.allowed_weather_cases)
    )
    invalid_weather = set(weather_cases).difference(protocol.allowed_weather_cases)
    if invalid_weather:
        raise ValueError(
            f"Protocol {protocol.name!r} does not support weather cases "
            f"{sorted(invalid_weather)}. Allowed: {protocol.allowed_weather_cases}."
        )
    common_windows = protocol.data.common_windows or len(weather_cases) > 1
    comparison_cases = (
        protocol.weather_cases
        if protocol.data.common_windows
        else tuple(weather_cases)
    )
    base_data_cfg = replace(
        protocol.data,
        selected_weather_cases=tuple(comparison_cases),
        common_windows=common_windows,
    )
    if args.fold is not None:
        if protocol.name != "paper":
            raise ValueError("--fold is only defined for the paper protocol.")
        base_data_cfg = replace(base_data_cfg, **PAPER_FOLDS[args.fold])
    if args.start_date is not None:
        base_data_cfg = replace(base_data_cfg, start_date=args.start_date)
    if args.calendar_tz is not None:
        base_data_cfg = replace(base_data_cfg, calendar_timezone=args.calendar_tz)
    variant = args.baseline or args.version
    choice = VARIANTS[variant]
    data_fields = {key: value for key, value in choice.items() if key in ("window_load_scaling", "decoder_lag24")}
    base_data_cfg = replace(base_data_cfg, **data_fields)
    epochs = protocol.epochs if args.epochs is None else args.epochs
    seed = protocol.seed if args.seed is None else args.seed
    experiment_cfg = replace(
        ExperimentConfig(),
        epochs=epochs,
        seed=seed,
        **{key: value for key, value in choice.items() if key in ("output_skip", "architecture")},
    )

    seed_everything(experiment_cfg.seed)
    dataframe = load_complete_dataframe(base_data_cfg)
    prepared_runs: list[tuple[str, DataConfig, PreparedCase]] = []
    for normalization_col in protocol.target_normalization_columns:
        data_cfg = replace(
            base_data_cfg,
            target_normalization_col=normalization_col,
        )
        cases = [
            prepare_case(
                dataframe,
                data_cfg=data_cfg,
                weather_case=weather_case,
            )
            for weather_case in weather_cases
        ]
        if data_cfg.common_windows:
            assert_common_origins(cases)
        if normalization_col is None:
            normalization_name = "aggregate"
        elif normalization_col == "active_buildings":
            normalization_name = "per_building"
        else:
            normalization_name = f"normalized_by_{normalization_col}"
        prepared_runs.extend(
            (normalization_name, data_cfg, case) for case in cases
        )

    for weather_case in weather_cases:
        assert_common_origins(
            case
            for _, _, case in prepared_runs
            if case.weather_case == weather_case
        )

    is_default_weather = tuple(weather_cases) == protocol.weather_cases
    result_group = ROOT / "results" / protocol.name
    folder = model_variant_name(base_data_cfg, variant)
    if args.fold is not None:
        folder = f"fold{args.fold}_{folder}"
    result_group = result_group / folder
    if not is_default_weather:
        result_group = result_group / "_".join(weather_cases)
    output_dir = args.output_dir or result_group / f"seed_{seed}"
    protocol_metadata = protocol.metadata()
    protocol_metadata["resolved_weather_cases"] = weather_cases
    protocol_metadata["resolved_models"] = models
    protocol_metadata["resolved_target_normalization_columns"] = list(
        protocol.target_normalization_columns
    )
    protocol_metadata["resolved_epochs"] = epochs
    protocol_metadata["resolved_seed"] = seed
    print(
        f"Protocol: {protocol.name} v{protocol.version} — {protocol.description}"
    )
    print("Data split summaries:")
    for _, _, case in prepared_runs:
        print(split_summary(case))
    print(f"Results: {output_dir}")

    device = resolve_device(args.device)
    print(f"Training on {device}")
    all_results: list[dict[str, object]] = []
    summary_lines: list[str] = []
    total_runs = len(prepared_runs) * len(models)
    completed_runs = 0
    experiment_started = time.time()
    multiple_normalizations = len(protocol.target_normalization_columns) > 1
    for normalization_name, data_cfg, case in prepared_runs:
        for model_name in models:
            run_started = time.time()
            print(
                f"Starting run {completed_runs + 1}/{total_runs}: "
                f"{case.weather_case} / {normalization_name} / {model_name}"
            )
            seed_everything(experiment_cfg.seed)
            model = build_model(
                model_name,
                n_past_features=len(case.past_cols),
                future_dim=case.train_dataset.future.shape[-1],
                hidden_size=experiment_cfg.hidden_size,
                num_layers=experiment_cfg.num_layers,
                attention_heads=experiment_cfg.attn_heads,
                dropout=experiment_cfg.dropout,
                pred_len=data_cfg.pred_len,
                quantiles=experiment_cfg.quantiles,
                output_skip=experiment_cfg.output_skip,
                target_index=case.past_cols.index(data_cfg.target_col),
                architecture=experiment_cfg.architecture,
            ).to(device)
            model, training = train_model(
                model_name,
                model,
                case,
                experiment_cfg,
                device,
            )
            metrics, forecasts = evaluate_model(
                model_name,
                model,
                case,
                experiment_cfg,
                device,
            )
            label = " / ".join(
                [case.weather_case, *([normalization_name] if multiple_normalizations else []), model_name]
            )
            summary_lines.append(summary_line(label, metrics))
            print(summary_lines[-1])
            save_run(
                output_dir=output_dir,
                model_name=model_name,
                model=model,
                case=case,
                data_cfg=data_cfg,
                experiment_cfg=experiment_cfg,
                training=training,
                metrics=metrics,
                protocol_metadata=protocol_metadata,
                filename_suffix=(normalization_name if multiple_normalizations else None),
                forecasts=(forecasts if protocol.export_forecasts else None),
                forecast_case=(
                    f"{'norm' if case.target_normalization_col else 'raw'}_temp"
                    if protocol.export_forecasts
                    else None
                ),
            )
            all_results.append(
                {
                    "weather_case": case.weather_case,
                    "target_normalization": normalization_name,
                    "model_name": model_name,
                    "training": {
                        "elapsed_minutes": training["elapsed_minutes"],
                        "epochs_completed": training["epochs_completed"],
                        "best_validation_mae_kw": training["best_validation_mae_kw"],
                    },
                    "metrics": metrics,
                }
            )
            completed_runs += 1
            run_minutes = (time.time() - run_started) / 60.0
            total_minutes = (time.time() - experiment_started) / 60.0
            mean_run_minutes = total_minutes / completed_runs
            eta_minutes = mean_run_minutes * (total_runs - completed_runs)
            print(
                f"Completed run {completed_runs}/{total_runs} in {run_minutes:.2f} min | "
                f"total elapsed {total_minutes:.2f} min | "
                f"estimated remaining {eta_minutes:.2f} min"
            )
    write_json(
        output_dir / "summary.json",
        {
            "protocol": protocol_metadata,
            "seed": experiment_cfg.seed,
            "weather_cases": weather_cases,
            "models": models,
            "runs": all_results,
        },
    )
    total_minutes = (time.time() - experiment_started) / 60.0
    print(f"All {total_runs} runs completed in {total_minutes:.2f} min.")
    print(f"\nTest results ({output_dir}):")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
