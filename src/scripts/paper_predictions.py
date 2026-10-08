"""Test-set predictions of the final A/B models: horizon metrics and paper figures.

``predict`` reloads the saved Gaussian and quantile checkpoints of versions A
and B (folds 1 and 2, all seeds), predicts validation and test windows, applies
the validation-based calibration and stores the calibrated nine-quantile
forecasts in results/paper_predictions/. No model is trained.

``report`` pools both test periods and prints point and probabilistic metrics
per lead time (mean over seeds) and draws the horizon figure, the forecast
example (three consecutive day-ahead forecasts with an error panel), and the
reliability diagram.

Usage:
    python src/scripts/paper_predictions.py predict
    python src/scripts/paper_predictions.py report
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "src" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_probabilistic_figures import build_checkpoint_model, config_from_metadata, load_checkpoint
from load_forecasting.metrics import (
    expand_quantiles,
    fit_gaussian_scale_calibration,
    fit_quantile_interval_calibration,
    gaussian_quantiles,
)
from load_forecasting.pipeline import collect_predictions, load_complete_dataframe, prepare_case, resolve_device

TAGS = {"A": "localcal_A", "B": "localcal_B"}
HEADS = ("gaussian", "quantile")
FOLDS = (1, 2)
SEEDS = (42, 43, 44, 45, 46)
QUANTILES = np.round(np.arange(0.1, 1.0, 0.1), 1)
OUT_DIR = ROOT / "results" / "paper_predictions"
FIG_DIR = ROOT / "Paper" / "paper_energy_informatics" / "figures"


def checkpoint_path(variant: str, fold: int, seed: int, head: str) -> Path:
    return ROOT / "results" / "paper" / f"fold{fold}_{TAGS[variant]}" / "api_forecast" / f"seed_{seed}" / f"{head}_api_forecast.pt"


def calibrated_quantiles(head: str, validation: dict, test: dict) -> tuple[np.ndarray, np.ndarray]:
    """Raw and calibrated forecasts on the shared nine-quantile grid [N, 24, 9]."""

    if head == "gaussian":
        factor = fit_gaussian_scale_calibration(validation["target_kw"], validation["mean_kw"], validation["scale_kw"])
        raw = gaussian_quantiles(test["mean_kw"], test["scale_kw"], QUANTILES)
        return raw, gaussian_quantiles(test["mean_kw"], test["scale_kw"] * factor, QUANTILES)
    delta = fit_quantile_interval_calibration(validation["target_kw"], validation["quantiles_kw"], QUANTILES)
    return test["quantiles_kw"], expand_quantiles(test["quantiles_kw"], QUANTILES, delta)


def predict(device_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    for variant in TAGS:
        for fold in FOLDS:
            first = load_checkpoint(checkpoint_path(variant, fold, SEEDS[0], HEADS[0]))
            data_cfg, _ = config_from_metadata(first["metadata"])
            case = prepare_case(load_complete_dataframe(data_cfg), data_cfg=data_cfg, weather_case="api_forecast")
            origins = np.asarray(case.dataframe.index)[case.origins.test]
            for seed in SEEDS:
                for head in HEADS:
                    checkpoint = load_checkpoint(checkpoint_path(variant, fold, seed, head))
                    _, experiment_cfg = config_from_metadata(checkpoint["metadata"])
                    model = build_checkpoint_model(checkpoint, case, experiment_cfg, device)
                    _, validation_loader, test_loader = case.loaders(seed=experiment_cfg.seed)
                    validation = collect_predictions(head, model, validation_loader, case, device)
                    test = collect_predictions(head, model, test_loader, case, device)
                    raw, calibrated = calibrated_quantiles(head, validation, test)
                    path = OUT_DIR / f"{variant}_{head}_fold{fold}_seed{seed}.npz"
                    np.savez_compressed(path, target=test["target_kw"], raw=raw.astype(np.float32),
                                        calibrated=calibrated.astype(np.float32), first_target=origins)
                    print(f"saved {path.name}: {len(origins)} test origins")


def load(variant: str, head: str, seed: int) -> dict[str, np.ndarray]:
    parts = [np.load(OUT_DIR / f"{variant}_{head}_fold{fold}_seed{seed}.npz") for fold in FOLDS]
    return {key: np.concatenate([part[key] for part in parts]) for key in ("target", "raw", "calibrated", "first_target")}


def crps_grid(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CRPS approximation on the nine-quantile grid (2 x mean pinball loss), per window and lead."""

    diff = y[..., None] - q
    return 2.0 * np.mean(np.maximum(QUANTILES * diff, (QUANTILES - 1.0) * diff), axis=-1)


def horizon_table() -> pd.DataFrame:
    """Calibrated metrics per lead time, pooled over both test periods, mean over seeds."""

    median = int(np.argmin(np.abs(QUANTILES - 0.5)))
    rows = []
    for variant in TAGS:
        for head in HEADS:
            for seed in SEEDS:
                data = load(variant, head, seed)
                y, q = data["target"], data["calibrated"]
                error = np.abs(q[..., median] - y)
                for lead in range(y.shape[1]):
                    rows.append({"model": f"{variant} {head}", "seed": seed, "lead": lead + 1,
                                 "mae": error[:, lead].mean(), "mape": 100 * np.mean(error[:, lead] / np.abs(y[:, lead])),
                                 "crps": crps_grid(y[:, lead], q[:, lead]).mean(),
                                 "picp80": np.mean((y[:, lead] >= q[:, lead, 0]) & (y[:, lead] <= q[:, lead, -1])),
                                 "miw80": np.mean(q[:, lead, -1] - q[:, lead, 0])})
    return pd.DataFrame(rows).groupby(["model", "lead"], sort=False).mean(numeric_only=True).drop(columns="seed")


def horizon_figure(table: pd.DataFrame) -> None:
    colors = {"A": "#0e8088", "B": "#b85450"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for variant, color in colors.items():
        part = table.loc[f"{variant} gaussian"]
        axes[0].plot(part.index, part["mae"], color=color, label=f"{variant}, MAE")
        axes[0].plot(part.index, part["crps"], color=color, linestyle="--", label=f"{variant}, CRPS")
        axes[1].plot(part.index, part["picp80"], color=color, label=f"Version {variant}")
        axes[2].plot(part.index, part["miw80"], color=color, label=f"Version {variant}")
    axes[1].axhline(0.8, color="0.5", linewidth=0.8, linestyle=":")
    for ax, label in zip(axes, ("MAE and CRPS [kW]", r"PICP$_{80\%}$", r"MIW$_{80\%}$ [kW]")):
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel(label)
        ax.set_xticks([1, 6, 12, 18, 24])
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ab_horizon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecast_figure(seed: int, start: str, days: int) -> None:
    """Consecutive day-ahead forecasts (origins at local midnight) for A and B (Gaussian head)."""

    first_target = pd.Timestamp(start)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.2]})
    colors = {"A": "#0e8088", "B": "#b85450"}
    median = int(np.argmin(np.abs(QUANTILES - 0.5)))
    for ax, variant in zip(axes[:2], TAGS):
        data = load(variant, "gaussian", seed)
        index = pd.DatetimeIndex(data["first_target"])
        times, y, mean, lower, upper, raw_lower, raw_upper = [], [], [], [], [], [], []
        for day in range(days):
            hit = np.where(index == first_target + pd.Timedelta(days=day))[0]
            if not len(hit):
                raise ValueError(f"No test origin at {first_target + pd.Timedelta(days=day)}")
            i = hit[0]
            times.append(pd.date_range(index[i], periods=24, freq="h"))
            y.append(data["target"][i]); mean.append(data["calibrated"][i, :, median])
            lower.append(data["calibrated"][i, :, 0]); upper.append(data["calibrated"][i, :, -1])
            raw_lower.append(data["raw"][i, :, 0]); raw_upper.append(data["raw"][i, :, -1])
        times = times[0].append(times[1:]).tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)
        y, mean, lower, upper, raw_lower, raw_upper = map(
            np.concatenate, (y, mean, lower, upper, raw_lower, raw_upper)
        )
        ax.fill_between(times, raw_lower, raw_upper, color=colors[variant], alpha=0.25, linewidth=0,
                        label="80 % interval, raw")
        ax.plot(times, lower, color=colors[variant], linewidth=0.9, linestyle="--",
                label="80 % interval, calibrated")
        ax.plot(times, upper, color=colors[variant], linewidth=0.9, linestyle="--")
        ax.plot(times, y, color="black", linewidth=1.0, label="Observed load")
        ax.plot(times, mean, color=colors[variant], linewidth=1.2, label=f"Version {variant}, Gaussian mean")
        axes[2].plot(times, mean - y, color=colors[variant], linewidth=1.0, label=f"Version {variant}")
        ax.set_ylabel("Load [kW]")
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.2 * (high - low))  # room for the legend
        ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=7)
        for day in range(1, days):
            for a in axes:
                a.axvline(times[24 * day], color="0.75", linewidth=0.6, linestyle=":")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylabel("Mean $-$ observed [kW]")
    axes[2].legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ab_forecast_example.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reliability_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    styles = {"gaussian": ("#1f77b4", "Gaussian"), "quantile": ("#ff7f0e", "Quantile")}
    for ax, variant in zip(axes, TAGS):
        for head, (color, label) in styles.items():
            for stage, ls in (("raw", "--"), ("calibrated", "-")):
                freq = np.mean([[np.mean(d["target"] <= d[stage][..., k]) for k in range(len(QUANTILES))]
                                for d in (load(variant, head, s) for s in SEEDS)], axis=0)
                ax.plot(QUANTILES, freq, ls, marker="o", ms=3, color=color,
                        label=f"{label}, {'cal.' if stage == 'calibrated' else 'raw'}")
        ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8)
        ax.set_title(f"Version {variant}")
        ax.set_xlabel("Nominal quantile level")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ab_reliability_diagram.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("predict", "report"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="Seed shown in the forecast example.")
    parser.add_argument("--start", default="2026-01-13 23:00", help="First target hour (UTC) of the example.")
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.device)
        return
    pd.set_option("display.width", 160)
    table = horizon_table()
    print(table.loc[["A gaussian", "B gaussian"]].iloc[[0, 23, 24, 47]].round(3).to_string())
    table.to_csv(ROOT / "results" / "paper_horizon_metrics.csv")
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    horizon_figure(table)
    forecast_figure(args.seed, args.start, args.days)
    reliability_figure()
    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
