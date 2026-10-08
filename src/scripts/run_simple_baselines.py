"""Seasonal-naive baselines on the paper windows (no training).

For each forecast origin t (first target hour) and lead h = 0..23:

* seasonal naive: y(t+h-24)
* probabilistic seasonal naive: seasonal naive plus the empirical quantiles of
  its validation errors per lead (nine AEDL quantile levels).

The origins, splits and metric functions are those of ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_forecasting.metrics import point_metrics, quantile_quality_metrics
from load_forecasting.pipeline import load_complete_dataframe, prepare_case, split_summary, write_json
from load_forecasting.protocols import PAPER_FOLDS, get_protocol

QUANTILES = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))


def naive_forecasts(load: np.ndarray, origins: np.ndarray, pred_len: int) -> dict[str, np.ndarray]:
    rows = origins[:, None] + np.arange(pred_len)[None, :]
    forecasts = {
        "seasonal_naive_daily": load[rows - 24],
    }
    for name, values in forecasts.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Missing load values in the {name} forecast.")
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, choices=tuple(PAPER_FOLDS), required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/paper_baselines/fold<F>_localcal.")
    args = parser.parse_args()

    cfg = replace(get_protocol("paper").data, **PAPER_FOLDS[args.fold])
    df = load_complete_dataframe(cfg)
    case = prepare_case(df, data_cfg=cfg, weather_case="api_forecast")
    load = df[cfg.target_col].to_numpy(dtype=float)
    pred_len = cfg.pred_len
    rows = lambda origins: origins[:, None] + np.arange(pred_len)[None, :]
    y_val, y_test = load[rows(case.origins.validation)], load[rows(case.origins.test)]
    val = naive_forecasts(load, case.origins.validation, pred_len)
    test = naive_forecasts(load, case.origins.test, pred_len)
    output_root = args.output_dir or ROOT / "results" / "paper_baselines" / f"fold{args.fold}_localcal"

    results = {}
    for name in val:
        results[name] = {
            "val": float(np.mean(np.abs(val[name] - y_val))),
            "metrics": {"point": point_metrics(y_test, test[name])},
        }

    # Probabilistic seasonal naive: empirical validation error quantiles per lead.
    residuals = y_val - val["seasonal_naive_daily"]
    offsets = np.quantile(residuals, QUANTILES, axis=0).T  # [lead, quantile]
    q_test = test["seasonal_naive_daily"][..., None] + offsets[None, :, :]
    probabilistic = quantile_quality_metrics(y_test, q_test, QUANTILES)
    results["seasonal_naive_quantile"] = {
        "val": results["seasonal_naive_daily"]["val"],
        "metrics": {
            "point": point_metrics(y_test, q_test[..., int(np.argmin(np.abs(QUANTILES - 0.5)))]),
            # Already fitted on validation errors; no further calibration step.
            "probabilistic_raw": probabilistic,
            "probabilistic_calibrated": probabilistic,
        },
    }

    summary = split_summary(case)
    for name, result in results.items():
        write_json(
            output_root / name / "seed_0" / "metrics.json",
            {
                "model_name": name,
                "config": {"seed": 0, "fold": args.fold},
                "parameters": 0,
                "split_summary": summary,
                "training": {"best_validation_mae_kw": result["val"], "best_epoch": 0},
                "metrics": result["metrics"],
            },
        )
        point = result["metrics"]["point"]
        extra = ""
        if "probabilistic_calibrated" in result["metrics"]:
            q = result["metrics"]["probabilistic_calibrated"]
            extra = f" | CRPS {q['crps_shared_quantile_grid_kw']:.1f} kW | PICP80 {q['picp80']:.3f} | MIW80 {q['miw80_kw']:.0f} kW"
        print(f"fold {args.fold} {name:<24} MAE {point['mae_kw']:7.1f} kW | MAPE {point['mape_percent']:5.2f} %{extra}")


if __name__ == "__main__":
    main()
