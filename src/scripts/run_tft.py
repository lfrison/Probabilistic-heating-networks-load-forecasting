"""TFT baseline (NeuralForecast) on the AEDL forecast windows of a protocol (paper or ulm).

Forecast origins, split, target scaling, early stopping, and evaluation follow
``run_experiments.py --protocol paper``. Only the network is taken from
NeuralForecast. Its windowing and trainer are bypassed because it cannot
represent a weather-forecast trajectory issued at each forecast origin.

Inputs at forecast origin t (first target hour): load for t-48 ... t-1 and, for
t-48 ... t+23, temperature and GHI (observed in the past, archived forecast
issued at t in the future) and the calendar features (local time). Each window
is normalised with the median/MAD of its history, as NeuralForecast's robust
scaler. TFT is trained with the pinball loss on the nine AEDL quantile levels,
the checkpoint with the lowest validation CRPS is kept, and the quantile
calibration of Quantile-AEDL is applied.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_forecasting.metrics import (
    expand_quantiles,
    fit_quantile_interval_calibration,
    point_metrics,
    quantile_quality_metrics,
)
from load_forecasting.pipeline import (
    CALENDAR_COLS,
    PreparedCase,
    _future_values_for_origins,
    load_complete_dataframe,
    prepare_case,
    resolve_device,
    seed_everything,
    split_summary,
    write_json,
)
from load_forecasting.protocols import PAPER_FOLDS, get_protocol

RESULTS_DIR = ROOT / "results" / "paper_baselines"


@dataclass(frozen=True)
class BaselineConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 256
    epochs: int = 50
    es_patience: int = 20
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    # NeuralForecast defaults except hidden_size (128 -> 64, 0.52 M parameters).
    tft_hidden_size: int = 64
    tft_n_head: int = 4
    tft_dropout: float = 0.1
    seed: int = 42


@dataclass
class OriginArrays:
    insample_y: np.ndarray  # [N, L, 1], target-scaled
    futr_exog: np.ndarray  # [N, L+H, F], scaled
    target_scaled: np.ndarray  # [N, H]
    target_kw: np.ndarray  # [N, H]


def paper_data_config(fold: int):
    return replace(get_protocol("paper").data, **PAPER_FOLDS[fold], calendar_timezone="Europe/Berlin")


def futr_exog_names(weather_columns: tuple[str, ...]) -> list[str]:
    return [*weather_columns, *CALENDAR_COLS]


def build_origin_arrays(case: PreparedCase, origins: np.ndarray, target_col: str,
                        weather_columns: tuple[str, ...], past_len: int,
                        pred_len: int) -> OriginArrays:
    df = case.dataframe
    rows_hist = origins[:, None] + np.arange(-past_len, 0)[None, :]
    rows_all = origins[:, None] + np.arange(-past_len, pred_len)[None, :]
    rows_future = rows_all[:, past_len:]

    target_raw = df[target_col].to_numpy(dtype=np.float64)
    target_mean = float(case.scaler_target.mean_[0])
    target_scale = float(case.scaler_target.scale_[0])
    insample_y = (target_raw[rows_hist] - target_mean) / target_scale
    target_kw = target_raw[rows_future]

    # One scale per channel for history and future, from the AEDL training-set scaler.
    past_index = {column: i for i, column in enumerate(case.past_cols)}
    exog_columns = futr_exog_names(weather_columns)
    mean = np.asarray([case.scaler_past.mean_[past_index[c]] for c in exog_columns])
    scale = np.asarray([case.scaler_past.scale_[past_index[c]] for c in exog_columns])

    weather_hist = df[list(weather_columns)].to_numpy(dtype=np.float64)[rows_hist]
    weather_future = _future_values_for_origins(
        df, origins, case.future_cols_by_horizon
    ).astype(np.float64)
    weather = np.concatenate([weather_hist, weather_future], axis=1)
    calendar = df[list(CALENDAR_COLS)].to_numpy(dtype=np.float64)[rows_all]
    futr_exog = (np.concatenate([weather, calendar], axis=-1) - mean) / scale

    arrays = OriginArrays(
        insample_y=insample_y[..., None].astype(np.float32),
        futr_exog=futr_exog.astype(np.float32),
        target_scaled=((target_kw - target_mean) / target_scale).astype(np.float32),
        target_kw=target_kw,
    )
    for name in ("insample_y", "futr_exog", "target_scaled"):
        if not np.isfinite(getattr(arrays, name)).all():
            raise ValueError(f"Non-finite values in baseline input {name!r}.")
    return arrays


def build_model(cfg: BaselineConfig, *, past_len: int, pred_len: int, futr_names: list[str]) -> nn.Module:
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import TFT

    return TFT(
        h=pred_len,
        input_size=past_len,
        futr_exog_list=futr_names,
        scaler_type="identity",
        random_seed=cfg.seed,
        loss=MQLoss(quantiles=list(cfg.quantiles)),
        hidden_size=cfg.tft_hidden_size,
        n_head=cfg.tft_n_head,
        dropout=cfg.tft_dropout,
    )


def forward(model: nn.Module, insample_y: torch.Tensor, futr_exog: torch.Tensor) -> torch.Tensor:
    return model(
        {
            "insample_y": insample_y,
            "insample_mask": torch.ones_like(insample_y),
            "futr_exog": futr_exog,
            "hist_exog": None,
            "stat_exog": None,
        }
    )


def run_model(model: nn.Module, insample_y: torch.Tensor, futr_exog: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass on window-normalised inputs (median/MAD of the history).

    Returns the output in normalised units and the target shift and scale.
    """

    past_len = insample_y.shape[1]
    history = torch.cat([insample_y, futr_exog[:, :past_len]], dim=-1)  # [B, L, 1+F]
    loc = history.median(dim=1, keepdim=True).values
    scale = (history - loc).abs().median(dim=1, keepdim=True).values
    # NeuralForecast replaces MAD = 0 by the std-based estimate.
    std = history.std(dim=1, keepdim=True, unbiased=False)
    scale = torch.where(scale > 0, scale, std * 0.6744897501960817)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale) + 1e-6
    insample_n = (insample_y - loc[..., :1]) / scale[..., :1]
    futr_n = (futr_exog - loc[..., 1:]) / scale[..., 1:]
    return forward(model, insample_n, futr_n), loc[..., :1], scale[..., :1]


@torch.no_grad()
def predict_scaled(model: nn.Module, arrays: OriginArrays, device: torch.device,
                   batch_size: int = 1024) -> np.ndarray:
    """Predict in globally scaled target units."""

    model.eval()
    outputs = []
    for start in range(0, len(arrays.insample_y), batch_size):
        stop = start + batch_size
        output, loc, scale = run_model(
            model,
            torch.from_numpy(arrays.insample_y[start:stop]).to(device),
            torch.from_numpy(arrays.futr_exog[start:stop]).to(device),
        )
        outputs.append((output * scale + loc).float().cpu().numpy())
    return np.concatenate(outputs)


def to_kw(values: np.ndarray, case: PreparedCase) -> np.ndarray:
    return values * float(case.scaler_target.scale_[0]) + float(case.scaler_target.mean_[0])


def train_model(model: nn.Module, train: OriginArrays, validation: OriginArrays,
                case: PreparedCase, cfg: BaselineConfig, device: torch.device,
                on_epoch=None) -> tuple[nn.Module, dict[str, object]]:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.insample_y),
            torch.from_numpy(train.futr_exog),
            torch.from_numpy(train.target_scaled),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.3
    )
    quantiles = np.asarray(cfg.quantiles, dtype=float)
    quantiles_t = torch.as_tensor(cfg.quantiles, dtype=torch.float32, device=device)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))

    best_crps = best_mae = float("nan")
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        accumulated_loss = 0.0
        sample_count = 0
        for insample_y, futr_exog, target in loader:
            insample_y = insample_y.to(device)
            futr_exog = futr_exog.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, loc, scale = run_model(model, insample_y, futr_exog)
            # As in NeuralForecast, the loss is computed in window-normalised units.
            target = (target - loc[..., 0]) / scale[..., 0]
            error = target.unsqueeze(-1) - output
            loss = torch.maximum(quantiles_t * error, (quantiles_t - 1.0) * error).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            accumulated_loss += float(loss.item()) * len(target)
            sample_count += len(target)
        scheduler.step()

        output_val = predict_scaled(model, validation, device)
        current_mae = float(np.mean(np.abs(to_kw(output_val[..., median_index], case) - validation.target_kw)))
        current_crps = quantile_quality_metrics(
            validation.target_kw, np.sort(to_kw(output_val, case), axis=-1), quantiles
        )["crps_shared_quantile_grid_kw"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": accumulated_loss / max(sample_count, 1),
                "validation_mae_kw": current_mae,
                "validation_crps_kw": current_crps,
            }
        )
        print(f"tft | seed {cfg.seed} | epoch {epoch:02d} | val MAE {current_mae:.2f} kW | "
              f"val CRPS {current_crps:.2f} kW", flush=True)
        if on_epoch is not None:
            on_epoch(epoch, current_mae)
        if current_crps < best_score - 0.05:
            best_score = current_crps
            best_mae, best_crps = current_mae, current_crps
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
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
        "epochs_completed": len(history),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "history": history,
    }


def evaluate(model: nn.Module, validation: OriginArrays, test: OriginArrays,
             case: PreparedCase, cfg: BaselineConfig, device: torch.device) -> dict[str, object]:
    quantiles = np.asarray(cfg.quantiles, dtype=float)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    q_test = np.sort(to_kw(predict_scaled(model, test, device), case), axis=-1)
    q_val = np.sort(to_kw(predict_scaled(model, validation, device), case), axis=-1)
    delta = fit_quantile_interval_calibration(validation.target_kw, q_val, quantiles)
    y = test.target_kw
    return {
        "point": point_metrics(y, q_test[..., median_index]),
        "probabilistic_raw": quantile_quality_metrics(y, q_test, quantiles),
        "probabilistic_calibrated": quantile_quality_metrics(
            y, expand_quantiles(q_test, quantiles, delta), quantiles
        ),
        "calibration": {"symmetric_delta_kw": delta},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=("paper", "ulm"), default="paper")
    parser.add_argument("--fold", type=int, choices=tuple(PAPER_FOLDS), help="Required for the paper protocol.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/paper_baselines/fold<F>_localcal_ws-robust_selcrps (paper) "
                        "or results/ulm (ulm).")
    args = parser.parse_args()
    if args.protocol == "paper":
        if args.fold is None:
            parser.error("--fold is required for the paper protocol.")
        data_cfg, weather_case = paper_data_config(args.fold), "api_forecast"
        output_dir = args.output_dir or RESULTS_DIR / f"fold{args.fold}_localcal_ws-robust_selcrps"
    else:
        protocol = get_protocol(args.protocol)
        data_cfg, weather_case = protocol.data, protocol.weather_cases[0]
        output_dir = args.output_dir or ROOT / "results" / args.protocol

    warnings.filterwarnings("ignore", module="neuralforecast")
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    case = prepare_case(load_complete_dataframe(data_cfg), data_cfg=data_cfg, weather_case=weather_case)
    summary = split_summary(case)
    print(summary)
    arrays_kwargs = dict(
        target_col=data_cfg.target_col,
        weather_columns=tuple(data_cfg.weather_columns),
        past_len=data_cfg.past_len,
        pred_len=data_cfg.pred_len,
    )
    train = build_origin_arrays(case, case.origins.train, **arrays_kwargs)
    validation = build_origin_arrays(case, case.origins.validation, **arrays_kwargs)
    test = build_origin_arrays(case, case.origins.test, **arrays_kwargs)
    futr_names = futr_exog_names(tuple(data_cfg.weather_columns))
    device = resolve_device(args.device)

    for seed in args.seeds:
        cfg = BaselineConfig(seed=seed)
        if args.epochs is not None:
            cfg = replace(cfg, epochs=args.epochs)
        seed_everything(seed)
        model = build_model(cfg, past_len=data_cfg.past_len, pred_len=data_cfg.pred_len,
                            futr_names=futr_names).to(device)
        model, training = train_model(model, train, validation, case, cfg, device)
        metrics = evaluate(model, validation, test, case, cfg, device)
        write_json(
            output_dir / "tft" / f"seed_{seed}" / "metrics.json",
            {
                "model_name": "tft",
                "config": asdict(cfg),
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "split_summary": summary,
                "training": training,
                "metrics": metrics,
            },
        )
        point, cal = metrics["point"], metrics["probabilistic_calibrated"]
        print(f"tft seed {seed}: MAE {point['mae_kw']:.2f} kW | CRPS cal. {cal['crps_shared_quantile_grid_kw']:.2f} kW "
              f"| PICP80 cal. {cal['picp80']:.3f}", flush=True)


if __name__ == "__main__":
    main()
