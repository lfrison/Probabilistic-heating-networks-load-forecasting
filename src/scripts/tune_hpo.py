"""Hyperparameter optimisation for AEDL variant A, variant B and TFT (Optuna).

All studies use fold 1 only: training up to December 2024, validation
January-May 2025. Fold 2's validation period is fold 1's test period, so no
test data enter the search. The objective is the best validation MAE [kW];
AEDL trials use the deterministic head, and the selected backbone is then
shared by all heads. Every trial uses the final training protocol (AdamW,
cosine schedule, early stopping) with ``--max-epochs`` and ``--patience``.

Variants (fixed inputs: calendar and lag-24 decoder inputs, local-time
calendar, archived weather forecasts):

* A:   window load scaling, no output shortcut
* B:   global scaling, linear output shortcut
* tft: NeuralForecast TFT with robust window scaling

Usage:
    python src/scripts/tune_hpo.py search  --variant A --trials 40
    python src/scripts/tune_hpo.py recheck --variant A --top 3 --seeds 43 44 45
    python src/scripts/tune_hpo.py report  --variant A

Studies are stored in results/hpo/<variant>/study.db and can be interrupted
and resumed by repeating the ``search`` command.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "src" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from load_forecasting.models import build_model
from load_forecasting.pipeline import (
    ExperimentConfig,
    load_complete_dataframe,
    prepare_case,
    resolve_device,
    seed_everything,
    split_summary,
)
from load_forecasting.protocols import PAPER_FOLDS, get_protocol

HPO_DIR = ROOT / "results" / "hpo"
FOLD = 1

# Current configuration, evaluated as the first trial of every study.
CURRENT = {
    "aedl": dict(hidden_size=96, num_layers=1, attn_heads=8, dropout=0.2, lr=3e-4,
                 weight_decay=1e-2, batch_size=256),
    "tft": dict(tft_hidden_size=64, tft_n_head=4, tft_dropout=0.1, lr=3e-4, batch_size=256),
}


def suggest(trial: optuna.Trial, variant: str) -> dict:
    if variant == "tft":
        return dict(
            tft_hidden_size=trial.suggest_categorical("tft_hidden_size", [32, 64, 128]),
            tft_n_head=trial.suggest_categorical("tft_n_head", [2, 4, 8]),
            tft_dropout=trial.suggest_float("tft_dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        )
    return dict(
        hidden_size=trial.suggest_categorical("hidden_size", [64, 96, 128, 192]),
        num_layers=trial.suggest_int("num_layers", 1, 2),
        attn_heads=trial.suggest_categorical("attn_heads", [4, 8]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
    )


class Tuner:
    """Loads fold-1 data once and trains one configuration per call."""

    def __init__(self, variant: str, max_epochs: int, patience: int, device: str):
        self.variant = variant
        self.max_epochs = max_epochs
        self.patience = patience
        data_cfg = replace(
            get_protocol("paper").data,
            **PAPER_FOLDS[FOLD],
            calendar_timezone="Europe/Berlin",
            decoder_calendar=variant != "tft",
            decoder_lag24=variant != "tft",
            window_load_scaling=variant == "A",
        )
        self.data_cfg = data_cfg
        df = load_complete_dataframe(data_cfg)
        self.case = prepare_case(df, data_cfg=data_cfg, weather_case="api_forecast")
        self.summary = split_summary(self.case)
        if variant == "tft":
            import run_tft as nf

            self.nf = nf
            kwargs = dict(target_col=data_cfg.target_col, weather_columns=tuple(data_cfg.weather_columns),
                          past_len=data_cfg.past_len, pred_len=data_cfg.pred_len)
            self.train_arrays = nf.build_origin_arrays(self.case, self.case.origins.train, **kwargs)
            self.val_arrays = nf.build_origin_arrays(self.case, self.case.origins.validation, **kwargs)
            self.futr_names = nf.futr_exog_names(tuple(data_cfg.weather_columns))
        self.device = resolve_device(device)

    def train(self, params: dict, seed: int, on_epoch=None) -> dict:
        started = time.time()
        if self.variant == "tft":
            nf = self.nf
            cfg = replace(nf.BaselineConfig(), seed=seed, epochs=self.max_epochs, es_patience=self.patience, **params)
            seed_everything(seed)
            model = nf.build_model(cfg, past_len=self.data_cfg.past_len, pred_len=self.data_cfg.pred_len,
                                   futr_names=self.futr_names).to(self.device)
            _, training = nf.train_model(model, self.train_arrays, self.val_arrays, self.case,
                                         cfg, self.device, on_epoch=on_epoch)
        else:
            from run_experiments import train_model

            params = dict(params)
            batch_size = params.pop("batch_size")
            for dataset in (self.case.train_dataset, self.case.validation_dataset, self.case.test_dataset):
                dataset.batch_size = batch_size
            cfg = replace(ExperimentConfig(), seed=seed, epochs=self.max_epochs, es_patience=self.patience,
                          output_skip="linear" if self.variant == "B" else "none", **params)
            seed_everything(seed)
            model = build_model(
                "deterministic",
                n_past_features=len(self.case.past_cols),
                future_dim=self.case.train_dataset.future.shape[-1],
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                attention_heads=cfg.attn_heads,
                dropout=cfg.dropout,
                pred_len=self.data_cfg.pred_len,
                quantiles=cfg.quantiles,
                output_skip=cfg.output_skip,
                target_index=self.case.past_cols.index(self.data_cfg.target_col),
            ).to(self.device)
            _, training = train_model("deterministic", model, self.case, cfg, self.device, on_epoch=on_epoch)
        history = training["history"]
        best = int(np.argmin([h["validation_mae_kw"] for h in history])) + 1
        return {
            "best_validation_mae_kw": float(training["best_validation_mae_kw"]),
            "best_epoch": best,
            "epochs_completed": len(history),
            "minutes": (time.time() - started) / 60.0,
        }


def study_for(variant: str, create: bool = True) -> optuna.Study:
    folder = HPO_DIR / variant
    folder.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{folder / 'study.db'}"
    if not create:
        return optuna.load_study(study_name=f"{variant}_fold{FOLD}", storage=storage)
    return optuna.create_study(
        study_name=f"{variant}_fold{FOLD}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, "state": t.state.name, "value": t.value, **t.params, **t.user_attrs})
    return pd.DataFrame(rows)


def cmd_search(args: argparse.Namespace) -> None:
    tuner = Tuner(args.variant, args.max_epochs, args.patience, args.device)
    print(f"{args.variant}: {tuner.summary['train_origins']} train / {tuner.summary['validation_origins']} validation origins "
          f"(fold {FOLD}); test data are not used.", flush=True)
    study = study_for(args.variant)
    if not study.trials:
        study.enqueue_trial(CURRENT["tft" if args.variant == "tft" else "aedl"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, args.variant)

        def report(epoch: int, value: float) -> None:
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = tuner.train(params, seed=args.seed, on_epoch=report)
        for key, value in result.items():
            trial.set_user_attr(key, value)
        return result["best_validation_mae_kw"]

    remaining = max(0, args.trials - len([t for t in study.trials if t.state.is_finished()]))
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    cmd_report(args)


def cmd_recheck(args: argparse.Namespace) -> None:
    study = study_for(args.variant, create=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value)[: args.top]
    tuner = Tuner(args.variant, args.max_epochs, args.patience, args.device)
    results = []
    for rank, trial in enumerate(top, start=1):
        values = [trial.value]
        for seed in args.seeds:
            result = tuner.train(dict(trial.params), seed=seed)
            values.append(result["best_validation_mae_kw"])
            print(f"rank {rank} (trial {trial.number}) seed {seed}: validation MAE {values[-1]:.2f} kW", flush=True)
        results.append({"rank": rank, "trial": trial.number, "params": trial.params,
                        "validation_mae_seed42_kw": trial.value, "validation_mae_recheck_kw": values[1:],
                        "mean_kw": float(np.mean(values)), "std_kw": float(np.std(values, ddof=1))})
    path = HPO_DIR / args.variant / "recheck.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRe-check (seed 42 + {len(args.seeds)} further seeds), saved to {path}:")
    for r in sorted(results, key=lambda r: r["mean_kw"]):
        print(f"  trial {r['trial']:>3}: validation MAE {r['mean_kw']:.2f} ± {r['std_kw']:.2f} kW | {r['params']}")


def cmd_report(args: argparse.Namespace) -> None:
    study = study_for(args.variant, create=False)
    frame = trials_frame(study)
    folder = HPO_DIR / args.variant
    frame.to_csv(folder / "trials.csv", index=False)
    done = frame[frame.state == "COMPLETE"].sort_values("value")
    counts = frame.state.value_counts().to_dict()
    minutes = frame.get("minutes", pd.Series(dtype=float)).fillna(0).sum()
    print(f"\n{args.variant}: {len(frame)} trials {counts} | tuning time of completed trials {minutes / 60:.1f} h")
    if done.empty:
        return
    cols = [c for c in done.columns if c not in ("state",)]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(done[cols].head(10).round(4).to_string(index=False))
    current = frame.iloc[0]
    print(f"\ncurrent configuration (trial 0): {current['value']:.2f} kW" if pd.notna(current["value"]) else "")
    best = study.best_trial
    print(f"best trial {best.number}: {best.value:.2f} kW | {best.params}")
    (folder / "best.json").write_text(json.dumps({"trial": best.number, "value": best.value, "params": best.params},
                                                  indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("search", "recheck", "report"))
    parser.add_argument("--variant", choices=("A", "B", "tft"), required=True)
    parser.add_argument("--trials", type=int, default=40, help="Total number of trials in the study.")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hpo-dir", type=Path, default=HPO_DIR, help="Default: results/hpo.")
    args = parser.parse_args()
    globals()["HPO_DIR"] = args.hpo_dir

    warnings.filterwarnings("ignore")
    for name in ("lightning_fabric", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    {"search": cmd_search, "recheck": cmd_recheck, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
