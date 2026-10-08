#!/usr/bin/env bash
# All runs and outputs of the revised paper (rolling folds 1 and 2, seeds 42-46).
#
# Usage (from the repository root): bash src/scripts/paper_runs.sh <stage>
#   aedl       versions A and B, all heads, archived forecast and weather ablation
#   baselines  seasonal naive, plain AEDL, plain LSTM (global and window scaling), TFT
#   start      training-start sensitivity of A and B (deterministic heads)
#   hpo        hyperparameter search and re-check of A, B, and TFT (fold 1 only)
#   analysis   compute benchmark, data section, result tables, horizon metrics, figures
set -euo pipefail

PY=/opt/miniconda3/envs/dhn-predict/bin/python
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46}"
RUN=("$PY" src/scripts/run_experiments.py --protocol paper --calendar-tz Europe/Berlin)

case "${1:-}" in
  aedl)
    for F in 1 2; do for S in "${SEEDS[@]}"; do for V in A B; do
      for CASES in api_forecast "historical observed_future"; do
        "${RUN[@]}" --version "$V" --models all --weather-cases $CASES --fold "$F" --seed "$S"
      done
    done; done; done
    ;;
  baselines)
    for F in 1 2; do
      "$PY" src/scripts/run_simple_baselines.py --fold "$F"
      "$PY" src/scripts/run_tft.py --fold "$F" --seeds "${SEEDS[@]}"
      for S in "${SEEDS[@]}"; do
        for BASE in plain lstm lstm-ws; do
          "${RUN[@]}" --baseline "$BASE" --models all --weather-cases api_forecast --fold "$F" --seed "$S"
        done
      done
    done
    ;;
  start)
    for F in 1 2; do for S in "${SEEDS[@]}"; do for START in 2023-01-01 2024-01-01; do
      for V in A B; do
        "${RUN[@]}" --version "$V" --models deterministic --start-date "$START" --weather-cases api_forecast \
          --fold "$F" --seed "$S"
      done
    done; done; done
    ;;
  hpo)
    for V in A B tft; do
      "$PY" src/scripts/tune_hpo.py search --variant "$V" --trials 40
      "$PY" src/scripts/tune_hpo.py recheck --variant "$V"
    done
    ;;
  analysis)
    "$PY" src/scripts/benchmark_compute.py
    "$PY" src/scripts/paper_data.py
    "$PY" src/scripts/paper_predictions.py predict
    "$PY" src/scripts/paper_predictions.py report
    "$PY" src/scripts/paper_latex_tables.py
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
