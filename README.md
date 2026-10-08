# AEDL probabilistic load forecasting

This repository trains deterministic, Gaussian, and quantile variants of a
direct multi-horizon attention encoder-decoder LSTM (AEDL) for day-ahead
thermal-load forecasting. Two versions handle the growing load of expanding
district heating networks. Named experiment protocols bind each forecasting
task to its data, inputs, horizon, and chronological split.

Only the Ulm data are publicly available. All other protocols use proprietary
data that are not part of this repository.

## Methods

All variants encode the historical load, weather, and cyclical calendar inputs
with feature attention, an LSTM, and temporal self-attention. The decoder
produces the complete forecast trajectory in one pass without autoregressive
target feedback or teacher forcing.

- **Deterministic AEDL** predicts one value per horizon and uses MAE loss.
- **Gaussian AEDL** predicts a mean and standard deviation. An MAE warm-up is
  followed by Gaussian NLL with small MAE and variance penalties. Validation
  data determine one scale-calibration factor.
- **Quantile AEDL** predicts ordered quantiles 0.1 to 0.9 using pinball loss
  with median-MAE and width penalties. Validation data determine a symmetric
  interval adjustment.

The model has one LSTM layer with 96 hidden units. Its decoder additionally
receives the calendar features of the target hours and the values 24 h earlier.
Two versions handle the growing load:

- **Version A** (default) divides the load of each window by the mean load of
  its 48 h history (`--version A`).
- **Version B** adds a linear map of the last 24 h of load to the output
  (`--version B`).

The probabilistic heads keep the checkpoint with the lowest validation CRPS.

`observed_future` uses true future weather and is therefore a perfect-weather
benchmark, not an operational forecast. `historical` provides no decoder
weather. `api_forecast` uses an origin-coherent operational forecast
trajectory (paper data only).

## Installation

```bash
git clone https://github.com/lfrison/Probabilistic-heating-networks-load-forecasting.git load-forecasting
cd load-forecasting
conda env create -f env/environment.yml
conda activate dhn-predict
```

Prepare the public Ulm data once (creates `data/data_Ulm.pkl`):

```bash
python src/scripts/prepare_ulm_data.py
```

The Ulm source data are publicly available in the
[`finkenrm/deepDHC-user-guide`](https://github.com/finkenrm/deepDHC-user-guide)
repository. The CSV used here is its
[`data/data.csv`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/data/data.csv).
The upstream repository includes a permissive MIT-style
[`LICENSE`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/LICENSE).
See [`data/README.md`](data/README.md) for provenance, units, and corrections.

## Quick start with the Ulm data

All commands are run from the repository root in the activated environment.
Versions A and B with all three heads, and the TFT baseline:

```bash
python src/scripts/run_experiments.py --protocol ulm --version A
python src/scripts/run_experiments.py --protocol ulm --version B
python src/scripts/run_tft.py --protocol ulm --seeds 42
```

Each AEDL command trains the three heads in about half an hour on a laptop. Add
`--seed <n>` for further seeds and `--epochs 1` for a quick smoke test. Results
are written to `results/ulm/A/seed_<seed>/`, `results/ulm/B/seed_<seed>/`, and
`results/ulm/tft/seed_<seed>/`.

Reliability diagram and forecast example of version A:

```bash
python src/scripts/generate_probabilistic_figures.py --protocol ulm --version A
```

The figures are written to `figures/`. Use `--output-dir`, `--sample-index`, or
`--device` for plot-specific overrides.

## Ulm results

Seed 42, 48 h history, 24 h horizon, observed future temperature, test period
13 December 2021 to 4 October 2022 (7,086 forecast origins). CRPS and PICP80
refer to the calibrated forecasts.

| Model | Head | MAE [kW] | MAPE [%] | CRPS [kW] | PICP80 |
|---|---|---:|---:|---:|---:|
| Version A | deterministic | 400.0 | 5.41 | | |
| | Gaussian | 396.2 | 5.35 | 312.7 | 0.83 |
| | quantile | 401.6 | 5.42 | 318.0 | 0.81 |
| Version B | deterministic | 394.8 | 5.41 | | |
| | Gaussian | 398.9 | 5.47 | 315.2 | 0.81 |
| | quantile | 396.9 | 5.44 | 318.3 | 0.81 |
| TFT | quantile | 436.4 | 5.75 | 349.2 | 0.80 |

Versions A and B reduce MAE and CRPS by 8 to 10 % compared with TFT. These are
single-seed results.

## Experiment protocols

| Protocol | Inputs and experiments | Models | Split |
|---|---|---|---|
| `ulm` (default, public data) | Load and perfect future temperature | Deterministic, Gaussian, quantile | Chronological 80/10/10 |
| `paper` (proprietary data) | Load, temperature, and irradiance. Historical, perfect, and API weather | Deterministic, Gaussian, quantile | Rolling-origin folds (`--fold 1`, `--fold 2`) or the original manuscript split |
| `eurosun` (proprietary data) | Weil am Rhein demand, temperature, and active buildings. Aggregate and per-building targets | Deterministic | Train through 2023, validate on 2024, test on 2025 |
| `malmoe` (proprietary data) | Malmö demand, temperature, and active buildings. Aggregate and per-building targets | Deterministic | Train through 2023, validate on 2024, test on 2025 |

The versioned definitions are in
[`src/load_forecasting/protocols.py`](src/load_forecasting/protocols.py). Every
checkpoint stores the protocol name and its fully resolved configuration.

The `ulm` and `paper` protocols run all three forecast heads. The `paper`
protocol additionally runs all three weather cases on common valid forecast
origins. The two network-expansion protocols (`eurosun`, `malmoe`) run the
deterministic head twice: once for aggregate demand and once after dividing
demand by `active_buildings`. Per-building predictions are multiplied by the
building count at each target timestamp before computing kW metrics, which
assumes that the 24-hour network size trajectory is known. These protocols
require `--version B`, because version A cannot be combined with the
per-building normalization.

## Runner options

`src/scripts/run_experiments.py`:

| Option | Purpose | Default |
|---|---|---|
| `--protocol` | `ulm`, `paper`, `eurosun`, or `malmoe` | `ulm` |
| `--models` | Subset of `deterministic`, `gaussian`, `quantile` | Protocol default |
| `--weather-cases` | Subset of the protocol's weather cases | Protocol default |
| `--epochs`, `--seed` | Epoch count and random seed | `50`, `42` |
| `--version` | `A` (window load scaling) or `B` (linear shortcut) | `A` |
| `--baseline` | Paper baselines instead of a version: `plain` AEDL, `lstm`, or `lstm-ws` (LSTM with window scaling) | none |
| `--calendar-tz` | Calendar features in a local time zone, e.g. `Europe/Berlin` | UTC |
| `--fold` | Rolling-origin fold (`paper` only) | original split |
| `--start-date` | First training origin | all data |
| `--device` | `auto`, `cpu`, `cuda`, or `mps` | `auto` |
| `--output-dir` | Result directory | `results/<protocol>/<version>/seed_<seed>/` |

JSON results and checkpoints record features, splits, scaling, calibration,
metrics, and training history. Point results include MAE, RMSE, MAPE, MPE, and
WAPE overall, for every lead time and month, and separately for the heating
season (October to March) and summer (April to September).

## Revised paper (proprietary data)

[`src/scripts/paper_runs.sh`](src/scripts/paper_runs.sh) contains all runs of
the revised manuscript (rolling folds 1 and 2, seeds 42 to 46) and the analysis
that produces its tables and figures:

```bash
bash src/scripts/paper_runs.sh aedl
bash src/scripts/paper_runs.sh baselines
bash src/scripts/paper_runs.sh start
bash src/scripts/paper_runs.sh hpo
bash src/scripts/paper_runs.sh analysis
```

| Script | Content |
|---|---|
| `run_experiments.py` | AEDL versions A and B and the plain AEDL and LSTM baselines |
| `run_simple_baselines.py` | Seasonal naive and its probabilistic version |
| `run_tft.py` | TFT (NeuralForecast) on the same forecast windows |
| `tune_hpo.py` | Optuna hyperparameter search for A, B, and TFT |
| `benchmark_compute.py` | Parameters, FLOPs, memory, training time, and latency |
| `paper_data.py` | Missing-data shares, correlations, and data figures |
| `paper_predictions.py` | Metrics by lead time, forecast example, and reliability diagram |
| `paper_latex_tables.py` | Result tables in LaTeX |

## Network-expansion evaluation

The EuroSun and Malmö protocols write detailed `forecasts_<case>.csv` files in
each seed directory. The notebooks in `EuroSun_2026/` (not part of the public
repository) evaluate full-year, monthly, seasonal, and horizon-dependent errors
and combine all available `seed_*` directories using the mean and sample
standard deviation.

## Data contract

Portable protocols use one pickled pandas `DataFrame` with a unique hourly
`DatetimeIndex`, a target in kW, and the columns named by the protocol. Calendar
features are generated automatically. A forecast window is assigned to a split
only if its complete 24 h target trajectory lies within that split. Windows
crossing a train/validation/test boundary are excluded, but their historical
input may extend into the preceding split. API protocols additionally require
one column per weather variable and lead, named `<weather>_forecast_lead_00h`
through `<weather>_forecast_lead_23h`. Incomplete API trajectories are excluded
rather than interpolated across issue times.
