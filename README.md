# AEDL probabilistic load forecasting

This repository trains deterministic, Gaussian, and quantile variants of a
direct multi-horizon attention encoder-decoder LSTM. Named experiment protocols
bind each forecasting task to its data, inputs, horizons, and chronological
split, so switching tasks requires only one argument.

## Methods

All variants encode the historical load, weather, and cyclical calendar inputs
with feature attention, an LSTM, and temporal self-attention. The decoder
produces the complete forecast trajectory in one pass without autoregressive
target feedback or teacher forcing.

- **Deterministic AEDL** predicts one value per horizon and uses MAE loss.
- **Gaussian AEDL** predicts a mean and standard deviation. MAE warm-up is
  followed by Gaussian NLL with small MAE and variance penalties; validation
  data determine one scale-calibration factor.
- **Quantile AEDL** predicts ordered quantiles 0.1--0.9 using pinball loss with
  median-MAE and width penalties. Validation data determine a symmetric
  interval adjustment.

`observed_future` uses true future weather and is therefore a perfect-weather
benchmark, not an operational forecast. `historical` provides no decoder
weather. `api_forecast` uses an origin-coherent operational forecast trajectory (if avaiable).

## Installation

```bash
git clone https://github.com/lfrison/Probabilistic-heating-networks-load-forecasting.git load-forecasting
cd load-forecasting
conda env create -f env/environment.yml
conda activate dhn-predict
```

Prepare the public csv data once (create pkl file):

```bash
python src/scripts/prepare_ulm_data.py
```

The Ulm source data are publicly available in the
[`finkenrm/deepDHC-user-guide`](https://github.com/finkenrm/deepDHC-user-guide)
repository; the CSV used here is its
[`data/data.csv`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/data/data.csv).
The upstream repository includes a permissive MIT-style
[`LICENSE`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/LICENSE).

## Experiment protocols

| Protocol | Inputs and experiments | Models | Split |
|---|---|---|---|
| `ulm` (default) | Load and perfect future temperature | Deterministic, Gaussian, quantile | Chronological 80/10/10 |
| `paper` (proprietary data) | Load, temperature and irradiance; historical, perfect, and API weather | Deterministic, Gaussian, quantile | Manuscript split with common API-valid origins and fixed test period |
| `eurosun` (proprietary data) | Weil am Rhein demand, temperature and active buildings; aggregate and per-building targets | Deterministic | Train through 2023, validate on 2024, test on 2025 |
| `malmoe` (proprietary data)| Malmö demand, temperature and active buildings; aggregate and per-building targets | Deterministic | Train through 2023, validate on 2024, test on 2025 |

The exact versioned definitions are in
[`src/load_forecasting/protocols.py`](src/load_forecasting/protocols.py). Add or
modify a protocol there when a task needs a different dataset, feature set,
horizon, or split. Every checkpoint stores both the protocol name and its fully
resolved configuration.

### Run a protocol

Ulm is the default:

```bash
python src/scripts/run_experiments.py --protocol ulm
```

Switch datasets/tasks with one option:

```bash
python src/scripts/run_experiments.py --protocol paper
python src/scripts/run_experiments.py --protocol eurosun
python src/scripts/run_experiments.py --protocol malmoe
```

Run multiple seeds for any protocol by setting `protocol` accordingly:

```bash
protocol=eurosun
for seed in 42 43 44 45 46; do
  python src/scripts/run_experiments.py --protocol "$protocol" --seed "$seed" || exit 1
done
```

The `ulm` and `paper` protocols run all three forecast heads. The `paper`
protocol additionally runs all three weather cases, giving nine runs. The two
network-expansion protocols (eurosun, malmoe) run the deterministic head twice: once for
aggregate demand and once after dividing demand by `active_buildings`.
Per-building predictions are multiplied by the building count at each target
timestamp before computing kW metrics; this assumes that the 24-hour network
size trajectory is known. A subset can be selected without changing the
protocol's dataset or split:

```bash
python src/scripts/run_experiments.py \
  --protocol paper \
  --weather-cases api_forecast \
  --models gaussian quantile
```

When several weather cases are selected, the runner automatically filters them
to common valid forecast origins.

A quick smoke run is:

```bash
python src/scripts/run_experiments.py \
  --protocol ulm \
  --models deterministic \
  --epochs 1
```

Default outputs are separated by task and seed:

```text
results/<protocol>/seed_<seed>/
```

If a non-default weather subset is selected, its name is inserted before the
seed directory. Existing result files are therefore not mixed across datasets.

## Ulm example results

The currently stored Ulm results use seed 42, a 48-hour history, a 24-hour
forecast horizon, and observed future temperature. They are therefore a
perfect-weather benchmark rather than an operational weather-forecast
experiment. The three heads have nearly identical point accuracy:

| Model | MAE [kW] | RMSE [kW] | MAPE [%] | WAPE/NMAE [%] |
|---|---:|---:|---:|---:|
| Deterministic | 431.68 | 655.88 | 5.93 | 5.53 |
| Gaussian | 431.77 | 660.29 | 5.86 | 5.53 |
| Quantile | 430.52 | 652.37 | 5.87 | 5.51 |

Probabilistic performance is:

| Model | CRPS on shared quantile grid [kW] | PICP 80 [%] | Mean 80% interval width [kW] |
|---|---:|---:|---:|
| Gaussian | 340.55 | 83.37 | 1,399.86 |
| Quantile, raw | 343.30 | 76.02 | 1,243.62 |
| Quantile, calibrated | 342.92 | 79.94 | 1,328.97 |

These are single-seed example results, not mean values over repeated runs. The
Gaussian intervals are slightly conservative, whereas calibration brings the
quantile model close to its nominal 80% coverage. The files currently reside
in the legacy-compatible directory `results/observed_future/seed_42/` and their
metadata confirm that they were generated from `data/data_Ulm.pkl`.

## Probabilistic figures

After Gaussian and quantile training, generate the default Ulm perfect-weather
figures with:

```bash
python src/scripts/generate_probabilistic_figures.py
```

For another seed:

```bash
python src/scripts/generate_probabilistic_figures.py --protocol ulm --seed 43
```

EuroSun and Malmö do not produce probabilistic figures because their protocols
run only the deterministic head.

For the manuscript API results:

```bash
python src/scripts/generate_probabilistic_figures.py \
  --protocol paper \
  --weather-case api_forecast \
  --results-dir results/decoder_comparison/total_only/all_weather/direct/seed_43
```

The script writes PDF and PNG reliability and forecast-example figures plus a
provenance JSON file to `figures/`. The example titles report trajectory-level
MAE, RMSE, and MAPE. Use `--output-dir`, `--sample-index`, or `--device` for
plot-specific overrides.

## Network-expansion evaluation

The EuroSun and Malmö protocols write detailed `forecasts_<case>.csv` files in
each seed directory. Open
[`EuroSun_2026/plots_notebook_aedl-v2.ipynb`](EuroSun_2026/plots_notebook_aedl-v2.ipynb)
after training to evaluate full-year, monthly, heating-season/summer, and
horizon-dependent errors. Change `PROTOCOL` in its configuration cell from
`eurosun` to `malmoe` to evaluate the Malmö runs. The notebook automatically
combines all available `seed_*` directories using the mean and sample standard
deviation.

## Data contract

Portable protocols use one pickled pandas `DataFrame` with a unique hourly
`DatetimeIndex`, a target in kW, and the columns named by the protocol. Calendar
features are generated automatically. For every protocol, a forecast window is
assigned to a split only if its complete target trajectory (24 hours in the
provided protocols) lies within that split. Windows crossing a
train/validation/test boundary are excluded; their historical input may extend
into the preceding split. API protocols additionally require one column per
weather variable and lead, named
`<weather>_forecast_lead_00h` through `<weather>_forecast_lead_23h`.
Incomplete API trajectories are excluded rather than interpolated across issue
times. See [`data/README.md`](data/README.md) for Ulm provenance, units, and
corrections.

## Runner options

| Option | Purpose | Default |
|---|---|---|
| `--protocol` | Select `ulm`, `paper`, `eurosun`, or `malmoe` | `ulm` |
| `--weather-cases` | Run an allowed subset of the protocol's weather cases | Protocol default: `observed_future` for `ulm`, `eurosun`, and `malmoe`; all three cases for `paper` |
| `--models` | Run an allowed subset of deterministic, Gaussian, and quantile heads | Protocol default: all three for `ulm` and `paper`; deterministic only for `eurosun` and `malmoe` |
| `--epochs` | Override the protocol's epoch count | `50` |
| `--seed` | Override the protocol's random seed | `42` |
| `--device` | Select `auto`, `cpu`, `cuda`, or `mps` | `auto` |
| `--output-dir` | Override the protocol-based result directory | `results/<protocol>/seed_<seed>/`; a selected weather subset adds a subdirectory |

Training output reports per-epoch and total elapsed time. JSON results and
checkpoints record features, splits, scaling, calibration, metrics, training
history, and the resolved versioned protocol. Point results include MAE, RMSE,
MAPE, MPE, and WAPE overall, for every forecast lead and month, and separately
for the heating season (October--March) and summer (April--September). The
EuroSun and Malmö runs additionally write `forecasts_raw_temp.csv` and
`forecasts_norm_temp.csv` for the evaluation notebook.
