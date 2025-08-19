# Reproducible Notebooks and Models for District Heating Forecasting

_AEDL (Attention‑Encoder‑Decoder‑LSTM) with deterministic and probabilistic (Gaussian and quantile) heads_

## Why this repo exists
This repository provides **Jupyter notebooks and reference PyTorch implementations** used in our Smart Energy Journal submission on **probabilistic building energy load prediction** for **district heating networks (DHNs)**. The goal is to make our workflow **transparent, reproducible, and extensible** for practitioners and researchers.

**Key points at a glance**
- **Models**: AEDL baseline (deterministic point‑forecast), **AEDL‑Prob (Gaussian head)**, **AEDL‑Quant (quantile head)**.
- **Scope**: Short‑term day‑ahead forecasting (default 24 h horizon) from a sliding history window (default 48 h).
- **Data**: Open data from the **district heating network of the City of Ulm** (replaces our closed‑source dataset used in the paper).
- **Reproducibility**: Notebooks + scripts + pinned environment; no redistribution of upstream data.
- **Outputs**: Metrics, figures, and model checkpoints for downstream use in MPC/operations studies.

---

## High‑level code map (what each file does)
> This summary orients reviewers and users in seconds. Details live in code comments and notebooks.

**Notebooks (reference pipeline)**
- `aedl.ipynd` — **AEDL‑forecast (Deterministic)**
- `aedl-prob.ipynd` — **AEDL‑Prob (Gaussian head)**
  - Predicts **mean (μ)** and **log‑std (log σ)** for each horizon step.
  - Loss: **Gaussian Negative Log‑Likelihood (NLL)** + small **MAE term** + variance regularizer to deter over‑inflation.
  - Computes **MAE/MAPE** on inverse‑scaled predictions; supports **CRPS (Gaussian)** for distributional quality.
  - Saves checkpoint to `../checkpoints/prob_aedl.pt` (path configurable).
- `aedl-quant.ipynb` — **AEDL‑Quant (quantile head)**
  - Predicts an array of **quantiles** (e.g., 0.1…0.9). Median (τ=0.5) acts as the point forecast.
  - Loss: **Pinball (quantile) loss**, with utilities for plotting quantile fans and evaluating MAE/MAPE on median.
  - Designed for **sharper uncertainty bands**; can be augmented with post‑hoc calibration if needed.


---

## Dataset (open source): District Heating Ulm
To enable open review and reuse, we **replace closed‑source data** with the open dataset from **Ulm’s district heating network**:

- **Repository:** https://github.com/finkenrm/deepDHC-user-guide/tree/main  
- **License:** As stated by the upstream repository. By using the data, you agree to those terms.
- **Local placement:** After downloading, place files  `data/data.csv` as described below.

### Expected CSV schema (default as structure in  `data/data.csv`)
`aedl-*.ipynb` scripts expect a CSV at `../data/data.csv` (relative to the script), with:
- **Timestamp column:** `MESS_DATUM` (will be renamed to `timestamp` and used as the index).
- **Target:** `load_mw` (network load in MW).
- **Exogenous weather features (typical):** `temperature`, `dewpoint`, `pressure_nn`, `windspeed`.
- **Known future feature:** `temperature_forecast` (optional but supported).
- The scripts add **cyclic time features**: hour/day‑of‑week/month sin/cos encodings.

> If your CSV uses different column names, adjust the `CFG` dictionary at the top of each script (see next section).

---

## Configuration (what to tweak and where)
Both training scripts start with a `CFG = dict(...)` block. Common keys:
- **Paths**: `csv_path` (default `../data/data.csv`), `save_name` (checkpoint path).
- **Columns**: `target_col` (`"load_mw"`), `weather_cols`, `known_future_cols`.
- **Windowing**: `past_len=48`, `pred_len=24`.
- **Training**: `batch_size`, `hidden_size`, `num_layers`, `attn_heads`, `dropout`, `lr`, `epochs`, `es_patience`.
- **Splits**: `train_intervals`, `test_intervals` (date strings), or `train_pct` / `val_pct` for rolling splits.

Change these to suit your environment and evaluation window.

---

## Quick start

```bash
# 1) Create env (conda example)
conda env create -f env/environment.yml
conda activate se-notebooks

# 2) Obtain Ulm dataset and place under:
#    data/raw/      (upstream structure)  and/or
#    data/data.csv  (compiled CSV with columns described above)

# Edit the CFG dict at the top of each script to change paths, columns, or hyper‑parameters.
```

**Outputs**
- Checkpoints: `checkpoints/*.pt` (relative to scripts’ `save_name`).

- Printed metrics: MAE, MAPE (and probabilistic metrics).

---

## Benchmarking Results & Practical Implications (Ulm dataset)

| Model Variant                         | TEST MAE (MW) | TEST MAPE (%) |
|--------------------------------------|---------------|---------------|
| AEDL (deterministic point‑forecast)  | **13.28**     | **5.96**      |
| AEDL‑Prob (Gaussian head)            | 13.71         | 9.37          |
| AEDL‑Quant (quantile head)           | 13.64         | 8.42          |

### What these numbers mean operationally
- **MAE (MW)** — Average absolute MW deviation from truth. At **13.28 MW**, the deterministic AEDL provides the tightest point‑forecast, which directly supports **day‑ahead unit commitment and pump scheduling** with fewer manual corrections.
- **MAPE (%)** — Average **relative** error. A **5.96 %** MAPE indicates robust accuracy across varying load magnitudes; useful when comparing performance across seasons or scaled networks.
- **Gaussian head vs Quantile head** — The **Gaussian head** typically yields **better calibration** (credible intervals match observed frequencies more closely), which is crucial for **risk‑aware MPC** and **reserve margins**. The **quantile head** produces **sharper** (narrower) intervals but can be **overconfident**; it’s valuable for **scenario exploration** (e.g., P10/P90 planning) and **sensitivity analysis**.
- **Practical takeaway** — Use **deterministic AEDL** for baseline scheduling; deploy **AEDL‑Prob** when **robustness under uncertainty** matters (e.g., cold spells, expansions); use **AEDL‑Quant** to probe **extremes** and to design **stress‑tests** for dispatch policies.

---

## Reproducing our table
1. Prepare `data/data.csv` with the schema above (or run `00_…` → `03_…` notebooks).
2. Ensure `CFG["train_intervals"]` and `CFG["test_intervals"]` in the scripts match your evaluation window.
3. Run notebooks; record **MAE/MAPE** from script output (and CRPS for Gaussian).

---

## Licensing & attribution
- **Code**: MIT 
- **Data**: Governed solely by the **deepDHC‑user‑guide** license. This repository does **not** redistribute the dataset.
- **Cite when using**: our paper (once available), this repository, and the Ulm dataset repository.

```bibtex
@misc{AEDLCode2025,
  title        = {Reproducible Notebooks and Models for District Heating Forecasting (AEDL)},
  author       = {Lilli Frison},
  year         = {2025},
  howpublished = {GitHub},
  note         = {v1.0}
}
```

---

## Changelog
- **v1.0** — Initial public release synced with manuscript submission; added open‑data pathway (Ulm), deterministic and probabilistic heads, and benchmarking table with operational interpretation.

---

## Questions / contributions
Issues and PRs are welcome. For research collaboration or operational pilots, please open an issue or contact the maintainers.
