"""LaTeX rows for the results tables of the revised paper.

Metrics are pooled over both test periods (folds 1 and 2) per seed: MAE, MAPE,
CRPS, MIW, PICP and PIT mean are weighted by the number of test origins, RMSE
via the weighted squared error and NMAE via the pooled mean load. Cells give
mean +- sample std over seeds. The best value per column is set in bold.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOLDS = (1, 2)
A, B = "localcal_A", "localcal_B"
PLAIN_AEDL, LSTM_GLOBAL, LSTM_WINDOW = "localcal_plain", "localcal_lstm", "localcal_lstm-ws"
CASE_DIR = {"api_forecast": "api_forecast", "historical": "historical_observed_future",
            "observed_future": "historical_observed_future"}


def fold_rows(paths: list[str], n_key) -> pd.DataFrame:
    rows = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        m = payload["metrics"]
        point = m["point"]
        row = {"n": n_key(payload), "mae": point["mae_kw"], "rmse": point["rmse_kw"], "mape": point["mape_percent"],
               "mean_load": point["mae_kw"] / point["nmae_percent"] * 100}
        for stage in ("raw", "cal"):
            prob = m.get("probabilistic_raw" if stage == "raw" else "probabilistic_calibrated")
            if prob:
                row.update({f"crps_{stage}": prob["crps_shared_quantile_grid_kw"], f"picp_{stage}": prob["picp80"],
                            f"miw_{stage}": prob["miw80_kw"], f"pit_{stage}": prob.get("pit_mean", prob.get("pit_mean_approx"))})
        rows[Path(path).parent.name] = row
    return pd.DataFrame(rows).T


def aedl(tag: str, head: str, case: str = "api_forecast", fold_prefix: str = "") -> dict[int, pd.DataFrame]:
    return {f: fold_rows(sorted(glob.glob(str(ROOT / "results" / "paper" / f"fold{f}_{fold_prefix}{tag}" / CASE_DIR[case]
                                              / "seed_*" / f"{head}_{case}.json"))),
                         lambda p: p["metadata"]["split_summary"]["test_origins"]) for f in FOLDS}


def baseline(tag: str, model: str) -> dict[int, pd.DataFrame]:
    return {f: fold_rows(sorted(glob.glob(str(ROOT / "results" / "paper_baselines" / f"fold{f}_{tag}" / model
                                              / "seed_*" / "metrics.json"))),
                         lambda p: p["split_summary"]["test_origins"]) for f in FOLDS}


def pooled(per_fold: dict[int, pd.DataFrame]) -> pd.DataFrame:
    seeds = per_fold[1].index.intersection(per_fold[2].index)
    a, b = (per_fold[f].loc[seeds].astype(float) for f in FOLDS)
    wa, wb = a["n"] / (a["n"] + b["n"]), b["n"] / (a["n"] + b["n"])
    out = a.mul(wa, axis=0) + b.mul(wb, axis=0)
    out["rmse"] = np.sqrt(a["rmse"] ** 2 * wa + b["rmse"] ** 2 * wb)
    out["nmae"] = out["mae"] / out["mean_load"] * 100
    return out


def cell(frame: pd.DataFrame, key: str, digits: int) -> tuple[str, float]:
    if key not in frame or frame[key].isna().all():
        return "n.a.", np.nan
    values = frame[key].astype(float)
    text = f"{values.mean():.{digits}f}"
    if len(values) > 1:
        text += f"\\pm{values.std(ddof=1):.{digits}f}"
    return text, values.mean()


def table(rows: list[tuple[str, pd.DataFrame]], columns: list[tuple[str, int, str]]) -> str:
    """columns: (key, digits, best) with best in {"min", "max", "0.8", "0.5", ""}."""

    cells = [[cell(frame, key, digits) for key, digits, _ in columns] for _, frame in rows]
    best = []
    for j, (_, _, rule) in enumerate(columns):
        values = np.array([c[j][1] for c in cells], dtype=float)
        if rule == "min":
            best.append(np.nanmin(values))
        elif rule in ("0.8", "0.5"):
            best.append(values[np.nanargmin(np.abs(values - float(rule)))])
        else:
            best.append(np.nan)
    lines = []
    for (label, _), row in zip(rows, cells):
        parts = []
        for j, (text, value) in enumerate(row):
            bold = np.isfinite(best[j]) and np.isfinite(value) and np.isclose(
                round(value, columns[j][1]), round(best[j], columns[j][1]))
            if text != "n.a.":
                text = f"$\\boldsymbol{{{text}}}$" if bold else f"${text}$"
            parts.append(text)
        lines.append(f"    {label} & " + " & ".join(parts) + " \\\\")
    return "\n".join(lines)


def main() -> None:
    a_det, b_det = pooled(aedl(A, "deterministic")), pooled(aedl(B, "deterministic"))
    heads = {(v, h): pooled(aedl(tag, h)) for v, tag in (("A", A), ("B", B)) for h in ("gaussian", "quantile")}
    plain = {h: pooled(aedl(PLAIN_AEDL, h)) for h in ("deterministic", "gaussian")}
    lstm_w = {h: pooled(aedl(LSTM_WINDOW, h)) for h in ("deterministic", "gaussian")}
    lstm_g = {h: pooled(aedl(LSTM_GLOBAL, h)) for h in ("deterministic", "gaussian")}
    tft = pooled(baseline("localcal_ws-robust_selcrps", "tft"))
    naive = pooled(baseline("localcal", "seasonal_naive_daily"))
    naive_q = pooled(baseline("localcal", "seasonal_naive_quantile"))

    print("% Table: point forecasts")
    print(table([
        ("A, deterministic", a_det), ("A, Gaussian", heads["A", "gaussian"]), ("A, quantile", heads["A", "quantile"]),
        ("B, deterministic", b_det), ("B, Gaussian", heads["B", "gaussian"]), ("B, quantile", heads["B", "quantile"]),
        ("Plain AEDL", plain["deterministic"]), ("LSTM, window scaling", lstm_w["deterministic"]),
        ("Plain LSTM", lstm_g["deterministic"]), ("TFT", tft), ("Seasonal naive", naive),
    ], [("mae", 1, "min"), ("rmse", 1, "min"), ("mape", 2, "min"), ("nmae", 2, "min")]))

    print("\n% Table: probabilistic forecasts")
    prob_rows = []
    for v in "AB":
        for h, name in (("gaussian", "Gaussian"), ("quantile", "quantile")):
            frame = heads[v, h]
            prob_rows.append((f"{v}, {name}", frame.rename(columns=lambda c: c.replace("_raw", ""))))
            prob_rows.append((f"{v}, {name}, cal.", frame.rename(columns=lambda c: c.replace("_cal", ""))))
    for label, frame in (("Plain AEDL, Gaussian, cal.", plain["gaussian"]), ("LSTM, window sc., Gaussian, cal.", lstm_w["gaussian"]),
                         ("Plain LSTM, Gaussian, cal.", lstm_g["gaussian"]), ("TFT, cal.", tft),
                         ("Seasonal naive, quantile", naive_q)):
        prob_rows.append((label, frame.rename(columns=lambda c: c.replace("_cal", ""))))
    print(table(prob_rows, [("pit", 3, "0.5"), ("miw", 1, ""), ("picp", 3, "0.8"), ("crps", 1, "min")]))

    print("\n% Table: future-weather ablation (calibrated), relative change of MAE and CRPS vs. archived forecast")
    for v, tag in (("A", A), ("B", B)):
        for h in ("deterministic", "gaussian", "quantile"):
            frames = {c: pooled(aedl(tag, h, c)) for c in ("historical", "observed_future", "api_forecast")}
            ref = frames["api_forecast"]
            for c, label in (("historical", "hist."), ("observed_future", "hist. + obs. future"), ("api_forecast", "hist. + API forecast")):
                f = frames[c]
                d_mae = 100 * (f["mae"].mean() / ref["mae"].mean() - 1)
                crps = f"${cell(f, 'crps_cal', 1)[0]}$" if "crps_cal" in f else "n.a."
                d_crps = f"${100 * (f['crps_cal'].mean() / ref['crps_cal'].mean() - 1):+.1f}$" if "crps_cal" in f else "n.a."
                picp = f"${cell(f, 'picp_cal', 3)[0]}$" if "picp_cal" in f else "n.a."
                print(f"    {v}, {h[:5] if h != 'deterministic' else 'det.'} & {label} & ${cell(f, 'mae', 1)[0]}$ & ${d_mae:+.1f}$ & "
                      f"{crps} & {d_crps} & {picp} \\\\")

    print("\n% Table: training start (deterministic heads), MAE per fold and pooled")
    for v, tag in (("A", A), ("B", B)):
        for prefix, label in (("", "Apr. 2020"), ("from2023-01-01_", "Jan. 2023"), ("from2024-01-01_", "Jan. 2024")):
            per_fold = aedl(tag, "deterministic", fold_prefix=prefix)
            cells = [f"${cell(per_fold[f], 'mae', 1)[0]}$" for f in FOLDS] + [f"${cell(pooled(per_fold), 'mae', 1)[0]}$"]
            print(f"    {v} & {label} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
