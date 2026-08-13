from __future__ import annotations

import numpy as np
from scipy.stats import norm


def point_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - y
    nonzero = np.abs(y) > 1e-6
    mae = float(np.mean(np.abs(error)))
    wape = float(100.0 * np.sum(np.abs(error)) / np.sum(np.abs(y)))
    return {
        "mae_kw": mae,
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(100.0 * np.mean(np.abs(error[nonzero]) / np.abs(y[nonzero]))),
        "mpe_percent": float(100.0 * np.mean(-error[nonzero] / y[nonzero])),
        "wape_percent": wape,
        # Retained for compatibility with existing manuscript result files.
        "nmae_percent": wape,
    }


def pinball_loss(
    y: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    quantile_values = np.asarray(quantile_values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    error = y[..., None] - quantile_values
    return np.maximum(quantiles * error, (quantiles - 1.0) * error)


def quantile_grid_crps(
    y: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Discrete CRPS approximation on a shared quantile grid.

    CRPS = 2 * integral_0^1 pinball_tau d tau. With equally spaced quantiles,
    twice the mean pinball loss is the corresponding simple grid approximation.
    Both Gaussian and quantile models are evaluated by this same function.
    """

    return float(2.0 * np.mean(pinball_loss(y, quantile_values, quantiles)))


def gaussian_crps(y: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.clip(np.asarray(scale, dtype=float), 1e-9, np.inf)
    standardized = (y - mean) / scale
    score = scale * (
        standardized * (2.0 * norm.cdf(standardized) - 1.0)
        + 2.0 * norm.pdf(standardized)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(score))


def gaussian_quantiles(
    mean: np.ndarray,
    scale: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    return np.asarray(mean)[..., None] + np.asarray(scale)[..., None] * norm.ppf(quantiles)


def interpolate_quantile(
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
    target_quantile: float,
) -> np.ndarray:
    quantile_values = np.asarray(quantile_values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    order = np.argsort(quantiles)
    sorted_values = quantile_values[..., order]
    sorted_quantiles = quantiles[order]
    flat = sorted_values.reshape(-1, sorted_values.shape[-1])
    result = np.asarray(
        [np.interp(float(target_quantile), sorted_quantiles, row) for row in flat]
    )
    return result.reshape(sorted_values.shape[:-1])


def central_quantile_interval(
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
    coverage: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - float(coverage)
    return (
        interpolate_quantile(quantile_values, quantiles, alpha / 2.0),
        interpolate_quantile(quantile_values, quantiles, 1.0 - alpha / 2.0),
    )


def picp(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float = 0.80,
) -> float:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    alpha = 1.0 - float(coverage)
    width = upper - lower
    under = (lower - y) * (y < lower)
    over = (y - upper) * (y > upper)
    return float(np.mean(width + 2.0 / alpha * (under + over)))


def gaussian_pit_mean(y: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    scale = np.clip(np.asarray(scale, dtype=float), 1e-9, np.inf)
    return float(np.mean(norm.cdf((np.asarray(y) - np.asarray(mean)) / scale)))


def quantile_pit(
    y: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Approximate PIT by inverting a piecewise-linear quantile function.

    Linear tail extrapolation is clipped to [0, 1], so all returned values are
    valid CDF values. Reliability curves should still be interpreted alongside
    this approximation because only the supplied quantiles are known.
    """

    y = np.asarray(y, dtype=float)
    quantile_values = np.asarray(quantile_values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    order = np.argsort(quantiles)
    quantiles = quantiles[order]
    quantile_values = quantile_values[..., order]
    quantile_values = np.maximum.accumulate(quantile_values, axis=-1)
    flat_y = y.reshape(-1)
    flat_q = quantile_values.reshape(-1, quantile_values.shape[-1])
    output = np.empty_like(flat_y, dtype=float)
    for row_index, (observed, row) in enumerate(zip(flat_y, flat_q)):
        output[row_index] = np.interp(observed, row, quantiles)
        if observed < row[0] and row[1] > row[0]:
            output[row_index] = quantiles[0] + (observed - row[0]) * (
                (quantiles[1] - quantiles[0]) / (row[1] - row[0])
            )
        elif observed > row[-1] and row[-1] > row[-2]:
            output[row_index] = quantiles[-1] + (observed - row[-1]) * (
                (quantiles[-1] - quantiles[-2]) / (row[-1] - row[-2])
            )
    return np.clip(output.reshape(y.shape), 0.0, 1.0)


def quantile_reliability_mae(
    y: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    observed = np.asarray(
        [np.mean(y <= quantile_values[..., index]) for index in range(len(quantiles))]
    )
    return float(np.mean(np.abs(observed - np.asarray(quantiles))))


def empirical_quantile(values: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    probability = float(np.clip(probability, 0.0, 1.0))
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def conformal_probability(sample_count: int, coverage: float) -> float:
    return min(1.0, float(np.ceil((sample_count + 1) * coverage) / max(sample_count, 1)))


def fit_gaussian_scale_calibration(
    y_validation: np.ndarray,
    mean_validation: np.ndarray,
    scale_validation: np.ndarray,
    coverage: float = 0.80,
) -> float:
    scale_validation = np.clip(np.asarray(scale_validation, dtype=float), 1e-9, np.inf)
    standardized_error = np.abs(y_validation - mean_validation) / scale_validation
    probability = conformal_probability(np.isfinite(standardized_error).sum(), coverage)
    nominal_z = float(norm.ppf(0.5 + coverage / 2.0))
    return max(1.0, empirical_quantile(standardized_error, probability) / nominal_z)


def fit_quantile_interval_calibration(
    y_validation: np.ndarray,
    quantile_validation: np.ndarray,
    quantiles: np.ndarray,
    coverage: float = 0.80,
) -> float:
    lower, upper = central_quantile_interval(quantile_validation, quantiles, coverage)
    nonconformity = np.maximum(lower - y_validation, y_validation - upper)
    probability = conformal_probability(np.isfinite(nonconformity).sum(), coverage)
    return max(0.0, empirical_quantile(nonconformity, probability))


def expand_quantiles(
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
    delta: float,
) -> np.ndarray:
    output = np.asarray(quantile_values, dtype=float).copy()
    quantiles = np.asarray(quantiles, dtype=float)
    output[..., quantiles < 0.5] -= float(delta)
    output[..., quantiles > 0.5] += float(delta)
    return output


def gaussian_quality_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    shared_quantiles: np.ndarray,
    coverage: float = 0.80,
) -> dict[str, float]:
    z_value = float(norm.ppf(0.5 + coverage / 2.0))
    lower = mean - z_value * scale
    upper = mean + z_value * scale
    q_values = gaussian_quantiles(mean, scale, shared_quantiles)
    return {
        "pit_mean": gaussian_pit_mean(y, mean, scale),
        "picp80": picp(y, lower, upper),
        "miw80_kw": mean_interval_width(lower, upper),
        "interval_score80_kw": interval_score(y, lower, upper, coverage),
        "crps_exact_kw": gaussian_crps(y, mean, scale),
        "crps_shared_quantile_grid_kw": quantile_grid_crps(y, q_values, shared_quantiles),
    }


def quantile_quality_metrics(
    y: np.ndarray,
    quantile_values: np.ndarray,
    quantiles: np.ndarray,
    coverage: float = 0.80,
) -> dict[str, float]:
    lower, upper = central_quantile_interval(quantile_values, quantiles, coverage)
    return {
        "pit_mean_approx": float(np.mean(quantile_pit(y, quantile_values, quantiles))),
        "quantile_reliability_mae": quantile_reliability_mae(y, quantile_values, quantiles),
        "picp80": picp(y, lower, upper),
        "miw80_kw": mean_interval_width(lower, upper),
        "interval_score80_kw": interval_score(y, lower, upper, coverage),
        "crps_shared_quantile_grid_kw": quantile_grid_crps(y, quantile_values, quantiles),
    }
