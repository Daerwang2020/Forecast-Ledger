"""Point-forecast metrics with explicit zero handling."""

from __future__ import annotations

import numpy as np

from .errors import AdapterError


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise AdapterError(f"Prediction shape {y_pred.shape} does not match target shape {y_true.shape}.")
    if y_true.size == 0:
        raise AdapterError("Cannot calculate metrics on an empty forecast.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise AdapterError("Metrics require finite targets and predictions.")
    if epsilon <= 0:
        raise ValueError("metric epsilon must be positive")

    error = y_pred.astype(np.float64) - y_true.astype(np.float64)
    absolute = np.abs(error)
    mse = float(np.mean(error**2))
    mae = float(np.mean(absolute))
    denominator = np.maximum(np.abs(y_true), epsilon)
    mape = float(100.0 * np.mean(absolute / denominator))
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    smape = float(100.0 * np.mean(2.0 * absolute / smape_denominator))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "mape": mape,
        "smape": smape,
    }
