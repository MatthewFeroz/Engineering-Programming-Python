"""Shared model helper functions."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return coefficient statistics and regression metrics for a trained model."""
    predictions = model.predict(X)
    coefficient_table = _coefficient_table(model, X, y, predictions)
    errors = y - predictions
    metrics = {
        "mae": mean_absolute_error(y, predictions),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": r2_score(y, predictions),
    }
    return coefficient_table, metrics


def _coefficient_table(model, X: pd.DataFrame, y: pd.Series, predictions) -> pd.DataFrame:
    """Build coefficient, t-statistic, and approximate p-value rows."""
    feature_names = ["intercept"] + list(X.columns)
    coefficients = np.concatenate(([model.intercept_], model.coef_))

    # Add the intercept column because LinearRegression stores it separately.
    design = np.column_stack((np.ones(len(X)), X.to_numpy()))
    residuals = y.to_numpy() - predictions

    # Standard errors come from the ordinary least squares covariance formula.
    degrees_of_freedom = max(len(X) - design.shape[1], 1)
    residual_variance = float(np.sum(residuals ** 2) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))

    t_stats = coefficients / standard_errors
    p_values = [_normal_p_value(t_stat) for t_stat in t_stats]

    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "t_stat": t_stats,
        "p_value": p_values,
    })


def _normal_p_value(t_stat: float) -> float:
    """Approximate a two-sided p-value from a t-statistic."""
    return erfc(abs(t_stat) / sqrt(2))
