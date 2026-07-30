"""
evaluate.py
-----------
Evaluation metrics, plots (Actual vs Predicted, Residuals),
and report generation for the House Price Prediction model.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from utils import (
    outputs_path,
    ensure_dir,
    save_metrics,
    print_metrics,
    save_figure,
    set_plot_style,
)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> dict:
    """
    Compute regression evaluation metrics.

    If *log_scale* is True the predictions are expm1-transformed before
    computing dollar-based MAE / RMSE (so metrics are interpretable).

    Parameters
    ----------
    y_true     : ground-truth target values
    y_pred     : model predictions
    log_scale  : whether y values are in log1p space

    Returns
    -------
    dict with keys: MAE, MSE, RMSE, R2
    """
    if log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    mse  = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)        # R² on log scale (model's scale)

    metrics = {
        "MAE ($)":  round(float(mae),  2),
        "MSE ($²)": round(float(mse),  2),
        "RMSE ($)": round(float(rmse), 2),
        "R² Score": round(float(r2),   4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Scatter plot of actual vs predicted sale prices (original dollar scale).

    Saves → outputs/actual_vs_predicted.png
    """
    set_plot_style()

    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_true_orig, y_pred_orig, alpha=0.45, color="#4C72B0", edgecolors="white", s=50)

    # Perfect-prediction reference line
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual SalePrice ($)", fontsize=12)
    ax.set_ylabel("Predicted SalePrice ($)", fontsize=12)
    ax.set_title("Actual vs Predicted Sale Prices", fontsize=14, fontweight="bold")
    ax.legend()
    save_figure(fig, "actual_vs_predicted.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Residual plot (predicted values vs residuals) + residual histogram.

    Saves → outputs/residual_plot.png
    """
    set_plot_style()

    residuals = y_true - y_pred          # in log space

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")

    # Scatter: predicted vs residual
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#DD8452", edgecolors="white", s=45)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted (log scale)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Predicted vs Residual")

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="#55A868")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    save_figure(fig, "residual_plot.png")


def plot_prediction_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot percentage prediction error distribution.

    Saves → outputs/prediction_error_distribution.png
    """
    set_plot_style()

    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    pct_error = (y_pred_orig - y_true_orig) / y_true_orig * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(pct_error, kde=True, ax=ax, color="#C44E52", bins=40)
    ax.axvline(0, color="navy", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Percentage Error (%)")
    ax.set_title("Prediction Error Distribution", fontsize=13)
    save_figure(fig, "prediction_error_distribution.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_evaluation_report(metrics: dict, cv_results: dict) -> None:
    """
    Write a plain-text evaluation report to outputs/evaluation_report.txt.
    """
    ensure_dir(outputs_path())
    filepath = outputs_path("evaluation_report.txt")
    with open(filepath, "w") as fh:
        fh.write("=" * 55 + "\n")
        fh.write("   House Price Prediction — Evaluation Report\n")
        fh.write("=" * 55 + "\n\n")
        fh.write("Hold-Out Test Set Metrics\n")
        fh.write("-" * 35 + "\n")
        for k, v in metrics.items():
            fh.write(f"  {k:<20}: {v}\n")
        fh.write("\n")
        fh.write("5-Fold Cross-Validation (Training Set)\n")
        fh.write("-" * 35 + "\n")
        for k, v in cv_results.items():
            fh.write(f"  {k:<20}: {v:.4f}\n")
    print(f"[INFO] Evaluation report saved → {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(model, X_test: np.ndarray, y_test: np.ndarray, cv_results: dict) -> dict:
    """
    Full evaluation workflow:
    1. Generate predictions.
    2. Compute metrics.
    3. Produce all plots.
    4. Save report + metrics JSON.

    Returns
    -------
    metrics : dict
    """
    print("\n[STEP] Running evaluation …")

    y_pred = model.predict(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, log_scale=True)
    print_metrics(metrics, title="Test Set Evaluation Metrics")
    save_metrics({**metrics, **cv_results})

    # Plots
    print("[STEP] Generating evaluation plots …")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_prediction_error_distribution(y_test, y_pred)

    # Text report
    save_evaluation_report(metrics, cv_results)

    return metrics
