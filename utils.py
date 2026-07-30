"""
utils.py
--------
Helper / utility functions shared across the pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    """Return the absolute path of the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(directory: str) -> None:
    """Create *directory* (and any parent dirs) if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def data_path(*parts: str) -> str:
    """Build a path relative to the project's data/ folder."""
    return os.path.join(get_project_root(), "data", *parts)


def models_path(*parts: str) -> str:
    """Build a path relative to the project's models/ folder."""
    return os.path.join(get_project_root(), "models", *parts)


def outputs_path(*parts: str) -> str:
    """Build a path relative to the project's outputs/ folder."""
    return os.path.join(get_project_root(), "outputs", *parts)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, filename: str = "metrics.json") -> None:
    """
    Persist an evaluation-metrics dictionary to outputs/<filename>.

    Parameters
    ----------
    metrics  : dict   Key-value pairs of metric name → value.
    filename : str    Target file name inside outputs/.
    """
    ensure_dir(outputs_path())
    filepath = outputs_path(filename)
    with open(filepath, "w") as fh:
        json.dump(metrics, fh, indent=4)
    print(f"[INFO] Metrics saved → {filepath}")


def print_metrics(metrics: dict, title: str = "Evaluation Metrics") -> None:
    """Pretty-print a metrics dictionary to stdout."""
    border = "=" * 45
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:.4f}")
        else:
            print(f"  {key:<25}: {value}")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save *fig* to outputs/<filename> and close it to free memory.

    Parameters
    ----------
    fig      : matplotlib.figure.Figure
    filename : str   e.g. 'residuals.png'
    """
    ensure_dir(outputs_path())
    filepath = outputs_path(filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved → {filepath}")


def set_plot_style() -> None:
    """Apply a clean, consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def check_dataset_exists(filepath: str) -> None:
    """
    Raise a descriptive FileNotFoundError if *filepath* is missing.

    Guides the user to download the dataset from Kaggle.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found: {filepath}\n"
            "Please download the dataset manually from Kaggle:\n"
            "  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data\n"
            "Place train.csv and test.csv inside the data/ folder."
        )


def summarise_dataframe(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print a quick structural summary of *df*."""
    print(f"\n{'─'*50}")
    print(f"  {label}  —  shape: {df.shape}")
    print(f"{'─'*50}")
    print(df.dtypes.value_counts().rename("dtype count").to_string())
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(f"\n  Columns with missing values ({len(missing)}):")
        print(missing.head(20).to_string())
    print()
