"""
train.py
--------
Model creation, training, and persistence for House Price Prediction.
"""

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

from utils import models_path, outputs_path, ensure_dir, save_figure, set_plot_style


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def build_linear_regression() -> LinearRegression:
    """Return a plain OLS Linear Regression model."""
    return LinearRegression()


def build_ridge(alpha: float = 10.0) -> Ridge:
    """Return a Ridge Regression model (L2 regularisation)."""
    return Ridge(alpha=alpha, random_state=42)


def build_lasso(alpha: float = 0.001) -> Lasso:
    """Return a Lasso Regression model (L1 regularisation)."""
    return Lasso(alpha=alpha, max_iter=10_000, random_state=42)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit *model* on the training data.

    Returns
    -------
    Fitted model.
    """
    model.fit(X_train, y_train)
    model_name = type(model).__name__
    print(f"[INFO] Model trained  →  {model_name}")
    return model


def cross_validate(model, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> dict:
    """
    Run k-fold cross-validation and return mean / std of R² and RMSE.

    Returns
    -------
    dict with keys: cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std
    """
    r2_scores   = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    rmse_scores = np.sqrt(
        -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    cv_results = {
        "cv_r2_mean":   float(r2_scores.mean()),
        "cv_r2_std":    float(r2_scores.std()),
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std":  float(rmse_scores.std()),
    }

    print(
        f"[INFO] {cv}-Fold CV  →  "
        f"R² = {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}  |  "
        f"RMSE = {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}"
    )
    return cv_results


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(X_train: np.ndarray, y_train: np.ndarray) -> str:
    """
    Train and cross-validate all candidate models; return the name of the best.

    Returns
    -------
    str   Name of the best-performing model (highest mean CV R²).
    """
    candidates = {
        "LinearRegression": build_linear_regression(),
        "Ridge":            build_ridge(),
        "Lasso":            build_lasso(),
    }

    results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        results[name] = scores.mean()
        print(f"  {name:<22}  CV R² = {scores.mean():.4f}")

    best_name = max(results, key=results.get)
    print(f"[INFO] Best model → {best_name}  (R² = {results[best_name]:.4f})")
    return best_name


# ---------------------------------------------------------------------------
# Plot training summary
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """
    Plot the absolute coefficient magnitudes (proxy for feature importance
    in linear models) and save to outputs/.

    Parameters
    ----------
    model         : fitted sklearn linear model with .coef_ attribute
    feature_names : list[str]
    top_n         : number of top features to display
    """
    if not hasattr(model, "coef_"):
        print("[WARN] Model has no .coef_ attribute — skipping feature importance plot.")
        return

    set_plot_style()
    coefs = np.abs(model.coef_)
    indices = np.argsort(coefs)[::-1][:top_n]

    rev_indices = indices[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feature_names[i] for i in rev_indices],
        coefs[rev_indices],
        color="#4C72B0",
    )
    ax.set_title(f"Top {top_n} Feature Importances\n(|Coefficient| — {type(model).__name__})", fontsize=13)
    ax.set_xlabel("|Coefficient|")
    save_figure(fig, "feature_importance.png")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(model, filename: str = "house_price_model.pkl") -> None:
    """Persist the trained model to models/<filename>."""
    ensure_dir(models_path())
    filepath = models_path(filename)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved → {filepath}")


def save_scaler(scaler, filename: str = "scaler.pkl") -> None:
    """Persist the fitted scaler to models/<filename>."""
    ensure_dir(models_path())
    filepath = models_path(filename)
    joblib.dump(scaler, filepath)
    print(f"[INFO] Scaler saved → {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training(X_train, y_train, feature_names, scaler):
    """
    Full training workflow:
    1. Compare candidate models via CV.
    2. Train the best model on full training data.
    3. Run cross-validation on the winner.
    4. Plot feature importance.
    5. Save model + scaler.

    Returns
    -------
    model         : fitted model
    cv_results    : dict
    """
    print("\n[STEP] Comparing candidate models …")
    best_name = compare_models(X_train, y_train)

    model_map = {
        "LinearRegression": build_linear_regression(),
        "Ridge":            build_ridge(),
        "Lasso":            build_lasso(),
    }
    model = model_map[best_name]

    print(f"\n[STEP] Training {best_name} on full training set …")
    model = train_model(model, X_train, y_train)

    print("\n[STEP] Cross-validation …")
    cv_results = cross_validate(model, X_train, y_train)

    print("\n[STEP] Plotting feature importance …")
    plot_feature_importance(model, feature_names)

    print("\n[STEP] Saving model and scaler …")
    save_model(model)
    save_scaler(scaler)

    return model, cv_results
