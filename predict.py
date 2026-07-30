"""
predict.py
----------
Load the saved model and scaler, then make predictions on new samples.
"""

import numpy as np
import pandas as pd
import joblib

from utils import models_path, check_dataset_exists


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model(filename: str = "house_price_model.pkl"):
    """Load the saved regression model from models/."""
    filepath = models_path(filename)
    if not __import__("os").path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Model not found: {filepath}\n"
            "Run main.py first to train and save the model."
        )
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded ← {filepath}")
    return model


def load_scaler(filename: str = "scaler.pkl"):
    """Load the saved StandardScaler from models/."""
    filepath = models_path(filename)
    if not __import__("os").path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Scaler not found: {filepath}\n"
            "Run main.py first to train and save the scaler."
        )
    scaler = joblib.load(filepath)
    print(f"[INFO] Scaler loaded ← {filepath}")
    return scaler


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_single(sample: dict, model, scaler, feature_names: list) -> float:
    """
    Predict the sale price for a single property.

    Parameters
    ----------
    sample       : dict  Feature name → value mapping.
                         Missing features will be filled with 0.
    model        : fitted sklearn model
    scaler       : fitted StandardScaler
    feature_names: list[str]  Ordered list of feature columns used in training.

    Returns
    -------
    float  Predicted sale price in dollars.
    """
    row = {col: sample.get(col, 0) for col in feature_names}
    X = pd.DataFrame([row])[feature_names]
    X_sc = scaler.transform(X)
    log_pred = model.predict(X_sc)[0]
    price = float(np.expm1(log_pred))
    return price


def predict_batch(df: pd.DataFrame, model, scaler, feature_names: list) -> np.ndarray:
    """
    Predict sale prices for a DataFrame of samples.

    Parameters
    ----------
    df           : pd.DataFrame  Must contain the same feature columns used at training.
    model        : fitted sklearn model
    scaler       : fitted StandardScaler
    feature_names: list[str]

    Returns
    -------
    np.ndarray  Predicted prices in dollars.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_names].fillna(0)
    X_sc = scaler.transform(X)
    log_preds = model.predict(X_sc)
    return np.expm1(log_preds)


# ---------------------------------------------------------------------------
# Demo predictions
# ---------------------------------------------------------------------------

def run_demo_predictions(model, scaler, feature_names: list) -> None:
    """
    Run predictions on a few hand-crafted example properties and print results.
    """
    print("\n" + "=" * 55)
    print("  Demo Predictions — Sample Properties")
    print("=" * 55)

    samples = [
        {
            "name": "Small Starter Home",
            "GrLivArea": 1200, "TotalBsmtSF": 600,
            "OverallQual": 5,  "OverallCond": 6,
            "YearBuilt": 1985, "YearRemodAdd": 1990,
            "GarageCars": 1,   "GarageArea": 300,
            "TotalSF": 1800,   "HouseAge": 39,
            "RemodelAge": 34,  "TotalBathrooms": 1.5,
            "HasGarage": 1,    "HasPool": 0,  "HasFireplace": 0,
        },
        {
            "name": "Mid-Range Family Home",
            "GrLivArea": 1800, "TotalBsmtSF": 900,
            "OverallQual": 7,  "OverallCond": 7,
            "YearBuilt": 2000, "YearRemodAdd": 2005,
            "GarageCars": 2,   "GarageArea": 480,
            "TotalSF": 2700,   "HouseAge": 24,
            "RemodelAge": 19,  "TotalBathrooms": 2.5,
            "HasGarage": 1,    "HasPool": 0,  "HasFireplace": 1,
        },
        {
            "name": "Luxury Property",
            "GrLivArea": 3200, "TotalBsmtSF": 1600,
            "OverallQual": 9,  "OverallCond": 8,
            "YearBuilt": 2010, "YearRemodAdd": 2015,
            "GarageCars": 3,   "GarageArea": 850,
            "TotalSF": 4800,   "HouseAge": 14,
            "RemodelAge": 9,   "TotalBathrooms": 4.0,
            "HasGarage": 1,    "HasPool": 1,  "HasFireplace": 1,
        },
    ]

    for s in samples:
        name = s.pop("name")
        price = predict_single(s, model, scaler, feature_names)
        print(f"  {name:<28}  →  Predicted Price: ${price:,.0f}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Feature names helper
# ---------------------------------------------------------------------------

def load_feature_names(filename: str = "feature_names.pkl") -> list:
    """Load the ordered feature name list saved during training."""
    filepath = models_path(filename)
    if not __import__("os").path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] feature_names.pkl not found: {filepath}\n"
            "Run main.py first."
        )
    return joblib.load(filepath)
