"""
main.py
-------
Entry point for the House Price Prediction pipeline.

Run
---
    python main.py

Steps executed
--------------
1. Preprocessing  — load, clean, engineer features, encode, scale
2. Training       — compare models, train best, save model + scaler
3. Evaluation     — metrics, plots, report
4. Prediction     — demo predictions on sample properties
"""

import sys
import joblib

from preprocess import run_preprocessing
from train      import run_training
from evaluate   import run_evaluation
from predict    import run_demo_predictions, load_model, load_scaler, load_feature_names
from utils      import models_path, ensure_dir, outputs_path


def main() -> None:
    print("=" * 60)
    print("  PRODIGY_ML_01 — House Price Prediction")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 1/4 — Preprocessing …")
    ensure_dir(outputs_path())
    ensure_dir(models_path())

    X_train, X_test, y_train, y_test, feature_names, scaler = run_preprocessing()

    # Persist feature names so predict.py can reload them independently
    joblib.dump(feature_names, models_path("feature_names.pkl"))
    print(f"[INFO] Feature names saved → {models_path('feature_names.pkl')}")

    # ------------------------------------------------------------------
    # Step 2: Training
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 2/4 — Training …")
    model, cv_results = run_training(X_train, y_train, feature_names, scaler)

    # ------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 3/4 — Evaluation …")
    metrics = run_evaluation(model, X_test, y_test, cv_results)

    # ------------------------------------------------------------------
    # Step 4: Demo Predictions
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 4/4 — Demo Predictions …")
    run_demo_predictions(model, scaler, feature_names)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  R² Score : {metrics['R² Score']}")
    print(f"  RMSE ($) : {metrics['RMSE ($)']:,.0f}")
    print(f"  MAE  ($) : {metrics['MAE ($)']:,.0f}")
    print("\n  Saved artefacts:")
    print("    models/house_price_model.pkl")
    print("    models/scaler.pkl")
    print("    models/feature_names.pkl")
    print("    outputs/ — plots + evaluation report + metrics JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
