"""
preprocess.py
-------------
Handles all data loading, EDA, cleaning, encoding, and feature engineering
for the House Price Prediction task.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from utils import (
    data_path,
    outputs_path,
    ensure_dir,
    check_dataset_exists,
    summarise_dataframe,
    save_figure,
    set_plot_style,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "SalePrice"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# Columns with >40 % missing values → drop entirely
HIGH_MISSING_THRESHOLD = 0.40

# Numeric columns to fill with median; categorical with mode
FILL_STRATEGY_MAP = {
    "LotFrontage":   "median",
    "MasVnrArea":    "zero",
    "GarageYrBlt":   "median",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw training CSV from data/train.csv.

    Returns
    -------
    pd.DataFrame  Raw, unmodified training data.
    """
    filepath = data_path("train.csv")
    check_dataset_exists(filepath)
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset  →  {df.shape[0]} rows × {df.shape[1]} columns")
    summarise_dataframe(df, label="Raw Training Data")
    return df


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """
    Generate and save exploratory data-analysis plots to outputs/.

    Plots produced
    --------------
    1. Target distribution (SalePrice histogram + KDE)
    2. Log-transformed target distribution
    3. Correlation heat-map (top 15 numeric features)
    4. Missing-value bar chart
    """
    set_plot_style()
    ensure_dir(outputs_path())

    # 1 — SalePrice distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Target Variable — SalePrice", fontsize=14, fontweight="bold")

    sns.histplot(df[TARGET_COL], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Original Scale")
    axes[0].set_xlabel("SalePrice ($)")

    sns.histplot(np.log1p(df[TARGET_COL]), kde=True, ax=axes[1], color="#DD8452")
    axes[1].set_title("Log-Transformed Scale")
    axes[1].set_xlabel("log(SalePrice + 1)")

    save_figure(fig, "eda_saleprice_distribution.png")

    # 2 — Correlation heat-map
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[TARGET_COL].abs().sort_values(ascending=False)
    top_cols = corr.index[:16].tolist()   # target + top 15 features
    corr_matrix = numeric_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heat-Map (Top 15 Numeric Features)", fontsize=13, pad=14)
    save_figure(fig, "eda_correlation_heatmap.png")

    # 3 — Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(30)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing.plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_title("Missing Values per Column", fontsize=13)
        ax.set_ylabel("Count")
        ax.set_xlabel("Column")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, "eda_missing_values.png")

    print("[INFO] EDA plots saved to outputs/")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.

    Strategy
    --------
    - Columns with > HIGH_MISSING_THRESHOLD missing → drop.
    - Specific numeric columns → see FILL_STRATEGY_MAP.
    - Remaining numeric columns → median.
    - Remaining categorical columns → mode (most frequent).

    Returns
    -------
    pd.DataFrame  Cleaned dataframe (copy).
    """
    df = df.copy()

    # Drop high-missing columns
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > HIGH_MISSING_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"[INFO] Dropping {len(drop_cols)} high-missing columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

    # Apply explicit fill strategies
    for col, strategy in FILL_STRATEGY_MAP.items():
        if col not in df.columns:
            continue
        if strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "zero":
            df[col].fillna(0, inplace=True)

    # Auto-fill remaining numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Auto-fill remaining categorical columns with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[INFO] Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features from the existing columns.

    New features
    ------------
    TotalSF         : GrLivArea + TotalBsmtSF  (total usable area)
    HouseAge        : YrSold − YearBuilt
    RemodelAge      : YrSold − YearRemodAdd
    TotalBathrooms  : Full + 0.5 × Half + BsmtFull + 0.5 × BsmtHalf
    HasGarage       : binary flag
    HasPool         : binary flag
    HasFireplace    : binary flag

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    df["TotalSF"]        = df.get("GrLivArea", 0) + df.get("TotalBsmtSF", 0)
    df["HouseAge"]       = df.get("YrSold", 0)    - df.get("YearBuilt", 0)
    df["RemodelAge"]     = df.get("YrSold", 0)    - df.get("YearRemodAdd", 0)
    df["TotalBathrooms"] = (
        df.get("FullBath", 0)
        + 0.5 * df.get("HalfBath", 0)
        + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("BsmtHalfBath", 0)
    )
    df["HasGarage"]    = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasPool"]      = (df.get("PoolArea", 0)   > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)

    print("[INFO] Feature engineering complete — added 7 new features")
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all remaining object/category columns.

    Returns
    -------
    pd.DataFrame  With only numeric columns.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[INFO] Label-encoded {len(cat_cols)} categorical columns")
    return df


# ---------------------------------------------------------------------------
# Log transform target
# ---------------------------------------------------------------------------

def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to SalePrice to reduce skewness."""
    df = df.copy()
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    print("[INFO] Applied log1p transform to SalePrice")
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X from target vector y.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[TARGET_COL, "Id"], errors="ignore")
    y = df[TARGET_COL]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(
        f"[INFO] Train size: {len(X_train)}   Test size: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing():
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray arrays (scaled)
    feature_names                    : list[str]
    """
    # Load
    df = load_raw_data()

    # EDA
    run_eda(df)

    # Clean
    df = handle_missing_values(df)

    # Feature engineering
    df = engineer_features(df)

    # Log-transform target
    df = log_transform_target(df)

    # Encode categoricals
    df = encode_categorical(df)

    # Split X / y
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("[INFO] Feature scaling applied (StandardScaler)")

    return X_train_sc, X_test_sc, y_train.values, y_test.values, feature_names, scaler
