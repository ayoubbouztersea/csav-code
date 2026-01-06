#!/usr/bin/env python3
"""
LightGBM classifier training script for QteConso prediction (0, 1, 2, 3).

LightGBM advantages over XGBoost:
- Native categorical feature support (no encoding needed)
- Lower memory usage (histogram-based algorithm)
- Faster training on large datasets
- Built-in class balancing

Usage:
    python train_LightGBM.py --data-path data/DF_ML_M1.csv --test-size 0.2

Example nohup command for GCP c2d-standard-32 VM:
    nohup python3 train_LightGBM.py --data-path data/DF_ML_M1.csv > train_lgbm.log 2>&1 &

Dependencies:
    pip install pandas numpy scikit-learn lightgbm
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TARGET_COLUMN = "QteConso"
EXCLUDE_COLUMNS = [
    "LIBEL_ARTICLE",
    "LIBEL_ARTICLE_length",
    "LIBEL_ARTICLE_Length",
    "len_LIBEL_ARTICLE",
    "Dossier",
    "NB_INTERV",
    "Clot_1er_Pa",
]
CLASS_LABELS = [0, 1, 2, 3]

# LightGBM configuration for CPU training on c2d-standard-32 (32 vCPU, 128 GB RAM)
LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "class_weight": "balanced",  # Handle class imbalance
    "n_jobs": 32,
    "random_state": RANDOM_STATE,
    "verbose": 1,
    "force_col_wise": True,  # Better for datasets with many features
}

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seeds set to {seed}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifier for QteConso prediction (0,1,2,3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/DF_ML_M1.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file with memory optimization."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from: {data_path}")
    
    # Load with optimized dtypes to reduce memory
    df = pd.read_csv(data_path, low_memory=False)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in data")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    logger.info(f"Target column '{TARGET_COLUMN}' found")


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, exclude specified columns."""
    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows} rows")

    # Drop excluded columns if they exist
    cols_to_drop = [col for col in EXCLUDE_COLUMNS if col in df.columns]
    if cols_to_drop:
        logger.info(f"Excluding columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # =========================================================================
    # DATA VALIDATION: Clean target column before conversion
    # =========================================================================
    logger.info("Validating target column...")

    # Check for missing values in target
    na_count = df[TARGET_COLUMN].isna().sum()
    if na_count > 0:
        logger.warning(f"Found {na_count} NA/NaN values in target column '{TARGET_COLUMN}'")

    # Check for infinite values in target (if numeric)
    if pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        inf_count = np.isinf(df[TARGET_COLUMN].replace([np.nan], 0)).sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in target column '{TARGET_COLUMN}'")
    else:
        inf_count = 0

    # Create mask for valid rows (not NA, not infinite)
    valid_mask = df[TARGET_COLUMN].notna()
    if pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        valid_mask &= ~np.isinf(df[TARGET_COLUMN])

    # Filter to valid rows only
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} rows with invalid target values (NA or inf)")
        df = df[valid_mask].reset_index(drop=True)
        logger.info(f"Dataset size after cleaning: {len(df)} rows")

    # Validate we still have data
    if len(df) == 0:
        logger.error("No valid data remaining after cleaning target column!")
        sys.exit(1)

    # Separate target - now safe to convert to int
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    # Clip target values to valid range [0, 3]
    y = y.clip(0, 3)

    # Log target distribution
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")

    return X, y


def identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")

    return numeric_cols, categorical_cols


def preprocess_for_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for LightGBM.
    
    LightGBM handles categorical features natively, so we only need to:
    1. Impute missing values
    2. Normalize numeric features
    3. Convert categorical columns to 'category' dtype
    """
    logger.info("Preprocessing data for LightGBM...")

    # Process numeric columns: impute + normalize
    if numeric_cols:
        logger.info(f"Processing {len(numeric_cols)} numeric columns...")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Process categorical columns: convert to category dtype (LightGBM native support)
    if categorical_cols:
        logger.info(f"Converting {len(categorical_cols)} categorical columns to 'category' dtype...")
        for col in categorical_cols:
            # Fill missing with a placeholder
            X_train[col] = X_train[col].fillna("__MISSING__").astype("category")
            X_test[col] = X_test[col].fillna("__MISSING__")
            
            # Ensure test categories are subset of train categories
            train_categories = X_train[col].cat.categories
            X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

    logger.info(f"Preprocessing complete. Shape: {X_train.shape}")
    return X_train, X_test


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> dict:
    """Compute per-class accuracy from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = {}

    for i, label in enumerate(labels):
        total = cm[i, :].sum()
        if total > 0:
            per_class_acc[f"accuracy_class_{label}"] = float(cm[i, i] / total)
        else:
            per_class_acc[f"accuracy_class_{label}"] = "N/A"

    return per_class_acc, cm


def save_metrics(
    overall_accuracy: float,
    per_class_accuracy: dict,
    confusion_mat: np.ndarray,
    output_prefix: str = "metrics_lgbm",
) -> None:
    """Save metrics to JSON and CSV files."""
    # Prepare metrics dictionary
    metrics = {
        "overall_accuracy": overall_accuracy,
        **per_class_accuracy,
        "confusion_matrix": confusion_mat.tolist(),
        "timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "lgbm_params": LGBM_PARAMS,
    }

    # Save to JSON
    json_path = f"{output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {json_path}")

    # Save to CSV (flat format)
    csv_metrics = {
        "overall_accuracy": [overall_accuracy],
    }
    for key, value in per_class_accuracy.items():
        csv_metrics[key] = [value if value != "N/A" else np.nan]

    csv_path = f"{output_prefix}.csv"
    pd.DataFrame(csv_metrics).to_csv(csv_path, index=False)
    logger.info(f"Metrics saved to: {csv_path}")

    # Save confusion matrix separately
    cm_path = "confusion_matrix_lgbm.csv"
    cm_df = pd.DataFrame(
        confusion_mat,
        index=[f"true_{i}" for i in CLASS_LABELS],
        columns=[f"pred_{i}" for i in CLASS_LABELS],
    )
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")


def main() -> None:
    """Main training pipeline."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("LightGBM Training Script for QteConso Classification")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_args()
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Test size: {args.test_size}")

    # Set seeds for reproducibility
    set_seeds(RANDOM_STATE)

    # Load and validate data
    df = load_data(args.data_path)
    validate_data(df)

    # Prepare features and target
    X, y = prepare_features_target(df)

    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X)

    # Split data with stratification
    logger.info(f"Splitting data (test_size={args.test_size}, stratified)...")

    # For stratification, bin rare classes to avoid errors
    y_stratify = y.copy()
    value_counts = y_stratify.value_counts()
    rare_classes = value_counts[value_counts < 2].index.tolist()
    if rare_classes:
        logger.warning(f"Rare classes with <2 samples found: {rare_classes}. Binning for stratification.")
        for rare_class in rare_classes:
            nearest = min(CLASS_LABELS, key=lambda x: abs(x - rare_class))
            y_stratify = y_stratify.replace(rare_class, nearest)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y_stratify,
    )
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Free memory
    del X, df, y_stratify
    gc.collect()

    # Preprocess data
    X_train, X_test = preprocess_for_lightgbm(
        X_train.copy(), X_test.copy(), numeric_cols, categorical_cols
    )

    logger.info(f"Memory usage - X_train: {X_train.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Get categorical feature names for LightGBM
    cat_feature_names = categorical_cols if categorical_cols else "auto"

    # Initialize LightGBM classifier
    logger.info("Initializing LightGBM classifier...")
    logger.info(f"LightGBM parameters: {LGBM_PARAMS}")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Train model
    logger.info("Starting model training...")
    train_start = datetime.now()
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        categorical_feature=cat_feature_names,
    )
    
    train_duration = datetime.now() - train_start
    logger.info(f"Training completed in {train_duration}")

    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    overall_accuracy = accuracy_score(y_test, y_pred)
    per_class_accuracy, confusion_mat = compute_per_class_accuracy(
        y_test.values, y_pred, CLASS_LABELS
    )

    # Print metrics
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    for key, value in per_class_accuracy.items():
        if value == "N/A":
            logger.info(f"{key}: N/A (no samples)")
        else:
            logger.info(f"{key}: {value:.4f}")

    logger.info("\nConfusion Matrix:")
    logger.info(f"Rows: True labels, Columns: Predicted labels")
    logger.info(f"Labels: {CLASS_LABELS}")
    for i, row in enumerate(confusion_mat):
        logger.info(f"  Class {CLASS_LABELS[i]}: {row}")

    # Print classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, labels=CLASS_LABELS))

    # Save metrics
    logger.info("\nSaving metrics...")
    save_metrics(overall_accuracy, per_class_accuracy, confusion_mat)

    # Save model
    model_path = "model_qteconso_lgbm.txt"
    logger.info(f"Saving model to: {model_path}")
    model.booster_.save_model(model_path)
    logger.info("Model saved successfully")

    # Feature importance
    logger.info("\nTop 20 Feature Importances:")
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']}")
    
    # Save feature importance
    feature_importance.to_csv("feature_importance_lgbm.csv", index=False)
    logger.info("Feature importance saved to: feature_importance_lgbm.csv")

    # Final summary
    total_duration = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {total_duration}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: metrics_lgbm.json, metrics_lgbm.csv")
    logger.info(f"Confusion matrix saved to: confusion_matrix_lgbm.csv")
    logger.info(f"Feature importance saved to: feature_importance_lgbm.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

