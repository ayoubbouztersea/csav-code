#!/usr/bin/env python3
"""
XGBoost classifier training script for QteConso prediction.

Usage:
    python train_Xgboost.py --data-path df_ml2.csv --test-size 0.2

Example nohup command for GCP c2d-standard-32 VM:
    nohup python3 train_Xgboost.py --data-path df_ml2.csv > train.log 2>&1 &

Dependencies:
    pip install pandas numpy scikit-learn xgboost
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TARGET_COLUMN = "QteConso"
EXCLUDE_COLUMNS = ["LIBEL_ARTICLE", "LIBEL_ARTICLE_length"]
CLASS_LABELS = [0, 1, 2, 3]

# XGBoost configuration for CPU training on c2d-standard-32 (32 vCPU, 128 GB RAM)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_jobs": 32,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "verbosity": 1,
    "tree_method": "hist",  # Efficient for CPU
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
        description="Train XGBoost classifier for QteConso prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="df_ml2.csv",
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
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
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
    # Drop excluded columns if they exist
    cols_to_drop = [col for col in EXCLUDE_COLUMNS if col in df.columns]
    if cols_to_drop:
        logger.info(f"Excluding columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Separate target
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    """Build preprocessing pipeline for numeric and categorical columns."""
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")

    # Numeric pipeline: median imputation
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical pipeline: most frequent imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: list, categorical_cols: list) -> list[str]:
    """Extract feature names after preprocessing."""
    feature_names = numeric_cols.copy()

    # Get one-hot encoded feature names if categorical columns exist
    if categorical_cols:
        cat_transformer = preprocessor.named_transformers_.get("cat")
        if cat_transformer is not None:
            onehot = cat_transformer.named_steps["onehot"]
            cat_feature_names = onehot.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_feature_names)

    return feature_names


def compute_class_weights(y: pd.Series) -> np.ndarray:
    """Compute sample weights to handle class imbalance."""
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    logger.info("Sample weights computed for class balancing")

    # Log class distribution and weights
    unique_classes = np.unique(y)
    for cls in unique_classes:
        mask = y == cls
        avg_weight = sample_weights[mask].mean()
        logger.info(f"  Class {cls}: count={mask.sum()}, avg_weight={avg_weight:.4f}")

    return sample_weights


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
    output_prefix: str = "metrics",
) -> None:
    """Save metrics to JSON and CSV files."""
    # Prepare metrics dictionary
    metrics = {
        "overall_accuracy": overall_accuracy,
        **per_class_accuracy,
        "confusion_matrix": confusion_mat.tolist(),
        "timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "xgb_params": XGB_PARAMS,
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
    cm_path = "confusion_matrix.csv"
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
    logger.info("XGBoost Training Script for QteConso Classification")
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

    # Build preprocessor
    logger.info("Building preprocessing pipeline...")
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Split data with stratification
    logger.info(f"Splitting data (test_size={args.test_size}, stratified)...")

    # #region agent log H1,H4,H5
    import json as _json
    _log_path = "/Users/ayoubbouz/Documents/DS_Projects/csav-code/.cursor/debug.log"
    _vc = y.value_counts()
    _rare_classes = [(int(k), int(v)) for k, v in _vc.items() if v < 2]
    _debug_data3 = {"hypothesisId": "H1,H4,H5", "location": "train_Xgboost.py:before_split", "message": "Pre-split analysis", "data": {"total_samples": len(y), "num_unique_classes": int(y.nunique()), "min_class_count": int(_vc.min()), "max_class_count": int(_vc.max()), "rare_classes_lt_2": _rare_classes, "expected_classes": CLASS_LABELS, "actual_unique": sorted(y.unique().tolist())}, "timestamp": int(datetime.now().timestamp() * 1000), "sessionId": "debug-session", "runId": "run1"}
    with open(_log_path, "a") as _f: _f.write(_json.dumps(_debug_data3) + "\n")
    # #endregion

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Fit preprocessor and transform data
    logger.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    logger.info(f"Total features after preprocessing: {len(feature_names)}")

    # Convert to DataFrame with proper column names for XGBoost
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    # Compute sample weights for class balancing
    logger.info("Computing sample weights for class balancing...")
    sample_weights = compute_class_weights(y_train)

    # Initialize XGBoost classifier
    logger.info("Initializing XGBoost classifier...")
    logger.info(f"XGBoost parameters: {XGB_PARAMS}")
    model = xgb.XGBClassifier(
        **XGB_PARAMS,
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
    )

    # Train model
    logger.info("Starting model training...")
    train_start = datetime.now()
    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_processed, y_test)],
        verbose=True,
    )
    train_duration = datetime.now() - train_start
    logger.info(f"Training completed in {train_duration}")

    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test_processed)

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

    # Save metrics
    logger.info("\nSaving metrics...")
    save_metrics(overall_accuracy, per_class_accuracy, confusion_mat)

    # Save model
    model_path = "model_qteconso.xgb"
    logger.info(f"Saving model to: {model_path}")
    model.save_model(model_path)
    logger.info("Model saved successfully")

    # Final summary
    total_duration = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {total_duration}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: metrics.json, metrics.csv")
    logger.info(f"Confusion matrix saved to: confusion_matrix.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

