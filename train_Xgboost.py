#!/usr/bin/env python3
"""
XGBoost regressor training script for QteConso prediction.

Predicts QteConso as a continuous value (0-3), then rounds/clips for classification metrics.
This approach avoids stratification issues with rare classes.

Usage:
    python train_Xgboost.py --data-path data/DF_ML_M1.csv --test-size 0.2

Example nohup command for GCP c2d-standard-32 VM:
    nohup python3 train_Xgboost.py --data-path data/DF_ML_M1.csv > train.log 2>&1 &

Dependencies:
    pip install pandas numpy scikit-learn xgboost
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
import xgboost as xgb

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TARGET_COLUMN = "QteConso"
EXCLUDE_COLUMNS = [
    "LIBEL_ARTICLE",
    "LIBEL_ARTICLE_length",
    "LIBEL_ARTICLE_Length",  # Case variant
    "Dossier",
    "NB_INTERV",
    "Clot_1er_Pa",
]
CLASS_LABELS = [0, 1, 2, 3]

# XGBoost configuration for CPU training on c2d-standard-32 (32 vCPU, 128 GB RAM)
# Using regressor to predict values 0-3 continuously, then round for classification
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_jobs": 32,
    "eval_metric": "rmse",  # Regression metric
    "random_state": RANDOM_STATE,
    "verbosity": 1,
    "tree_method": "hist",  # Efficient for CPU
    "objective": "reg:squarederror",  # Regression objective
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
        description="Train XGBoost classifier for QteConso prediction (0,1,2,3)",
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


class MemoryEfficientPreprocessor:
    """
    Memory-efficient preprocessor using Label Encoding instead of One-Hot Encoding.
    This prevents memory explosion with high-cardinality categorical columns.
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "MemoryEfficientPreprocessor":
        """Fit the preprocessor on training data."""
        # Identify column types
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(f"Numeric columns ({len(self.numeric_cols)}): {self.numeric_cols[:10]}{'...' if len(self.numeric_cols) > 10 else ''}")
        logger.info(f"Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols[:10]}{'...' if len(self.categorical_cols) > 10 else ''}")

        # Fit numeric imputer and scaler
        if self.numeric_cols:
            numeric_data = X[self.numeric_cols].values
            numeric_data = self.numeric_imputer.fit_transform(numeric_data)
            self.scaler.fit(numeric_data)

        # Fit label encoders for each categorical column
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Handle missing values by treating them as a category
            col_data = X[col].fillna("__MISSING__").astype(str)
            le.fit(col_data)
            self.label_encoders[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        result_arrays = []

        # Process numeric columns: impute + normalize
        if self.numeric_cols:
            numeric_data = X[self.numeric_cols].values
            numeric_data = self.numeric_imputer.transform(numeric_data)
            numeric_data = self.scaler.transform(numeric_data)
            result_arrays.append(numeric_data)

        # Process categorical columns: label encode
        if self.categorical_cols:
            cat_encoded = np.zeros((len(X), len(self.categorical_cols)), dtype=np.float32)
            for i, col in enumerate(self.categorical_cols):
                col_data = X[col].fillna("__MISSING__").astype(str)
                le = self.label_encoders[col]
                # Handle unseen categories by assigning -1
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ], dtype=np.float32)
                cat_encoded[:, i] = encoded
            result_arrays.append(cat_encoded)

        return np.hstack(result_arrays).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self) -> list[str]:
        """Get feature names after preprocessing."""
        feature_names = [sanitize_feature_name(col) for col in self.numeric_cols]
        feature_names.extend([sanitize_feature_name(col) for col in self.categorical_cols])
        return feature_names


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature name for XGBoost compatibility (no [, ], or <)."""
    return name.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")


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
    rmse: float,
    mae: float,
    output_prefix: str = "metrics",
) -> None:
    """Save metrics to JSON and CSV files."""
    # Prepare metrics dictionary
    metrics = {
        "rmse": rmse,
        "mae": mae,
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
        "rmse": [rmse],
        "mae": [mae],
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

    # Split data with stratification
    logger.info(f"Splitting data (test_size={args.test_size}, stratified)...")
    
    # For stratification, bin rare classes to avoid errors with classes having <2 samples
    y_stratify = y.copy()
    value_counts = y_stratify.value_counts()
    rare_classes = value_counts[value_counts < 2].index.tolist()
    if rare_classes:
        logger.warning(f"Rare classes with <2 samples found: {rare_classes}. Binning to nearest valid class for stratification.")
        # Map rare classes to the nearest common class (0, 1, 2, or 3)
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

    # Build and fit memory-efficient preprocessor (Label Encoding + StandardScaler)
    logger.info("Building memory-efficient preprocessing pipeline...")
    logger.info("Using Label Encoding (not One-Hot) to prevent memory explosion")
    preprocessor = MemoryEfficientPreprocessor()
    
    logger.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names()
    logger.info(f"Total features after preprocessing: {len(feature_names)}")
    logger.info(f"Memory usage - X_train: {X_train_processed.nbytes / 1e9:.2f} GB")

    # Free up memory from raw DataFrames
    del X_train, X_test, X, df
    gc.collect()
    logger.info("Freed memory from raw DataFrames")

    # Convert to DataFrame with proper column names for XGBoost
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    # Initialize XGBoost regressor
    logger.info("Initializing XGBoost regressor...")
    logger.info(f"XGBoost parameters: {XGB_PARAMS}")
    model = xgb.XGBRegressor(**XGB_PARAMS)

    # Train model
    logger.info("Starting model training...")
    train_start = datetime.now()
    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=True,
    )
    train_duration = datetime.now() - train_start
    logger.info(f"Training completed in {train_duration}")

    # Make predictions (regression output)
    logger.info("Making predictions on test set...")
    y_pred_raw = model.predict(X_test_processed)

    # Round and clip predictions to valid class range [0, 3]
    y_pred = np.clip(np.round(y_pred_raw), 0, 3).astype(int)
    logger.info(f"Raw predictions range: [{y_pred_raw.min():.3f}, {y_pred_raw.max():.3f}]")
    logger.info(f"Rounded predictions distribution: {pd.Series(y_pred).value_counts().sort_index().to_dict()}")

    # Compute metrics
    logger.info("Computing evaluation metrics...")

    # Regression metrics (on raw predictions)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_raw))
    mae = mean_absolute_error(y_test, y_pred_raw)

    # Classification metrics (on rounded predictions)
    overall_accuracy = accuracy_score(y_test, y_pred)
    per_class_accuracy, confusion_mat = compute_per_class_accuracy(
        y_test.values, y_pred, CLASS_LABELS
    )

    # Print metrics
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"RMSE (raw): {rmse:.4f}")
    logger.info(f"MAE (raw): {mae:.4f}")
    logger.info(f"Overall Accuracy (rounded): {overall_accuracy:.4f}")
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
    save_metrics(overall_accuracy, per_class_accuracy, confusion_mat, rmse, mae)

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

