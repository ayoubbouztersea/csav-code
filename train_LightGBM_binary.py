#!/usr/bin/env python3
"""
LightGBM Binary Classifier: Predicts 0 vs Non-0 from QteConso.

This script trains a LightGBM binary classifier to predict whether QteConso is 0 or non-0.
It exports an accuracy table showing performance for each class.

Usage:
    python train_LightGBM_binary.py --data-path data/DF_ML_M1.csv

Dependencies:
    pip install pandas numpy scikit-learn lightgbm
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TARGET_COLUMN = "QteConso"

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    "QteConso",  # Target column
    "Qteconso",  # Possible variant spelling
    "len_LIBEL_ARTICLE",
    "LIBEL_ARTICLE",
    "Dossier",
]

CLASS_LABELS = [0, 1]  # 0 = Zero, 1 = Non-Zero
CLASS_NAMES = {0: "Zero (0)", 1: "Non-Zero (1, 2, 3)"}

# LightGBM optimized parameters for high accuracy binary classification
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 30,
    "min_child_weight": 0.001,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 0.1,  # L2 regularization
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "force_col_wise": True,
    "is_unbalance": True,  # Handle class imbalance
}

# Early stopping configuration
EARLY_STOPPING_ROUNDS = 50

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
        description="Train LightGBM binary classifier for 0 vs Non-0 prediction",
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
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="binary_accuracy",
        help="Prefix for output files",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and binary target (0 vs Non-0).
    
    Returns:
        X: Feature DataFrame
        y: Binary target Series (0 = zero, 1 = non-zero)
    """
    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows} rows")

    # Validate target column exists
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found!")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Clean target column - remove NA/NaN values
    na_count = df[TARGET_COLUMN].isna().sum()
    if na_count > 0:
        logger.warning(f"Found {na_count} NA values in target column, removing...")
        df = df[df[TARGET_COLUMN].notna()].reset_index(drop=True)

    # Create binary target: 0 stays 0, everything else becomes 1
    y_original = df[TARGET_COLUMN].astype(float)
    y = (y_original != 0).astype(int)

    logger.info(f"Original target distribution:\n{y_original.value_counts().sort_index()}")
    logger.info(f"Binary target distribution (0 vs Non-0):\n{y.value_counts().sort_index()}")

    # Drop excluded columns
    cols_to_drop = [col for col in EXCLUDE_COLUMNS if col in df.columns]
    if cols_to_drop:
        logger.info(f"Excluding columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # All remaining columns are features
    X = df

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    return X, y


def identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")

    return numeric_cols, categorical_cols


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for LightGBM.
    
    - Impute missing numeric values with median
    - Normalize numeric features
    - Convert categorical columns to 'category' dtype (LightGBM native support)
    """
    logger.info("Preprocessing data...")

    # Process numeric columns
    if numeric_cols:
        logger.info(f"Processing {len(numeric_cols)} numeric columns...")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Process categorical columns
    if categorical_cols:
        logger.info(f"Converting {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            X_train[col] = X_train[col].fillna("__MISSING__").astype("category")
            X_test[col] = X_test[col].fillna("__MISSING__")

            # Ensure test categories are subset of train categories
            train_categories = X_train[col].cat.categories
            X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

    logger.info(f"Preprocessing complete. Shape: {X_train.shape}")
    return X_train, X_test


def compute_accuracy_table(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute detailed accuracy table for binary classification.
    
    Returns a DataFrame with accuracy metrics for class 0 and class 1 (non-0).
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    rows = []
    for i, label in enumerate(CLASS_LABELS):
        true_count = cm[i, :].sum()  # Total actual samples
        pred_count = cm[:, i].sum()  # Total predicted as this class
        correct = cm[i, i]           # Correctly predicted
        accuracy = correct / true_count if true_count > 0 else 0.0

        rows.append({
            "Class": CLASS_NAMES[label],
            "Label": label,
            "Support (Actual Count)": int(true_count),
            "Predicted Count": int(pred_count),
            "Correct Predictions": int(correct),
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision (%)": round(precision[i] * 100, 2),
            "Recall (%)": round(recall[i] * 100, 2),
            "F1-Score": round(f1[i], 4),
        })

    # Add overall metrics row
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    rows.append({
        "Class": "OVERALL",
        "Label": "-",
        "Support (Actual Count)": int(len(y_true)),
        "Predicted Count": int(len(y_pred)),
        "Correct Predictions": int((y_true == y_pred).sum()),
        "Accuracy (%)": round(overall_accuracy * 100, 2),
        "Precision (%)": round(macro_precision * 100, 2),
        "Recall (%)": round(macro_recall * 100, 2),
        "F1-Score": round(macro_f1, 4),
    })

    df = pd.DataFrame(rows)
    return df, cm, auc


def print_accuracy_table(accuracy_df: pd.DataFrame, cm: np.ndarray, auc: float) -> None:
    """Print formatted accuracy table to console."""
    logger.info("\n" + "=" * 80)
    logger.info("ACCURACY TABLE: 0 vs Non-0 Classification")
    logger.info("=" * 80)
    
    # Print the table
    print("\n" + accuracy_df.to_string(index=False))
    
    logger.info("\n" + "-" * 80)
    logger.info("CONFUSION MATRIX")
    logger.info("-" * 80)
    logger.info("                  Predicted 0    Predicted Non-0")
    logger.info(f"Actual 0          {cm[0, 0]:>10,}    {cm[0, 1]:>15,}")
    logger.info(f"Actual Non-0      {cm[1, 0]:>10,}    {cm[1, 1]:>15,}")
    
    if not np.isnan(auc):
        logger.info(f"\nAUC-ROC Score: {auc:.4f}")
    
    logger.info("=" * 80)


def save_results(
    accuracy_df: pd.DataFrame,
    cm: np.ndarray,
    auc: float,
    feature_importance: pd.DataFrame,
    output_prefix: str,
) -> None:
    """Save all results to files."""
    
    # Save accuracy table
    accuracy_path = f"{output_prefix}_table.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    logger.info(f"Accuracy table saved to: {accuracy_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_Non0"],
        columns=["Predicted_0", "Predicted_Non0"],
    )
    cm_path = f"{output_prefix}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Save feature importance
    fi_path = f"{output_prefix}_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    logger.info(f"Feature importance saved to: {fi_path}")
    
    # Save summary metrics
    summary = {
        "overall_accuracy": accuracy_df[accuracy_df["Class"] == "OVERALL"]["Accuracy (%)"].values[0],
        "accuracy_class_0": accuracy_df[accuracy_df["Label"] == 0]["Accuracy (%)"].values[0],
        "accuracy_class_non0": accuracy_df[accuracy_df["Label"] == 1]["Accuracy (%)"].values[0],
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
        "timestamp": datetime.now().isoformat(),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")


def main() -> None:
    """Main training pipeline."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("LightGBM Binary Classifier: 0 vs Non-0 Prediction")
    logger.info("=" * 80)

    # Parse arguments
    args = parse_args()
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Test size: {args.test_size}")

    # Set seeds
    set_seeds(RANDOM_STATE)

    # Load data
    df = load_data(args.data_path)

    # Prepare features and binary target
    X, y = prepare_features_target(df)

    # Free memory
    del df
    gc.collect()

    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X)

    # Split data
    logger.info(f"Splitting data (test_size={args.test_size}, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Free memory
    del X
    gc.collect()

    # Preprocess data
    X_train, X_test = preprocess_data(
        X_train.copy(), X_test.copy(), numeric_cols, categorical_cols
    )

    # Get categorical feature names
    cat_feature_names = categorical_cols if categorical_cols else "auto"

    # Initialize and train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LIGHTGBM BINARY CLASSIFIER")
    logger.info("=" * 80)
    logger.info(f"LightGBM parameters: {LGBM_PARAMS}")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    train_start = datetime.now()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        categorical_feature=cat_feature_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    train_duration = datetime.now() - train_start
    logger.info(f"Training completed in {train_duration}")
    logger.info(f"Best iteration: {model.best_iteration_}")

    # Make predictions
    logger.info("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (non-zero)

    # Compute accuracy table
    logger.info("Computing accuracy metrics...")
    accuracy_df, cm, auc = compute_accuracy_table(y_test.values, y_pred, y_prob)

    # Print accuracy table
    print_accuracy_table(accuracy_df, cm, auc)

    # Feature importance
    logger.info("\nTop 20 Feature Importances:")
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    for _, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']}")

    # Save results
    logger.info("\nSaving results...")
    save_results(accuracy_df, cm, auc, feature_importance, args.output_prefix)

    # Save model
    model_path = f"{args.output_prefix}_model.txt"
    model.booster_.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Final summary
    total_duration = datetime.now() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {total_duration}")
    logger.info(f"Overall Accuracy: {accuracy_df[accuracy_df['Class'] == 'OVERALL']['Accuracy (%)'].values[0]}%")
    logger.info(f"Accuracy for Class 0: {accuracy_df[accuracy_df['Label'] == 0]['Accuracy (%)'].values[0]}%")
    logger.info(f"Accuracy for Non-0: {accuracy_df[accuracy_df['Label'] == 1]['Accuracy (%)'].values[0]}%")
    if not np.isnan(auc):
        logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

