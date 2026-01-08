#!/usr/bin/env python3
"""
Evaluate LightGBM Binary Model: Accuracy per Cardinality (0, 1, 2, 3).

This script loads a trained LightGBM binary model (0 vs Non-0) and evaluates
its performance on a test dataset, reporting accuracy for each original
cardinality value (0, 1, 2, 3).

Usage:
    python evaluate_binary_model.py --model-path binary_accuracy_model.txt \
                                    --test-data data/test_dataset_prediag_auto_55K.csv

Dependencies:
    pip install pandas numpy scikit-learn lightgbm
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_COLUMN = "QteConso"

# Columns to exclude from features (same as training script)
EXCLUDE_COLUMNS = [
    "QteConso",  # Target column
    "Qteconso",  # Possible variant spelling
    "len_LIBEL_ARTICLE",
    "LIBEL_ARTICLE",
    "Dossier",
]

# Cardinality values to evaluate
CARDINALITIES = [0, 1, 2, 3]

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM binary model accuracy per cardinality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="binary_accuracy_model.txt",
        help="Path to the trained LightGBM model file",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test_dataset_prediag_auto_55K.csv",
        help="Path to the test dataset CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="accuracy_per_cardinality.csv",
        help="Path for the output accuracy table",
    )
    return parser.parse_args()


def load_model(model_path: str) -> lgb.Booster:
    """Load trained LightGBM model."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model from: {model_path}")
    model = lgb.Booster(model_file=model_path)
    logger.info(f"Model loaded successfully. Features: {model.num_feature()}")
    return model


def load_test_data(data_path: str) -> pd.DataFrame:
    """Load test data from CSV file."""
    if not os.path.exists(data_path):
        logger.error(f"Test data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading test data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    logger.info(f"Test data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def prepare_features_and_target(df: pd.DataFrame, model: lgb.Booster) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and extract original target (QteConso).
    
    Returns:
        X: Feature DataFrame aligned with model features
        y_original: Original target Series (0, 1, 2, 3)
    """
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

    # Extract original target
    y_original = df[TARGET_COLUMN].astype(float).astype(int)
    logger.info(f"Original target distribution:\n{y_original.value_counts().sort_index()}")

    # Drop excluded columns
    cols_to_drop = [col for col in EXCLUDE_COLUMNS if col in df.columns]
    if cols_to_drop:
        logger.info(f"Excluding columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Fix column names: replace spaces with underscores to match model features
    df.columns = df.columns.str.replace(' ', '_')
    logger.info("Column names normalized (spaces replaced with underscores)")

    # Get expected feature names from model
    model_features = model.feature_name()
    logger.info(f"Model expects {len(model_features)} features")

    # Align features with model
    missing_features = set(model_features) - set(df.columns)
    extra_features = set(df.columns) - set(model_features)

    if missing_features:
        logger.warning(f"Missing features (will be filled with 0): {missing_features}")
        for feat in missing_features:
            df[feat] = 0

    if extra_features:
        logger.info(f"Extra features in test data (will be dropped): {extra_features}")
        df = df.drop(columns=list(extra_features))

    # Reorder columns to match model
    X = df[model_features]

    logger.info(f"Features shape: {X.shape}")
    return X, y_original


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features for prediction.
    
    Note: In production, you should save and load the fitted imputer/scaler
    from training. This is a simplified version.
    
    For LightGBM Booster prediction, categorical columns must be encoded as
    integers (label encoding) rather than using pandas category dtype.
    """
    logger.info("Preprocessing features...")

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Categorical columns: {len(categorical_cols)}")

    # Process numeric columns
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Process categorical columns - convert to integer codes (label encoding)
    # LightGBM Booster expects numeric data, not pandas category dtype
    if categorical_cols:
        for col in categorical_cols:
            X[col] = X[col].fillna("__MISSING__")
            # Convert to category then extract integer codes
            X[col] = pd.Categorical(X[col]).codes.astype(float)

    # Convert all columns to float to avoid any categorical dtype issues
    X = X.astype(float)
    
    return X


def compute_accuracy_per_cardinality(
    y_original: np.ndarray,
    y_pred_binary: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Compute accuracy for each original cardinality value.
    
    The binary model predicts:
    - 0 = Zero
    - 1 = Non-Zero
    
    So:
    - For cardinality 0: correct if predicted 0
    - For cardinality 1, 2, 3: correct if predicted 1 (non-zero)
    """
    rows = []

    for cardinality in CARDINALITIES:
        # Get samples with this cardinality
        mask = y_original == cardinality
        count = mask.sum()

        if count == 0:
            rows.append({
                "Cardinality": cardinality,
                "Sample Count": 0,
                "Expected Binary": "N/A",
                "Correct Predictions": 0,
                "Incorrect Predictions": 0,
                "Accuracy (%)": "N/A",
                "Avg Probability": "N/A",
            })
            continue

        # Get predictions for this cardinality
        preds = y_pred_binary[mask]
        probs = y_prob[mask]

        # Expected binary class: 0 for cardinality 0, 1 for cardinality 1/2/3
        expected_binary = 0 if cardinality == 0 else 1

        # Calculate accuracy
        correct = (preds == expected_binary).sum()
        incorrect = count - correct
        accuracy = (correct / count) * 100

        # Average probability of being non-zero
        avg_prob = probs.mean()

        rows.append({
            "Cardinality": cardinality,
            "Sample Count": int(count),
            "Expected Binary": expected_binary,
            "Correct Predictions": int(correct),
            "Incorrect Predictions": int(incorrect),
            "Accuracy (%)": round(accuracy, 2),
            "Avg Prob (Non-Zero)": round(avg_prob, 4),
        })

    # Add overall row
    total_samples = len(y_original)
    # Expected: 0 stays 0, non-0 becomes 1
    expected_all = (y_original != 0).astype(int)
    total_correct = (y_pred_binary == expected_all).sum()
    overall_accuracy = (total_correct / total_samples) * 100

    rows.append({
        "Cardinality": "OVERALL",
        "Sample Count": int(total_samples),
        "Expected Binary": "-",
        "Correct Predictions": int(total_correct),
        "Incorrect Predictions": int(total_samples - total_correct),
        "Accuracy (%)": round(overall_accuracy, 2),
        "Avg Prob (Non-Zero)": round(y_prob.mean(), 4),
    })

    return pd.DataFrame(rows)


def compute_accuracy_binary_grouped(
    y_original: np.ndarray,
    y_pred_binary: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Compute accuracy for binary groups: 0 vs Non-0 (1, 2, 3 combined).
    """
    rows = []

    # Group 0: Cardinality = 0
    mask_zero = y_original == 0
    count_zero = mask_zero.sum()
    preds_zero = y_pred_binary[mask_zero]
    probs_zero = y_prob[mask_zero]
    correct_zero = (preds_zero == 0).sum()
    accuracy_zero = (correct_zero / count_zero) * 100 if count_zero > 0 else 0

    rows.append({
        "Cardinality": "0",
        "Sample Count": int(count_zero),
        "Correct Predictions": int(correct_zero),
        "Incorrect Predictions": int(count_zero - correct_zero),
        "Accuracy (%)": round(accuracy_zero, 2),
        "Avg Prob (Non-Zero)": round(probs_zero.mean(), 4) if count_zero > 0 else 0,
    })

    # Group Non-0: Cardinality = 1, 2, 3
    mask_nonzero = y_original != 0
    count_nonzero = mask_nonzero.sum()
    preds_nonzero = y_pred_binary[mask_nonzero]
    probs_nonzero = y_prob[mask_nonzero]
    correct_nonzero = (preds_nonzero == 1).sum()
    accuracy_nonzero = (correct_nonzero / count_nonzero) * 100 if count_nonzero > 0 else 0

    rows.append({
        "Cardinality": "Non-0 (1,2,3)",
        "Sample Count": int(count_nonzero),
        "Correct Predictions": int(correct_nonzero),
        "Incorrect Predictions": int(count_nonzero - correct_nonzero),
        "Accuracy (%)": round(accuracy_nonzero, 2),
        "Avg Prob (Non-Zero)": round(probs_nonzero.mean(), 4) if count_nonzero > 0 else 0,
    })

    # Overall
    total_samples = len(y_original)
    expected_all = (y_original != 0).astype(int)
    total_correct = (y_pred_binary == expected_all).sum()
    overall_accuracy = (total_correct / total_samples) * 100

    rows.append({
        "Cardinality": "OVERALL",
        "Sample Count": int(total_samples),
        "Correct Predictions": int(total_correct),
        "Incorrect Predictions": int(total_samples - total_correct),
        "Accuracy (%)": round(overall_accuracy, 2),
        "Avg Prob (Non-Zero)": round(y_prob.mean(), 4),
    })

    return pd.DataFrame(rows)


def print_accuracy_table(accuracy_df: pd.DataFrame) -> None:
    """Print formatted accuracy table to console."""
    logger.info("\n" + "=" * 90)
    logger.info("ACCURACY TABLE BY CARDINALITY (0, 1, 2, 3)")
    logger.info("=" * 90)
    logger.info("Binary Model: Predicts 0 vs Non-0")
    logger.info("- Cardinality 0 → Expected prediction: 0")
    logger.info("- Cardinality 1, 2, 3 → Expected prediction: 1 (Non-Zero)")
    logger.info("-" * 90)

    print("\n" + accuracy_df.to_string(index=False))

    logger.info("\n" + "=" * 90)


def main() -> None:
    """Main evaluation pipeline."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("LightGBM Binary Model Evaluation: Accuracy per Cardinality")
    logger.info("=" * 80)

    # Parse arguments
    args = parse_args()
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data path: {args.test_data}")

    # Load model
    model = load_model(args.model_path)

    # Load test data
    df = load_test_data(args.test_data)

    # Prepare features and target
    X, y_original = prepare_features_and_target(df, model)

    # Preprocess features
    X = preprocess_features(X.copy())

    # Make predictions using numpy array to avoid pandas categorical issues
    logger.info("\nMaking predictions...")
    X_values = X.values  # Convert to numpy array
    y_prob = model.predict(X_values)  # Probability of class 1 (non-zero)
    y_pred_binary = (y_prob >= 0.5).astype(int)

    logger.info(f"Predictions made for {len(y_pred_binary)} samples")
    logger.info(f"Predicted 0: {(y_pred_binary == 0).sum()}")
    logger.info(f"Predicted Non-0: {(y_pred_binary == 1).sum()}")

    # Compute accuracy per cardinality (0, 1, 2, 3)
    logger.info("\nComputing accuracy per cardinality...")
    accuracy_df = compute_accuracy_per_cardinality(
        y_original.values,
        y_pred_binary,
        y_prob,
    )

    # Compute binary grouped accuracy (0 vs Non-0)
    accuracy_binary_df = compute_accuracy_binary_grouped(
        y_original.values,
        y_pred_binary,
        y_prob,
    )

    # Print accuracy tables
    print_accuracy_table(accuracy_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("ACCURACY TABLE: 0 vs Non-0 (Binary Grouped)")
    logger.info("=" * 70)
    print("\n" + accuracy_binary_df.to_string(index=False))
    logger.info("\n" + "=" * 70)

    # Save results
    accuracy_df.to_csv(args.output_path, index=False)
    logger.info(f"\nAccuracy table (per cardinality) saved to: {args.output_path}")
    
    # Save binary grouped table
    binary_output_path = args.output_path.replace(".csv", "_binary.csv")
    accuracy_binary_df.to_csv(binary_output_path, index=False)
    logger.info(f"Accuracy table (0 vs Non-0) saved to: {binary_output_path}")

    # Summary
    total_duration = datetime.now() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {total_duration}")

    # Print binary grouped summary
    logger.info("Binary Grouped Results (0 vs Non-0):")
    for _, row in accuracy_binary_df.iterrows():
        logger.info(f"  {row['Cardinality']}: {row['Accuracy (%)']}% accuracy ({row['Sample Count']} samples)")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()

