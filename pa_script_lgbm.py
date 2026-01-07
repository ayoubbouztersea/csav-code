"""
Evaluation script for LightGBM model V2_K2 on test_df_prediag_auto.csv
Calculates containment accuracy per cardinality (1, 2, 3+)

Inputs:
- ./lightgbmV2/lightgbm_multilabel_model_V2_K2.joblib
- ./lightgbmV2/label_encoder.joblib
- ./lightgbmV2/test_df_prediag_auto.csv

Outputs:
- ./lightgbmV2/pa_cardinality_accuracy.csv
- Console logs with detailed metrics
"""

import os
import ast
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_labels(df, label_column='LIBEL_ARTICLE'):
    """
    Parse the label column which contains string representations of lists.
    
    Args:
        df: Input dataframe
        label_column: Name of the column containing labels
        
    Returns:
        list: List of label lists (multi-label format)
    """
    logger.info(f"Parsing labels from column: {label_column}")
    
    # Parse string representation of lists
    labels = df[label_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Keep at most 3 labels per row
    labels = labels.apply(lambda x: x[:3] if len(x) > 3 else x)
    
    logger.info(f"Labels parsed. Sample: {labels.iloc[0]}")
    return labels.tolist()


def prepare_features(df):
    """
    Prepare feature columns for prediction.
    Must match the feature preparation from training.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Feature matrix ready for prediction
    """
    logger.info("Preparing features...")
    
    # Exclude non-feature columns
    exclude_cols = ['Dossier', 'LIBEL_ARTICLE', 'LIBEL_ARTICLE_Length']
    
    # Get all columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df_features = df[feature_cols].copy()
    
    # Handle categorical columns with label encoding
    categorical_cols = df_features.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
    
    # Downcast numerics to save memory
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].apply(
        pd.to_numeric, errors='coerce', downcast='float'
    )

    # Fill any missing values after conversions
    df_features = df_features.fillna(-1)
    
    logger.info(f"Features prepared. Number of features: {len(df_features.columns)}")
    
    return df_features


def predict_top_k(model, label_encoder, X, k=3):
    """
    Predict top k labels with their probabilities for each sample.
    
    Args:
        model: Trained LightGBM MultiOutputClassifier
        label_encoder: Fitted MultiLabelBinarizer
        X: Feature matrix
        k: Number of top labels to predict
        
    Returns:
        tuple: (predicted_labels, predicted_probabilities)
    """
    logger.info(f"Predicting top {k} labels with probabilities...")
    
    # Get probability predictions for all labels
    probabilities = []
    for estimator in model.estimators_:
        proba = estimator.predict_proba(X)
        # Get probability for class 1 (positive class)
        probabilities.append(proba[:, 1])
    
    # Stack probabilities (samples x labels)
    all_probabilities = np.column_stack(probabilities).astype(np.float32)
    
    # Get top k indices for each sample
    top_k_indices = np.argsort(all_probabilities, axis=1)[:, -k:][:, ::-1]
    
    # Get corresponding probabilities
    top_k_probas = np.take_along_axis(all_probabilities, top_k_indices, axis=1)
    
    # Convert indices to label names
    predicted_labels = []
    predicted_probas = []
    
    for i in range(len(X)):
        # Get label names for top k predictions
        labels = [label_encoder.classes_[idx] for idx in top_k_indices[i]]
        probas = top_k_probas[i].tolist()
        
        predicted_labels.append(labels)
        predicted_probas.append(probas)
    
    logger.info(f"Prediction completed for {len(X)} samples")
    
    return predicted_labels, predicted_probas


def bucket_key(n):
    """Categorize cardinality into buckets: 1, 2, or 3+."""
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def evaluate_by_cardinality(true_labels, pred_labels):
    """
    Evaluate containment accuracy per true-label cardinality bucket (1, 2, 3+).
    Containment: all true labels are present in the predicted labels (order-agnostic).
    
    Args:
        true_labels: List of true label lists
        pred_labels: List of predicted label lists
        
    Returns:
        pd.DataFrame: Metrics per cardinality bucket
    """
    buckets = ["1", "2", "3+"]
    totals = {b: 0 for b in buckets}
    contain_hits = {b: 0 for b in buckets}
    exact_hits = {b: 0 for b in buckets}

    for t_labels, p_labels in zip(true_labels, pred_labels):
        true_set = set(t_labels)
        pred_set = set(p_labels)
        key = bucket_key(len(true_set))
        totals[key] += 1
        
        # Containment: true_set is a subset of pred_set
        if true_set.issubset(pred_set):
            contain_hits[key] += 1
        
        # Exact match: true_set equals pred_set
        if true_set == pred_set:
            exact_hits[key] += 1

    rows = []
    for b in buckets:
        total = totals[b]
        contain_acc = contain_hits[b] / total if total else 0.0
        exact_acc = exact_hits[b] / total if total else 0.0
        rows.append({
            "cardinality": b,
            "total_samples": total,
            "containment_accuracy": round(contain_acc, 6),
            "exact_match_accuracy": round(exact_acc, 6),
            "wrong_rate": round(1 - contain_acc, 6),
        })

    return pd.DataFrame(rows)


def main():
    """
    Main evaluation function.
    Loads model and test data, makes predictions, and exports accuracy by cardinality.
    """
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    MODEL_PATH = os.path.join(script_dir, './lightgbmV2/lightgbm_multilabel_model_V2_K2.joblib')
    ENCODER_PATH = os.path.join(script_dir, './lightgbmV2/label_encoder.joblib')
    TEST_DATA_PATH = os.path.join(script_dir, './data/test_df_prediag_auto.csv')
    OUTPUT_PATH = os.path.join(script_dir, './lightgbmV2/pa_cardinality_accuracy.csv')
    
    # Check required files exist
    for path, name in [
        (MODEL_PATH, "Model"),
        (ENCODER_PATH, "Label encoder"),
        (TEST_DATA_PATH, "Test data")
    ]:
        if not os.path.exists(path):
            logger.error(f"{name} not found at: {path}")
            raise FileNotFoundError(f"{name} not found at: {path}")
    
    # Load model and encoder
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully. Number of estimators: {len(model.estimators_)}")
    
    logger.info(f"Loading label encoder from {ENCODER_PATH}")
    label_encoder = joblib.load(ENCODER_PATH)
    logger.info(f"Label encoder loaded. Number of classes: {len(label_encoder.classes_)}")
    
    # Load test data
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    df_test = pd.read_csv(TEST_DATA_PATH)
    logger.info(f"Test data loaded. Shape: {df_test.shape}")
    
    # Parse true labels
    true_labels = parse_labels(df_test)
    logger.info(f"Parsed {len(true_labels)} true label sets")
    
    # Prepare features
    X_test = prepare_features(df_test)
    logger.info(f"Features prepared. Shape: {X_test.shape}")
    
    # Make predictions
    pred_labels, pred_probas = predict_top_k(model, label_encoder, X_test, k=3)
    
    # Display sample predictions
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS (First 5 samples)")
    logger.info("="*80)
    for i in range(min(5, len(pred_labels))):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  True Labels: {true_labels[i]}")
        logger.info(f"  Predicted Labels: {pred_labels[i]}")
        logger.info(f"  Probabilities: {[f'{p:.4f}' for p in pred_probas[i]]}")
        # Check if correct
        true_set = set(true_labels[i])
        pred_set = set(pred_labels[i])
        is_correct = true_set.issubset(pred_set)
        logger.info(f"  Containment Match: {is_correct}")
    
    # Evaluate by cardinality
    logger.info("\n" + "="*80)
    logger.info("EVALUATING BY CARDINALITY")
    logger.info("="*80)
    
    df_metrics = evaluate_by_cardinality(true_labels, pred_labels)
    
    # Save results
    df_metrics.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Cardinality accuracy saved to: {OUTPUT_PATH}")
    
    # Print results table
    logger.info("\n" + "="*80)
    logger.info("ACCURACY BY CARDINALITY")
    logger.info("="*80)
    print("\n")
    print(df_metrics.to_string(index=False))
    print("\n")
    
    # Calculate overall metrics
    total_correct = sum(1 for t, p in zip(true_labels, pred_labels) if set(t).issubset(set(p)))
    total_exact = sum(1 for t, p in zip(true_labels, pred_labels) if set(t) == set(p))
    overall_containment = total_correct / len(true_labels)
    overall_exact = total_exact / len(true_labels)
    
    logger.info("="*80)
    logger.info("OVERALL METRICS")
    logger.info("="*80)
    logger.info(f"Total samples:            {len(true_labels)}")
    logger.info(f"Containment accuracy:     {overall_containment:.4f} ({overall_containment*100:.2f}%)")
    logger.info(f"Exact match accuracy:     {overall_exact:.4f} ({overall_exact*100:.2f}%)")
    logger.info(f"Correct predictions:      {total_correct}")
    logger.info(f"Wrong predictions:        {len(true_labels) - total_correct}")
    logger.info("="*80)
    
    logger.info("\nEVALUATION COMPLETED SUCCESSFULLY!")
    
    return df_metrics


if __name__ == "__main__":
    main()

