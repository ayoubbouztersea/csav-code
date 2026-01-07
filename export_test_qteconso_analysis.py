"""
Script: Export Test Dataset & Calculate QteConso Percentage per Cardinality

This script:
1. Recreates the exact train/test split from train_predict_lightgbm.py
2. Exports the test dataset with original columns (including QteConso, Clot_1er_Pa)
3. Calculates the percentage of QteConso per cardinality (1, 2, 3+) using the formula:
   - False: CLOT_1ER_PASSAGE = FALSE OR (CLOT_1ER_PASSAGE = TRUE AND QTTE_CONSO > 0 AND PREDIAG_SANS_PIECE)
   - True: the inverse

Author: ML Engineering Team
Date: January 2026
"""

import os
import ast
import logging
from collections import Counter
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_labels(df, label_column='LIBEL_ARTICLE'):
    """Parse the label column which contains string representations of lists."""
    labels = df[label_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    labels = labels.apply(lambda x: x[:3] if len(x) > 3 else x)
    return labels.tolist()


def get_top_labels(labels_list, top_k=4001):
    """Compute the most frequent labels."""
    flat = [label for labels in labels_list for label in labels]
    counter = Counter(flat)
    top_labels = [label for label, _ in counter.most_common(top_k)]
    return set(top_labels)


def filter_labels_to_top(labels_list, top_labels, k=3):
    """Filter labels to the top set and pad/truncate to k items."""
    filtered = []
    kept_indices = []

    for idx, labels in enumerate(labels_list):
        kept = [l for l in labels if l in top_labels][:k]
        if len(kept) == 0:
            continue
        filtered.append(kept)
        kept_indices.append(idx)

    return filtered, kept_indices


def get_cardinality_bucket(n):
    """Get cardinality bucket (1, 2, 3+) based on label count."""
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def calculate_qteconso_condition(row):
    """
    Calculate the QteConso condition based on the formula:
    False: CLOT_1ER_PASSAGE = FALSE OR (CLOT_1ER_PASSAGE = TRUE AND QTTE_CONSO > 0 AND PREDIAG_SANS_PIECE)
    True: the inverse
    
    Returns True if the condition is True (inverse of False condition)
    """
    clot_1er_passage = row['Clot_1er_Pa'] == 1.0
    qtte_conso = row['QteConso']
    type_prediag = row['Type_Prediag']
    
    # Check if it's a "prediag sans pièce"
    prediag_sans_piece = type_prediag in ['Auto sans pièces', 'Manuel sans pièces']
    
    # False condition: CLOT_1ER_PASSAGE = FALSE OR (CLOT_1ER_PASSAGE = TRUE AND QTTE_CONSO > 0 AND PREDIAG_SANS_PIECE)
    false_condition = (not clot_1er_passage) or (clot_1er_passage and qtte_conso > 0 and prediag_sans_piece)
    
    # True is the inverse
    return not false_condition


def main():
    """Main execution function."""
    # Configuration - same as train_predict_lightgbm.py
    DATA_PATH = './data/df_ml2.csv'
    OUTPUT_DIR = './test_export'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TOP_K_LABELS = 4001
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    # Store original columns we need for analysis
    original_df = df[['Dossier', 'QteConso', 'Clot_1er_Pa', 'Type_Prediag', 'LIBEL_ARTICLE', 'LIBEL_ARTICLE_Length']].copy()
    
    # Parse labels
    labels_list = parse_labels(df)
    
    # Keep only the top K frequent labels
    top_labels = get_top_labels(labels_list, top_k=TOP_K_LABELS)
    filtered_labels, kept_indices = filter_labels_to_top(labels_list, top_labels, k=3)
    
    # Filter dataframe to kept rows
    df_filtered = df.iloc[kept_indices].reset_index(drop=True)
    original_filtered = original_df.iloc[kept_indices].reset_index(drop=True)
    logger.info(f"Data filtered to {df_filtered.shape[0]} rows after top-label selection")
    
    # Prepare features (same as train_predict_lightgbm.py)
    exclude_cols = [
        'Dossier', 'LIBEL_ARTICLE', 'LIBEL_ARTICLE_Length',
        'NB_INTERV', 'QteConso', 'Clot_1er_Pa'
    ]
    feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
    X = df_filtered[feature_cols].copy()
    
    # Handle categorical columns with label encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Downcast numerics
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce', downcast='float')
    X = X.fillna(-1)
    
    # Encode labels
    label_encoder = MultiLabelBinarizer(sparse_output=True)
    y_encoded = label_encoder.fit_transform(filtered_labels)
    
    # Create stratification key (same as train_predict_lightgbm.py)
    stratify_key = ['|'.join(sorted(labels)) for labels in filtered_labels]
    combo_counts = Counter(stratify_key)
    min_samples = max(2, int(1 / TEST_SIZE) + 1)
    stratify_key = [
        combo if combo_counts[combo] >= min_samples else '__RARE__'
        for combo in stratify_key
    ]
    
    # Split data with the SAME random state and parameters
    X_train, X_test, y_train, y_test, orig_train, orig_test, labels_train, labels_test = train_test_split(
        X, y_encoded, original_filtered, filtered_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_key
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Create test export dataframe with original columns
    test_export = orig_test.copy()
    test_export['True_Labels'] = [str(lst) for lst in labels_test]
    test_export['True_Labels_Count'] = [len(lst) for lst in labels_test]
    test_export['Cardinality_Bucket'] = test_export['True_Labels_Count'].apply(get_cardinality_bucket)
    
    # Calculate QteConso condition
    test_export['QteConso_Condition'] = test_export.apply(calculate_qteconso_condition, axis=1)
    
    # Export full test dataset
    test_export_path = os.path.join(OUTPUT_DIR, 'test_dataset_with_qteconso.csv')
    test_export.to_csv(test_export_path, index=False)
    logger.info(f"Test dataset exported to {test_export_path}")
    
    # Calculate percentage of QteConso condition per cardinality bucket
    logger.info("\n" + "="*80)
    logger.info("QteConso Condition Percentage per Cardinality")
    logger.info("="*80)
    logger.info("\nFormula:")
    logger.info("  False: CLOT_1ER_PASSAGE = FALSE OR (CLOT_1ER_PASSAGE = TRUE AND QTTE_CONSO > 0 AND PREDIAG_SANS_PIECE)")
    logger.info("  True: the inverse")
    logger.info("")
    
    results = []
    for bucket in ["1", "2", "3+"]:
        bucket_data = test_export[test_export['Cardinality_Bucket'] == bucket]
        total = len(bucket_data)
        true_count = bucket_data['QteConso_Condition'].sum()
        false_count = total - true_count
        
        true_pct = (true_count / total * 100) if total > 0 else 0
        false_pct = (false_count / total * 100) if total > 0 else 0
        
        results.append({
            'Cardinality': bucket,
            'Total_Samples': total,
            'True_Count': int(true_count),
            'False_Count': int(false_count),
            'True_Percentage': round(true_pct, 2),
            'False_Percentage': round(false_pct, 2)
        })
        
        logger.info(f"\nCardinality {bucket}:")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  True: {true_count} ({true_pct:.2f}%)")
        logger.info(f"  False: {false_count} ({false_pct:.2f}%)")
    
    # Add overall totals
    total_all = len(test_export)
    true_all = test_export['QteConso_Condition'].sum()
    false_all = total_all - true_all
    
    results.append({
        'Cardinality': 'ALL',
        'Total_Samples': total_all,
        'True_Count': int(true_all),
        'False_Count': int(false_all),
        'True_Percentage': round(true_all / total_all * 100, 2) if total_all > 0 else 0,
        'False_Percentage': round(false_all / total_all * 100, 2) if total_all > 0 else 0
    })
    
    logger.info(f"\nOverall:")
    logger.info(f"  Total samples: {total_all}")
    logger.info(f"  True: {true_all} ({true_all/total_all*100:.2f}%)")
    logger.info(f"  False: {false_all} ({false_all/total_all*100:.2f}%)")
    
    # Export results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, 'qteconso_percentage_by_cardinality.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults exported to {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Export and analysis completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

