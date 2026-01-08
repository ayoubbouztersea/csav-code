#!/usr/bin/env python3
"""
Data Preprocessing for Transformer Piece Predictor.

This script:
- Loads and parses LIBEL_ARTICLE lists from df_ml2.csv
- Builds piece vocabulary with frequency filtering
- Encodes categorical features to indices
- Creates train/val/test splits (80/10/10)
- Saves preprocessed data and vocabulary mappings

Usage:
    python preprocess.py --data-path ../data/df_ml2.csv --min-freq 5
"""

import argparse
import ast
import json
import logging
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Categorical features to use as input
CATEGORICAL_FEATURES = [
    "Symptome",
    "Marque",
    "Fam1",
    "Fam2",
    "Fam3",
    "Fam4",
    "Version",
    "Type_Prediag",
    "CodeError",
]

# Target column containing list of pieces
TARGET_COLUMN = "LIBEL_ARTICLE"

# Random seed for reproducibility
RANDOM_STATE = 42


def parse_piece_list(value: str) -> list[str]:
    """Parse a string representation of a list into actual list of pieces."""
    if pd.isna(value):
        return []
    try:
        pieces = ast.literal_eval(value)
        if isinstance(pieces, list):
            return [str(p).strip() for p in pieces if p]
        return []
    except (ValueError, SyntaxError):
        return []


def build_vocabularies(
    df: pd.DataFrame,
    categorical_features: list[str],
    min_piece_freq: int = 5,
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[int, str]]:
    """
    Build vocabularies for categorical features and pieces.
    
    Args:
        df: DataFrame with the data
        categorical_features: List of categorical feature column names
        min_piece_freq: Minimum frequency for a piece to be included
        
    Returns:
        feature_vocabs: Dict mapping feature name to {value: index}
        piece_to_idx: Dict mapping piece name to index
        idx_to_piece: Dict mapping index to piece name
    """
    logger.info("Building vocabularies...")
    
    # Build vocabulary for each categorical feature
    feature_vocabs = {}
    for feat in categorical_features:
        if feat not in df.columns:
            logger.warning(f"Feature '{feat}' not found in DataFrame, skipping...")
            continue
        
        # Get unique values, handle NaN
        unique_vals = df[feat].fillna("__MISSING__").unique()
        # Reserve index 0 for unknown/padding
        vocab = {"__PAD__": 0, "__UNK__": 1}
        for i, val in enumerate(sorted(unique_vals), start=2):
            vocab[str(val)] = i
        feature_vocabs[feat] = vocab
        logger.info(f"  {feat}: {len(vocab)} unique values")
    
    # Build piece vocabulary
    logger.info("Building piece vocabulary...")
    all_pieces = []
    for val in df[TARGET_COLUMN]:
        pieces = parse_piece_list(val)
        all_pieces.extend(pieces)
    
    piece_counts = Counter(all_pieces)
    logger.info(f"  Total unique pieces before filtering: {len(piece_counts)}")
    
    # Filter by minimum frequency
    filtered_pieces = [p for p, c in piece_counts.items() if c >= min_piece_freq]
    filtered_pieces = sorted(filtered_pieces)  # Sort for reproducibility
    
    piece_to_idx = {piece: idx for idx, piece in enumerate(filtered_pieces)}
    idx_to_piece = {idx: piece for piece, idx in piece_to_idx.items()}
    
    logger.info(f"  Pieces after filtering (min_freq={min_piece_freq}): {len(piece_to_idx)}")
    
    return feature_vocabs, piece_to_idx, idx_to_piece


def encode_features(
    df: pd.DataFrame,
    feature_vocabs: dict[str, dict[str, int]],
) -> np.ndarray:
    """
    Encode categorical features to indices.
    
    Args:
        df: DataFrame with the data
        feature_vocabs: Dict mapping feature name to {value: index}
        
    Returns:
        Numpy array of shape (n_samples, n_features) with encoded indices
    """
    logger.info("Encoding categorical features...")
    
    n_samples = len(df)
    n_features = len(feature_vocabs)
    encoded = np.zeros((n_samples, n_features), dtype=np.int64)
    
    for feat_idx, (feat_name, vocab) in enumerate(feature_vocabs.items()):
        unk_idx = vocab.get("__UNK__", 1)
        for row_idx, val in enumerate(df[feat_name].fillna("__MISSING__")):
            encoded[row_idx, feat_idx] = vocab.get(str(val), unk_idx)
    
    return encoded


def encode_targets(
    df: pd.DataFrame,
    piece_to_idx: dict[str, int],
) -> np.ndarray:
    """
    Encode target pieces as multi-hot vectors.
    
    Args:
        df: DataFrame with the data
        piece_to_idx: Dict mapping piece name to index
        
    Returns:
        Numpy array of shape (n_samples, n_pieces) with multi-hot encoding
    """
    logger.info("Encoding target pieces as multi-hot vectors...")
    
    n_samples = len(df)
    n_pieces = len(piece_to_idx)
    targets = np.zeros((n_samples, n_pieces), dtype=np.float32)
    
    skipped_pieces = 0
    for row_idx, val in enumerate(df[TARGET_COLUMN]):
        pieces = parse_piece_list(val)
        for piece in pieces:
            if piece in piece_to_idx:
                targets[row_idx, piece_to_idx[piece]] = 1.0
            else:
                skipped_pieces += 1
    
    logger.info(f"  Skipped {skipped_pieces} piece occurrences (not in vocabulary)")
    
    return targets


class PieceDataset(Dataset):
    """PyTorch Dataset for piece prediction."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Args:
            features: Encoded categorical features (n_samples, n_features)
            targets: Multi-hot encoded targets (n_samples, n_pieces)
        """
        self.features = torch.from_numpy(features).long()
        self.targets = torch.from_numpy(targets).float()
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def preprocess_and_save(
    data_path: str,
    output_dir: str,
    min_piece_freq: int = 5,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> dict[str, Any]:
    """
    Main preprocessing pipeline.
    
    Args:
        data_path: Path to the input CSV file
        output_dir: Directory to save preprocessed data
        min_piece_freq: Minimum frequency for a piece to be included
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        
    Returns:
        Dictionary with preprocessing statistics
    """
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"  Shape: {df.shape}")
    
    # Filter to only rows with valid LIBEL_ARTICLE
    valid_mask = df[TARGET_COLUMN].apply(lambda x: len(parse_piece_list(x)) > 0)
    df = df[valid_mask].reset_index(drop=True)
    logger.info(f"  Rows with valid pieces: {len(df)}")
    
    # Filter categorical features to only those present in the data
    available_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    logger.info(f"Using features: {available_features}")
    
    # Build vocabularies
    feature_vocabs, piece_to_idx, idx_to_piece = build_vocabularies(
        df, available_features, min_piece_freq
    )
    
    # Encode features and targets
    X = encode_features(df, feature_vocabs)
    y = encode_targets(df, piece_to_idx)
    
    # Split data: train/val/test
    logger.info("Splitting data into train/val/test...")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=RANDOM_STATE
    )
    
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Compute class weights for imbalanced pieces
    logger.info("Computing class weights...")
    piece_frequencies = y_train.sum(axis=0)
    # Avoid division by zero
    piece_frequencies = np.maximum(piece_frequencies, 1)
    # Inverse frequency weighting, capped
    pos_weights = len(y_train) / (piece_frequencies * len(piece_to_idx))
    pos_weights = np.clip(pos_weights, 0.1, 10.0)  # Cap weights
    
    # Save everything
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "pos_weights.npy"), pos_weights)
    
    # Save vocabularies
    vocab_data = {
        "feature_vocabs": feature_vocabs,
        "piece_to_idx": piece_to_idx,
        "idx_to_piece": idx_to_piece,
        "feature_names": available_features,
    }
    with open(os.path.join(output_dir, "vocabularies.pkl"), "wb") as f:
        pickle.dump(vocab_data, f)
    
    # Save config/stats as JSON
    config = {
        "n_samples_train": len(X_train),
        "n_samples_val": len(X_val),
        "n_samples_test": len(X_test),
        "n_features": len(available_features),
        "n_pieces": len(piece_to_idx),
        "feature_names": available_features,
        "feature_vocab_sizes": {k: len(v) for k, v in feature_vocabs.items()},
        "min_piece_freq": min_piece_freq,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved preprocessed data to: {output_dir}")
    logger.info(f"  Features: {len(available_features)}")
    logger.info(f"  Pieces vocabulary: {len(piece_to_idx)}")
    
    return config


def load_preprocessed_data(data_dir: str) -> dict[str, Any]:
    """
    Load preprocessed data from directory.
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        Dictionary with all loaded data
    """
    logger.info(f"Loading preprocessed data from: {data_dir}")
    
    data = {
        "X_train": np.load(os.path.join(data_dir, "X_train.npy")),
        "X_val": np.load(os.path.join(data_dir, "X_val.npy")),
        "X_test": np.load(os.path.join(data_dir, "X_test.npy")),
        "y_train": np.load(os.path.join(data_dir, "y_train.npy")),
        "y_val": np.load(os.path.join(data_dir, "y_val.npy")),
        "y_test": np.load(os.path.join(data_dir, "y_test.npy")),
        "pos_weights": np.load(os.path.join(data_dir, "pos_weights.npy")),
    }
    
    with open(os.path.join(data_dir, "vocabularies.pkl"), "rb") as f:
        vocab_data = pickle.load(f)
    data.update(vocab_data)
    
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = json.load(f)
    data["config"] = config
    
    logger.info(f"  Train: {len(data['X_train'])}")
    logger.info(f"  Val: {len(data['X_val'])}")
    logger.info(f"  Test: {len(data['X_test'])}")
    logger.info(f"  Pieces: {len(data['piece_to_idx'])}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for Transformer Piece Predictor"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/df_ml2.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save preprocessed data",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum frequency for piece to be included in vocabulary",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction for test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction for validation set",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    config = preprocess_and_save(
        data_path=str(data_path),
        output_dir=str(output_dir),
        min_piece_freq=args.min_freq,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    
    logger.info("Preprocessing complete!")
    logger.info(f"Config: {config}")


if __name__ == "__main__":
    main()

