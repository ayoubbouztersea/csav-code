#!/usr/bin/env python3
"""
Evaluation by Cardinality for Transformer Piece Predictor.

This script evaluates model performance broken down by the number of pieces
per sample (cardinality 1, 2, 3, etc.).

Usage:
    python evaluate_by_cardinality.py --data-dir data --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import create_model
from preprocess import PieceDataset, load_preprocessed_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load model from checkpoint."""
    model = create_model(
        feature_vocab_sizes=config["feature_vocab_sizes"],
        n_pieces=config["n_pieces"],
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def compute_metrics_for_subset(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute evaluation metrics for a subset of data."""
    if len(probs) == 0:
        return {
            "count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "has_correct": 0.0,
        }
    
    preds = (probs > threshold).astype(np.float32)
    
    # Micro-averaged metrics
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Exact match ratio
    exact_match = (preds == targets).all(axis=1).mean()
    
    # At least one correct prediction
    has_correct = ((preds * targets).sum(axis=1) > 0).mean()
    
    # Per-sample F1 (macro)
    sample_f1s = []
    for i in range(len(targets)):
        tp_i = (preds[i] * targets[i]).sum()
        fp_i = (preds[i] * (1 - targets[i])).sum()
        fn_i = ((1 - preds[i]) * targets[i]).sum()
        
        prec_i = tp_i / (tp_i + fp_i + 1e-8)
        rec_i = tp_i / (tp_i + fn_i + 1e-8)
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i + 1e-8)
        sample_f1s.append(f1_i)
    
    macro_f1 = np.mean(sample_f1s)
    
    # Average predictions vs actual
    avg_preds = preds.sum(axis=1).mean()
    avg_targets = targets.sum(axis=1).mean()
    
    return {
        "count": len(probs),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "macro_f1": float(macro_f1),
        "exact_match": float(exact_match),
        "has_correct": float(has_correct),
        "avg_predictions": float(avg_preds),
        "avg_targets": float(avg_targets),
    }


def compute_precision_recall_at_k(
    probs: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> dict:
    """Compute Precision@K and Recall@K."""
    if len(probs) == 0:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0}
    
    precisions = []
    recalls = []
    
    for i in range(len(probs)):
        top_k_indices = np.argsort(probs[i])[-k:]
        preds_at_k = np.zeros_like(probs[i])
        preds_at_k[top_k_indices] = 1.0
        
        tp = (preds_at_k * targets[i]).sum()
        n_positives = targets[i].sum()
        
        precisions.append(tp / k)
        recalls.append(tp / n_positives if n_positives > 0 else 0.0)
    
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
    }


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Get model predictions."""
    model.eval()
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features).long()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    for (batch_features,) in dataloader:
        batch_features = batch_features.to(device)
        logits = model(batch_features)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs)


def optimize_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Find optimal threshold on validation set."""
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs > thresh).astype(np.float32)
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold


def evaluate_by_cardinality(
    data_dir: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 128,
    max_cardinality: int = 5,
) -> pd.DataFrame:
    """
    Evaluate model performance by cardinality (number of pieces per sample).
    
    Args:
        data_dir: Directory with preprocessed data
        checkpoint_path: Path to model checkpoint
        output_path: Optional path to save results CSV
        batch_size: Batch size for inference
        max_cardinality: Maximum cardinality to evaluate separately
        
    Returns:
        DataFrame with metrics per cardinality
    """
    device = get_device()
    
    # Load data
    logger.info("Loading data...")
    data = load_preprocessed_data(data_dir)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(checkpoint_path, data["config"], device)
    
    # Get cardinality for each sample (number of pieces)
    y_test = data["y_test"]
    cardinalities = y_test.sum(axis=1).astype(int)
    
    logger.info(f"Cardinality distribution in test set:")
    unique, counts = np.unique(cardinalities, return_counts=True)
    for c, cnt in zip(unique, counts):
        logger.info(f"  Cardinality {c}: {cnt} samples ({cnt/len(cardinalities)*100:.1f}%)")
    
    # Optimize threshold on validation set
    logger.info("Optimizing threshold on validation set...")
    val_probs = get_predictions(model, data["X_val"], device, batch_size)
    best_threshold = optimize_threshold(val_probs, data["y_val"])
    logger.info(f"Optimal threshold: {best_threshold:.2f}")
    
    # Get test predictions
    logger.info("Getting test predictions...")
    test_probs = get_predictions(model, data["X_test"], device, batch_size)
    
    # Evaluate by cardinality
    results = []
    
    # Cardinalities 1, 2, 3 (and optionally more)
    for card in range(1, max_cardinality + 1):
        mask = cardinalities == card
        n_samples = mask.sum()
        
        if n_samples == 0:
            logger.info(f"Cardinality {card}: No samples, skipping...")
            continue
        
        logger.info(f"Evaluating cardinality {card} ({n_samples} samples)...")
        
        card_probs = test_probs[mask]
        card_targets = y_test[mask]
        
        metrics = compute_metrics_for_subset(card_probs, card_targets, best_threshold)
        
        # Add Precision@K metrics
        for k in [1, 3, 5]:
            if k <= card + 2:  # Only compute relevant K values
                pk_metrics = compute_precision_recall_at_k(card_probs, card_targets, k)
                metrics.update(pk_metrics)
        
        metrics["cardinality"] = card
        results.append(metrics)
    
    # Add "4+" category
    mask_4plus = cardinalities >= 4
    n_4plus = mask_4plus.sum()
    if n_4plus > 0:
        logger.info(f"Evaluating cardinality 4+ ({n_4plus} samples)...")
        
        card_probs = test_probs[mask_4plus]
        card_targets = y_test[mask_4plus]
        
        metrics = compute_metrics_for_subset(card_probs, card_targets, best_threshold)
        for k in [1, 3, 5]:
            pk_metrics = compute_precision_recall_at_k(card_probs, card_targets, k)
            metrics.update(pk_metrics)
        
        metrics["cardinality"] = "4+"
        results.append(metrics)
    
    # Add overall metrics
    logger.info("Computing overall metrics...")
    overall_metrics = compute_metrics_for_subset(test_probs, y_test, best_threshold)
    for k in [1, 3, 5]:
        pk_metrics = compute_precision_recall_at_k(test_probs, y_test, k)
        overall_metrics.update(pk_metrics)
    overall_metrics["cardinality"] = "ALL"
    results.append(overall_metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols_order = [
        "cardinality", "count",
        "micro_precision", "micro_recall", "micro_f1", "macro_f1",
        "exact_match", "has_correct",
        "precision@1", "recall@1",
        "precision@3", "recall@3",
        "avg_predictions", "avg_targets",
    ]
    cols_present = [c for c in cols_order if c in df.columns]
    df = df[cols_present]
    
    # Print results
    logger.info("\n" + "=" * 100)
    logger.info("ACCURACY BY CARDINALITY")
    logger.info("=" * 100)
    logger.info(f"Threshold: {best_threshold:.2f}")
    logger.info("-" * 100)
    
    # Format for display
    display_df = df.copy()
    for col in display_df.columns:
        if col not in ["cardinality", "count"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    
    print("\n" + display_df.to_string(index=False))
    
    logger.info("\n" + "=" * 100)
    
    # Detailed breakdown
    logger.info("\nDETAILED BREAKDOWN:")
    logger.info("-" * 100)
    
    for _, row in df.iterrows():
        card = row["cardinality"]
        logger.info(f"\nCardinality: {card} ({int(row['count'])} samples)")
        logger.info(f"  Micro F1:      {row['micro_f1']:.4f}")
        logger.info(f"  Macro F1:      {row['macro_f1']:.4f}")
        logger.info(f"  Exact Match:   {row['exact_match']:.4f}")
        logger.info(f"  Has Correct:   {row['has_correct']:.4f}")
        if "precision@1" in row:
            logger.info(f"  Precision@1:   {row['precision@1']:.4f}")
        if "precision@3" in row:
            logger.info(f"  Precision@3:   {row['precision@3']:.4f}")
    
    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")
        
        # Also save as JSON
        json_path = output_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "threshold": best_threshold,
                "results": results,
            }, f, indent=2)
        logger.info(f"JSON saved to: {json_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer Piece Predictor by Cardinality"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="accuracy_by_cardinality.csv",
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-cardinality",
        type=int,
        default=3,
        help="Maximum cardinality to evaluate separately (higher grouped as N+)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / checkpoint_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    evaluate_by_cardinality(
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        batch_size=args.batch_size,
        max_cardinality=args.max_cardinality,
    )


if __name__ == "__main__":
    main()

