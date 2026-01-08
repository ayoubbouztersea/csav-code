#!/usr/bin/env python3
"""
Evaluation and Inference for Transformer Piece Predictor.

Features:
- Comprehensive metrics: Precision@K, Recall@K, F1, Exact Match Ratio
- Threshold optimization on validation set
- predict() function that returns predicted piece names

Usage:
    python evaluate.py --data-dir data --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
) -> nn.Module:
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
    if "metrics" in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions for all samples in dataloader.
    
    Returns:
        probs: Predicted probabilities (n_samples, n_pieces)
        targets: Ground truth labels (n_samples, n_pieces)
    """
    model.eval()
    
    all_probs = []
    all_targets = []
    
    for features, targets in tqdm(dataloader, desc="Predicting"):
        features = features.to(device)
        logits = model(features)
        probs = torch.sigmoid(logits)
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_targets)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        probs: Predicted probabilities (n_samples, n_pieces)
        targets: Ground truth labels (n_samples, n_pieces)
        threshold: Probability threshold for positive prediction
        
    Returns:
        Dictionary with all metrics
    """
    preds = (probs > threshold).astype(np.float32)
    
    # Micro-averaged metrics (treating all predictions equally)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    # Per-sample metrics
    n_samples = len(targets)
    
    sample_precisions = []
    sample_recalls = []
    sample_f1s = []
    
    for i in range(n_samples):
        pred_i = preds[i]
        target_i = targets[i]
        
        tp_i = (pred_i * target_i).sum()
        fp_i = (pred_i * (1 - target_i)).sum()
        fn_i = ((1 - pred_i) * target_i).sum()
        
        prec_i = tp_i / (tp_i + fp_i + 1e-8)
        rec_i = tp_i / (tp_i + fn_i + 1e-8)
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i + 1e-8)
        
        sample_precisions.append(prec_i)
        sample_recalls.append(rec_i)
        sample_f1s.append(f1_i)
    
    macro_precision = np.mean(sample_precisions)
    macro_recall = np.mean(sample_recalls)
    macro_f1 = np.mean(sample_f1s)
    
    # Exact match ratio
    exact_match = (preds == targets).all(axis=1).mean()
    
    # Subset accuracy (at least one correct prediction)
    has_correct = ((preds * targets).sum(axis=1) > 0).mean()
    
    # Average number of predictions per sample
    avg_preds = preds.sum(axis=1).mean()
    avg_targets = targets.sum(axis=1).mean()
    
    return {
        "threshold": threshold,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "exact_match": float(exact_match),
        "has_correct": float(has_correct),
        "avg_predictions": float(avg_preds),
        "avg_targets": float(avg_targets),
    }


def compute_precision_at_k(
    probs: np.ndarray,
    targets: np.ndarray,
    k: int = 3,
) -> dict:
    """
    Compute Precision@K and Recall@K metrics.
    
    For each sample, take top-k predictions and compute precision/recall.
    """
    n_samples = len(probs)
    
    precisions = []
    recalls = []
    
    for i in range(n_samples):
        # Get top-k predictions
        top_k_indices = np.argsort(probs[i])[-k:]
        preds_at_k = np.zeros_like(probs[i])
        preds_at_k[top_k_indices] = 1.0
        
        target_i = targets[i]
        
        # True positives in top-k
        tp = (preds_at_k * target_i).sum()
        
        # Precision@K = TP / K
        precision = tp / k
        
        # Recall@K = TP / number of actual positives
        n_positives = target_i.sum()
        recall = tp / n_positives if n_positives > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
    }


def optimize_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> tuple[float, dict]:
    """
    Find the optimal threshold that maximizes F1 score.
    
    Args:
        probs: Predicted probabilities
        targets: Ground truth labels
        thresholds: List of thresholds to try
        
    Returns:
        best_threshold: Optimal threshold
        best_metrics: Metrics at optimal threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05).tolist()
    
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = None
    
    logger.info("Optimizing threshold...")
    for thresh in thresholds:
        metrics = compute_metrics(probs, targets, threshold=thresh)
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = thresh
            best_metrics = metrics
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return best_threshold, best_metrics


class PiecePredictor:
    """
    High-level class for piece prediction.
    
    Usage:
        predictor = PiecePredictor.load("checkpoints/best_model.pt", "data")
        pieces = predictor.predict(features)
    """
    
    def __init__(
        self,
        model: nn.Module,
        idx_to_piece: dict,
        feature_vocabs: dict,
        feature_names: list[str],
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model
        self.idx_to_piece = idx_to_piece
        self.feature_vocabs = feature_vocabs
        self.feature_names = feature_names
        self.device = device
        self.threshold = threshold
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        data_dir: str,
        threshold: float = 0.5,
    ) -> "PiecePredictor":
        """Load predictor from checkpoint and data directory."""
        device = get_device()
        data = load_preprocessed_data(data_dir)
        
        model = load_model(checkpoint_path, data["config"], device)
        
        return cls(
            model=model,
            idx_to_piece=data["idx_to_piece"],
            feature_vocabs=data["feature_vocabs"],
            feature_names=data["feature_names"],
            device=device,
            threshold=threshold,
        )
    
    def encode_features(self, features_dict: dict) -> torch.Tensor:
        """
        Encode a dictionary of feature values to tensor.
        
        Args:
            features_dict: Dict mapping feature name to value
            
        Returns:
            Tensor of shape (1, n_features)
        """
        encoded = []
        for feat_name in self.feature_names:
            vocab = self.feature_vocabs[feat_name]
            value = features_dict.get(feat_name, "__MISSING__")
            idx = vocab.get(str(value), vocab.get("__UNK__", 1))
            encoded.append(idx)
        
        return torch.tensor([encoded], dtype=torch.long)
    
    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor | dict,
        top_k: Optional[int] = None,
    ) -> list[list[str]]:
        """
        Predict piece names for given features.
        
        Args:
            features: Either a tensor of encoded features or a dict of raw values
            top_k: If set, return top-k predictions regardless of threshold
            
        Returns:
            List of lists containing predicted piece names
        """
        self.model.eval()
        
        # Handle dict input
        if isinstance(features, dict):
            features = self.encode_features(features)
        
        features = features.to(self.device)
        
        logits = self.model(features)
        probs = torch.sigmoid(logits)
        
        results = []
        for i in range(probs.size(0)):
            if top_k is not None:
                # Get top-k predictions
                _, indices = probs[i].topk(min(top_k, len(probs[i])))
                piece_indices = indices.cpu().tolist()
            else:
                # Threshold-based predictions
                piece_indices = (probs[i] > self.threshold).nonzero(as_tuple=True)[0]
                piece_indices = piece_indices.cpu().tolist()
            
            # Convert indices to piece names
            pieces = [self.idx_to_piece[idx] for idx in piece_indices if idx in self.idx_to_piece]
            results.append(pieces)
        
        return results
    
    def predict_with_scores(
        self,
        features: torch.Tensor | dict,
        top_k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """
        Predict pieces with confidence scores.
        
        Returns:
            List of lists of (piece_name, probability) tuples
        """
        self.model.eval()
        
        if isinstance(features, dict):
            features = self.encode_features(features)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.sigmoid(logits)
        
        results = []
        for i in range(probs.size(0)):
            scores, indices = probs[i].topk(min(top_k, len(probs[i])))
            pieces_with_scores = [
                (self.idx_to_piece[idx.item()], score.item())
                for idx, score in zip(indices, scores)
                if idx.item() in self.idx_to_piece
            ]
            results.append(pieces_with_scores)
        
        return results


def evaluate(
    data_dir: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 128,
) -> dict:
    """
    Full evaluation on test set.
    
    Args:
        data_dir: Directory with preprocessed data
        checkpoint_path: Path to model checkpoint
        output_path: Optional path to save results
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with all evaluation metrics
    """
    device = get_device()
    
    # Load data
    logger.info("Loading data...")
    data = load_preprocessed_data(data_dir)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(checkpoint_path, data["config"], device)
    
    # Create test dataloader
    test_dataset = PieceDataset(data["X_test"], data["y_test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Get predictions
    logger.info("Getting predictions...")
    probs, targets = get_predictions(model, test_loader, device)
    
    # Optimize threshold on validation set
    val_dataset = PieceDataset(data["X_val"], data["y_val"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_probs, val_targets = get_predictions(model, val_loader, device)
    
    best_threshold, _ = optimize_threshold(val_probs, val_targets)
    
    # Compute metrics on test set
    logger.info("Computing metrics...")
    metrics = compute_metrics(probs, targets, threshold=best_threshold)
    
    # Add Precision@K metrics
    for k in [1, 3, 5]:
        pk_metrics = compute_precision_at_k(probs, targets, k=k)
        metrics.update(pk_metrics)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Optimal Threshold: {best_threshold:.2f}")
    logger.info("-" * 60)
    logger.info(f"Micro Precision:  {metrics['micro_precision']:.4f}")
    logger.info(f"Micro Recall:     {metrics['micro_recall']:.4f}")
    logger.info(f"Micro F1:         {metrics['micro_f1']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Macro Precision:  {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall:     {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1:         {metrics['macro_f1']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Exact Match:      {metrics['exact_match']:.4f}")
    logger.info(f"Has Correct:      {metrics['has_correct']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Precision@1:      {metrics['precision@1']:.4f}")
    logger.info(f"Precision@3:      {metrics['precision@3']:.4f}")
    logger.info(f"Precision@5:      {metrics['precision@5']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Recall@1:         {metrics['recall@1']:.4f}")
    logger.info(f"Recall@3:         {metrics['recall@3']:.4f}")
    logger.info(f"Recall@5:         {metrics['recall@5']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Avg Predictions:  {metrics['avg_predictions']:.2f}")
    logger.info(f"Avg Targets:      {metrics['avg_targets']:.2f}")
    logger.info("=" * 60)
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer Piece Predictor"
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
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
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
    
    output_path = args.output
    if output_path and not Path(output_path).is_absolute():
        output_path = str(script_dir / output_path)
    
    evaluate(
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint_path),
        output_path=output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

