#!/usr/bin/env python3
"""
Training Script for Transformer Piece Predictor.

Features:
- CUDA/GPU support with automatic device detection
- Mixed precision training (FP16) for T4 optimization
- BCE loss with positive class weighting
- AdamW optimizer with cosine learning rate schedule
- Early stopping on validation loss
- Checkpoint saving

Usage:
    python train.py --data-dir data --epochs 25 --batch-size 128
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TransformerPiecePredictor, create_model
from preprocess import PieceDataset, load_preprocessed_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
        # Set number of threads for CPU
        torch.set_num_threads(min(32, os.cpu_count() or 4))
    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for features, targets in pbar:
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                logits = model(features)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return {"loss": total_loss / n_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_probs = []
    all_targets = []
    
    for features, targets in tqdm(dataloader, desc="Validating", leave=False):
        features = features.to(device)
        targets = targets.to(device)
        
        if use_amp:
            with autocast():
                logits = model(features)
                loss = criterion(logits, targets)
        else:
            logits = model(features)
            loss = criterion(logits, targets)
        
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
        
        total_loss += loss.item()
        n_batches += 1
    
    # Compute metrics
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Threshold-based predictions
    preds = (all_probs > 0.5).astype(np.float32)
    
    # Compute metrics
    # Per-sample metrics
    n_samples = len(all_targets)
    
    # Precision, Recall, F1 (micro-averaged)
    tp = (preds * all_targets).sum()
    fp = (preds * (1 - all_targets)).sum()
    fn = ((1 - preds) * all_targets).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Exact match ratio
    exact_match = (preds == all_targets).all(axis=1).mean()
    
    return {
        "loss": total_loss / n_batches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }, path)


def train(
    data_dir: str,
    checkpoint_dir: str,
    embed_dim: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    batch_size: int = 128,
    epochs: int = 25,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 10,
    seed: int = 42,
) -> dict:
    """
    Main training function.
    
    Args:
        data_dir: Directory with preprocessed data
        checkpoint_dir: Directory to save checkpoints
        embed_dim: Model embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        patience: Early stopping patience
        seed: Random seed
        
    Returns:
        Dictionary with training history
    """
    set_seed(seed)
    device = get_device()
    use_amp = torch.cuda.is_available()
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    data = load_preprocessed_data(data_dir)
    
    # Create datasets and dataloaders
    train_dataset = PieceDataset(data["X_train"], data["y_train"])
    val_dataset = PieceDataset(data["X_val"], data["y_val"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        feature_vocab_sizes=data["config"]["feature_vocab_sizes"],
        n_pieces=data["config"]["n_pieces"],
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    model = model.to(device)
    
    params = model.get_num_parameters()
    logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Loss function with class weighting
    pos_weights = torch.from_numpy(data["pos_weights"]).float().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Mixed precision: {use_amp}")
    logger.info("=" * 60)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_exact_match": [],
        "lr": [],
    }
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history (convert numpy types to native Python for JSON serialization)
        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_exact_match"].append(float(val_metrics["exact_match"]))
        history["lr"].append(float(current_lr))
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(checkpoint_dir, "best_model.pt")
            )
            logger.info(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics,
            os.path.join(checkpoint_dir, "latest_model.pt")
        )
        
        # Early stopping
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    # Save training history (convert to native Python types for JSON)
    history["total_time_seconds"] = float(total_time)
    history["best_val_loss"] = float(best_val_loss)
    history["final_epoch"] = int(epoch)
    
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Final epoch: {epoch}")
    logger.info("=" * 60)
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer Piece Predictor"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=1024,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = script_dir / checkpoint_dir
    
    train(
        data_dir=str(data_dir),
        checkpoint_dir=str(checkpoint_dir),
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

