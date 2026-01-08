#!/bin/bash
# =============================================================================
# Transformer Piece Predictor - Full Pipeline
# =============================================================================
# This script runs preprocessing, training, and evaluation sequentially.
# Use with nohup: nohup ./run_all.sh > pipeline.log 2>&1 &
# =============================================================================

set -e  # Exit on error

# Project directory on VM
PROJECT_DIR="./trans_csav"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Transformer Piece Predictor Pipeline"
echo "============================================================"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# =============================================================================
# Step 1: Preprocessing
# =============================================================================
echo ""
echo "[Step 1/3] Preprocessing data..."
echo "------------------------------------------------------------"
python preprocess.py \
    --data-path df_ml2.csv \
    --min-freq 5 \
    --output-dir data \
    --test-size 0.1 \
    --val-size 0.1

echo "Preprocessing complete: $(date)"

# =============================================================================
# Step 2: Training
# =============================================================================
echo ""
echo "[Step 2/3] Training model..."
echo "------------------------------------------------------------"
python train.py \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 4 \
    --ff-dim 1024 \
    --dropout 0.1 \
    --batch-size 128 \
    --epochs 25 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --patience 10 \
    --seed 42

echo "Training complete: $(date)"

# =============================================================================
# Step 3: Evaluation
# =============================================================================
echo ""
echo "[Step 3/3] Evaluating model..."
echo "------------------------------------------------------------"
python evaluate.py \
    --data-dir data \
    --checkpoint checkpoints/best_model.pt \
    --output evaluation_results.json \
    --batch-size 128

echo "Evaluation complete: $(date)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - data/                     (preprocessed data)"
echo "  - checkpoints/best_model.pt (trained model)"
echo "  - checkpoints/history.json  (training history)"
echo "  - evaluation_results.json   (evaluation metrics)"
echo "============================================================"

