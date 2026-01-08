# Transformer Piece Predictor

A transformer-based model for predicting repair pieces from categorical appliance features.

## Overview

This package implements a multi-label classification model using a transformer encoder architecture. Given categorical features about an appliance (symptom, brand, product family, etc.), it predicts which repair pieces are needed.

## Architecture

- **Input**: Categorical features (Symptome, Marque, Fam1-Fam4, Version, Type_Prediag, CodeError)
- **Embeddings**: Each feature gets its own embedding layer
- **Transformer**: 4-layer encoder with 8 attention heads, 256 dim
- **Output**: Multi-label classification over ~10K-15K pieces

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Preprocess Data

```bash
cd trans_csav
python preprocess.py --data-path ../data/df_ml2.csv --min-freq 5
```

This creates:
- `data/X_train.npy`, `data/X_val.npy`, `data/X_test.npy` - Encoded features
- `data/y_train.npy`, `data/y_val.npy`, `data/y_test.npy` - Multi-hot targets
- `data/vocabularies.pkl` - Feature and piece vocabularies
- `data/config.json` - Dataset configuration

### 2. Train Model

```bash
python train.py --data-dir data --epochs 25 --batch-size 128
```

Key options:
- `--epochs`: Maximum training epochs (default: 25)
- `--batch-size`: Batch size (default: 128, optimized for T4 GPU)
- `--learning-rate`: Initial learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 10)

Training outputs:
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/latest_model.pt` - Latest checkpoint
- `checkpoints/history.json` - Training history

### 3. Evaluate Model

```bash
python evaluate.py --data-dir data --checkpoint checkpoints/best_model.pt
```

This outputs:
- Micro/Macro Precision, Recall, F1
- Exact Match Ratio
- Precision@K and Recall@K (K=1,3,5)

### 4. Inference

```python
from trans_csav.evaluate import PiecePredictor

# Load predictor
predictor = PiecePredictor.load(
    checkpoint_path="trans_csav/checkpoints/best_model.pt",
    data_dir="trans_csav/data"
)

# Predict from feature dictionary
features = {
    "Symptome": "SYMP_002 Fuit",
    "Marque": "BEKO",
    "Fam1": "BLANC",
    "Fam2": "LAVAGE",
    "Fam3": "LAVE VAISSELLE",
    "Fam4": "LAVE VAISSELLE POSE LIBRE",
    "Version": "NO_VERSION",
    "Type_Prediag": "Auto avec pièces",
    "CodeError": "NO_ERROR",
}

# Get predicted pieces
pieces = predictor.predict(features)
print(f"Predicted pieces: {pieces}")

# Get predictions with confidence scores
pieces_with_scores = predictor.predict_with_scores(features, top_k=5)
for piece, score in pieces_with_scores[0]:
    print(f"  {piece}: {score:.3f}")
```

## Training Time Estimates

| Hardware | Estimated Time |
|----------|---------------|
| NVIDIA T4 (16GB) | 1-2 hours |
| NVIDIA A100 | 30-45 minutes |
| CPU (32 cores) | 4-8 hours |

## File Structure

```
trans_csav/
├── __init__.py          # Package initialization
├── model.py             # Transformer model architecture
├── preprocess.py        # Data preprocessing
├── train.py             # Training script
├── evaluate.py          # Evaluation and inference
├── requirements.txt     # Dependencies
├── README.md            # This file
├── data/                # Preprocessed data (generated)
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── ...
│   ├── vocabularies.pkl
│   └── config.json
└── checkpoints/         # Model checkpoints (generated)
    ├── best_model.pt
    ├── latest_model.pt
    └── history.json
```

## Model Details

- **Embedding Dimension**: 256
- **Attention Heads**: 8
- **Transformer Layers**: 4
- **Feed-forward Dimension**: 1024
- **Dropout**: 0.1
- **Parameters**: ~8M (varies with vocabulary size)

## Loss Function

Uses Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`) with:
- Positive class weighting to handle imbalanced piece frequencies
- Weights computed from training set piece frequencies

