"""
Transformer-based Piece Predictor for CSAV.

This package contains modules for predicting repair pieces based on
categorical appliance features using a transformer encoder architecture.
"""

from .model import TransformerPiecePredictor, create_model
from .preprocess import PieceDataset, load_preprocessed_data
from .evaluate import PiecePredictor

__version__ = "1.0.0"
__all__ = [
    "TransformerPiecePredictor",
    "create_model",
    "PieceDataset",
    "load_preprocessed_data",
    "PiecePredictor",
]

