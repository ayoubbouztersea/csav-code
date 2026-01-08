#!/usr/bin/env python3
"""
Transformer Model Architecture for Piece Prediction.

This module defines:
- FeatureEmbeddings: Embedding layer for categorical features
- TransformerPiecePredictor: Main model with transformer encoder and multi-label head
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeatureEmbeddings(nn.Module):
    """
    Embedding layer for multiple categorical features.
    
    Each feature gets its own embedding table, and all embeddings
    are projected to the same dimension.
    """
    
    def __init__(
        self,
        vocab_sizes: list[int],
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_sizes: List of vocabulary sizes for each feature
            embed_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_features = len(vocab_sizes)
        
        # Create embedding table for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx].zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, n_features) with feature indices
            
        Returns:
            Tensor of shape (batch_size, n_features, embed_dim)
        """
        # Embed each feature
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            feat_emb = emb(x[:, i])  # (batch_size, embed_dim)
            embeddings.append(feat_emb)
        
        # Stack along sequence dimension
        x = torch.stack(embeddings, dim=1)  # (batch_size, n_features, embed_dim)
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class TransformerPiecePredictor(nn.Module):
    """
    Transformer-based model for multi-label piece prediction.
    
    Architecture:
    1. Embed categorical features
    2. Add CLS token and positional encoding
    3. Pass through transformer encoder layers
    4. Use CLS token output for classification
    5. MLP head outputs logits for each piece
    """
    
    def __init__(
        self,
        vocab_sizes: list[int],
        n_pieces: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_sizes: List of vocabulary sizes for each categorical feature
            n_pieces: Number of unique pieces (output dimension)
            embed_dim: Embedding/hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            ff_dim: Feed-forward dimension in transformer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_pieces = n_pieces
        self.n_features = len(vocab_sizes)
        
        # Feature embeddings
        self.feature_embeddings = FeatureEmbeddings(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=embed_dim,
            max_len=len(vocab_sizes) + 1,  # +1 for CLS token
            dropout=dropout,
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_pieces),
        )
        
        # Initialize classifier
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, n_features) with feature indices
            return_embeddings: If True, also return CLS embeddings
            
        Returns:
            logits: Tensor of shape (batch_size, n_pieces) with logits
            embeddings (optional): Tensor of shape (batch_size, embed_dim)
        """
        batch_size = x.size(0)
        
        # Embed features: (batch_size, n_features, embed_dim)
        x = self.feature_embeddings(x)
        
        # Prepend CLS token: (batch_size, 1 + n_features, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        
        # Classification head
        logits = self.classifier(cls_output)  # (batch_size, n_pieces)
        
        if return_embeddings:
            return logits, cls_output
        return logits
    
    def predict_pieces(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Predict piece indices for a batch of inputs.
        
        Args:
            x: Tensor of shape (batch_size, n_features)
            threshold: Probability threshold for positive prediction
            top_k: If set, return top-k predictions regardless of threshold
            
        Returns:
            List of lists, each containing predicted piece indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            predictions = []
            for i in range(probs.size(0)):
                if top_k is not None:
                    # Get top-k predictions
                    _, indices = probs[i].topk(top_k)
                    predictions.append(indices.cpu().tolist())
                else:
                    # Threshold-based predictions
                    indices = (probs[i] > threshold).nonzero(as_tuple=True)[0]
                    predictions.append(indices.cpu().tolist())
            
            return predictions
    
    def get_num_parameters(self) -> dict[str, int]:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }


def create_model(
    feature_vocab_sizes: dict[str, int],
    n_pieces: int,
    embed_dim: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    ff_dim: int = 1024,
    dropout: float = 0.1,
) -> TransformerPiecePredictor:
    """
    Factory function to create model from config.
    
    Args:
        feature_vocab_sizes: Dict mapping feature name to vocab size
        n_pieces: Number of unique pieces
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        
    Returns:
        Initialized TransformerPiecePredictor model
    """
    vocab_sizes = list(feature_vocab_sizes.values())
    
    model = TransformerPiecePredictor(
        vocab_sizes=vocab_sizes,
        n_pieces=n_pieces,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing TransformerPiecePredictor...")
    
    # Simulate config
    vocab_sizes = [120, 220, 10, 30, 90]  # 5 features
    n_pieces = 10000
    
    model = TransformerPiecePredictor(
        vocab_sizes=vocab_sizes,
        n_pieces=n_pieces,
        embed_dim=256,
        n_heads=8,
        n_layers=4,
    )
    
    # Print parameter count
    params = model.get_num_parameters()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Test forward pass
    batch_size = 4
    x = torch.randint(0, 10, (batch_size, len(vocab_sizes)))
    
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    predictions = model.predict_pieces(x, threshold=0.5)
    print(f"Predictions: {predictions}")
    
    print("Test passed!")

