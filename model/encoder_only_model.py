from __future__ import annotations

import torch
import torch.nn as nn
from config.config import load_config
from embedding import TokenEmbedding
from encoding_block import EncodingBlock


class EncoderOnlyModel(nn.Module):
    """Encoder-only Transformer model for BTR prediction."""

    def __init__(self) -> None:
        super().__init__()

        config = load_config()
        self.embedding_dim = int(config.get("d_model"))
        self.vocab_size = int(config.get("vocab_size"))
        self.dropout = float(config.get("dropout"))

        # Embedding layer
        self.embedding = TokenEmbedding()

        # Encoder block
        self.encoder_block = EncodingBlock()

        # TODO: lets define this together. 
        # Classification head for BTR prediction
        self.classification_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len) with token ids

        Returns:
            Output predictions of shape (batch_size, 1) for BTR
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, d_model)

        # Encoder
        encoded = self.encoder_block(embedded)  # (batch_size, seq_len, d_model)

        # TODO: lets define this together. 
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)

        # TODO: lets define this together. 
        # Classification head
        output = self.classification_head(pooled)  # (batch_size, 1)

        return output
