from __future__ import annotations

import torch
import torch.nn as nn
from config.config import load_config
from model.encoding_block import EncodingBlock
from model.feature_tokenizer import FeatureTokenizer


class EncoderOnlyModel(nn.Module):
    """Encoder-only Transformer model for BTR prediction."""

    def __init__(
        self, cardinalities: dict[str, int], config: dict | None = None
    ) -> None:
        super().__init__()

        config_data = config or load_config()
        self.embedding_dim = int(config_data.get("d_model"))
        self.vocab_size = int(config_data.get("vocab_size"))
        self.dropout = float(config_data.get("dropout"))

        # Multimodal feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(cardinalities, config_data)

        # Encoder block
        self.encoder_block = EncodingBlock(config_data)

        # Classification head for BTR prediction (returns logits)
        self.classification_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        title_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            numeric: Tensor of shape (batch_size, n_numeric)
            categorical: Tensor of shape (batch_size, n_categorical)
            title_ids: Tensor of shape (batch_size, max_title_len)

        Returns:
            Logits of shape (batch_size, 1) for BTR
        """
        # Multimodal tokenization
        tokens, padding_mask = self.feature_tokenizer(
            numeric=numeric,
            categorical=categorical,
            title_ids=title_ids,
        )

        # Encoder with key padding mask
        encoded = self.encoder_block(tokens, src_key_padding_mask=padding_mask)

        # CLS representation
        cls_representation = encoded[:, 0, :]  # (batch_size, d_model)

        # Classification head
        output = self.classification_head(cls_representation)  # (batch_size, 1)

        return output
