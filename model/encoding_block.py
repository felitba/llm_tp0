from __future__ import annotations

import torch
import torch.nn as nn

from config.config import load_config


class EncodingBlock(nn.Module):
    """Transformer encoder block configured via config.json."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        config = load_config()

        self.embedding_dim = int(config.get("d_model"))
        self.num_heads = int(config.get("n_heads"))
        self.num_layers = int(config.get("num_layers"))
        self.dropout = float(config.get("dropout"))
        self.dim_feedforward = int(config.get("dim_feedforward"))
        self.activation = config.get("activation")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            # batch_first=True,
            dropout=self.dropout,
            activation=self.activation,
            dim_feedforward=self.dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers, # number of transformer encoder layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

