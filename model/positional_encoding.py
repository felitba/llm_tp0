"""Sinusoidal positional encoding for Transformer inputs."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
	"""Add fixed sinusoidal positional information to token embeddings.

	Args:
		d_model: Embedding dimension.
		max_len: Maximum supported sequence length.
		dropout: Dropout probability applied after adding the encoding.
		batch_first: If True, inputs have shape (batch_size, sequence_length, embedding_dim)

	"""

	def __init__(
		self,
		d_model: int,
		max_len: int = 5000,
		dropout: float = 0.0,
		batch_first: bool = True,
	) -> None:
		super().__init__()
		if d_model <= 0:
			raise ValueError("d_model must be positive")
		if max_len <= 0:
			raise ValueError("max_len must be positive")

		self.dropout = nn.Dropout(dropout)
		self.batch_first = batch_first

		position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2, dtype=torch.float32)
			* (-math.log(10000.0) / d_model)
		)
		encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
		encoding[:, 0::2] = torch.sin(position * div_term)
		encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

		# A buffer follows the module across devices but is not trainable.
		self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return ``x`` with positional encodings added."""
		if x.ndim != 3:
			raise ValueError("x must have shape (batch, sequence, features) or (sequence, batch, features)")

		sequence_length = x.shape[1] if self.batch_first else x.shape[0]
		if sequence_length > self.encoding.shape[1]:
			raise ValueError("sequence length exceeds the configured max_len")

		if self.batch_first:
			x = x + self.encoding[:, :sequence_length].to(dtype=x.dtype)
		else:
			x = x + self.encoding[:, :sequence_length].transpose(0, 1).to(dtype=x.dtype)
		return self.dropout(x)
