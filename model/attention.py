"""Recover the attention weights nn.TransformerEncoder throws away.

``nn.TransformerEncoderLayer`` calls its ``self_attn`` with
``need_weights=False`` and returns only the transformed sequence, so a plain
forward hook sees the output and never the weights. The way through is to wrap
each layer's ``self_attn.forward`` for the duration of one forward pass, ask it
for the weights, and stash them -- the wrapper is removed afterwards, so the
model is left exactly as it was found.

Wrapping alone is not enough. In eval mode under no_grad the layer takes a fused
native kernel that never touches the ``self_attn`` module, so the wrapper is
simply never called and nothing is captured; the fast path has to be turned off
around the pass. It is only an optimisation, so the numbers do not change.

What comes back is, per layer, a (batch, positions, positions) matrix where
entry [i, j] is how much position i read from position j. Row 0 is the one the
report is about: [CLS] carries no input of its own, so what it attends to is
what the classification head ends up reading.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from model.encoder_only_model import EncoderOnlyModel


@contextmanager
def capture_attention(model: EncoderOnlyModel):
	"""Yield a dict that fills with {layer_index: weights} during the forward pass."""
	captured: dict[int, torch.Tensor] = {}
	layers = list(model.encoder_block.encoder.layers)
	originals = [layer.self_attn.forward for layer in layers]
	fastpath_was_enabled = torch.backends.mha.get_fastpath_enabled()

	def make_wrapper(index, original):
		def wrapper(*args, **kwargs):
			# average_attn_weights collapses the heads. The report's claim is about
			# which COLUMN the encoder reads, not which head reads it, and a
			# per-head figure needs one panel per head to be honest.
			kwargs["need_weights"] = True
			kwargs["average_attn_weights"] = True
			output, weights = original(*args, **kwargs)
			if weights is not None:
				captured[index] = weights.detach().cpu()
			return output, weights
		return wrapper

	for index, (layer, original) in enumerate(zip(layers, originals)):
		layer.self_attn.forward = make_wrapper(index, original)
	torch.backends.mha.set_fastpath_enabled(False)
	try:
		yield captured
	finally:
		torch.backends.mha.set_fastpath_enabled(fastpath_was_enabled)
		for layer, original in zip(layers, originals):
			layer.self_attn.forward = original


def cls_attention(
	model: EncoderOnlyModel,
	numeric: torch.Tensor,
	categorical: torch.Tensor,
	title_ids: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
	"""Mean attention from [CLS] to every position, per layer.

	Returns ``(layers x positions)`` averaged over the batch, and the name of
	each position from the tokenizer, so the axis labels are the column names.
	"""
	model.eval()
	with torch.no_grad(), capture_attention(model) as captured:
		model(numeric=numeric, categorical=categorical, title_ids=title_ids)
	if not captured:
		raise RuntimeError("no attention captured: does this model have encoder layers?")
	rows = [captured[index][:, 0, :].mean(dim=0).numpy() for index in sorted(captured)]
	return np.vstack(rows), model.feature_tokenizer.token_names()
