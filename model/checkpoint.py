"""Trained weights, saved only when ``main.py --save-weights`` asks for them.

Off by default on purpose. A d_model=96 checkpoint is ~20 MB, and 4.82M of its
4.99M parameters are the tiktoken embedding table for 50,257 subwords that a
10-token product name barely touches — while a full retrain of that config is
about six minutes. So the numbers a run produced always go to disk
(``metrics/run_results.py``); the weights only when you intend to reopen the
model, which is what these buy:

* attention from ``[CLS]`` over the feature tokens — which columns the encoder
  actually reads. ``nn.TransformerEncoder`` drops the weights, so this still
  needs a hook on each layer's ``self_attn``; the checkpoint is what keeps that
  work from costing a retrain per attempt.
* the learned per-column embedding tables, read straight off the state dict —
  a column with no signal stays near its ``d_model**-0.5`` init.
* inference on a row you make up (flip ``title_tag`` and watch the probability
  move).

What is stored is the state restored by early stopping, not the last epoch's:
the weights the reported test metrics actually came from. ``cardinalities`` and
the merged ``config`` travel with them because ``EncoderOnlyModel`` cannot be
rebuilt without both, and ``categorical_id_mapping`` because an id is
meaningless without the value it addresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from metrics.run_results import run_dir
from model.encoder_only_model import EncoderOnlyModel

FILENAME = "model.pt"


def checkpoint_path(name: str) -> Path:
	return run_dir(name) / FILENAME


def save_checkpoint(
	name: str,
	model: EncoderOnlyModel,
	cardinalities: dict[str, int],
	config: dict[str, Any],
	categorical_id_mapping: dict[str, dict[int, str]] | None = None,
	best_epoch: int | None = None,
	test: dict[str, float] | None = None,
) -> Path:
	"""Write one experiment's weights next to its run.json."""
	path = checkpoint_path(name)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"name": name,
			"state_dict": model.state_dict(),
			"cardinalities": {column: int(value) for column, value in cardinalities.items()},
			# The experiment list would drag every other config in with it.
			"config": {key: value for key, value in config.items() if key != "experiments"},
			"categorical_id_mapping": {
				column: {int(index): str(value) for index, value in mapping.items()}
				for column, mapping in (categorical_id_mapping or {}).items()
			},
			"best_epoch": best_epoch,
			"test": {key: float(value) for key, value in (test or {}).items()},
		},
		path,
	)
	return path


def load_checkpoint(
	name: str, device: str | torch.device = "cpu"
) -> tuple[EncoderOnlyModel, dict[str, Any]]:
	"""Rebuild one experiment's model in eval mode, with the payload beside it.

		model, payload = load_checkpoint("medium_d96_l2_baseline")
		model.feature_tokenizer.token_names()           # what each position holds
		table = model.feature_tokenizer.categorical_tokenizer.embeddings["brand"].weight
		payload["categorical_id_mapping"]["brand"][1]   # what row 1 of it stands for
	"""
	path = checkpoint_path(name)
	if not path.exists():
		raise FileNotFoundError(
			f"No weights for '{name}' ({path}). "
			f"Rerun it with: python main.py --experiment {name} --save-weights"
		)
	payload = torch.load(path, map_location=device, weights_only=True)
	model = EncoderOnlyModel(
		cardinalities=payload["cardinalities"], config=payload["config"]
	).to(device)
	model.load_state_dict(payload["state_dict"])
	model.eval()
	return model, payload
