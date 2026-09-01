"""Draw what [CLS] attends to, from a saved checkpoint. No retraining.

    python scripts/attention_map.py s1_tabular_tokens_8col

Needs weights, so the run has to have been trained with --save-weights. The
batch it scores is the test split of that run's own config, rebuilt from the
config stored inside the checkpoint, so the figure always matches the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.preprocess_dataset import categorical_cardinalities, get_data_processed
from dataset.product_dataset import create_data_loaders
from metrics.run_results import run_dir
from model.attention import cls_attention
from model.checkpoint import load_checkpoint
from plots.attention_map import plot_cls_attention
from plots.plot_theme import save


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("name", help="experiment name under output/experiments/")
	parser.add_argument("--batches", type=int, default=4,
	                    help="test batches to average over (default 4)")
	args = parser.parse_args()

	model, payload = load_checkpoint(args.name)
	config = payload["config"]
	loaders = create_data_loaders(get_data_processed(config), config)

	numeric, categorical, title_ids = [], [], []
	for index, batch in enumerate(loaders.test):
		if index >= args.batches:
			break
		numeric.append(batch["numeric"])
		categorical.append(batch["categorical"])
		title_ids.append(batch["title_ids"])

	weights, names = cls_attention(
		model, torch.cat(numeric), torch.cat(categorical), torch.cat(title_ids)
	)

	figure, _ = plot_cls_attention(
		weights, names,
		title=f"{args.name} — atención de [CLS] por posición",
		hyperparameters=(
			f"d_model={config.get('d_model')} · {config.get('num_layers')} capas · "
			f"{config.get('n_heads')} heads · promedio sobre {len(torch.cat(numeric))} filas de test"
		),
	)
	path = save(figure, run_dir(args.name) / "attention_cls.jpg")
	print(f"Figura: {path}")

	uniform = 1.0 / weights.shape[1]
	print(f"Nivel uniforme (sin selección): {uniform:.4f}")
	for layer in range(weights.shape[0]):
		order = weights[layer].argsort()[::-1][:5]
		top = ", ".join(f"{names[i]} {weights[layer][i]:.3f}" for i in order)
		print(f"  capa {layer + 1}: {top}")


if __name__ == "__main__":
	main()
