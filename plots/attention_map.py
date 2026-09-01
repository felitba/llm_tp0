"""What [CLS] reads, per layer, with the column names on the axis.

The claim the whole architecture rests on is that treating a row as a sequence
makes "which feature matters, given the rest of the row" an explicit, inspectable
operation. This is the figure that either shows that or fails to: if attention
from [CLS] is flat across the columns, the encoder is not selecting anything and
the ablation with no encoder should confirm it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plots.plot_theme import BODY, apply_theme, legend_top_left, set_title


def plot_cls_attention(
	weights: np.ndarray, token_names: list[str],
	title: str = "Attention from [CLS] by position",
	hyperparameters: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
	weights = np.atleast_2d(np.asarray(weights, dtype=float))
	layers, positions = weights.shape

	apply_theme()
	figure, axes = plt.subplots(figsize=(max(6.0, 0.42 * positions), 1.1 * layers + 2.0))
	# Sequential, not the categorical palette: this is one quantity on an
	# ordered scale, and a categorical map would invent boundaries in it.
	image = axes.imshow(weights, aspect="auto", cmap="Blues", vmin=0.0)

	axes.set_xticks(range(positions))
	axes.set_xticklabels(token_names[:positions], rotation=60, ha="right", fontsize=8)
	axes.set_yticks(range(layers))
	axes.set_yticklabels([f"capa {i + 1}" for i in range(layers)])
	axes.grid(False)

	# The uniform level is the reference that makes the figure readable: above it
	# the position is being selected, below it, ignored.
	uniform = 1.0 / positions
	bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
	bar.set_label(f"peso de atención (uniforme = {uniform:.3f})", fontsize=8, color=BODY)

	for row in range(layers):
		for column in range(positions):
			value = weights[row, column]
			if value >= 2 * uniform:
				axes.annotate(f"{value:.2f}", xy=(column, row), ha="center", va="center",
				              fontsize=7, color="white")

	set_title(axes, title)
	# Same stacking as every other figure: title, then the run's hyperparameters,
	# then the data. There is no legend on a heatmap, so this only places the
	# subtitle, but it keeps the spacing identical to the rest of the deck.
	legend_top_left(axes, subtitle=hyperparameters)
	figure.tight_layout()
	return figure, axes
