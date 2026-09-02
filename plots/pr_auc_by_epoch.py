"""PR-AUC per epoch for train and validation, the pair read against the loss.

The loss curve says when the model stops fitting; this one says when it stops
*ranking* better, which is the thing the report is graded on and the thing the
checkpoint is selected by. The two disagree on this dataset -- BCE bottoms out
and starts rising while PR-AUC is still climbing -- so showing only the loss
would argue for stopping far earlier than the metric wants.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt

from plots.plot_theme import (
	BASELINE, DASH, MUTED, SPLIT_COLORS, apply_theme, legend_top_left, save, set_title,
)


def plot_pr_auc_by_epoch(
	epoch_metrics: Sequence[dict],
	title: str = "PR-AUC por época",
	hyperparameters: str | None = None,
	chance_level: float | None = None,
	baseline_level: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.2, 4))
	epochs = [row["epoch"] for row in epoch_metrics]

	for split, key, label in (
		("train", "train_pr_auc", "entrenamiento"),
		("validation", "val_pr_auc", "validación"),
	):
		if epoch_metrics and key in epoch_metrics[0]:
			axes.plot(
				epochs, [row[key] for row in epoch_metrics],
				color=SPLIT_COLORS[split], label=label,
			)

	if chance_level is not None:
		axes.axhline(chance_level, color=BASELINE, linestyle=DASH, linewidth=1)
		axes.annotate(
			f"azar = {chance_level:.3f}", xy=(epochs[-1] if epochs else 0, chance_level),
			xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED,
		)
	if baseline_level is not None:
		axes.axhline(baseline_level, color=BASELINE, linestyle=DASH, linewidth=1)
		axes.annotate(
			f"baseline title_tag = {baseline_level:.3f}",
			xy=(epochs[-1] if epochs else 0, baseline_level), xytext=(-4, 4),
			textcoords="offset points", ha="right", fontsize=8, color=MUTED,
		)
	# CHANGED (2026-09-01): aca se marcaba la epoca del checkpoint con una linea
	# vertical y su anotacion. Sale de las figuras: la regla de seleccion se
	# explica en sus propias filminas (scripts/slide_seleccion.py), no encima de
	# cada curva. La curva queda como lo que es, una curva por epoca.

	axes.set_xlabel("Época")
	axes.set_ylabel("PR-AUC")
	set_title(axes, title)
	legend_top_left(axes, subtitle=hyperparameters)
	figure.tight_layout()
	return figure, axes
