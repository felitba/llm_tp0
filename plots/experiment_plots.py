"""Figures for one or many experiments, drawn from what a run wrote to disk.

Both ``main.py`` (right after training) and ``replot.py`` (from
``output/experiments/*/run.json``) go through here, so a styling change lands on
the whole deck without a single forward pass being repeated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from plots.pr_auc import plot_pr_auc, plot_pr_auc_by_config
from plots.roc_auc import plot_roc_auc, plot_roc_auc_by_config
from plots.train_vs_val_error import plot_training_progress
from metrics.run_results import EXPERIMENTS_DIR, RunResults
from plots.plot_theme import save


def training_hyperparameters(config: dict) -> str:
	"""Render the effective run configuration beneath its loss curve."""
	return (
		f"d_model={int(config.get('d_model', 0))} · "
		f"heads={int(config.get('n_heads', 0))} · "
		f"layers={int(config.get('num_layers', 0))} · "
		f"FFN={int(config.get('dim_feedforward', 0))}\n"
		f"lr={float(config.get('learning_rate', 0.0)):.0e} · "
		f"dropout={float(config.get('dropout', 0.0)):.2f} · "
		f"weight decay={float(config.get('weight_decay', 0.0)):.2g} · "
		f"batch={int(config.get('batch_size', 0))} · "
		f"max text length={int(config.get('max_title_len', 0))}"
	)


def plot_run(results: RunResults) -> list[Path]:
	"""Save loss, PR and ROC figures for one experiment next to its run.json."""
	output_dir = results.directory
	output_paths = []

	if results.history.get("train"):
		figure, _ = plot_training_progress(
			results.history,
			best_epoch=results.selection.get("best_epoch"),
			title=f"{results.name} — Training vs. Validation Loss",
			hyperparameters=training_hyperparameters(results.config),
		)
		output_paths.append(save(figure, output_dir / "loss.jpg"))

	if not results.has_curves:
		return output_paths

	labels = np.asarray(results.labels, dtype=int)
	probs = np.asarray(results.probs, dtype=float)
	figure, _, _ = plot_pr_auc(labels, probs)
	output_paths.append(save(figure, output_dir / "pr_auc.jpg"))
	figure, _, _ = plot_roc_auc(labels, probs)
	output_paths.append(save(figure, output_dir / "roc_auc.jpg"))
	return output_paths


def plot_runs_combined(runs: Sequence[RunResults], suffix: str = "") -> list[Path]:
	"""Save ROC and PR figures holding every run's test curve, one color per run.

	``suffix`` names the file when the figure covers one config file's runs
	rather than everything on disk (``roc_auc_all_configs_<suffix>.jpg``).
	"""
	curves_input = [
		(run.name, np.asarray(run.labels, dtype=int), np.asarray(run.probs, dtype=float))
		for run in runs
		if run.has_curves
	]
	if not curves_input:
		return []

	tail = f"_{suffix}" if suffix else ""
	output_paths = []
	for plot_curves, filename in (
		(plot_roc_auc_by_config, f"roc_auc_all_configs{tail}.jpg"),
		(plot_pr_auc_by_config, f"pr_auc_all_configs{tail}.jpg"),
	):
		figure, _ = plot_curves(curves_input)
		output_paths.append(save(figure, EXPERIMENTS_DIR / filename))
	return output_paths
