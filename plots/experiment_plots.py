"""Figures for one or many experiments, drawn from what a run wrote to disk.

Both ``main.py`` (right after training) and ``replot.py`` (from
``output/experiments/*/run.json``) go through here, so a styling change lands on
the whole deck without a single forward pass being repeated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from plots.calibration import plot_btr_by_query, plot_reliability, plot_score_histogram
from plots.config_comparison import (
	BASELINE_PR_AUC,
	plot_cost_vs_score,
	plot_metric_by_epoch,
	plot_seed_spread,
	plot_sensitivity,
	plot_test_scores,
)
from plots.pr_auc import plot_pr_auc, plot_pr_auc_by_config, positive_rate
from plots.pr_auc_by_epoch import plot_pr_auc_by_epoch
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
	"""Every figure that reads one experiment, saved next to its run.json."""
	output_dir = results.directory
	output_paths = []
	caption = training_hyperparameters(results.config)
	best_epoch = results.selection.get("best_epoch")

	if results.history.get("train"):
		figure, _ = plot_training_progress(
			results.history,
			best_epoch=best_epoch,
			title=f"{results.name} — Training vs. Validation Loss",
			hyperparameters=caption,
		)
		output_paths.append(save(figure, output_dir / "loss.jpg"))

	# Next to the loss curve rather than next to the test PR curve: both are
	# per-epoch views of training, and reading them as a pair is the point.
	if results.epoch_metrics:
		figure, _ = plot_pr_auc_by_epoch(
			results.epoch_metrics,
			best_epoch=best_epoch,
			title=f"{results.name} — PR-AUC by epoch",
			hyperparameters=caption,
			chance_level=positive_rate(results.labels) if len(results.labels) else None,
			baseline_level=BASELINE_PR_AUC,
		)
		output_paths.append(save(figure, output_dir / "pr_auc_by_epoch.jpg"))

	if not results.has_curves:
		return output_paths

	labels = np.asarray(results.labels, dtype=int)
	probs = np.asarray(results.probs, dtype=float)
	figure, _, _ = plot_pr_auc(labels, probs)
	output_paths.append(save(figure, output_dir / "pr_auc.jpg"))
	figure, _, _ = plot_roc_auc(labels, probs)
	output_paths.append(save(figure, output_dir / "roc_auc.jpg"))

	# Calibration is read on validation, not on test: it is a diagnostic that
	# invites acting on what it shows, and acting on the test split is how the
	# test split stops being held out. Falls back to test for runs saved before
	# validation scores were persisted.
	calibration_split = "validation" if "validation" in results.split_predictions else "test"
	cal_labels, cal_probs = results.split_predictions.get(calibration_split, (labels, probs))
	figure, _ = plot_reliability(
		np.asarray(cal_labels, dtype=int), np.asarray(cal_probs, dtype=float),
		title=f"{results.name} — Reliability diagram ({calibration_split})",
		hyperparameters=caption,
	)
	output_paths.append(save(figure, output_dir / "calibration.jpg"))
	figure, _ = plot_score_histogram(
		labels, probs, title=f"{results.name} — Score distribution by class",
		hyperparameters=caption,
	)
	output_paths.append(save(figure, output_dir / "score_histogram.jpg"))

	# One PR curve per split from the same selected weights: the generalisation
	# gap measured in the metric the report is graded on, not in the loss.
	split_curves = [
		(split, np.asarray(l, dtype=int), np.asarray(pr, dtype=float))
		for split, (l, pr) in sorted(results.split_predictions.items())
		if len(l) and len(np.unique(l)) > 1
	]
	if len(split_curves) > 1:
		figure, _ = plot_pr_auc_by_config(split_curves)
		output_paths.append(save(figure, output_dir / "pr_auc_by_split.jpg"))

	drawn = plot_btr_by_query(
		results.query_ids, labels, probs,
		title=f"{results.name} — BTR by query: predicted vs. observed",
		hyperparameters=caption,
	)
	if drawn:
		output_paths.append(save(drawn[0], output_dir / "btr_by_query.jpg"))
	return output_paths


def plot_runs_combined(runs: Sequence[RunResults], suffix: str = "") -> list[Path]:
	"""Every comparative figure for one config file's runs.

	``suffix`` names the files when the figures cover one config file rather than
	everything on disk (``pr_auc_all_configs_<suffix>.jpg``). Without it each
	batch overwrites the previous batch's figures, so pass it.
	"""
	tail = f"_{suffix}" if suffix else ""
	subtitle = shared_hyperparameters(runs)
	output_paths = []

	curves_input = [
		(run.name, np.asarray(run.labels, dtype=int), np.asarray(run.probs, dtype=float))
		for run in runs
		if run.has_curves
	]
	if curves_input:
		for plot_curves, filename in (
			(plot_roc_auc_by_config, f"roc_auc_all_configs{tail}.jpg"),
			(plot_pr_auc_by_config, f"pr_auc_all_configs{tail}.jpg"),
		):
			figure, _ = plot_curves(curves_input)
			output_paths.append(save(figure, EXPERIMENTS_DIR / filename))

	# The test curves say which one won; these say why, which is where capacity
	# and overfitting can be told apart.
	for metric_key, y_label, title, limits, filename in (
		("val_loss", "Validation loss", "Validation loss by configuration",
		 None, f"val_loss_all_configs{tail}.jpg"),
		("val_pr_auc", "Validation PR-AUC", "Validation PR-AUC by configuration",
		 (0.0, 1.0), f"val_pr_auc_all_configs{tail}.jpg"),
	):
		drawn = plot_metric_by_epoch(runs, metric_key, y_label, title, subtitle, limits)
		if drawn:
			output_paths.append(save(drawn[0], EXPERIMENTS_DIR / filename))

	for plot_fn, filename in (
		(plot_test_scores, f"test_scores_all_configs{tail}.jpg"),
		(plot_cost_vs_score, f"cost_vs_score_all_configs{tail}.jpg"),
		(plot_sensitivity, f"sensitivity_all_configs{tail}.jpg"),
		(plot_seed_spread, f"seed_spread_all_configs{tail}.jpg"),
	):
		drawn = plot_fn(runs, subtitle=subtitle)
		if drawn:
			output_paths.append(save(drawn[0], EXPERIMENTS_DIR / filename))
	return output_paths


def shared_hyperparameters(runs: Sequence[RunResults]) -> str:
	"""Only the hyperparameters every run agrees on.

	In a comparative figure what matters is not what each run used -- the legend
	says that -- but what was held fixed, which is what makes the visible
	difference attributable.
	"""
	if not runs:
		return ""
	keys = ("d_model", "n_heads", "num_layers", "dim_feedforward",
	        "learning_rate", "dropout", "weight_decay", "batch_size")
	shared = [
		f"{key}={runs[0].config[key]}"
		for key in keys
		if key in runs[0].config and len({str(r.config.get(key)) for r in runs}) == 1
	]
	return ("fijo: " + " · ".join(shared)) if shared else ""
