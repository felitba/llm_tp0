"""Precision-recall curve calculated over model-score thresholds."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from metrics.metrics import precision, recall
from graph.threshold_curves import (
	ConfusionCounts,
	ThresholdCurve,
	calculate_threshold_curve,
	make_realistic_demo_samples,
	plot_combined_threshold_curves,
	plot_threshold_curve,
)


def _precision_from_counts(counts: ConfusionCounts) -> float:
	if counts.true_positive + counts.false_positive == 0:
		return 1.0
	return precision(counts.true_positive, counts.false_positive)


def _recall_from_counts(counts: ConfusionCounts) -> float:
	return recall(counts.true_positive, counts.false_negative)


def pr_curve(y_true: Sequence[int], y_scores: Sequence[float]) -> ThresholdCurve:
	"""Return the threshold-based precision-recall curve points and its AUC."""
	return calculate_threshold_curve(
		y_true=y_true,
		y_scores=y_scores,
		x_metric=_recall_from_counts,
		y_metric=_precision_from_counts,
	)


def pr_auc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
	"""Return threshold-based PR AUC without creating a plot."""
	return pr_curve(y_true, y_scores).auc


def plot_pr_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a precision-recall curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	curve = pr_curve(y_true, y_scores)
	return plot_threshold_curve(
		curve=curve,
		x_label="Recall",
		y_label="Precision",
		title="Precision-Recall Curve",
		auc_label="PR AUC",
		step=True,
	)


def plot_pr_auc_by_config(
	results_by_config: Sequence[tuple[str, Sequence[int], Sequence[float]]],
) -> tuple[plt.Figure, plt.Axes]:
	"""Plot every config's PR curve on one figure, one color per config.

	Each entry is ``(config_name, y_true, y_scores)``. Curves are drawn from
	best to worst AUC so the legend reads as a ranking.
	"""
	curves = [(name, pr_curve(y_true, y_scores)) for name, y_true, y_scores in results_by_config]
	curves.sort(key=lambda item: item[1].auc, reverse=True)
	return plot_combined_threshold_curves(
		curves=curves,
		x_label="Recall",
		y_label="Precision",
		title="Precision-Recall Curve by Configuration",
		auc_label="AUC",
		step=True,
	)


if __name__ == "__main__":
	y_true, y_scores = make_realistic_demo_samples()
	figure, axes, auc = plot_pr_auc(y_true, y_scores)
	figure.savefig(Path(__file__).with_name("pr_auc.jpg"), dpi=300, bbox_inches="tight")
	print(f"Realistic demo PR AUC: {auc:.3f}")
	plt.show()
