"""ROC curve calculated over model-score thresholds."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from metrics.metrics import fall_out, recall
from graph.threshold_curves import (
	ConfusionCounts,
	calculate_threshold_curve,
	make_realistic_demo_samples,
	plot_threshold_curve,
)


def _false_positive_rate_from_counts(counts: ConfusionCounts) -> float:
	return fall_out(counts.true_negative, counts.false_positive)


def _true_positive_rate_from_counts(counts: ConfusionCounts) -> float:
	return recall(counts.true_positive, counts.false_negative)


def roc_auc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
	"""Return threshold-based ROC AUC without creating a plot."""
	curve = calculate_threshold_curve(
		y_true=y_true,
		y_scores=y_scores,
		x_metric=_false_positive_rate_from_counts,
		y_metric=_true_positive_rate_from_counts,
	)
	return curve.auc


def plot_roc_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a ROC curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	curve = calculate_threshold_curve(
		y_true=y_true,
		y_scores=y_scores,
		x_metric=_false_positive_rate_from_counts,
		y_metric=_true_positive_rate_from_counts,
	)
	return plot_threshold_curve(
		curve=curve,
		x_label="False Positive Rate",
		y_label="True Positive Rate",
		title="ROC Curve",
		auc_label="ROC AUC",
		show_random_baseline=True,
	)


if __name__ == "__main__":
	y_true, y_scores = make_realistic_demo_samples()
	figure, axes, auc = plot_roc_auc(y_true, y_scores)
	figure.savefig(Path(__file__).with_name("roc_auc.jpg"), dpi=300, bbox_inches="tight")
	print(f"Realistic demo ROC AUC: {auc:.3f}")
	plt.show()
