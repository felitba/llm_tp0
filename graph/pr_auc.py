"""Precision-recall curve calculated over sampled thresholds."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from metrics.metrics import precision, recall
from graph.threshold_curves import (
	ConfusionCounts,
	calculate_threshold_curve,
	make_realistic_demo_samples,
	plot_threshold_curve,
)


def _precision_from_counts(counts: ConfusionCounts) -> float:
	if counts.true_positive + counts.false_positive == 0:
		return 1.0
	return precision(counts.true_positive, counts.false_positive)


def _recall_from_counts(counts: ConfusionCounts) -> float:
	return recall(counts.true_positive, counts.false_negative)

def plot_pr_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a precision-recall curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	curve = calculate_threshold_curve(
		y_true=y_true,
		y_scores=y_scores,
		x_metric=_recall_from_counts,
		y_metric=_precision_from_counts,
	)
	return plot_threshold_curve(
		curve=curve,
		x_label="Recall",
		y_label="Precision",
		title="Precision-Recall Curve",
		auc_label="PR AUC",
		step=True,
	)


if __name__ == "__main__":
	y_true, y_scores = make_realistic_demo_samples()
	figure, axes, auc = plot_pr_auc(y_true, y_scores)
	figure.savefig(Path(__file__).with_name("pr_auc.jpg"), dpi=300, bbox_inches="tight")
	print(f"Realistic demo PR AUC: {auc:.3f}")
	plt.show()
