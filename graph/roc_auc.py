"""ROC curve, computed with scikit-learn."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from graph.threshold_curves import (
	ThresholdCurve,
	make_realistic_demo_samples,
	plot_threshold_curve,
)

def plot_roc_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a ROC curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_scores)
	curve = ThresholdCurve(
		thresholds=thresholds,
		x_values=false_positive_rate,
		y_values=true_positive_rate,
		auc=float(roc_auc_score(y_true, y_scores)),
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
