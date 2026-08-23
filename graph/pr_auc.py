"""Precision-recall curve computed with scikit-learn"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


from graph.threshold_curves import (
	ThresholdCurve,
	make_realistic_demo_samples,
	plot_threshold_curve,
)

def plot_pr_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a precision-recall curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	curve = ThresholdCurve(
		# precision_recall_curve returns one more point than thresholds: the
		# final (recall 0, precision 1) corner has no threshold behind it.
		thresholds=np.append(thresholds, np.inf),
		x_values=recall,
		y_values=precision,
		auc=float(average_precision_score(y_true, y_scores)),
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
