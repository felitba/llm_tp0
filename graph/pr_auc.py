"""Precision-recall curve calculated over sampled thresholds."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics.metrics import precision, recall


def make_realistic_demo_samples(
	n_samples: int = 500, positive_rate: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
	"""Create reproducible labels and overlapping model scores for demo plots."""
	rng = np.random.default_rng(seed)
	y_true = rng.binomial(1, positive_rate, size=n_samples)
	y_scores = np.empty(n_samples, dtype=float)

	positive_mask = y_true == 1
	negative_mask = ~positive_mask
	y_scores[positive_mask] = rng.beta(5.0, 2.5, size=int(np.sum(positive_mask)))
	y_scores[negative_mask] = rng.beta(2.0, 5.0, size=int(np.sum(negative_mask)))
	return y_true, y_scores

def plot_pr_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a precision-recall curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	if len(y_true) != len(y_scores):
		raise ValueError("y_true and y_scores must have the same length")

	labels = np.asarray(y_true)
	scores = np.asarray(y_scores, dtype=float)
	thresholds = np.linspace(1.0, 0.0, 101)
	precisions = []
	recalls = []

	for threshold in thresholds:
		predicted = scores >= threshold
		true_positive = int(np.sum(predicted & (labels == 1)))
		false_positive = int(np.sum(predicted & (labels == 0)))
		false_negative = int(np.sum(~predicted & (labels == 1)))
		if true_positive + false_positive == 0:
			precisions.append(1.0)
		else:
			precisions.append(precision(true_positive, false_positive))
		recalls.append(recall(true_positive, false_negative))

	# Recall is not necessarily monotonic when thresholds are sampled, so sort
	# points before applying the trapezoidal rule.
	order = np.argsort(recalls)
	auc = float(np.trapezoid(np.asarray(precisions)[order], np.asarray(recalls)[order]))

	figure, axes = plt.subplots()
	axes.step(recalls, precisions, where="post", label=f"PR AUC = {auc:.3f}")
	axes.plot(recalls, precisions, "o", markersize=3)
	axes.set_xlabel("Recall")
	axes.set_ylabel("Precision")
	axes.set_title("Precision-Recall Curve")
	axes.set_xlim(0.0, 1.0)
	axes.set_ylim(0.0, 1.05)
	axes.grid(True, alpha=0.3)
	axes.legend()
	figure.tight_layout()
	return figure, axes, auc


if __name__ == "__main__":
	y_true, y_scores = make_realistic_demo_samples()
	figure, axes, auc = plot_pr_auc(y_true, y_scores)
	figure.savefig(Path(__file__).with_name("pr_auc.jpg"), dpi=300, bbox_inches="tight")
	print(f"Realistic demo PR AUC: {auc:.3f}")
	plt.show()
