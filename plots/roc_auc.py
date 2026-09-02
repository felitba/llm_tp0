"""ROC curve calculated over model-score thresholds."""

from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from config.config import PROJECT_ROOT

from metrics.metrics import fall_out, recall
from plots.plot_theme import save
from plots.threshold_curves import (
	ConfusionCounts,
	ThresholdCurve,
	calculate_threshold_curve,
	make_realistic_demo_samples,
	plot_combined_threshold_curves,
	plot_threshold_curve,
)


def _false_positive_rate_from_counts(counts: ConfusionCounts) -> float:
	return fall_out(counts.true_negative, counts.false_positive)


def _true_positive_rate_from_counts(counts: ConfusionCounts) -> float:
	return recall(counts.true_positive, counts.false_negative)


def roc_curve(y_true: Sequence[int], y_scores: Sequence[float]) -> ThresholdCurve:
	"""Return the threshold-based ROC curve points and its AUC."""
	return calculate_threshold_curve(
		y_true=y_true,
		y_scores=y_scores,
		x_metric=_false_positive_rate_from_counts,
		y_metric=_true_positive_rate_from_counts,
	)


def roc_auc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
	"""ROC AUC, from sklearn. The single reported ROC number.

	Delegated rather than integrated from ``roc_curve`` so the reported metric
	is the reference implementation and needs no defending in the report. The
	hand-built curve is still what gets drawn; its area now agrees with this
	(see ``plots.threshold_curves.area_under_curve``).

	Trapezoidal integration is correct in ROC space -- unlike PR space, linear
	interpolation between ROC points is achievable (Davis & Goadrich 2006), so
	the estimator was never the problem here; the sort order was.
	"""
	return float(sk_roc_auc_score(np.asarray(y_true, dtype=int), np.asarray(y_scores, dtype=float)))


def plot_roc_auc(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot and return a ROC curve and its area under the curve.

	``y_true`` contains binary labels and ``y_scores`` contains the score or
	probability used for thresholding.
	"""
	# Legend shows the sklearn number, so figure and tables cannot disagree.
	curve = replace(roc_curve(y_true, y_scores), auc=roc_auc_score(y_true, y_scores))
	return plot_threshold_curve(
		curve=curve,
		x_label="Tasa de falsos positivos",
		y_label="Tasa de verdaderos positivos",
		title="Curva ROC",
		auc_label="ROC AUC",
		show_random_baseline=True,
	)


def plot_roc_auc_by_config(
	results_by_config: Sequence[tuple[str, Sequence[int], Sequence[float]]],
) -> tuple[plt.Figure, plt.Axes]:
	"""Plot every config's ROC curve on one figure, one color per config.

	Each entry is ``(config_name, y_true, y_scores)``. Curves are drawn from
	best to worst AUC so the legend reads as a ranking.
	"""
	curves = [
		(name, replace(roc_curve(y_true, y_scores), auc=roc_auc_score(y_true, y_scores)))
		for name, y_true, y_scores in results_by_config
	]
	curves.sort(key=lambda item: item[1].auc, reverse=True)
	return plot_combined_threshold_curves(
		curves=curves,
		x_label="Tasa de falsos positivos",
		y_label="Tasa de verdaderos positivos",
		title="Curva ROC por configuración",
		auc_label="AUC",
		show_random_baseline=True,
	)


if __name__ == "__main__":
	y_true, y_scores = make_realistic_demo_samples()
	figure, axes, auc = plot_roc_auc(y_true, y_scores)
	# save() closes the figure, so the demo writes the file instead of blocking
	# on a window; open output/demo/roc_auc.jpg to look at it.
	print(f"Realistic demo ROC AUC: {auc:.3f}")
	print(f"Wrote: {save(figure, PROJECT_ROOT / 'output' / 'demo' / 'roc_auc.jpg')}")
