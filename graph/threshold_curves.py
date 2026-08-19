"""Shared helpers for binary classification curves built from thresholds."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
	true_positive: int
	false_positive: int
	true_negative: int
	false_negative: int


@dataclass(frozen=True)
class ThresholdCurve:
	thresholds: np.ndarray
	x_values: np.ndarray
	y_values: np.ndarray
	auc: float


MetricFromCounts = Callable[[ConfusionCounts], float]


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


def calculate_threshold_curve(
	y_true: Sequence[int],
	y_scores: Sequence[float],
	x_metric: MetricFromCounts,
	y_metric: MetricFromCounts,
	thresholds: Sequence[float] | None = None,
) -> ThresholdCurve:
	"""Calculate curve points by varying the classification threshold."""
	labels, scores = _validate_binary_inputs(y_true, y_scores)
	if thresholds is None:
		threshold_values = np.linspace(1.0, 0.0, 101)
	else:
		threshold_values = np.asarray(thresholds, dtype=float)

	x_values = []
	y_values = []
	for threshold in threshold_values:
		counts = _confusion_counts(labels, scores, threshold)
		x_values.append(x_metric(counts))
		y_values.append(y_metric(counts))

	x_array = np.asarray(x_values, dtype=float)
	y_array = np.asarray(y_values, dtype=float)
	return ThresholdCurve(
		thresholds=threshold_values,
		x_values=x_array,
		y_values=y_array,
		auc=_area_under_curve(x_array, y_array),
	)


def plot_threshold_curve(
	curve: ThresholdCurve,
	x_label: str,
	y_label: str,
	title: str,
	auc_label: str,
	step: bool = False,
	show_random_baseline: bool = False,
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot a calculated threshold curve and return its figure, axes, and AUC."""
	figure, axes = plt.subplots()
	if step:
		axes.step(
			curve.x_values,
			curve.y_values,
			where="post",
			label=f"{auc_label} = {curve.auc:.3f}",
		)
	else:
		axes.plot(curve.x_values, curve.y_values, label=f"{auc_label} = {curve.auc:.3f}")
	axes.plot(curve.x_values, curve.y_values, "o", markersize=3)

	if show_random_baseline:
		axes.plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", label="Random baseline")

	axes.set_xlabel(x_label)
	axes.set_ylabel(y_label)
	axes.set_title(title)
	axes.set_xlim(0.0, 1.0)
	axes.set_ylim(0.0, 1.05)
	axes.grid(True, alpha=0.3)
	axes.legend()
	figure.tight_layout()
	return figure, axes, curve.auc


def _validate_binary_inputs(
	y_true: Sequence[int], y_scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
	if len(y_true) != len(y_scores):
		raise ValueError("y_true and y_scores must have the same length")

	labels = np.asarray(y_true)
	if not np.isin(labels, [0, 1]).all():
		raise ValueError("y_true must contain only binary labels: 0 and 1")

	return labels, np.asarray(y_scores, dtype=float)


def _confusion_counts(
	labels: np.ndarray, scores: np.ndarray, threshold: float
) -> ConfusionCounts:
	predicted = scores >= threshold
	positive_labels = labels == 1
	negative_labels = labels == 0
	return ConfusionCounts(
		true_positive=int(np.sum(predicted & positive_labels)),
		false_positive=int(np.sum(predicted & negative_labels)),
		true_negative=int(np.sum(~predicted & negative_labels)),
		false_negative=int(np.sum(~predicted & positive_labels)),
	)


def _area_under_curve(x_values: np.ndarray, y_values: np.ndarray) -> float:
	order = np.argsort(x_values)
	return float(np.trapezoid(y_values[order], x_values[order]))
