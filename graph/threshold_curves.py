"""Shared helpers for binary classification curves built from thresholds."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ThresholdCurve:
	thresholds: np.ndarray
	x_values: np.ndarray
	y_values: np.ndarray
	auc: float


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
	# One marker per point was readable on the old 101-point grid. A real curve
	# has one point per distinct score, so markers would bury the line.
	if len(curve.x_values) <= 100:
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
