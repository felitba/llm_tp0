"""Shared helpers for binary classification curves built from thresholds."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from plots.plot_theme import (
	ACCENT,
	BASELINE,
	DASH,
	apply_theme,
	legend_top_left,
	series_colors,
	set_title,
)


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
) -> ThresholdCurve:
	"""Calculate curve points by varying the classification threshold."""
	labels, scores = _validate_binary_inputs(y_true, y_scores)
	thresholds = np.concatenate(([np.inf], np.sort(np.unique(scores))[::-1]))

	x_values = []
	y_values = []
	for threshold in thresholds:
		counts = _confusion_counts(labels, scores, threshold)
		x_values.append(x_metric(counts))
		y_values.append(y_metric(counts))

	x_array = np.asarray(x_values, dtype=float)
	y_array = np.asarray(y_values, dtype=float)
	return ThresholdCurve(
		thresholds=thresholds,
		x_values=x_array,
		y_values=y_array,
		auc=area_under_curve(x_array, y_array),
	)


def area_under_curve(x_values: np.ndarray, y_values: np.ndarray) -> float:
	"""Integrate a curve after sorting by the x-axis, ties broken by y.

	``np.lexsort`` and not ``np.argsort``: an ROC curve has vertical segments
	(a run of consecutive positives raises TPR while FPR stays put), so several
	points share an x. ``np.argsort`` defaults to an unstable quicksort, which
	is free to scramble the y order inside such a run; the trapezoid terms that
	cross into and out of the run then use the wrong heights and the area comes
	out wrong -- up to 0.023 against sklearn on random inputs. Sorting by
	(x, y) restores the curve's own order and matches sklearn to 1e-16.

	The reported scalars come from sklearn regardless (see plots/roc_auc.py and
	plots/pr_auc.py); this keeps the number printed in a figure legend equal to
	the number in the tables.
	"""
	order = np.lexsort((y_values, x_values))
	return float(np.trapezoid(y_values[order], x_values[order]))


def plot_threshold_curve(
	curve: ThresholdCurve,
	x_label: str,
	y_label: str,
	title: str,
	auc_label: str,
	step: bool = False,
	show_random_baseline: bool = False,
	baseline_y: float | None = None,
) -> tuple[plt.Figure, plt.Axes, float]:
	"""Plot a calculated threshold curve and return its figure, axes, and AUC.

	``show_random_baseline`` draws the ROC diagonal; ``baseline_y`` draws the
	no-skill line of a PR curve, which sits at the positive rate rather than at
	zero. Both go in the legend along with the AUC, which is where a reader of
	this kind of figure looks for them.
	"""
	apply_theme()
	figure, axes = _unit_square_figure(5.0)

	draw = axes.step if step else axes.plot
	draw(
		curve.x_values, curve.y_values,
		**({"where": "post"} if step else {}),
		color=ACCENT, label=f"{auc_label} = {curve.auc:.3f}",
	)
	_draw_baseline(axes, show_random_baseline, baseline_y)

	_style_curve_axes(axes, x_label, y_label, title)
	legend_top_left(axes)
	figure.tight_layout()
	return figure, axes, curve.auc


def plot_combined_threshold_curves(
	curves: Sequence[tuple[str, ThresholdCurve]],
	x_label: str,
	y_label: str,
	title: str,
	auc_label: str,
	step: bool = False,
	show_random_baseline: bool = False,
	baseline_y: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
	"""Plot several named curves on one axes, one color per name."""
	apply_theme()
	figure, axes = _unit_square_figure(5.6)
	colors = series_colors(len(curves))

	for index, (name, curve) in enumerate(curves):
		draw = axes.step if step else axes.plot
		draw(
			curve.x_values, curve.y_values,
			**({"where": "post"} if step else {}),
			color=colors[index], label=f"{name} ({auc_label} = {curve.auc:.3f})",
		)
	_draw_baseline(axes, show_random_baseline, baseline_y)

	_style_curve_axes(axes, x_label, y_label, title)
	# One column: config names are long, and two columns of them are wider than
	# the square they sit under.
	legend_top_left(axes, ncols=1)
	figure.tight_layout()
	return figure, axes


def _unit_square_figure(side: float) -> tuple[plt.Figure, plt.Axes]:
	"""ROC and PR both live on the unit square; drawing them wide distorts them.

	A square box also makes "above the diagonal" and "how far from the top-left
	corner" mean what the reader thinks they mean.
	"""
	figure, axes = plt.subplots(figsize=(side, side))
	axes.set_box_aspect(1)
	return figure, axes


def _draw_baseline(axes: plt.Axes, diagonal: bool, level: float | None) -> None:
	"""The no-skill reference: the diagonal for ROC, the positive rate for PR."""
	if diagonal:
		axes.plot(
			[0.0, 1.0], [0.0, 1.0],
			color=BASELINE, linestyle=DASH, linewidth=1.2, label="Chance",
		)
	if level is not None:
		axes.axhline(
			level, color=BASELINE, linestyle=DASH, linewidth=1.2,
			label=f"Chance ({level:.3f})",
		)


def _style_curve_axes(axes: plt.Axes, x_label: str, y_label: str, title: str) -> None:
	"""Axis limits and labels shared by the single-curve and combined figures."""
	axes.set_xlabel(x_label)
	axes.set_ylabel(y_label)
	set_title(axes, title)
	# A hair past the unit square: at exactly 1.0 the stroke is half-clipped by
	# the frame, which reads as the curve stopping short.
	axes.set_xlim(-0.015, 1.015)
	axes.set_ylim(-0.015, 1.015)
	ticks = np.linspace(0.0, 1.0, 6)
	axes.set_xticks(ticks)
	axes.set_yticks(ticks)
	# The theme's default grid is horizontal-only, which suits a time series. On
	# a unit square you read both coordinates off the same point, so both rule.
	axes.grid(True, axis="both")


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
