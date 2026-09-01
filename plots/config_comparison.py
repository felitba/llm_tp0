"""Figures that put several runs on the same axes.

The per-run figures answer "how did this one do". These answer "which one is
better, and where does the difference come from" -- the question a config file
with more than one experiment is asking.

Three of them exist because a bare bar chart cannot express what the deck needs:

``plot_test_scores``  bars, but with the two floors drawn on them. A PR-AUC of
    0.70 means nothing to a listener until the chance level (the base rate,
    0.13) and the groupby(title_tag) baseline (0.679) are on the same axis.

``plot_sensitivity``  for the one-factor-at-a-time steps. Ten bars hide the
    trend; a line over the values of the factor that actually varied shows it,
    and shows whether the curve is flat -- which is itself the finding.

``plot_seed_spread``  mean and range over runs that differ only in seed. Without
    it no gap in this project can be defended against "isn't that just noise?".
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from config.experiments import base_name
from metrics.run_results import RunResults
from plots.plot_theme import (
	ACCENT, BASELINE, BODY, DASH, MUTED, ORANGE, SPLIT_COLORS, SURFACE,
	apply_theme, legend_top_left, series_colors, set_title,
)

# The groupby(title_tag) baseline, from scripts/eda_columns.py. Hard-coded on
# purpose: it is a property of the dataset, not of any run, and a figure that
# omits it invites reading 0.70 as good without a reference.
BASELINE_PR_AUC = 0.679
BASELINE_ROC_AUC = 0.958


def _short(name: str, keep: int = 26) -> str:
	name = base_name(name)
	return name if len(name) <= keep else name[: keep - 1] + "…"


def seed_groups(runs: Sequence[RunResults]) -> list[list[RunResults]]:
	"""Runs that differ only in seed, grouped, in first-seen order.

	Grouping is by every config value except the seed, so a ``seeds`` batch
	collapses back to one entry per experiment without any naming convention;
	a batch without seeds yields one group per run and the figures look as before.
	"""
	groups: dict[tuple, list[RunResults]] = {}
	for run in runs:
		signature = tuple(sorted(
			(k, str(v)) for k, v in run.config.items() if k not in ("seed", "seeds", "experiments")
		))
		groups.setdefault(signature, []).append(run)
	return list(groups.values())


def _group_label(members: Sequence[RunResults]) -> str:
	label = _short(members[0].name)
	if len(members) > 1:
		return f"{label}\n({len(members)} semillas)"
	selected = (members[0].selection or {}).get("epoch")
	return f"{label}\nep {selected}" if selected else label


def _reference_line(axes, level: float, text: str) -> None:
	"""A dashed horizontal reference with its label pinned to the left margin.

	Left, not right: the bars are sorted best-first, so the right side is where
	the low bars and their value labels are, and a label anchored there lands on
	top of them. The opaque backing is what keeps the text off the gridline.
	"""
	axes.axhline(level, color=BASELINE, linestyle=DASH, linewidth=1)
	axes.annotate(
		text, xy=(0.004, level), xycoords=axes.get_yaxis_transform(),
		xytext=(0, 3), textcoords="offset points",
		ha="left", va="bottom", fontsize=7.5, color=MUTED,
		bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8),
	)


def plot_test_scores(
	runs: Sequence[RunResults], title: str = "Test scores by configuration",
	subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	# One bar per experiment: the mean over its seeds, with the min..max range as
	# a whisker. A single-seed batch has no whiskers and shows the epoch instead.
	rows = []
	for members in seed_groups([r for r in runs if r.test and r.test.get("pr_auc") is not None]):
		roc = np.array([m.test["roc_auc"] for m in members])
		pr = np.array([m.test["pr_auc"] for m in members])
		rows.append((_group_label(members), roc.mean(), pr.mean(),
		             (pr.mean() - pr.min(), pr.max() - pr.mean()) if len(members) > 1 else None,
		             (roc.mean() - roc.min(), roc.max() - roc.mean()) if len(members) > 1 else None))
	if not rows:
		return None
	rows.sort(key=lambda row: row[2], reverse=True)
	names = [r[0] for r in rows]
	positions = np.arange(len(rows))

	apply_theme()
	figure, axes = plt.subplots(figsize=(max(6.2, 1.15 * len(rows)), 4.4))
	width = 0.38
	axes.bar(positions - width / 2, [r[1] for r in rows], width, color=ACCENT, label="ROC-AUC")
	axes.bar(positions + width / 2, [r[2] for r in rows], width, color=ORANGE, label="PR-AUC")
	for x, row in zip(positions, rows):
		if row[3] is not None:
			axes.errorbar(x + width / 2, row[2], yerr=[[row[3][0]], [row[3][1]]],
			              fmt="none", ecolor=BODY, elinewidth=1.1, capsize=3, zorder=3)
			axes.errorbar(x - width / 2, row[1], yerr=[[row[4][0]], [row[4][1]]],
			              fmt="none", ecolor=BODY, elinewidth=1.1, capsize=3, zorder=3)

	base_rate = float(np.mean(runs[0].labels)) if len(runs[0].labels) else None
	_reference_line(axes, BASELINE_PR_AUC, f"baseline title_tag  {BASELINE_PR_AUC:.3f}")
	if base_rate:
		_reference_line(axes, base_rate, f"azar (base rate)  {base_rate:.3f}")

	# Values on both series: with every ROC-AUC around 0.96 and every PR-AUC
	# between 0.67 and 0.75, the bar heights alone cannot be compared by eye on a
	# 0-1 axis, and the axis has to stay 0-1 for the two floors to mean anything.
	for x, value in zip(positions - width / 2, [r[1] for r in rows]):
		if value is not None:
			axes.annotate(f"{value:.3f}", xy=(x, value), xytext=(0, 2),
			              textcoords="offset points", ha="center", fontsize=7, color=BODY)
	for x, value in zip(positions + width / 2, [r[2] for r in rows]):
		if value is not None:
			axes.annotate(f"{value:.3f}", xy=(x, value), xytext=(0, 2),
			              textcoords="offset points", ha="center", fontsize=7,
			              color=BODY, fontweight="bold")

	axes.set_xticks(positions)
	axes.set_xticklabels(names, rotation=30, ha="right")
	axes.set_ylabel("Score en test")
	axes.set_ylim(0, 1.02)
	set_title(axes, title)
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_metric_by_epoch(
	runs: Sequence[RunResults], metric_key: str, y_label: str, title: str,
	subtitle: str | None = None, y_limits: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""One validation curve per run. Validation only: two lines per run is
	unreadable past two runs, and validation is what the comparison turns on."""
	# One curve per experiment. With seeds, the line is the mean over seeds and
	# the band its min..max; a lone run keeps its selected-epoch marker.
	series = []
	for members in seed_groups([r for r in runs if r.epoch_metrics and metric_key in r.epoch_metrics[0]]):
		length = min(len(m.epoch_metrics) for m in members)
		epochs = [int(row["epoch"]) for row in members[0].epoch_metrics[:length]]
		matrix = np.array([[row[metric_key] for row in m.epoch_metrics[:length]] for m in members])
		selected = (members[0].selection or {}).get("epoch") if len(members) == 1 else None
		series.append((_short(members[0].name), epochs, matrix, selected, len(members)))
	if not series:
		return None
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.6, 4.2))
	for (name, epochs, matrix, selected, count), color in zip(series, series_colors(len(series))):
		values = matrix.mean(axis=0)
		label = name + (f" · ep {selected}" if selected else "") + (f" · {count} semillas" if count > 1 else "")
		axes.plot(epochs, values, color=color, label=label)
		if count > 1:
			axes.fill_between(epochs, matrix.min(axis=0), matrix.max(axis=0), color=color, alpha=0.15, linewidth=0)
		# The checkpoint the test row reports, marked on the curve it was read from.
		if isinstance(selected, int) and selected in epochs:
			axes.plot(
				[selected], [values[epochs.index(selected)]],
				marker="o", markersize=5.5, color=color,
				markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle="none",
			)
	# A PR-AUC axis without the floor invites reading 0.70 as good on its own.
	# A loss axis has no such reference, hence the guard.
	if "pr_auc" in metric_key:
		_reference_line(axes, BASELINE_PR_AUC, f"baseline title_tag  {BASELINE_PR_AUC:.3f}")
	axes.set_xlabel("Epoch")
	axes.set_ylabel(y_label)
	if y_limits:
		axes.set_ylim(*y_limits)
	set_title(axes, title)
	legend_top_left(axes, ncols=2, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_cost_vs_score(
	runs: Sequence[RunResults], title: str = "Cost vs. score",
	subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Wall clock against test PR-AUC. Turns "the text arm costs 20x for the same
	number" from an argument about sequence length into a point on a plot."""
	# One point per experiment: mean time and mean score over its seeds.
	rows = []
	for members in seed_groups([r for r in runs if r.duration_seconds and r.test.get("pr_auc") is not None]):
		rows.append((
			_short(members[0].name) + (f" ({len(members)} semillas)" if len(members) > 1 else ""),
			float(np.mean([m.duration_seconds for m in members])),
			float(np.mean([m.test["pr_auc"] for m in members])),
		))
	if len(rows) < 2:
		return None
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.2, 4.2))
	for (name, seconds, score), color in zip(rows, series_colors(len(rows))):
		axes.scatter(seconds / 60.0, score, color=color, s=44, label=name, zorder=3)
	axes.set_xlabel("Tiempo de entrenamiento (minutos)")
	axes.set_ylabel("Test PR-AUC")
	_reference_line(axes, BASELINE_PR_AUC, f"baseline title_tag  {BASELINE_PR_AUC:.3f}")
	set_title(axes, title)
	# Two columns: one row of six long run names is wider than the figure, which
	# is what makes tight_layout give up on the margins.
	legend_top_left(axes, ncols=2, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


# Hyperparameters a one-factor-at-a-time step is allowed to vary. Order fixes
# which one gets the x axis when a config happens to move more than one.
SWEEP_KEYS = ("d_model", "num_layers", "n_heads", "dim_feedforward",
              "dropout", "weight_decay", "learning_rate")


def varying_key(runs: Sequence[RunResults]) -> str | None:
	"""The single hyperparameter these runs disagree on, if there is exactly one."""
	varying = [k for k in SWEEP_KEYS if len({r.config.get(k) for r in runs}) > 1]
	return varying[0] if len(varying) == 1 else None


def plot_sensitivity(
	runs: Sequence[RunResults], key: str | None = None, subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	key = key or varying_key(runs)
	if key is None:
		return None
	points = sorted(
		(float(r.config[key]), r.test.get("pr_auc"), r.test.get("roc_auc"))
		for r in runs if key in r.config and r.test.get("pr_auc") is not None
	)
	if len(points) < 2:
		return None
	xs = [p[0] for p in points]
	apply_theme()
	figure, axes = plt.subplots(figsize=(5.8, 4))
	axes.plot(xs, [p[1] for p in points], color=ACCENT, marker="o", label="Test PR-AUC")
	axes.plot(xs, [p[2] for p in points], color=ORANGE, marker="s", label="Test ROC-AUC")
	axes.axhline(BASELINE_PR_AUC, color=BASELINE, linestyle=DASH, linewidth=1)
	axes.annotate("baseline title_tag", xy=(xs[-1], BASELINE_PR_AUC), xytext=(-2, 3),
	              textcoords="offset points", ha="right", fontsize=8, color=MUTED)
	axes.set_xlabel(key)
	axes.set_ylabel("Score en test")
	set_title(axes, f"Sensibilidad a {key}")
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_seed_spread(
	runs: Sequence[RunResults], subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Mean and full range of test PR-AUC over runs that differ only in seed.

	Grouping is by every config value except the seed, so runs land together
	exactly when the seed is the only thing separating them -- no naming
	convention required.
	"""
	groups = [g for g in seed_groups([r for r in runs if r.test.get("pr_auc") is not None]) if len(g) > 1]
	if not groups:
		return None

	entries = []
	for members in groups:
		scores = np.array([m.test["pr_auc"] for m in members])
		entries.append((base_name(members[0].name), scores.mean(), scores.min(), scores.max(), len(scores)))
	entries.sort(key=lambda e: e[1], reverse=True)

	apply_theme()
	figure, axes = plt.subplots(figsize=(max(5.6, 1.3 * len(entries)), 4.2))
	positions = np.arange(len(entries))
	means = [e[1] for e in entries]
	lower = [e[1] - e[2] for e in entries]
	upper = [e[3] - e[1] for e in entries]
	axes.errorbar(positions, means, yerr=[lower, upper], fmt="o", color=ACCENT,
	              capsize=5, linewidth=1.4, label="media y rango entre semillas")
	_reference_line(axes, BASELINE_PR_AUC, f"baseline title_tag  {BASELINE_PR_AUC:.3f}")
	axes.set_xticks(positions)
	axes.set_xticklabels([f"{_short(e[0], 20)}\n({e[4]} semillas)" for e in entries],
	                     rotation=20, ha="right")
	axes.set_ylabel("Test PR-AUC")
	set_title(axes, "Variación entre semillas")
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes
