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

from metrics.run_results import RunResults
from plots.plot_theme import (
	ACCENT, BASELINE, BODY, DASH, MUTED, ORANGE, SPLIT_COLORS,
	apply_theme, legend_top_left, series_colors, set_title,
)

# The groupby(title_tag) baseline, from scripts/eda_columns.py. Hard-coded on
# purpose: it is a property of the dataset, not of any run, and a figure that
# omits it invites reading 0.70 as good without a reference.
BASELINE_PR_AUC = 0.679
BASELINE_ROC_AUC = 0.958


def _short(name: str, keep: int = 26) -> str:
	return name if len(name) <= keep else name[: keep - 1] + "…"


def plot_test_scores(
	runs: Sequence[RunResults], title: str = "Test scores by configuration",
	subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	rows = [(r.name, r.test.get("roc_auc"), r.test.get("pr_auc")) for r in runs if r.test]
	if not rows:
		return None
	rows.sort(key=lambda row: (row[2] is None, row[2]), reverse=True)
	names = [_short(r[0]) for r in rows]
	positions = np.arange(len(rows))

	apply_theme()
	figure, axes = plt.subplots(figsize=(max(6.2, 1.15 * len(rows)), 4.4))
	width = 0.38
	axes.bar(positions - width / 2, [r[1] for r in rows], width, color=ACCENT, label="ROC-AUC")
	axes.bar(positions + width / 2, [r[2] for r in rows], width, color=ORANGE, label="PR-AUC")

	base_rate = float(np.mean(runs[0].labels)) if len(runs[0].labels) else None
	for level, text in (
		(BASELINE_PR_AUC, f"baseline title_tag PR-AUC {BASELINE_PR_AUC:.3f}"),
		(base_rate, f"azar PR-AUC {base_rate:.3f}" if base_rate else None),
	):
		if level is None or text is None:
			continue
		axes.axhline(level, color=BASELINE, linestyle=DASH, linewidth=1)
		axes.annotate(text, xy=(len(rows) - 0.5, level), xytext=(0, 3),
		              textcoords="offset points", ha="right", fontsize=8, color=MUTED)

	for x, value in zip(positions + width / 2, [r[2] for r in rows]):
		if value is not None:
			axes.annotate(f"{value:.3f}", xy=(x, value), xytext=(0, 2),
			              textcoords="offset points", ha="center", fontsize=7, color=BODY)

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
	series = [
		(r.name, [row["epoch"] for row in r.epoch_metrics], [row[metric_key] for row in r.epoch_metrics])
		for r in runs
		if r.epoch_metrics and metric_key in r.epoch_metrics[0]
	]
	if not series:
		return None
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.6, 4.2))
	for (name, epochs, values), color in zip(series, series_colors(len(series))):
		axes.plot(epochs, values, color=color, label=_short(name))
	# A PR-AUC axis without the floor invites reading 0.70 as good on its own.
	# A loss axis has no such reference, hence the guard.
	if "pr_auc" in metric_key:
		axes.axhline(BASELINE_PR_AUC, color=BASELINE, linestyle=DASH, linewidth=1)
		axes.annotate(
			f"baseline title_tag {BASELINE_PR_AUC:.3f}",
			xy=(axes.get_xlim()[1], BASELINE_PR_AUC), xytext=(-2, 3),
			textcoords="offset points", ha="right", fontsize=8, color=MUTED,
		)
	axes.set_xlabel("Epoch")
	axes.set_ylabel(y_label)
	if y_limits:
		axes.set_ylim(*y_limits)
	set_title(axes, title)
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_cost_vs_score(
	runs: Sequence[RunResults], title: str = "Cost vs. score",
	subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Wall clock against test PR-AUC. Turns "the text arm costs 20x for the same
	number" from an argument about sequence length into a point on a plot."""
	rows = [(r.name, r.duration_seconds, r.test.get("pr_auc")) for r in runs
	        if r.duration_seconds and r.test.get("pr_auc") is not None]
	if len(rows) < 2:
		return None
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.2, 4.2))
	for (name, seconds, score), color in zip(rows, series_colors(len(rows))):
		axes.scatter(seconds / 60.0, score, color=color, s=44, label=_short(name), zorder=3)
	axes.set_xlabel("Tiempo de entrenamiento (minutos)")
	axes.set_ylabel("Test PR-AUC")
	axes.axhline(BASELINE_PR_AUC, color=BASELINE, linestyle=DASH, linewidth=1)
	axes.annotate("baseline title_tag", xy=(axes.get_xlim()[1], BASELINE_PR_AUC),
	              xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
	set_title(axes, title)
	legend_top_left(axes, subtitle=subtitle)
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
	groups: dict[tuple, list[RunResults]] = {}
	for run in runs:
		if run.test.get("pr_auc") is None:
			continue
		signature = tuple(sorted(
			(k, str(v)) for k, v in run.config.items() if k not in ("seed", "experiments")
		))
		groups.setdefault(signature, []).append(run)
	groups = {k: v for k, v in groups.items() if len(v) > 1}
	if not groups:
		return None

	entries = []
	for members in groups.values():
		scores = np.array([m.test["pr_auc"] for m in members])
		entries.append((members[0].name, scores.mean(), scores.min(), scores.max(), len(scores)))
	entries.sort(key=lambda e: e[1], reverse=True)

	apply_theme()
	figure, axes = plt.subplots(figsize=(max(5.6, 1.3 * len(entries)), 4.2))
	positions = np.arange(len(entries))
	means = [e[1] for e in entries]
	lower = [e[1] - e[2] for e in entries]
	upper = [e[3] - e[1] for e in entries]
	axes.errorbar(positions, means, yerr=[lower, upper], fmt="o", color=ACCENT,
	              capsize=5, linewidth=1.4, label="media y rango entre semillas")
	axes.axhline(BASELINE_PR_AUC, color=BASELINE, linestyle=DASH, linewidth=1)
	axes.annotate("baseline title_tag", xy=(positions[-1], BASELINE_PR_AUC), xytext=(0, 3),
	              textcoords="offset points", ha="right", fontsize=8, color=MUTED)
	axes.set_xticks(positions)
	axes.set_xticklabels([f"{_short(e[0], 20)}\n({e[4]} semillas)" for e in entries],
	                     rotation=20, ha="right")
	axes.set_ylabel("Test PR-AUC")
	set_title(axes, "Variación entre semillas")
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes
