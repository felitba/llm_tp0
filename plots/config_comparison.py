"""Figures that put several runs on the same axes.

The per-run figures answer "how did this one do". These answer "which one is
better, and where does the difference come from" -- the question a config file
with more than one experiment is asking.

Three of them exist because a bare bar chart cannot express what the deck needs:

``plot_test_scores``  bars, but with the two floors drawn on them. A PR-AUC of
    0.70 means nothing to a listener until the chance level (the base rate,
    0.13) and the groupby(title_tag) baseline (0.600 on validation) are on the
    same axis.

``plot_sensitivity``  for the one-factor-at-a-time steps. Ten bars hide the
    trend; a line over the values of the factor that actually varied shows it,
    and shows whether the curve is flat -- which is itself the finding.

``plot_seed_spread``  mean and range over runs that differ only in seed. Without
    it no gap in this project can be defended against "isn't that just noise?".

``plot_score_matrix``  the same numbers as ``plot_test_scores`` pivoted back onto
    the two swept axes, for the matrix steps. Nine bars sorted by score shuffle
    the factor levels along one axis, so the question the matrix was run for --
    is there an interaction, or is the surface flat -- cannot be read off them.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from config.experiments import base_name
from metrics.run_results import RunResults, query_ids_for
from plots.pr_auc import pr_auc_score
from plots.plot_theme import (
	ACCENT, BASELINE, BODY, DASH, MUTED, ORANGE, SEQUENTIAL, SPLIT_COLORS, SURFACE,
	apply_theme, legend_top_left, sequential_text_color, series_colors, set_title,
)
from plots.roc_auc import roc_auc_score

# The groupby(title_tag) baseline, from scripts/eda_columns.py. Hard-coded on
# purpose: it is a property of the dataset, not of any run, and a figure that
# omits it invites reading 0.70 as good without a reference.
#
# CHANGED (2026-09-01): was one number, 0.679, drawn on validation and test
# alike. That value came from the old interpolated-trapezoid estimator; under
# Average Precision (sklearn, via plots/pr_auc.py) the same zero-parameter
# groupby scores 0.600 on validation and 0.634 on test. Every figure drawn
# before this date carries a baseline line that is too high by ~0.05-0.08, which
# is larger than most of the gaps the figures are asked to show. To undo:
# collapse these back to a single float and drop the `split` argument threaded
# through the plots below.
BASELINE_PR_AUC = {"validation": 0.6004, "test": 0.6342}
BASELINE_ROC_AUC = {"validation": 0.9495, "test": 0.9576}

# Where each split's numbers come from, and what the y axis should say.
SPLIT_LABELS = {"validation": "validación", "test": "test"}

# Display names for the x axis / legend, keyed by experiment name. Empty by
# default, so figures keep showing the config's own names -- which is what you
# want while iterating, because the label then matches what you type on the
# command line. Fill it to put readable labels on a deck figure without
# renaming the experiments (renaming them would orphan every run directory):
#
#     from plots import config_comparison as cc
#     cc.DISPLAY_NAMES.update({"cap_nl1_nh4": "1 capa · 4 cabezas"})
#
# scripts/*.py that build their own deck figures pass labels directly instead.
DISPLAY_NAMES: dict[str, str] = {}


def baseline_pr_auc(split: str = "test") -> float:
	return BASELINE_PR_AUC[split]


def _scores(run: RunResults, split: str) -> dict[str, float] | None:
	"""The run's PR-AUC and ROC-AUC on ``split``.

	Test comes from the stored ``test`` block, which is what the run reported.
	Validation is recomputed from the saved validation scores, because runs only
	persist the test metrics -- and recomputing keeps both splits on the same
	sklearn estimator whatever version trained the run (see
	scripts/reselect.py:per_epoch_val_metric for the same argument).
	"""
	if split == "test":
		return run.test if run.test and run.test.get("pr_auc") is not None else None
	stored = run.split_predictions.get(split)
	if stored is None or not len(stored[0]) or len(np.unique(stored[0])) < 2:
		return None
	labels, probs = np.asarray(stored[0], dtype=int), np.asarray(stored[1], dtype=float)
	return {"pr_auc": float(pr_auc_score(labels, probs)),
	        "roc_auc": float(roc_auc_score(labels, probs))}


def _predictions(run: RunResults, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""labels, probs and query ids of ``split``, for the bootstrap interval."""
	if split == "test":
		labels, probs = run.labels, run.probs
	else:
		labels, probs = run.split_predictions.get(split, (np.empty(0), np.empty(0)))
	queries = run.query_ids if split == "test" and len(run.query_ids) == len(labels) \
		else query_ids_for(run, split)
	return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float), queries


def _short(name: str, keep: int = 26) -> str:
	"""The label an arm shows on an axis: its DISPLAY_NAMES override, else its
	own name, truncated. An override is never truncated -- if you took the
	trouble to name it, the name is the length you wanted."""
	name = base_name(name)
	if name in DISPLAY_NAMES:
		return DISPLAY_NAMES[name]
	return name if len(name) <= keep else name[: keep - 1] + "…"


def seed_groups(runs: Sequence[RunResults]) -> list[list[RunResults]]:
	"""Runs that differ only in seed, grouped, in first-seen order.

	Grouping is by every config value except the seed, so a ``seeds`` batch
	collapses back to one entry per experiment without any naming convention;
	a batch without seeds yields one group per run and the figures look as before.

	CHANGED (2026-09-01): las claves que empiezan con ``_`` quedan afuera de la
	firma. Son documentacion (``_step``, ``_comment``, ``_selection``, ``_eje``,
	``_carry_forward``), no configuracion del modelo, y viajan dentro de cada
	run.json: editar un comentario del config despues de correr una semilla
	partia el brazo en dos series con el mismo nombre.
	"""
	groups: dict[tuple, list[RunResults]] = {}
	for run in runs:
		signature = tuple(sorted(
			(k, str(v)) for k, v in run.config.items()
			if k not in ("seed", "seeds", "experiments") and not k.startswith("_")
		))
		groups.setdefault(signature, []).append(run)
	return list(groups.values())


def _group_label(members: Sequence[RunResults]) -> str:
	# CHANGED (2026-09-01): un grupo de una sola corrida llevaba debajo "ep N", la
	# epoca de su checkpoint. Sale junto con el resto de los marcadores de
	# checkpoint: la regla de seleccion se explica en sus propias filminas.
	label = _short(members[0].name)
	return f"{label}\n({len(members)} semillas)" if len(members) > 1 else label


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
	runs: Sequence[RunResults], title: str | None = None,
	subtitle: str | None = None, split: str = "test",
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Bars per configuration on ``split``, with both floors and the seed range.

	``split="validation"`` is what the ablation steps are decided on: test is read
	once, at the end (docs/PROTOCOL.md), so a figure that ranks arms on test is
	answering a question the protocol has not asked yet.

	CHANGED (2026-09-01): also drew the per-arm query-bootstrap CI as a second,
	wider whisker. Removed -- nobody read it, and it was the weak statistic
	anyway: a MARGINAL interval per arm, whose overlap is a poor test of whether
	two arms differ. The comparison that decides ties is the PAIRED bootstrap of
	the difference (scripts/paired_bootstrap.py), which is reported in the tables
	and in docs/INFORME.md §8.2. The interval itself is still computed there and
	in metrics/final_table.py; only this figure stopped drawing it.
	"""
	title = title or f"Scores en {SPLIT_LABELS[split]} por configuración"
	# One bar per experiment: the mean over its seeds, with the min..max range as
	# a whisker. A single-seed batch has no whiskers and shows the epoch instead.
	scored = [r for r in runs if _scores(r, split) is not None]
	rows = []
	for members in seed_groups(scored):
		roc = np.array([_scores(m, split)["roc_auc"] for m in members])
		pr = np.array([_scores(m, split)["pr_auc"] for m in members])
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

	reference_labels = _predictions(scored[0], split)[0]
	base_rate = float(np.mean(reference_labels)) if len(reference_labels) else None
	if base_rate:
		_reference_line(axes, base_rate, f"azar (base rate)  {base_rate:.3f}")

	# Values on both series: with every ROC-AUC around 0.96 and every PR-AUC
	# between 0.67 and 0.75, the bar heights alone cannot be compared by eye on a
	# 0-1 axis, and the axis has to stay 0-1 for the two floors to mean anything.
	for x, value in zip(positions - width / 2, [r[1] for r in rows]):
		if value is not None:
			axes.annotate(f"{value:.3f}", xy=(x, value), xytext=(0, 2),
			              textcoords="offset points", ha="center", fontsize=7, color=BODY)
	for x, row in zip(positions + width / 2, rows):
		value = row[2]
		if value is None:
			continue
		# Cleared above the seed whisker, so the number never lands on its cap.
		top = value + (row[3][1] if row[3] is not None else 0.0)
		axes.annotate(f"{value:.3f}", xy=(x, top), xytext=(0, 4),
		              textcoords="offset points", ha="center", fontsize=7,
		              color=BODY, fontweight="bold")

	axes.set_xticks(positions)
	axes.set_xticklabels(names, rotation=30, ha="right")
	axes.set_ylabel(f"Score en {SPLIT_LABELS[split]}")
	axes.set_ylim(0, 1.02)
	set_title(axes, title)
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


# Above this many arms, drawing train as well doubles the lines into an
# unreadable tangle; at or below it the train/validation GAP is the whole point,
# because that gap is what overfitting looks like and a val-only curve can only
# hint at it. The per-run loss.jpg always shows both (plots/train_vs_val_error).
MAX_SERIES_WITH_TRAIN = 4

# Epochs skipped when auto-scaling the y axis: the untrained model's loss is not
# a quantity anyone reads off this figure, it just sets the scale badly.
BURN_IN = 2


def plot_metric_by_epoch(
	runs: Sequence[RunResults], metric_key: str, y_label: str, title: str,
	subtitle: str | None = None, y_limits: tuple[float, float] | None = None,
	show_train: bool | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""One curve per run, validation solid and (when few enough) train dashed.

	``show_train=None`` decides by arm count: with up to MAX_SERIES_WITH_TRAIN
	arms the train curve is drawn in the same colour as its validation curve but
	dashed, so the vertical distance between the pair reads directly as
	overfitting. Colour identifies the arm and line style identifies the split --
	never a second hue, which would collide with the arm palette.
	"""
	# One curve per experiment. With seeds, the line is the mean over seeds and the
	# band their min..max -- the full RANGE, not a standard deviation: with three
	# seeds an SD is estimated from two degrees of freedom and would imply a
	# precision the sample does not have, while min..max is exactly what was seen.
	train_key = metric_key.replace("val_", "train_", 1) if metric_key.startswith("val_") else None
	series = []
	for members in seed_groups([r for r in runs if r.epoch_metrics and metric_key in r.epoch_metrics[0]]):
		length = min(len(m.epoch_metrics) for m in members)
		epochs = [int(row["epoch"]) for row in members[0].epoch_metrics[:length]]
		matrix = np.array([[row[metric_key] for row in m.epoch_metrics[:length]] for m in members])
		has_train = bool(train_key) and train_key in members[0].epoch_metrics[0]
		train = np.array([[row[train_key] for row in m.epoch_metrics[:length]] for m in members]) \
			if has_train else None
		# CHANGED (2026-09-01): aca se calculaba la epoca del checkpoint de cada
		# semilla para marcarla sobre la curva. Los marcadores salieron de la
		# figura -- la regla de seleccion se explica en sus propias filminas.
		series.append((_short(members[0].name), epochs, matrix, len(members), train))
	if not series:
		return None
	if show_train is None:
		show_train = len(series) <= MAX_SERIES_WITH_TRAIN
	show_train = show_train and any(entry[4] is not None for entry in series)
	apply_theme()
	figure, axes = plt.subplots(figsize=(9.0, 4.4))
	for (name, epochs, matrix, count, train), color in zip(series, series_colors(len(series))):
		values = matrix.mean(axis=0)
		seed_note = "" if count == 1 else f" · {count} semillas"
		axes.plot(epochs, values, color=color, linewidth=1.2, label=name + seed_note)
		if show_train and train is not None:
			axes.plot(epochs, train.mean(axis=0), color=color, linestyle=(0, (5, 2)),
			          linewidth=1.0, alpha=0.7)
			# Train gets the same seed band as validation. Without it the dashed
			# line reads as a single run, or as a quantity with no seed variance,
			# and the whole point of the pair is comparing how far apart the two
			# splits are -- which is only honest if both carry their spread.
			# Fainter than the validation band so the two stay tellable apart
			# where they overlap.
			if count > 1:
				axes.fill_between(epochs, train.min(axis=0), train.max(axis=0),
				                  color=color, alpha=0.07, linewidth=0)
		if count > 1:
			axes.fill_between(epochs, matrix.min(axis=0), matrix.max(axis=0),
			                  color=color, alpha=0.12, linewidth=0)
	# CHANGED (2026-09-02): the "baseline title_tag" reference line is gone from
	# every figure -- it is a PR-AUC (Average Precision) number, so next to a
	# ROC-AUC series it read as a floor for the wrong metric. To restore it, draw
	# _reference_line(axes, baseline_pr_auc(split), ...) at the sites this
	# comment marks.
	floor_level = None
	# One key for the split, once, instead of doubling every arm's legend entry.
	# The axis carries both splits now, so a label that still says "Validation"
	# would be describing half the figure.
	if show_train:
		axes.plot([], [], color=BODY, linestyle="-", linewidth=1.2, label="validación")
		axes.plot([], [], color=BODY, linestyle=(0, (5, 2)), linewidth=1.0, label="entrenamiento")
		strip = lambda text: text.replace("Validation ", "").replace("validation ", "")
		y_label = strip(y_label)[:1].upper() + strip(y_label)[1:]
		title = strip(title)[:1].upper() + strip(title)[1:]
	axes.set_xlabel("Época")
	axes.set_ylabel(y_label)
	if y_limits:
		axes.set_ylim(*y_limits)
	else:
		# Focus on everything AFTER the first epoch. Epoch 1 is the untrained model
		# and is 2-3x the converged loss, so including it squeezes the whole
		# interesting range -- every checkpoint, every train/val gap -- into the
		# bottom tenth of the axis. The curves still run off the top edge, which is
		# what tells the reader the first epochs were cut.
		visible = [m[:, BURN_IN:] for _, _, m, _, _ in series if m.shape[1] > BURN_IN]
		if show_train:
			visible += [t[:, BURN_IN:] for *_, t in series if t is not None and t.shape[1] > BURN_IN]
		if visible:
			low = min(float(v.min()) for v in visible)
			high = max(float(v.max()) for v in visible)
			# A floor the figure draws must stay in view, or the reference the
			# reader is meant to judge against is off-screen.
			if floor_level is not None:
				low, high = min(low, floor_level), max(high, floor_level)
			pad = (high - low) * 0.10 or 0.01
			axes.set_ylim(low - pad, high + pad)
	# Both axes: reading a checkpoint off the curve means reading an EPOCH, and a
	# horizontal-only grid makes that a guess.
	axes.grid(True, axis="both", linewidth=0.6, alpha=0.5)
	set_title(axes, title)
	# Outside, stacked above the axes. CHANGED (2026-09-01): briefly drawn inside
	# at upper right, which is free on a decaying loss curve but is exactly where
	# a rising PR-AUC curve lives -- the legend covered the data it labelled. One
	# figure function draws both metrics, so the placement has to work for both.
	legend_top_left(axes, ncols=2, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_cost_vs_score(
	runs: Sequence[RunResults], title: str | None = None,
	subtitle: str | None = None, split: str = "test",
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Wall clock against PR-AUC. Turns "the text arm costs 20x for the same
	number" from an argument about sequence length into a point on a plot."""
	title = title or f"Costo vs. score ({SPLIT_LABELS[split]})"
	# One point per experiment: mean time and mean score over its seeds.
	rows = []
	eligible = [r for r in runs if r.duration_seconds and _scores(r, split) is not None]
	for members in seed_groups(eligible):
		rows.append((
			_short(members[0].name) + (f" ({len(members)} semillas)" if len(members) > 1 else ""),
			float(np.mean([m.duration_seconds for m in members])),
			float(np.mean([_scores(m, split)["pr_auc"] for m in members])),
		))
	if len(rows) < 2:
		return None
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.2, 4.2))
	for (name, seconds, score), color in zip(rows, series_colors(len(rows))):
		axes.scatter(seconds / 60.0, score, color=color, s=44, label=name, zorder=3)
	axes.set_xlabel("Tiempo de entrenamiento (minutos)")
	axes.set_ylabel(f"PR-AUC en {SPLIT_LABELS[split]}")
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
	split: str = "test",
) -> tuple[plt.Figure, plt.Axes] | None:
	key = key or varying_key(runs)
	if key is None:
		return None
	points = sorted(
		(float(r.config[key]), _scores(r, split)["pr_auc"], _scores(r, split)["roc_auc"])
		for r in runs if key in r.config and _scores(r, split) is not None
	)
	if len(points) < 2:
		return None
	xs = [p[0] for p in points]
	where = SPLIT_LABELS[split]
	apply_theme()
	figure, axes = plt.subplots(figsize=(5.8, 4))
	axes.plot(xs, [p[1] for p in points], color=ACCENT, marker="o", label=f"PR-AUC en {where}")
	axes.plot(xs, [p[2] for p in points], color=ORANGE, marker="s", label=f"ROC-AUC en {where}")
	axes.set_xlabel(key)
	axes.set_ylabel(f"Score en {where}")
	set_title(axes, f"Sensibilidad a {key} ({SPLIT_LABELS[split]})")
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes


def plot_score_matrix(
	runs: Sequence[RunResults], subtitle: str | None = None, split: str = "test",
) -> tuple[plt.Figure, np.ndarray] | None:
	"""Heatmap per swept pair: one cell per configuration, mean over its seeds.

	Only draws when the seed groups differ in exactly two SWEEP_KEYS -- the
	matrix steps -- and returns None for every other batch, so wiring it next to
	the bar figure costs the one-factor steps nothing. Each panel normalises its
	colour ramp to its own observed range: with every cell inside the bootstrap
	CI the ramp would otherwise be a single flat tint, and the point of the
	figure is to let rows, columns and diagonals be compared, with the printed
	mean and seed range keeping the tiny absolute spread honest.
	"""
	scored = [r for r in runs if _scores(r, split) is not None]
	groups = seed_groups(scored)
	keys = [k for k in SWEEP_KEYS if len({g[0].config.get(k) for g in groups}) > 1]
	if len(keys) != 2 or len(groups) < 4:
		return None
	row_key, col_key = keys
	row_values = sorted({g[0].config[row_key] for g in groups})
	col_values = sorted({g[0].config[col_key] for g in groups})

	cells: dict[str, np.ndarray] = {
		metric: np.full((len(row_values), len(col_values)), np.nan)
		for metric in ("pr_auc", "roc_auc")
	}
	ranges: dict[tuple[int, int], tuple[dict[str, tuple[float, float]], int]] = {}
	for members in groups:
		i = row_values.index(members[0].config[row_key])
		j = col_values.index(members[0].config[col_key])
		spread = {}
		for metric in cells:
			values = np.array([_scores(m, split)[metric] for m in members])
			cells[metric][i, j] = values.mean()
			spread[metric] = (float(values.min()), float(values.max()))
		ranges[(i, j)] = (spread, len(members))

	seed_counts = {count for _, count in ranges.values()}
	seed_note = (
		f"media de {seed_counts.pop()} semillas" if len(seed_counts) == 1
		else "media entre semillas"
	)

	apply_theme()
	figure, axes_pair = plt.subplots(1, 2, figsize=(9.0, 4.1))
	for axes, (metric, label) in zip(axes_pair, (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))):
		matrix = np.ma.masked_invalid(cells[metric])
		low, high = float(matrix.min()), float(matrix.max())
		span = (high - low) or 1e-9
		mesh = axes.pcolormesh(matrix, cmap=SEQUENTIAL, vmin=low, vmax=high,
		                       edgecolors=SURFACE, linewidth=2)
		for (i, j), (spread, count) in ranges.items():
			value = cells[metric][i, j]
			ink = sequential_text_color((value - low) / span)
			# Mean above, seed range below: the range is what keeps a cell that
			# LOOKS darker from being read as a win the seeds do not support.
			axes.text(j + 0.5, i + 0.5 + (0.11 if count > 1 else 0.0), f"{value:.3f}",
			          ha="center", va="center", fontsize=9.5, fontweight="bold", color=ink)
			if count > 1:
				lo, hi = spread[metric]
				axes.text(j + 0.5, i + 0.30, f"{lo:.3f}–{hi:.3f}",
				          ha="center", va="center", fontsize=6.5, color=ink)
		axes.set_xticks(np.arange(len(col_values)) + 0.5)
		axes.set_xticklabels([f"{v:g}" for v in col_values])
		axes.set_yticks(np.arange(len(row_values)) + 0.5)
		axes.set_yticklabels([f"{v:g}" for v in row_values])
		axes.set_xlabel(col_key)
		axes.set_ylabel(row_key)
		axes.set_aspect("equal")
		axes.grid(False)
		for spine in axes.spines.values():
			spine.set_visible(False)
		axes.tick_params(length=0)
		set_title(axes, label)
		bar = figure.colorbar(mesh, ax=axes, fraction=0.046, pad=0.05, ticks=[low, high])
		bar.ax.set_yticklabels([f"{low:.3f}", f"{high:.3f}"])
		bar.ax.tick_params(length=0, labelsize=7)
		bar.outline.set_visible(False)
	figure.tight_layout()
	# Above the axes, like legend_top_left does for the single-ax figures; that
	# helper anchors to one ax, so a two-panel figure titles itself here instead.
	figure.suptitle(
		f"Scores en {SPLIT_LABELS[split]}: {row_key} × {col_key} ({seed_note})",
		y=1.06,
	)
	if subtitle:
		figure.text(0.5, 0.99, subtitle, ha="center", va="bottom",
		            fontsize=plt.rcParams["font.size"] - 1, color=MUTED)
	return figure, axes_pair


def plot_seed_spread(
	runs: Sequence[RunResults], subtitle: str | None = None, split: str = "test",
) -> tuple[plt.Figure, plt.Axes] | None:
	"""Mean and full range of PR-AUC on ``split`` over runs differing only in seed.

	Grouping is by every config value except the seed, so runs land together
	exactly when the seed is the only thing separating them -- no naming
	convention required.
	"""
	scored = [r for r in runs if _scores(r, split) is not None]
	groups = [g for g in seed_groups(scored) if len(g) > 1]
	if not groups:
		return None

	entries = []
	for members in groups:
		scores = np.array([_scores(m, split)["pr_auc"] for m in members])
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
	axes.set_xticks(positions)
	axes.set_xticklabels([f"{_short(e[0], 20)}\n({e[4]} semillas)" for e in entries],
	                     rotation=20, ha="right")
	axes.set_ylabel(f"PR-AUC en {SPLIT_LABELS[split]}")
	set_title(axes, f"Variación entre semillas ({SPLIT_LABELS[split]})")
	legend_top_left(axes, subtitle=subtitle)
	figure.tight_layout()
	return figure, axes
