"""Do the probabilities mean what they say, and where do the errors live.

Three figures that read the raw scores rather than an aggregate:

``plot_reliability``  BCEWithLogitsLoss is a proper scoring rule, so training
    aims at calibrated probabilities and not only at the right order. Nothing in
    the report checks that until this figure does: bin the scores, and compare
    each bin's mean score against the fraction of that bin that was actually
    bought. On the diagonal means calibrated.

``plot_score_histogram``  why PR-AUC and not accuracy, in one picture. With a
    base rate of 0.13 the negative class buries the positive one, and the figure
    shows exactly how much of the positive mass sits under the negative mass.

``plot_btr_by_query``  the figure closest to the assignment's own words. The TP
    defines BTR as bought over impressions, i.e. a rate over a SET, while the
    model emits a probability per impression. Averaging the probabilities of one
    query and comparing against what that query actually bought is the claim
    "predicting per-impression probability is predicting BTR", drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plots.plot_theme import (
	ACCENT, BASELINE, DASH, LIGHT, MUTED, NEGATIVE, POSITIVE,
	apply_theme, legend_top_left, set_title,
)


def plot_reliability(
	labels: np.ndarray, probs: np.ndarray, bins: int = 10,
	title: str = "Reliability diagram", hyperparameters: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
	labels = np.asarray(labels, dtype=float)
	probs = np.asarray(probs, dtype=float)
	# Quantile edges, not equal width: with a base rate of 0.13 almost every score
	# lands in the first equal-width bin and nine bins come back empty.
	edges = np.unique(np.quantile(probs, np.linspace(0, 1, bins + 1)))
	if len(edges) < 3:
		edges = np.linspace(0.0, 1.0, bins + 1)
	index = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, len(edges) - 2)

	mean_score, observed, counts = [], [], []
	for b in range(len(edges) - 1):
		mask = index == b
		if mask.sum() == 0:
			continue
		mean_score.append(probs[mask].mean())
		observed.append(labels[mask].mean())
		counts.append(int(mask.sum()))

	apply_theme()
	figure, axes = plt.subplots(figsize=(5, 5))
	axes.plot([0, 1], [0, 1], color=BASELINE, linestyle=DASH, linewidth=1, label="Calibración perfecta")
	axes.plot(mean_score, observed, color=ACCENT, marker="o", label="Modelo")
	for x, y, n in zip(mean_score, observed, counts):
		axes.annotate(f"n={n}", xy=(x, y), xytext=(4, -9), textcoords="offset points",
		              fontsize=7, color=MUTED)
	axes.set_xlim(0, 1); axes.set_ylim(0, 1)
	axes.set_xlabel("Probabilidad predicha (media del bin)")
	axes.set_ylabel("Fracción realmente comprada")
	set_title(axes, title)
	legend_top_left(axes, subtitle=hyperparameters)
	figure.tight_layout()
	return figure, axes


def plot_score_histogram(
	labels: np.ndarray, probs: np.ndarray, bins: int = 40,
	title: str = "Score distribution by class", hyperparameters: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
	labels = np.asarray(labels, dtype=int)
	probs = np.asarray(probs, dtype=float)
	apply_theme()
	figure, axes = plt.subplots(figsize=(6.2, 4))
	edges = np.linspace(0, 1, bins + 1)
	for value, color, label in ((0, NEGATIVE, "no comprado"), (1, POSITIVE, "comprado")):
		axes.hist(probs[labels == value], bins=edges, color=color, alpha=0.75, label=label)
	# Log counts: the negative class outnumbers the positive one roughly 7 to 1,
	# so on a linear axis the positive histogram is invisible.
	axes.set_yscale("log")
	axes.set_xlabel("Probabilidad predicha")
	axes.set_ylabel("Impresiones (escala log)")
	set_title(axes, title)
	legend_top_left(axes, subtitle=hyperparameters)
	figure.tight_layout()
	return figure, axes


def plot_btr_by_query(
	query_ids: np.ndarray, labels: np.ndarray, probs: np.ndarray,
	title: str = "BTR by query: predicted vs. observed",
	hyperparameters: str | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
	"""One point per query: mean predicted probability vs. fraction bought."""
	query_ids = np.asarray(query_ids)
	if query_ids.size == 0:
		return None
	labels = np.asarray(labels, dtype=float)
	probs = np.asarray(probs, dtype=float)

	unique = np.unique(query_ids)
	predicted = np.array([probs[query_ids == q].mean() for q in unique])
	observed = np.array([labels[query_ids == q].mean() for q in unique])
	sizes = np.array([int((query_ids == q).sum()) for q in unique])

	apply_theme()
	figure, axes = plt.subplots(figsize=(5, 5))
	limit = max(predicted.max(), observed.max(), 1.0)
	axes.plot([0, limit], [0, limit], color=BASELINE, linestyle=DASH, linewidth=1,
	          label="BTR predicho = BTR observado")
	# Jitter on the observed axis only: with 1 to 8 impressions per query the
	# observed BTR can only take a handful of values, so the points stack into
	# horizontal lines and the density is unreadable without it.
	rng = np.random.default_rng(0)
	axes.scatter(
		predicted, observed + rng.uniform(-0.02, 0.02, size=len(observed)),
		s=6 + 4 * sizes, color=ACCENT, alpha=0.35, linewidths=0,
		label=f"{len(unique)} queries (área = impresiones)",
	)
	axes.set_xlabel("BTR predicho (promedio de probabilidades de la query)")
	axes.set_ylabel("BTR observado (fracción comprada)")
	set_title(axes, title)
	legend_top_left(axes, subtitle=hyperparameters)
	figure.tight_layout()
	return figure, axes
