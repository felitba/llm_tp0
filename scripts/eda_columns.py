"""Exploratory charts and statistics for the per-column feature decisions.

Regenerates every number quoted in ``comentarios_para_el_equipo/ANALISIS_COLUMNAS.md``
and writes the presentation figures to ``plots/eda/``.

Loads the CSV with the standard library instead of pandas on purpose: pandas
coerces the literal string ``"None"`` in ``allergens`` to ``NaN`` through its
default ``na_values`` list, which invents a 44.5% missing rate in a column that
has no missing values at all.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import sys

import matplotlib.pyplot as plt

# `python scripts/eda_columns.py` puts scripts/ on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plots.plot_theme import ACCENT, MUTED, NEG, POS, apply_theme, legend_top_left  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "dataset" / "supermarket_products.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "eda"

# The four columns that define what the user asked for. Every row of a query
# shares the same tuple, which is what makes the rows non-independent.
FILTER_TUPLE = ("filter_category", "filter_price_min", "filter_price_max", "filter_storage_type")

# Columns that survive ``drop_columns`` in config.json and therefore reach the
# model, either one-hot encoded or normalized.
KEPT_FEATURES = frozenset(
	{
		"category",
		"storage_type",
		"brand",
		"country_of_origin",
		"allergens",
		"price",
		"filter_price_min",
		"filter_price_max",
		"net_weight_oz",
		"nutrition_score",
	}
)

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PAIRS = (("train", "validation"), ("validation", "test"), ("train", "test"))

TAG_PATTERN = re.compile(r"\(([^)]+)\)\s*$")
HIGH_TAGS = frozenset({"Customer Favorite", "Best Seller", "Top Rated", "#1 Pick"})
LOW_TAGS = frozenset({"Well Reviewed", "Shopper Favorite", "Highly Rated", "Popular Choice"})

# Los hex viven en plots/plot_theme.py, que es el mismo tema que usa el deck.
POSITIVE_COLOR = POS
NEGATIVE_COLOR = NEG
NEUTRAL_COLOR = ACCENT
BASE_COLOR = MUTED


@dataclass(frozen=True)
class Row:
	"""One product impression with the derived fields the analysis needs."""

	values: dict[str, str]
	bought: int
	tag: str
	tier: str

	def __getitem__(self, column: str) -> str:
		return self.values[column]


@dataclass(frozen=True)
class GroupStat:
	"""Share of a group's impressions that ended in a purchase.

	Not a buy-through rate in the funnel sense: the denominator is every row in
	the group, not the rows that reached the cart. See ``purchase_rate``.
	"""

	name: str
	count: int
	purchase_rate: float


@dataclass(frozen=True)
class ColumnSignal:
	"""Chi-squared independence test of one column against ``bought``."""

	column: str
	groups: list[GroupStat]
	chi_square: float
	degrees_of_freedom: int
	z_score: float

	@property
	def spread(self) -> float:
		if not self.groups:
			return 0.0
		rates = [group.purchase_rate for group in self.groups]
		return max(rates) - min(rates)

	@property
	def is_signal(self) -> bool:
		return self.z_score > 3.0


def load_rows(dataset: Path = DATASET) -> list[Row]:
	"""Read the CSV and attach the derived title tag and tier to every row."""
	with dataset.open(encoding="utf-8") as handle:
		raw_rows = list(csv.DictReader(handle))

	rows = []
	for raw in raw_rows:
		match = TAG_PATTERN.search(raw["title"].strip())
		tag = match.group(1) if match else "(sin etiqueta)"
		if tag in HIGH_TAGS:
			tier = "ALTO"
		elif tag in LOW_TAGS:
			tier = "BAJO"
		else:
			tier = "CERO"
		bought = 1 if raw["bought"].strip().lower() == "true" else 0
		rows.append(Row(values=raw, bought=bought, tag=tag, tier=tier))
	return rows


def purchase_rate(rows: Sequence[Row]) -> float:
	"""Fraction of the given rows with ``bought=true``.

	One row is one product impression, so this is purchases per impression —
	P(bought | group). It is deliberately not a buy-through rate: that would be
	``bought / cart`` (43.3% overall against 13.0% here), a different quantity
	measured against a different denominator.
	"""
	if not rows:
		return 0.0
	return sum(row.bought for row in rows) / len(rows)


def group_by(
	rows: Sequence[Row], key: Callable[[Row], str], min_count: int = 1
) -> list[GroupStat]:
	"""Group rows by ``key`` and return each group's purchase rate, best first."""
	grouped: dict[str, list[int]] = defaultdict(list)
	for row in rows:
		grouped[key(row)].append(row.bought)

	stats = [
		GroupStat(name=name, count=len(labels), purchase_rate=sum(labels) / len(labels))
		for name, labels in grouped.items()
		if len(labels) >= min_count
	]
	return sorted(stats, key=lambda group: group.purchase_rate, reverse=True)


def chi_square_signal(
	rows: Sequence[Row], column: str, key: Callable[[Row], str] | None = None, min_count: int = 30
) -> ColumnSignal:
	"""Test whether ``column`` is independent of ``bought``.

	The chi-squared tail is converted to a z-score with the Wilson-Hilferty
	transform so columns with different group counts stay comparable: a column
	with 63 groups needs a much larger raw statistic than one with 3 to mean
	the same thing.
	"""
	groups = group_by(rows, key or (lambda row: row[column]), min_count=min_count)
	total = sum(group.count for group in groups)
	positives = sum(round(group.purchase_rate * group.count) for group in groups)

	degrees_of_freedom = len(groups) - 1
	if degrees_of_freedom < 1 or total == 0 or positives in (0, total):
		return ColumnSignal(column, groups, 0.0, max(degrees_of_freedom, 0), float("-inf"))

	rate = positives / total
	chi_square = sum(
		(group.purchase_rate * group.count - group.count * rate) ** 2
		/ (group.count * rate * (1 - rate))
		for group in groups
	)
	normalized = (chi_square / degrees_of_freedom) ** (1 / 3)
	expected = 1 - 2 / (9 * degrees_of_freedom)
	z_score = (normalized - expected) / math.sqrt(2 / (9 * degrees_of_freedom))
	return ColumnSignal(column, groups, chi_square, degrees_of_freedom, z_score)


def quartile_stats(rows: Sequence[Row], column: str, within: str | None = None) -> list[GroupStat]:
	"""Bucket a numeric column into quartiles and return the rate of each.

	With ``within`` set, rows are ranked inside each value of that column before
	bucketing, which removes the level differences between categories and leaves
	only the shape of the relationship.
	"""
	buckets: dict[int, list[int]] = defaultdict(list)
	partitions: dict[str, list[Row]] = defaultdict(list)
	for row in rows:
		partitions[row[within] if within else ""].append(row)

	for partition in partitions.values():
		ordered = sorted(partition, key=lambda row: float(row[column]))
		size = len(ordered)
		for index, row in enumerate(ordered):
			buckets[min(3, index * 4 // size)].append(row.bought)

	return [
		GroupStat(
			name=f"Q{index + 1}",
			count=len(buckets[index]),
			purchase_rate=sum(buckets[index]) / len(buckets[index]),
		)
		for index in sorted(buckets)
		if buckets[index]
	]


def pearson_correlation(rows: Sequence[Row], column: str) -> float:
	values = [float(row[column]) for row in rows]
	labels = [row.bought for row in rows]
	count = len(values)
	mean_x = sum(values) / count
	mean_y = sum(labels) / count
	covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, labels))
	deviation_x = math.sqrt(sum((x - mean_x) ** 2 for x in values))
	deviation_y = math.sqrt(sum((y - mean_y) ** 2 for y in labels))
	if deviation_x == 0 or deviation_y == 0:
		return 0.0
	return covariance / (deviation_x * deviation_y)


def identical_fraction(rows: Sequence[Row], left: str, right: str) -> float:
	matches = sum(1 for row in rows if row[left].strip() == row[right].strip())
	return matches / len(rows)


def contained_fraction(rows: Sequence[Row], needle: str, haystack: str) -> float:
	matches = sum(1 for row in rows if row[needle].strip().lower() in row[haystack].strip().lower())
	return matches / len(rows)


def _barh(
	axes: plt.Axes,
	groups: Sequence[GroupStat],
	colors: Sequence[str],
	baseline: float | None,
	show_labels: bool = True,
) -> None:
	"""Draw one horizontal bar per group, best at the top.

	Rows are placed at descending positions rather than inverting the axis,
	because inverting a shared axis twice flips the order back.
	"""
	positions = [-index for index in range(len(groups))]
	axes.barh(positions, [group.purchase_rate for group in groups], color=colors)
	axes.set_yticks(positions)
	axes.set_yticklabels(
		[f"{group.name}  (n={group.count})" for group in groups] if show_labels else [],
		fontsize=9,
	)
	axes.set_xlabel("compras / impresiones")
	for position, group in zip(positions, groups):
		axes.text(group.purchase_rate + 0.008, position, f"{group.purchase_rate:.3f}", va="center", fontsize=8)
	if baseline is not None:
		axes.axvline(baseline, color=BASE_COLOR, linestyle="--", linewidth=1)
		axes.text(baseline, 0.9, f" base {baseline:.3f}", color=BASE_COLOR, fontsize=8)
	axes.set_xlim(0, max(max(group.purchase_rate for group in groups), 0.01) * 1.18)
	axes.set_ylim(-len(groups) + 0.4, 1.4)
	axes.spines[["top", "right"]].set_visible(False)


def plot_group_rates(
	groups: Sequence[GroupStat],
	title: str,
	baseline: float | None = None,
	color_by_tier: Callable[[GroupStat], str] | None = None,
) -> plt.Figure:
	"""Horizontal bar chart of purchases per impression, one bar per group."""
	figure, axes = plt.subplots(figsize=(9, 0.42 * len(groups) + 1.6))
	colors = [color_by_tier(group) if color_by_tier else NEUTRAL_COLOR for group in groups]
	_barh(axes, groups, colors, baseline)
	axes.set_title(title, fontsize=12, pad=12)
	figure.tight_layout()
	return figure


def plot_marginal_vs_conditional(
	marginal: Sequence[GroupStat], conditional: Sequence[GroupStat], column: str
) -> plt.Figure:
	"""Show a column's rates before and after conditioning on the high-tag tier.

	This is the chart that separates a real secondary signal from one the title
	tag was providing all along.
	"""
	conditional_by_name = {group.name: group for group in conditional}
	ordered = [group for group in marginal if group.name in conditional_by_name]
	ordered.sort(key=lambda group: conditional_by_name[group.name].purchase_rate, reverse=True)

	figure, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(ordered) + 2.0))
	_barh(axes[0], ordered, [BASE_COLOR] * len(ordered), None)
	axes[0].set_title(f"{column} — marginal (n=10.000)", fontsize=11)
	conditional_ordered = [conditional_by_name[group.name] for group in ordered]
	_barh(axes[1], conditional_ordered, [NEUTRAL_COLOR] * len(ordered), None, show_labels=False)
	axes[1].set_title(
		f"{column} — dentro del tier ALTO (n={sum(group.count for group in conditional)})",
		fontsize=11,
	)
	figure.suptitle(f"'{column}' sobrevive a la etiqueta del titulo", fontsize=13, y=1.0)
	figure.tight_layout()
	return figure


def plot_price_paradox(rows: Sequence[Row]) -> plt.Figure:
	"""The headline chart: why Pearson's r says price is noise and binning says it is not."""
	high = [row for row in rows if row.tier == "ALTO"]
	correlation = pearson_correlation(rows, "price")
	quartiles = quartile_stats(high, "price", within="category")

	figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

	axes[0].scatter(
		[float(row["price"]) for row in rows],
		[row.bought + (0.06 if row.bought else -0.06) for row in rows],
		s=4,
		alpha=0.12,
		color=BASE_COLOR,
	)
	axes[0].set_title(f"Vista lineal: r = {correlation:+.3f}  $\\rightarrow$  'no hay señal'", fontsize=11)
	axes[0].set_xlabel("price")
	axes[0].set_ylabel("bought")
	axes[0].set_yticks([0, 1])
	axes[0].spines[["top", "right"]].set_visible(False)

	positions = range(len(quartiles))
	axes[1].bar(
		positions,
		[group.purchase_rate for group in quartiles],
		color=[POSITIVE_COLOR if group.purchase_rate > 0.65 else NEUTRAL_COLOR for group in quartiles],
	)
	axes[1].set_xticks(list(positions))
	axes[1].set_xticklabels([f"{group.name}\nn={group.count}" for group in quartiles], fontsize=9)
	for position, group in zip(positions, quartiles):
		axes[1].text(position, group.purchase_rate + 0.015, f"{group.purchase_rate:.3f}", ha="center", fontsize=9)
	axes[1].set_title(
		"Bineado por percentil de precio dentro de categoria\n(solo tier ALTO)", fontsize=11
	)
	axes[1].set_ylabel("compras / impresiones")
	axes[1].set_ylim(0, max(group.purchase_rate for group in quartiles) * 1.2)
	axes[1].spines[["top", "right"]].set_visible(False)

	figure.suptitle("price no es ruido: la relacion es no monotona, por eso r ≈ 0", fontsize=13)
	figure.tight_layout()
	return figure


def plot_price_by_category(rows: Sequence[Row], min_count: int = 80) -> plt.Figure:
	"""The inverted U reproduced inside each category, so it is not a mix artifact."""
	high = [row for row in rows if row.tier == "ALTO"]
	by_category: dict[str, list[Row]] = defaultdict(list)
	for row in high:
		by_category[row["category"]].append(row)
	categories = sorted(
		(name for name, group in by_category.items() if len(group) >= min_count),
		key=lambda name: -len(by_category[name]),
	)

	figure, axes = plt.subplots(figsize=(9, 5))
	for name in categories:
		quartiles = quartile_stats(by_category[name], "price")
		axes.plot(
			[group.name for group in quartiles],
			[group.purchase_rate for group in quartiles],
			marker="o",
			label=f"{name} (n={len(by_category[name])})",
			linewidth=1.6,
		)
	axes.set_ylabel("compras / impresiones")
	axes.set_xlabel("Cuartil de precio dentro de la categoria")
	axes.set_title("La U invertida se repite en cada categoria por separado (tier ALTO)", fontsize=12)
	legend_top_left(axes, ncols=2)
	axes.spines[["top", "right"]].set_visible(False)
	figure.tight_layout()
	return figure


def plot_signal_ranking(signals: Sequence[ColumnSignal]) -> plt.Figure:
	"""Every column on one axis: which ones beat chance, on a comparable scale."""
	ordered = sorted(signals, key=lambda signal: signal.z_score)
	figure, axes = plt.subplots(figsize=(9, 0.38 * len(ordered) + 1.8))
	positions = range(len(ordered))
	colors = [POSITIVE_COLOR if signal.is_signal else BASE_COLOR for signal in ordered]
	axes.barh(positions, [max(signal.z_score, -1.0) for signal in ordered], color=colors)
	axes.set_yticks(list(positions))
	axes.set_yticklabels([signal.column for signal in ordered], fontsize=9)
	axes.axvline(3.0, color=NEGATIVE_COLOR, linestyle="--", linewidth=1)
	axes.text(3.4, len(ordered) - 0.4, "umbral z=3", color=NEGATIVE_COLOR, fontsize=8)
	axes.set_xlabel("z de la prueba chi-cuadrado contra 'bought'  (escala log)")
	axes.set_xscale("symlog")
	axes.set_title("Señal por columna, corregida por cantidad de grupos", fontsize=12, pad=12)
	axes.spines[["top", "right"]].set_visible(False)
	figure.tight_layout()
	return figure


def plot_leakage(rows: Sequence[Row]) -> plt.Figure:
	"""The cart test: an input that is never false when the target is true."""
	counts = Counter((row["cart"], row["bought"]) for row in rows)
	cells = [
		("cart=false\nbought=false", counts[("false", "false")], BASE_COLOR),
		("cart=true\nbought=false", counts[("true", "false")], BASE_COLOR),
		("cart=true\nbought=true", counts[("true", "true")], POSITIVE_COLOR),
		("cart=false\nbought=true", counts[("false", "true")], NEGATIVE_COLOR),
	]
	figure, axes = plt.subplots(figsize=(8, 4.4))
	positions = range(len(cells))
	axes.bar(positions, [count for _, count, _ in cells], color=[color for _, _, color in cells])
	axes.set_xticks(list(positions))
	axes.set_xticklabels([label for label, _, _ in cells], fontsize=9)
	for position, (_, count, _) in zip(positions, cells):
		axes.text(position, count + 90, f"{count:,}", ha="center", fontsize=10)
	axes.set_ylabel("filas")
	axes.set_title("'cart' es fuga: bought=true implica cart=true, 0 excepciones", fontsize=12)
	axes.spines[["top", "right"]].set_visible(False)
	figure.tight_layout()
	return figure


def plot_allergens_nulls(rows: Sequence[Row]) -> plt.Figure:
	"""The 44.5% missing rate that only exists after pandas reads the file."""
	counts = Counter(row["allergens"] for row in rows)
	none_count = counts["None"]
	other_count = len(rows) - none_count

	figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
	axes[0].bar(
		["'None'\n(string literal)", "otros alergenos"],
		[none_count, other_count],
		color=[POSITIVE_COLOR, NEUTRAL_COLOR],
	)
	axes[0].set_title(
		f"En el CSV: 0 celdas vacias\n'None' = {none_count / len(rows):.1%}", fontsize=11
	)
	axes[0].set_ylabel("filas")

	axes[1].bar(
		["NaN\n(inventado)", "con valor"],
		[none_count, other_count],
		color=[NEGATIVE_COLOR, NEUTRAL_COLOR],
	)
	axes[1].set_title(
		f"Tras pandas.read_csv por defecto:\n{none_count / len(rows):.1%} 'faltante'", fontsize=11
	)
	for axis in axes:
		axis.spines[["top", "right"]].set_visible(False)
	figure.suptitle("El 44,5% de nulos de 'allergens' es un artefacto de parseo", fontsize=13)
	figure.tight_layout()
	return figure


def plot_redundancy(rows: Sequence[Row]) -> plt.Figure:
	"""Columns that carry no information the kept columns do not already have."""
	checks = [
		("filter_category\n== category", identical_fraction(rows, "filter_category", "category")),
		(
			"filter_storage_type\n== storage_type",
			identical_fraction(rows, "filter_storage_type", "storage_type"),
		),
		("brand\ndentro de title", contained_fraction(rows, "brand", "title")),
		("package_size\ndentro de title", contained_fraction(rows, "package_size", "title")),
		(
			"price dentro del\nrango de filtros",
			sum(
				1
				for row in rows
				if float(row["filter_price_min"])
				<= float(row["price"])
				<= float(row["filter_price_max"])
			)
			/ len(rows),
		),
	]
	figure, axes = plt.subplots(figsize=(9, 4.4))
	positions = range(len(checks))
	axes.bar(positions, [value for _, value in checks], color=NEGATIVE_COLOR)
	axes.set_xticks(list(positions))
	axes.set_xticklabels([label for label, _ in checks], fontsize=9)
	for position, (_, value) in zip(positions, checks):
		axes.text(position, value + 0.015, f"{value:.1%}", ha="center", fontsize=10)
	axes.set_ylim(0, 1.15)
	axes.set_ylabel("filas que cumplen")
	axes.set_title("Redundancia: informacion ya presente en otra columna", fontsize=12)
	axes.spines[["top", "right"]].set_visible(False)
	figure.tight_layout()
	return figure


def rows_by_query(rows: Sequence[Row]) -> dict[str, list[Row]]:
	"""Group rows by ``query_id``, preserving file order inside each query."""
	queries: dict[str, list[Row]] = defaultdict(list)
	for row in rows:
		queries[row["query_id"]].append(row)
	return queries


def _cut(group: Sequence[Row], ratios: tuple[float, float, float]) -> dict[str, list[Row]]:
	"""Reproduce ``dataset.preprocess_dataset.separate`` exactly, including int()."""
	total = sum(ratios)
	train_end = int(len(group) * ratios[0] / total)
	valid_end = train_end + int(len(group) * ratios[1] / total)
	return {
		"train": list(group[:train_end]),
		"validation": list(group[train_end:valid_end]),
		"test": list(group[valid_end:]),
	}


def split_positional(rows: Sequence[Row]) -> dict[str, list[Row]]:
	"""The split in use today: stratify by ``bought``, then cut each stratum by position.

	Not a plain 80/10/10 over the frame — purchases and non-purchases are cut
	separately and concatenated, so both strata have to be reproduced to get the
	real split boundaries.
	"""
	bought = [row for row in rows if row.bought]
	not_bought = [row for row in rows if not row.bought]
	bought_splits = _cut(bought, (0.8, 0.1, 0.1))
	not_bought_splits = _cut(not_bought, (0.8, 0.1, 0.1))
	return {name: bought_splits[name] + not_bought_splits[name] for name in SPLIT_NAMES}


def split_shuffled(rows: Sequence[Row], seed: int = 0) -> dict[str, list[Row]]:
	"""The same positional cut applied after a row-level shuffle.

	This is the counterfactual that matters: the current split only avoids the
	leak because the CSV happens to arrive sorted by ``query_id``.
	"""
	shuffled = list(rows)
	random.Random(seed).shuffle(shuffled)
	return _cut(shuffled, (0.8, 0.1, 0.1))


def split_grouped(rows: Sequence[Row]) -> dict[str, list[Row]]:
	"""Assign whole queries to splits, so no query can straddle a boundary."""
	queries = rows_by_query(rows)
	names = sorted(queries)
	random.Random(0).shuffle(names)

	train_end = int(len(names) * 0.8)
	valid_end = train_end + int(len(names) * 0.1)
	assignment = {
		"train": names[:train_end],
		"validation": names[train_end:valid_end],
		"test": names[valid_end:],
	}
	return {name: [row for query in group for row in queries[query]] for name, group in assignment.items()}


def overlap_stats(splits: dict[str, list[Row]]) -> dict[tuple[str, str], tuple[int, int]]:
	"""For each split pair, the shared query count and the rows sitting in them."""
	queries = {name: {row["query_id"] for row in group} for name, group in splits.items()}
	stats = {}
	for left, right in SPLIT_PAIRS:
		shared = queries[left] & queries[right]
		affected = sum(
			1 for name in (left, right) for row in splits[name] if row["query_id"] in shared
		)
		stats[(left, right)] = (len(shared), affected)
	return stats


def contaminated_eval_fraction(splits: dict[str, list[Row]]) -> tuple[int, int]:
	"""Evaluation rows whose query also appears in train, over all evaluation rows."""
	train_queries = {row["query_id"] for row in splits["train"]}
	evaluation = splits["validation"] + splits["test"]
	leaked = sum(1 for row in evaluation if row["query_id"] in train_queries)
	return leaked, len(evaluation)


def plot_query_structure(rows: Sequence[Row]) -> plt.Figure:
	"""Queries vary in both size and outcome, so a cut through one is arbitrary."""
	queries = rows_by_query(rows)
	sizes = Counter(len(group) for group in queries.values())
	purchases = Counter(sum(row.bought for row in group) for group in queries.values())
	zero_purchase = purchases[0]

	figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))

	size_keys = sorted(sizes)
	axes[0].bar(size_keys, [sizes[key] for key in size_keys], color=NEUTRAL_COLOR)
	for key in size_keys:
		axes[0].text(key, sizes[key] + 6, f"{sizes[key]:,}", ha="center", fontsize=8)
	axes[0].set_xticks(size_keys)
	axes[0].set_xlabel("filas por query")
	axes[0].set_ylabel("cantidad de queries")
	axes[0].set_title(
		f"Tamaño de query: 1 a {max(size_keys)} filas\n"
		f"(media {len(rows) / len(queries):.2f}, {len(queries):,} queries)",
		fontsize=11,
	)
	axes[0].set_ylim(0, max(sizes.values()) * 1.16)

	purchase_keys = sorted(purchases)
	colors = [NEGATIVE_COLOR if key == 0 else NEUTRAL_COLOR for key in purchase_keys]
	axes[1].bar(purchase_keys, [purchases[key] for key in purchase_keys], color=colors)
	for key in purchase_keys:
		axes[1].text(key, purchases[key] + 12, f"{purchases[key]:,}", ha="center", fontsize=8)
	axes[1].set_xticks(purchase_keys)
	axes[1].set_xlabel("compras por query")
	axes[1].set_ylabel("cantidad de queries")
	axes[1].set_title("Resultado de query: de 0 a 4 compras", fontsize=11)
	axes[1].set_ylim(0, max(purchases.values()) * 1.30)
	axes[1].annotate(
		f"{zero_purchase:,} queries ({zero_purchase / len(queries):.1%})\nno terminan en ninguna compra",
		xy=(0, zero_purchase),
		xytext=(1.15, zero_purchase * 0.93),
		fontsize=9,
		color=NEGATIVE_COLOR,
		arrowprops={"arrowstyle": "->", "color": NEGATIVE_COLOR, "linewidth": 1.2},
	)

	for axis in axes:
		axis.spines[["top", "right"]].set_visible(False)
	figure.suptitle(
		"Las queries varian en tamaño y en resultado: cortar por el medio de una es arbitrario",
		fontsize=13,
	)
	figure.tight_layout()
	return figure


def plot_query_leakage(rows: Sequence[Row]) -> plt.Figure:
	"""The leak is near zero today only because the CSV arrives sorted by query_id."""
	schemes = [
		("Posicional\n(actual)", split_positional(rows), NEUTRAL_COLOR),
		("Posicional tras\nmezclar filas", split_shuffled(rows), NEGATIVE_COLOR),
		("Agrupado por\nquery_id", split_grouped(rows), POSITIVE_COLOR),
	]
	stats = [(label, overlap_stats(splits), color) for label, splits, color in schemes]

	figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

	width = 0.26
	positions = range(len(SPLIT_PAIRS))
	for index, (label, overlaps, color) in enumerate(stats):
		offsets = [position + (index - 1) * width for position in positions]
		counts = [overlaps[pair][0] for pair in SPLIT_PAIRS]
		axes[0].bar(offsets, counts, width=width, color=color, label=label)
		for offset, pair in zip(offsets, SPLIT_PAIRS):
			shared, affected = overlaps[pair]
			axes[0].text(
				offset,
				shared + 12,
				f"{shared}\n({affected:,} filas)" if shared else "0",
				ha="center",
				fontsize=7.5,
			)
	axes[0].set_xticks(list(positions))
	axes[0].set_xticklabels([f"{left}\n$\\cap$ {right}" for left, right in SPLIT_PAIRS], fontsize=9)
	axes[0].set_ylabel("queries compartidas entre splits")
	axes[0].set_title("Queries que caen en dos splits a la vez\n(entre parentesis: filas afectadas)", fontsize=11)
	legend_top_left(axes[0])
	axes[0].set_ylim(0, 950)

	fractions = []
	for label, splits, color in schemes:
		leaked, total = contaminated_eval_fraction(splits)
		fractions.append((label, leaked, total, color))
	axes[1].bar(
		range(len(fractions)),
		[leaked / total for _, leaked, total, _ in fractions],
		color=[color for *_, color in fractions],
	)
	axes[1].set_xticks(range(len(fractions)))
	axes[1].set_xticklabels([label for label, *_ in fractions], fontsize=9)
	for index, (_, leaked, total, _) in enumerate(fractions):
		axes[1].text(
			index,
			leaked / total + 0.022,
			f"{leaked / total:.1%}\n({leaked:,} de {total:,} filas)",
			ha="center",
			fontsize=9,
		)
	axes[1].set_ylabel("filas de evaluacion contaminadas")
	axes[1].set_ylim(0, 1.16)
	axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
	axes[1].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
	axes[1].set_title(
		"Filas de val/test cuya query tambien esta en train", fontsize=11
	)

	for axis in axes:
		axis.spines[["top", "right"]].set_visible(False)
	figure.suptitle(
		"El split actual no filtra porque el CSV viene ordenado por query_id, no por diseño",
		fontsize=13,
	)
	figure.tight_layout()
	return figure


def plot_within_query_constancy(rows: Sequence[Row]) -> plt.Figure:
	"""Six columns never vary inside a query, so its rows are not independent draws."""
	queries = [group for group in rows_by_query(rows).values() if len(group) > 1]
	columns = [column for column in rows[0].values if column != "query_id"]
	constancy = [
		(
			column,
			sum(1 for group in queries if len({row[column] for row in group}) == 1) / len(queries),
		)
		for column in columns
	]
	constancy.sort(key=lambda item: item[1])

	figure, axes = plt.subplots(figsize=(9.5, 0.34 * len(constancy) + 2.2))
	positions = range(len(constancy))
	colors = [
		POSITIVE_COLOR if fraction == 1.0 else BASE_COLOR for _, fraction in constancy
	]
	axes.barh(positions, [fraction for _, fraction in constancy], color=colors)
	axes.set_yticks(list(positions))
	axes.set_yticklabels(
		[
			f"{column}  $\\star$" if column in KEPT_FEATURES else column
			for column, _ in constancy
		],
		fontsize=9,
	)
	for position, (_, fraction) in zip(positions, constancy):
		axes.text(fraction + 0.012, position, f"{fraction:.1%}", va="center", fontsize=8)
	axes.set_xlim(0, 1.16)
	axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
	axes.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
	axes.set_xlabel("queries en las que la columna toma un solo valor")
	axes.set_title(
		f"Constancia dentro de la query  (n={len(queries):,} queries de mas de una fila)\n"
		"$\\star$ = columna que sobrevive a drop_columns y llega al modelo",
		fontsize=12,
		pad=12,
	)
	axes.axvline(1.0, color=NEGATIVE_COLOR, linestyle="--", linewidth=1)

	tuples_per_query = {
		len({tuple(row[column] for column in FILTER_TUPLE) for row in group})
		for group in rows_by_query(rows).values()
	}
	assert tuples_per_query == {1}, f"expected one filter tuple per query, got {tuples_per_query}"
	# Anchored in the empty lower-right quadrant left by the near-zero bars.
	axes.text(
		0.30,
		4.2,
		f"Las {len(rows_by_query(rows)):,} queries tienen exactamente\n"
		f"1 tupla de filtros distinta ({len(rows_by_query(rows)):,} / {len(rows_by_query(rows)):,}).\n\n"
		"Cuatro de las columnas 100% constantes\n"
		"llegan al modelo: category, storage_type,\n"
		"filter_price_min, filter_price_max.",
		fontsize=9.5,
		color=POSITIVE_COLOR,
		va="center",
		bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": POSITIVE_COLOR},
	)
	axes.spines[["top", "right"]].set_visible(False)
	figure.tight_layout()
	return figure


def _tier_color(group: GroupStat) -> str:
	if group.purchase_rate > 0.5:
		return POSITIVE_COLOR
	if group.purchase_rate > 0.005:
		return NEUTRAL_COLOR
	return NEGATIVE_COLOR


def build_report(rows: Sequence[Row]) -> list[ColumnSignal]:
	"""Compute the chi-squared signal for every candidate column."""
	categorical = [
		"category",
		"allergens",
		"brand",
		"country_of_origin",
		"storage_type",
		"unit_of_measure",
		"ingredients",
		"package_size",
	]
	signals = [chi_square_signal(rows, column) for column in categorical]
	signals.append(chi_square_signal(rows, "title tag", key=lambda row: row.tag))
	signals.append(chi_square_signal(rows, "timestamp (mes)", key=lambda row: row["timestamp"][:7]))
	return signals


def print_summary(rows: Sequence[Row], signals: Sequence[ColumnSignal]) -> None:
	"""Print every statistic quoted in the write-up, so the two cannot drift apart."""
	high = [row for row in rows if row.tier == "ALTO"]
	line = "-" * 78

	print(f"\n{line}\nDATASET\n{line}")
	print(f"filas={len(rows)}  columnas={len(rows[0].values)}  compras/impresiones base={purchase_rate(rows):.4f}")
	empty = {column: sum(1 for row in rows if not row[column].strip()) for column in rows[0].values}
	print(f"columnas con celdas vacias: {sum(1 for count in empty.values() if count)} de {len(empty)}")
	print(f"allergens == 'None' (string literal): {Counter(row['allergens'] for row in rows)['None']}")

	print(f"\n{line}\nTIERS DE LA ETIQUETA DEL TITULO\n{line}")
	for tier in ("ALTO", "BAJO", "CERO"):
		subset = [row for row in rows if row.tier == tier]
		print(f"{tier:<5} n={len(subset):<6} compras/impresiones={purchase_rate(subset):.4f}")

	print(f"\n{line}\nSEÑAL POR COLUMNA (marginal)\n{line}")
	print(f"{'columna':<20}{'grupos':>7}{'spread':>9}{'chi2':>10}{'dof':>5}{'z':>8}  veredicto")
	for signal in sorted(signals, key=lambda item: -item.z_score):
		verdict = "SEÑAL" if signal.is_signal else "ruido"
		print(
			f"{signal.column:<20}{len(signal.groups):>7}{signal.spread:>9.3f}"
			f"{signal.chi_square:>10.1f}{signal.degrees_of_freedom:>5}"
			f"{signal.z_score:>8.2f}  {verdict}"
		)

	print(f"\n{line}\nSEÑAL CONDICIONADA AL TIER ALTO (n={len(high)})\n{line}")
	for column in ("category", "allergens", "brand", "country_of_origin", "ingredients"):
		signal = chi_square_signal(high, column)
		verdict = "SEÑAL" if signal.is_signal else "ruido"
		print(
			f"{column:<20} spread={signal.spread:>6.3f}  chi2={signal.chi_square:>7.1f}"
			f"  z={signal.z_score:>6.2f}  {verdict}"
		)

	print(f"\n{line}\nCOLUMNAS NUMERICAS\n{line}")
	for column in ("price", "net_weight_oz", "nutrition_score", "filter_price_min", "filter_price_max"):
		correlation = pearson_correlation(rows, column)
		shape = " ".join(f"{group.purchase_rate:.3f}" for group in quartile_stats(high, column))
		print(f"{column:<20} r={correlation:+.4f}   cuartiles en tier ALTO: {shape}")
	ranked = quartile_stats(high, "price", within="category")
	print(
		"price por percentil dentro de categoria (tier ALTO): "
		+ " ".join(f"{group.purchase_rate:.3f}" for group in ranked)
	)

	print(f"\n{line}\nREDUNDANCIA Y FUGA\n{line}")
	print(f"filter_category == category:          {identical_fraction(rows, 'filter_category', 'category'):.1%}")
	print(f"filter_storage_type == storage_type:  {identical_fraction(rows, 'filter_storage_type', 'storage_type'):.1%}")
	print(f"brand contenido en title:             {contained_fraction(rows, 'brand', 'title'):.1%}")
	print(f"package_size contenido en title:      {contained_fraction(rows, 'package_size', 'title'):.1%}")
	counts = Counter((row["cart"], row["bought"]) for row in rows)
	print(f"cart: {dict(counts)}")
	print(f"bought=true con cart=false:           {counts[('false', 'true')]}  (fuga si es 0)")

	print(f"\n{line}\nQUERY_ID\n{line}")
	queries = rows_by_query(rows)
	sizes = Counter(len(group) for group in queries.values())
	print(f"queries={len(queries)}  tamaños={dict(sorted(sizes.items()))}")
	purchases = Counter(sum(row.bought for row in group) for group in queries.values())
	print(f"compras por query={dict(sorted(purchases.items()))}  sin compras={purchases[0]}")
	tuples_per_query = Counter(
		len({tuple(row[column] for column in FILTER_TUPLE) for row in group})
		for group in queries.values()
	)
	print(f"tuplas de filtro distintas por query={dict(tuples_per_query)}")

	for label, splits in (
		("posicional (actual)", split_positional(rows)),
		("posicional tras mezclar", split_shuffled(rows)),
		("agrupado por query_id", split_grouped(rows)),
	):
		leaked, total = contaminated_eval_fraction(splits)
		overlaps = overlap_stats(splits)
		detail = "  ".join(
			f"{left[:2]}∩{right[:2]}={overlaps[(left, right)][0]}q/{overlaps[(left, right)][1]}f"
			for left, right in SPLIT_PAIRS
		)
		print(f"{label:<26} eval contaminado={leaked:>5}/{total} ({leaked / total:>6.1%})  {detail}")


def generate_figures(
	rows: Sequence[Row], signals: Sequence[ColumnSignal], output_dir: Path
) -> list[Path]:
	"""Write every presentation figure and return the paths written."""
	output_dir.mkdir(parents=True, exist_ok=True)
	high = [row for row in rows if row.tier == "ALTO"]
	baseline = purchase_rate(rows)

	figures = {
		"01_title_tag": plot_group_rates(
			group_by(rows, lambda row: row.tag),
			"La etiqueta del titulo explica casi todo el target",
			baseline=baseline,
			color_by_tier=_tier_color,
		),
		"02_allergens_nulls": plot_allergens_nulls(rows),
		"03_price_paradox": plot_price_paradox(rows),
		"04_price_by_category": plot_price_by_category(rows),
		"05_category_conditional": plot_marginal_vs_conditional(
			group_by(rows, lambda row: row["category"]),
			group_by(high, lambda row: row["category"], min_count=30),
			"category",
		),
		"06_allergens_conditional": plot_marginal_vs_conditional(
			group_by(rows, lambda row: row["allergens"]),
			group_by(high, lambda row: row["allergens"], min_count=30),
			"allergens",
		),
		"07_signal_ranking": plot_signal_ranking(signals),
		"08_leakage_cart": plot_leakage(rows),
		"09_redundancy": plot_redundancy(rows),
	}

	written = []
	for name, figure in figures.items():
		path = output_dir / f"{name}.jpg"
		figure.savefig(path, dpi=300, bbox_inches="tight")
		plt.close(figure)
		written.append(path)
	return written


def generate_query_figures(rows: Sequence[Row], output_dir: Path) -> list[Path]:
	"""Write the three figures that justify grouping the split by ``query_id``.

	Saved as PNG rather than JPG: these go into slides, where the flat fills and
	thin rules of a bar chart pick up visible ringing under JPEG compression.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	figures = {
		"10_query_structure": plot_query_structure(rows),
		"11_query_leakage": plot_query_leakage(rows),
		"12_within_query_constancy": plot_within_query_constancy(rows),
	}

	written = []
	for name, figure in figures.items():
		path = output_dir / f"{name}.png"
		figure.savefig(path, dpi=200, bbox_inches="tight")
		plt.close(figure)
		written.append(path)
	return written


def main() -> None:
	apply_theme()
	rows = load_rows()
	signals = build_report(rows)
	print_summary(rows, signals)
	written = generate_figures(rows, signals, OUTPUT_DIR)
	written += generate_query_figures(rows, OUTPUT_DIR)
	print(f"\n{len(written)} figuras escritas en {OUTPUT_DIR}:")
	for path in written:
		print(f"  {path.name}")


if __name__ == "__main__":
	main()
