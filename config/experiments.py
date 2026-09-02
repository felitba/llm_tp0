"""Expand a config file's ``experiments`` list into the runs it actually means.

Three axes live in the config: the ``experiments`` entries (what varies), an
optional per-entry ``matrix`` (the cross product of several factors at once, see
``expand_matrix``), and the optional top-level ``"seeds": [42, 7, 1234]`` (how
many times each is trained).
With more than one seed, every experiment becomes one run per seed, named
``<experiment>_seed<n>``, and the figures group them back by everything except
the seed (``plots/config_comparison.py``). Without ``seeds`` nothing changes:
one run per experiment, named as declared.

Every reader of a config (main, replot, reselect, the analysis scripts) goes
through ``experiment_names`` so they all agree on what a batch contains.
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any

SEED_SUFFIX = re.compile(r"_seed\d+$")


def _abbreviate(key: str) -> str:
	"""``num_layers`` -> ``nl``, ``n_heads`` -> ``nh``, ``dropout`` -> ``d``.

	Initials of the underscore-separated parts. Short enough that a nine-cell
	grid still fits on an axis label, and stable, so the same cell keeps the same
	run directory across re-runs.
	"""
	return "".join(part[0] for part in key.split("_") if part)


def expand_matrix(
	experiment: dict[str, Any], base_config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
	"""One entry with a ``matrix`` -> the cross product of its axes.

	    {"name": "cap", "matrix": {"num_layers": [1, 2], "n_heads": [2, 4]}}

	becomes four entries, ``cap_nl1_nh2`` .. ``cap_nl2_nh4``, each with the cell's
	values merged over the entry's own ``overrides``. Axis order in the JSON fixes
	the order the runs come out in, so figures stay comparable between re-runs.
	``abbrev`` overrides the generated short names per key.

	An entry without ``matrix`` is returned unchanged, so the flat one-factor-at-
	a-time style keeps working and the two can be mixed in one config.
	"""
	matrix = experiment.get("matrix")
	if not matrix:
		return [experiment]
	abbrev = experiment.get("abbrev") or {}
	prefix = str(experiment.get("name", "")).strip()
	keys = list(matrix)
	entries = []
	for combination in itertools.product(*(matrix[key] for key in keys)):
		cell = dict(zip(keys, combination))
		# nn.MultiheadAttention requires d_model % n_heads == 0, and a grid that
		# crosses the two will generate cells that cannot be built. Fail here,
		# naming the cell, rather than several minutes into the batch with a
		# shape error from inside torch. The base config must be consulted too
		# (2026-09-01): a matrix over n_heads alone, with d_model only in the
		# base, previously skipped this check and died inside torch anyway.
		merged_cell = {**experiment.get("overrides", {}), **cell}
		effective = {**(base_config or {}), **merged_cell}
		width = effective.get("d_model")
		heads = effective.get("n_heads")
		if width and heads and int(width) % int(heads):
			raise ValueError(
				f"matrix cell d_model={width} x n_heads={heads} is not buildable "
				f"({width} % {heads} != 0). Constrain the matrix to head counts that "
				f"divide every d_model in it."
			)
		suffix = "_".join(
			f"{abbrev.get(key, _abbreviate(key))}{value}" for key, value in cell.items()
		)
		entries.append({
			"name": f"{prefix}_{suffix}" if prefix else suffix,
			"overrides": merged_cell,
			# Kept so a figure can label the cell by its axes without re-parsing
			# the name, and so the config comment survives into run.json.
			"_cell": cell,
			"_comment": experiment.get("_comment"),
		})
	return entries


def merged_config(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	"""A copy of base_config with one experiment's overrides applied."""
	config = copy.deepcopy(base_config)
	config.update(overrides)
	return config


def seed_list(config: dict[str, Any]) -> list[int]:
	seeds = config.get("seeds")
	if not seeds:
		return [int(config.get("seed", config.get("split_seed", 42)))]
	return [int(seed) for seed in seeds]


def expand_experiments(
	config: dict[str, Any], selected: str | None = None
) -> list[tuple[str, dict[str, Any]]]:
	"""``[(run_name, merged_config), ...]`` in declaration order, seeds innermost.

	``selected`` may be a run name (``x_seed7``) or an experiment name (``x``,
	which selects all of its seeds).
	"""
	declared = config.get("experiments") or [{"name": "base", "overrides": {}}]
	experiments = [cell for entry in declared for cell in expand_matrix(entry, config)]
	seeds = seed_list(config)
	multi = bool(config.get("seeds")) and len(seeds) > 1
	runs = []
	for index, experiment in enumerate(experiments, start=1):
		experiment_name = str(experiment.get("name", f"experiment_{index}"))
		overrides = dict(experiment.get("overrides", {}))
		for seed in seeds:
			run_name = f"{experiment_name}_seed{seed}" if multi else experiment_name
			if selected is not None and selected not in (experiment_name, run_name):
				continue
			merged = merged_config(config, overrides)
			if multi:
				merged["seed"] = seed
			runs.append((run_name, merged))
	if selected is not None and not runs:
		available = ", ".join(str(e.get("name")) for e in experiments)
		raise ValueError(f"Unknown experiment '{selected}'. Available: {available}")
	return runs


def experiment_names(config: dict[str, Any], selected: str | None = None) -> list[str]:
	return [name for name, _ in expand_experiments(config, selected)]


def base_name(run_name: str) -> str:
	"""``x_seed7`` -> ``x``; anything else unchanged."""
	return SEED_SUFFIX.sub("", run_name)
