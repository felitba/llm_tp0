"""Expand a config file's ``experiments`` list into the runs it actually means.

Two axes live in the config: the ``experiments`` entries (what varies) and the
optional top-level ``"seeds": [42, 7, 1234]`` (how many times each is trained).
With more than one seed, every experiment becomes one run per seed, named
``<experiment>_seed<n>``, and the figures group them back by everything except
the seed (``plots/config_comparison.py``). Without ``seeds`` nothing changes:
one run per experiment, named as declared.

Every reader of a config (main, replot, reselect, the analysis scripts) goes
through ``experiment_names`` so they all agree on what a batch contains.
"""

from __future__ import annotations

import copy
import re
from typing import Any

SEED_SUFFIX = re.compile(r"_seed\d+$")


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
	experiments = config.get("experiments") or [{"name": "base", "overrides": {}}]
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
