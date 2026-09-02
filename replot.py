"""Rebuild every figure from what the runs already wrote to disk.
	python replot.py                                            # every saved run
	python replot.py medium_d96_l2_baseline medium_d128_l2      # only these
	python replot.py --config config/text_vs_feature_tokens.json --suffix text_vs_feature_tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.config import resolve_path
from config.experiments import base_name, experiment_names
from plots.experiment_plots import plot_run, plot_runs_combined
from metrics.run_results import (
	experiments_dir, load_runs, output_dir_from_config, saved_run_names, set_experiments_dir,
)


def names_from_config(config_file: str | Path) -> list[str]:
	"""The experiment names one config file declares, in declaration order.

	Also points the reader at that config's ``output_dir``, so the runs are
	looked up where main.py wrote them.
	"""
	with resolve_path(config_file).open(encoding="utf-8") as file:
		config = json.load(file)
	output_dir_from_config(config)
	return experiment_names(config)


def selected_names(args: argparse.Namespace) -> list[str]:
	"""Resolve which saved runs to redraw, keeping the caller's order."""
	if args.config:
		requested = names_from_config(args.config)
		if args.names:
			requested = args.names
	elif args.names:
		requested = args.names
	else:
		return saved_run_names()

	saved = saved_run_names()
	available = set(saved)
	# An experiment name stands for all of its seeds, the same way
	# `main.py --experiment <arm>` runs all three. Without this, redrawing one arm
	# of a seeded batch means spelling out `<arm>_seed42 <arm>_seed7 <arm>_seed1234`,
	# and forgetting one silently produces a figure whose "3 semillas" label is a lie.
	resolved: list[str] = []
	for name in requested:
		if name in available:
			resolved.append(name)
			continue
		seeded = [run for run in saved if base_name(run) == name]
		resolved.extend(seeded if seeded else [name])

	missing = [name for name in resolved if name not in available]
	if missing:
		raise SystemExit(
			f"No saved results for: {', '.join(missing)}. "
			f"Available: {', '.join(sorted(available)) or '(none)'}"
		)
	return resolved


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"names",
		nargs="*",
		help="Experiment names to redraw. Defaults to every saved run.",
	)
	parser.add_argument(
		"--config",
		help="Redraw the experiments declared in this config file instead.",
		default=None,
	)
	parser.add_argument(
		"--suffix",
		help="Name the combined figures roc/pr_auc_all_configs_<suffix>.jpg.",
		default="",
	)
	parser.add_argument(
		"--output-dir",
		help="Batch folder to read and write (default: the config's output_dir, else output/experiments).",
		default=None,
	)
	parser.add_argument(
		"--no-combined",
		action="store_true",
		help="Skip the all-configs ROC/PR figures.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.output_dir:
		set_experiments_dir(args.output_dir)
	names = selected_names(args)
	if args.output_dir:
		set_experiments_dir(args.output_dir)  # an explicit flag beats the config's key
	runs = load_runs(names)
	if not runs:
		raise SystemExit(f"Nothing to replot: no run.json under {experiments_dir()}.")

	for run in runs:
		for path in plot_run(run):
			print(f"Wrote: {path}")
		print(
			f"[{run.name}] test loss={run.test.get('loss', float('nan')):.4f} "
			f"roc_auc={run.test.get('roc_auc', float('nan')):.4f} "
			f"pr_auc={run.test.get('pr_auc', float('nan')):.4f}"
		)

	if not args.no_combined:
		for path in plot_runs_combined(runs, suffix=args.suffix):
			print(f"Wrote: {path}")


if __name__ == "__main__":
	main()
