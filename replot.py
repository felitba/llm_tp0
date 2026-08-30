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
from plots.experiment_plots import plot_run, plot_runs_combined
from metrics.run_results import load_runs, saved_run_names


def names_from_config(config_file: str | Path) -> list[str]:
	"""The experiment names one config file declares, in declaration order."""
	with resolve_path(config_file).open(encoding="utf-8") as file:
		config = json.load(file)
	return [str(experiment.get("name")) for experiment in config.get("experiments", [])]


def selected_names(args: argparse.Namespace) -> list[str]:
	"""Resolve which saved runs to redraw, keeping the caller's order."""
	if args.names:
		requested = args.names
	elif args.config:
		requested = names_from_config(args.config)
	else:
		return saved_run_names()

	available = set(saved_run_names())
	missing = [name for name in requested if name not in available]
	if missing:
		raise SystemExit(
			f"No saved results for: {', '.join(missing)}. "
			f"Available: {', '.join(sorted(available)) or '(none)'}"
		)
	return requested


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
		"--no-combined",
		action="store_true",
		help="Skip the all-configs ROC/PR figures.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	runs = load_runs(selected_names(args))
	if not runs:
		raise SystemExit("Nothing to replot: no run.json under output/experiments/.")

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
