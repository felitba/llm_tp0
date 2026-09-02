"""Apply a checkpoint rule to saved runs and re-score test, without retraining.

    python scripts/reselect.py --config config/01_entrada.json                    # table only
    python scripts/reselect.py --config config/01_entrada.json --rule val_pr_auc  # another rule
    python scripts/reselect.py --config config/01_entrada.json --rule epoch --epoch 20
    python scripts/reselect.py --config config/01_entrada.json --apply            # rewrite runs + figures

Works on runs that stored per-epoch predictions (``epoch_predictions.npz``,
written by main.py since the min-val-loss change). The rule reads validation
only; test is looked up at the chosen epoch afterwards. ``--rule epoch`` is the
one exception, it is for exploring the curves, not for reporting a result.

``--apply`` rewrites each run's test block, selection, summary row,
test/validation prediction files and error analysis for the chosen epoch, then
redraws the per-run and combined figures. Train predictions and any saved
weights belong to the originally selected checkpoint and cannot be regenerated
from probabilities, so the train file is removed and model.pt is flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import resolve_path  # noqa: E402
from config.experiments import base_name, experiment_names  # noqa: E402
from metrics.error_analysis import build_error_analysis, write_error_analysis  # noqa: E402
from metrics.final_table import aggregate, final_results, format_table, write_final_table  # noqa: E402
from metrics.run_results import (  # noqa: E402
	RunResults, SPLIT_PREDICTION_FILES, experiments_dir, load_run, output_dir_from_config,
	query_ids_for, run_dir, save_run, write_summary_csv,
)
from plots.config_comparison import plot_score_matrix, plot_seed_spread, plot_test_scores  # noqa: E402
from plots.experiment_plots import plot_run, plot_runs_combined, shared_hyperparameters  # noqa: E402
from plots.plot_theme import save  # noqa: E402
from plots.pr_auc import pr_auc_score  # noqa: E402
from plots.roc_auc import roc_auc_score  # noqa: E402

RULES = ("val_loss", "val_pr_auc", "val_pr_auc_smooth", "val_roc_auc", "epoch", "ensemble")
# ``val_pr_auc_smooth``: argmax of val_pr_auc after a centred moving average over
# SMOOTH_WINDOW epochs. Still returns ONE real epoch whose weights exist on disk,
# but it cannot land on an isolated one-epoch spike, which is the failure mode of
# plain argmax on 131 validation positives. In the validation-only bake-off
# (select on half the queries, score on the other half, 200 splits x 10 arms) it
# scored 0.7546 against 0.7495 for plain argmax and 0.7459 for min val_loss.
SMOOTH_WINDOW = 5
# ``ensemble``: no checkpoint at all. The reported probabilities are the mean of
# the per-epoch probabilities from --from to the last epoch (no upper cutoff).
# No epoch is chosen, so no epoch can be chosen luckily. This is a checkpoint
# ensemble / "horizontal voting" (Xie et al. 2013; Chen et al. 2017).
#
# Why not pick an epoch: the query-level bootstrap SE of validation AP is
# 0.032-0.050, while the epoch-to-epoch SD of validation AP over epochs 10-50 is
# only 0.007-0.029. The validation set cannot resolve epochs, so argmax over 41
# of them is a maximum of noisy estimates and is optimistically biased by
# construction (Jensen & Cohen 2000; Cawley & Talbot 2010).
#
# Window: discard the first BURN_IN_FRACTION of the epoch budget, average the
# rest. With the standard 50 epochs that is epochs 11..50.
#
# Stated as a fraction of the budget on purpose. The window start is NOT derived
# from the data: the sensitivity analysis says the data cannot inform it (any
# start in 1..41 gives the same arm ranking, and the largest AP change between
# start 10 and start 13 is 0.009, against a bootstrap CI half-width of ~0.075).
# When a choice cannot be informed by the data, a fixed convention is strictly
# more defensible than a data-derived value: there is nothing to accuse of being
# tuned, and nothing to leak. A fraction also survives a change of budget --
# 30 epochs would give a burn-in of 6 with no new decision to justify.
#
# 20% mirrors the "discard the early transient, average the converged tail"
# convention of checkpoint averaging (Vaswani et al. 2017 average the last N;
# Izmailov et al. 2018 start SWA at 75% of budget). The exact fraction is a
# design choice, and the report says so.
BURN_IN_FRACTION = 0.2


def ensemble_start(total_epochs: int, fraction: float = BURN_IN_FRACTION) -> int:
	"""First epoch averaged: the one after the burn-in fraction of the budget."""
	return int(total_epochs * fraction) + 1


# Kept for the CLI default; overridden per-run once the epoch count is known.
DEFAULT_ENSEMBLE_FROM = ensemble_start(50)


def bce_loss(labels: np.ndarray, probs: np.ndarray) -> float:
	"""Mean binary cross-entropy from probabilities, same quantity main.py reports."""
	clipped = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
	return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def per_epoch_val_metric(run: RunResults, rule: str) -> np.ndarray | None:
	"""Recompute a per-epoch validation metric from the stored probabilities.

	``run.epoch_metrics`` is whatever the code wrote at training time, so runs
	trained before the trapezoid -> Average Precision fix carry a different
	quantity under the same key (up to 0.063 apart on the coarse-scored arms).
	Recomputing here makes the rule mean the same thing for every run in a batch,
	whatever version trained it. Returns None when the run predates
	``epoch_predictions.npz`` and cannot be rescored.
	"""
	stored = run.epoch_predictions
	if not stored or rule == "val_roc_auc":
		return None
	labels = stored["val_labels"].astype(int)
	probs = stored["val_probs"].astype(float)
	if rule == "val_loss":
		return np.array([bce_loss(labels, p) for p in probs])
	return np.array([pr_auc_score(labels, p) for p in probs])


def choose_epoch(run: RunResults, rule: str, fixed_epoch: int | None) -> int:
	metrics = run.epoch_metrics
	if rule == "epoch":
		if fixed_epoch is None:
			raise SystemExit("--rule epoch needs --epoch N")
		available = [int(row["epoch"]) for row in metrics]
		if fixed_epoch not in available:
			raise SystemExit(f"{run.name}: epoch {fixed_epoch} not in 1..{max(available)}")
		return fixed_epoch
	fresh = per_epoch_val_metric(run, rule)
	if rule == "val_pr_auc_smooth":
		values = fresh if fresh is not None else np.array([row["val_pr_auc"] for row in metrics], dtype=float)
		half = SMOOTH_WINDOW // 2
		# Edge-padded so early and late epochs stay eligible instead of dropping out.
		padded = np.pad(values, half, mode="edge")
		smoothed = np.convolve(padded, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")
		return int(metrics[int(np.argmax(smoothed))]["epoch"])
	if fresh is not None:
		index = int(np.argmin(fresh) if rule == "val_loss" else np.argmax(fresh))
		return int(metrics[index]["epoch"])
	pick = min if rule == "val_loss" else max
	return int(pick(metrics, key=lambda row: row[rule])["epoch"])


def selected_probs(run: RunResults, epoch: int | None, ensemble_from: int) -> dict[str, np.ndarray]:
	"""Validation and test probabilities for one epoch, or their mean from ``ensemble_from`` on."""
	stored = run.epoch_predictions
	if epoch is None:
		mask = stored["epochs"] >= ensemble_from
		return {
			"val": stored["val_probs"][mask].astype(float).mean(axis=0),
			"test": stored["test_probs"][mask].astype(float).mean(axis=0),
		}
	index = int(np.where(stored["epochs"] == epoch)[0][0])
	return {
		"val": stored["val_probs"][index].astype(float),
		"test": stored["test_probs"][index].astype(float),
	}


def rescore(run: RunResults, epoch: int | None, ensemble_from: int = DEFAULT_ENSEMBLE_FROM) -> dict[str, float]:
	labels = run.epoch_predictions["test_labels"].astype(int)
	probs = selected_probs(run, epoch, ensemble_from)["test"]
	return {
		"loss": bce_loss(labels, probs),
		"roc_auc": float(roc_auc_score(labels, probs)),
		"pr_auc": float(pr_auc_score(labels, probs)),
	}


def apply_selection(
	run: RunResults, rule: str, epoch: int | None, test: dict[str, float],
	ensemble_from: int = DEFAULT_ENSEMBLE_FROM,
) -> RunResults:
	"""Rewrite the run in memory so that every derived artifact describes the selection."""
	stored = run.epoch_predictions
	min_val_loss_row = min(run.epoch_metrics, key=lambda item: item["val_loss"])
	probs = selected_probs(run, epoch, ensemble_from)
	val_labels = stored["val_labels"].astype(int)

	if epoch is None:
		last = int(stored["epochs"].max())
		metric_name = f"ensemble_epochs_{ensemble_from}_{last}"
		epoch_label = f"{ensemble_from}–{last}"
		val_loss, val_pr = bce_loss(val_labels, probs["val"]), float(pr_auc_score(val_labels, probs["val"]))
		val_roc = float(roc_auc_score(val_labels, probs["val"]))
	else:
		row = next(item for item in run.epoch_metrics if int(item["epoch"]) == epoch)
		metric_name = rule if rule != "epoch" else "fixed_epoch"
		epoch_label = epoch
		val_loss, val_pr, val_roc = row["val_loss"], row["val_pr_auc"], row["val_roc_auc"]

	run.test = dict(test)
	run.selection = {
		"metric": metric_name,
		"epoch": epoch_label,
		"val_loss": val_loss,
		"val_pr_auc": val_pr,
		"val_roc_auc": val_roc,
		"min_val_loss_epoch": int(min_val_loss_row["epoch"]),
		"min_val_loss": min_val_loss_row["val_loss"],
		"reselected": True,
	}
	run.summary.update({
		"selection_metric": metric_name,
		"selected_epoch": epoch_label,
		"min_val_loss_epoch": run.selection["min_val_loss_epoch"],
		"test_loss": test["loss"],
		"test_roc_auc": test["roc_auc"],
		"test_pr_auc": test["pr_auc"],
	})
	run.labels = stored["test_labels"].astype(int)
	run.probs = probs["test"]
	run.row_ids = stored["test_row_ids"].astype(int)
	run.split_predictions = {
		"test": (run.labels, run.probs),
		"validation": (val_labels, probs["val"]),
	}
	run.split_row_ids = {"test": run.row_ids, "validation": stored["val_row_ids"].astype(int)}
	return run


def names_from_config(config_file: str) -> list[str]:
	with resolve_path(config_file).open(encoding="utf-8") as file:
		config = json.load(file)
	output_dir_from_config(config)
	return experiment_names(config)


def selected_names(declared: list[str], requested: list[str]) -> list[str]:
	"""Resolve which of the config's runs to reselect, keeping the caller's order.

	Same contract as replot.py: an experiment name stands for all of its seeds,
	so a subset comparison never silently drops a seed.
	"""
	if not requested:
		return declared
	available = set(declared)
	resolved: list[str] = []
	for name in requested:
		if name in available:
			resolved.append(name)
			continue
		seeded = [run for run in declared if base_name(run) == name]
		resolved.extend(seeded if seeded else [name])
	unknown = [name for name in resolved if name not in available]
	if unknown:
		raise SystemExit(
			f"Not declared in the config: {', '.join(unknown)}. "
			f"Declared: {', '.join(declared)}"
		)
	return resolved


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"names",
		nargs="*",
		help="Experiment names to reselect (an arm name covers all its seeds). "
		     "Defaults to every experiment the config declares.",
	)
	parser.add_argument("--config", required=True, help="config whose runs to reselect")
	parser.add_argument("--rule", choices=RULES, default="val_loss")
	parser.add_argument("--epoch", type=int, default=None, help="with --rule epoch")
	parser.add_argument("--from", dest="ensemble_from", type=int, default=None,
	                    # argparse runs help through %-formatting, so % must be doubled.
	                    help=f"with --rule ensemble: first epoch averaged, through the last. "
	                         f"Default: after the first {BURN_IN_FRACTION:.0%} of each run's epoch "
	                         f"budget ({DEFAULT_ENSEMBLE_FROM} for 50 epochs). "
	                         f"See docs/PROTOCOL.md".replace("%", "%%"))
	parser.add_argument("--apply", action="store_true", help="rewrite runs, summary and figures")
	parser.add_argument("--final", action="store_true",
	                    help="read and report the TEST predictions. Without it only validation is "
	                         "scored, so exploring rules cannot leak test. Protocol frozen "
	                         "2026-09-01; see docs/PROTOCOL.md")
	parser.add_argument("--suffix", default="", help="suffix for the combined figure names")
	args = parser.parse_args()

	names = selected_names(names_from_config(args.config), args.names)
	print(f"batch folder: {experiments_dir()}   rule: {args.rule}"
	      + (f" (epoch {args.epoch})" if args.rule == "epoch" else "")
	      + (f" (mean of the epochs after the first {BURN_IN_FRACTION:.0%} of the budget)"
	         if args.rule == "ensemble" and args.ensemble_from is None
	         else f" (mean of epochs {args.ensemble_from}..end)" if args.rule == "ensemble" else ""))
	header = (f"{'experiment':<30}{'was':>10}{'now':>6}{'val_loss':>10}{'val_ap':>8}"
	          + (f"{'test_loss':>11}{'test_roc':>10}{'test_ap':>9}{'was_ap':>9}"
	             if args.final else f"{'test':>39}"))
	print(header)
	print("-" * len(header))

	updated: list[RunResults] = []
	rescored: list[RunResults] = []
	validation_rows: list[dict] = []
	for name in names:
		try:
			run = load_run(name)
		except FileNotFoundError:
			print(f"{name:<30} (no run.json)")
			continue
		if not run.epoch_predictions:
			print(f"{name:<30} (no epoch_predictions.npz: re-run to store per-epoch scores)")
			continue
		ensemble_from = args.ensemble_from or ensemble_start(int(run.epoch_predictions["epochs"].max()))
		epoch = None if args.rule == "ensemble" else choose_epoch(run, args.rule, args.epoch)
		# Test stays closed unless --final. The rule is chosen on validation; reading
		# test while exploring rules is what turns test into a second validation set.
		test = rescore(run, epoch, ensemble_from) if args.final else None
		val_labels = run.epoch_predictions["val_labels"].astype(int)
		val = selected_probs(run, epoch, ensemble_from)["val"]
		val_queries = query_ids_for(run, "validation")
		if epoch is None:
			val_loss, val_pr = bce_loss(val_labels, val), float(pr_auc_score(val_labels, val))
			now = f"{ensemble_from}+"
		else:
			row = next(item for item in run.epoch_metrics if int(item["epoch"]) == epoch)
			val_loss, val_pr, now = row["val_loss"], row["val_pr_auc"], str(epoch)
		validation_rows.append({
			"config": base_name(name),
			"seed": int(run.config.get("seed", 0)),
			"epoch": epoch if epoch is not None else now,
			"pr_auc": float(pr_auc_score(val_labels, val)),
			"roc_auc": float(roc_auc_score(val_labels, val)),
			"labels": val_labels, "probs": val, "query_ids": val_queries,
		})
		# The validation figures are drawn from THIS epoch, not from the
		# validation_predictions.csv on disk: that file belongs to whatever
		# checkpoint the run was saved with, and reselecting is exactly the case
		# where the two differ. In memory only -- nothing is written unless --apply.
		run.split_predictions["validation"] = (val_labels, val)
		run.split_row_ids["validation"] = run.epoch_predictions["val_row_ids"].astype(int)
		rescored.append(run)
		was_epoch = (run.selection or {}).get("epoch")
		test_columns = (
			f"{test['loss']:>11.4f}{test['roc_auc']:>10.3f}{test['pr_auc']:>9.3f}"
			f"{run.test.get('pr_auc', float('nan')):>9.3f}"
			if test is not None else f"{'(--final)':>39}"
		)
		print(
			f"{name:<30}{str(was_epoch or '?'):>10}{now:>6}{val_loss:>10.4f}"
			f"{val_pr:>8.3f}{test_columns}"
		)
		if args.apply and test is None:
			print(f"{'':<30} --apply needs --final: nothing written")
			continue
		if args.apply:
			run = apply_selection(run, args.rule, epoch, test, ensemble_from)
			stale_train = run_dir(name) / SPLIT_PREDICTION_FILES["train"]
			if stale_train.exists():
				stale_train.unlink()
			if (run_dir(name) / "model.pt").exists():
				print(f"{'':<30} model.pt holds the ORIGINAL checkpoint's weights, not epoch {epoch}")
			save_run(run)
			# The readable error table has to describe the same scores as the test row.
			from dataset.preprocess_dataset import get_data_processed  # noqa: PLC0415  (torch-free path stays torch-free)
			splits = get_data_processed(run.config)
			errors = build_error_analysis(splits["test"], run.row_ids, run.labels, run.probs)
			write_error_analysis(run_dir(name), errors)
			plot_run(run)
			updated.append(run)

	# The comparison the ablation is decided on: one row per configuration, mean
	# and range across seeds, computed on VALIDATION. Printed whether or not
	# --final was passed, because this is the table used while test stays closed.
	if validation_rows:
		print("\n" + format_table(aggregate(validation_rows), split="val"))

	# ...and the figure of that table. Drawn here rather than in
	# plot_runs_combined because only this path knows the reselected epoch, and
	# because it must stay available while test is closed: --final is not needed.
	if rescored:
		subtitle = shared_hyperparameters(rescored)
		# Same naming rule as plot_runs_combined: `_<suffix>` when one is given.
		tail = f"_{args.suffix}" if args.suffix else ""
		for plot_fn, filename in (
			(plot_test_scores, f"val_scores_all_configs{tail}.jpg"),
			(plot_seed_spread, f"val_seed_spread_all_configs{tail}.jpg"),
			# Only drawn for the matrix steps (two swept keys); None elsewhere.
			(plot_score_matrix, f"val_scores_matrix_all_configs{tail}.jpg"),
		):
			drawn = plot_fn(rescored, subtitle=subtitle, split="validation")
			if drawn:
				print(f"figure:  {save(drawn[0], experiments_dir() / filename)}")

	if args.apply and updated:
		print(f"summary: {write_summary_csv([run.summary for run in updated])}")
		# The reporting table: one row per configuration with the three seeds
		# visible, not one row per run. This is the table the informe quotes.
		print("\n" + format_table(final_results(updated)))
		print(f"\nfinal table: {write_final_table(updated, experiments_dir() / 'resultados_finales.csv')}")
		for path in plot_runs_combined(updated, suffix=args.suffix):
			print(f"figure:  {path}")


if __name__ == "__main__":
	main()
