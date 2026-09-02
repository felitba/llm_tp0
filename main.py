from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config.config import load_config
from config.experiments import expand_experiments, experiment_names
from dataset.preprocess_dataset import categorical_cardinalities, get_data_processed
from dataset.product_dataset import ProductDataLoaders, create_data_loaders
from plots.experiment_plots import plot_run, plot_runs_combined
from plots.pr_auc import pr_auc_score
from plots.roc_auc import roc_auc_score
from metrics.error_analysis import build_error_analysis, error_summary, write_error_analysis
from metrics.run_results import (
	RunResults, experiments_dir, output_dir_from_config, run_dir, save_run, write_summary_csv,
)
from model.checkpoint import save_checkpoint
from model.encoder_only_model import EncoderOnlyModel


def set_seed(seed: int = 42) -> None:
	"""Make runs repeatable enough for small experiment comparisons."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def experiment_configs(
	base_config: dict[str, Any], selected_name: str | None = None
) -> list[tuple[str, dict[str, Any]]]:
	"""The runs a config file declares: experiments x seeds (config/experiments.py)."""
	return expand_experiments(base_config, selected_name)


def run_epoch(
	model: EncoderOnlyModel,
	loader,
	criterion,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None = None,
	print_cls_btr: bool = False,
	split_name: str = "",
	cls_output_path: Path | None = None,
	epoch: int | None = None,
) -> dict[str, float | np.ndarray]:
	"""Run one train or evaluation epoch."""
	is_train = optimizer is not None
	model.train(is_train)
	total_loss = 0.0
	all_logits = []
	all_labels = []
	all_row_ids = []
	cls_rows = []

	for batch in loader:
		numeric = batch["numeric"].to(device)
		categorical = batch["categorical"].to(device)
		title_ids = batch["title_ids"].to(device)
		labels = batch["label"].to(device)

		if is_train:
			optimizer.zero_grad()

		logits = model(numeric=numeric, categorical=categorical, title_ids=title_ids)
		loss = criterion(logits, labels)

		if print_cls_btr:
			product_ids = next(
				(batch[key] for key in ("product_id", "product_ids", "id") if key in batch),
				range(len(labels)),
			)
			if torch.is_tensor(product_ids):
				product_ids = product_ids.detach().cpu().tolist()
			for product_id, cls_btr, label in zip(
				product_ids,
				logits.detach().cpu().reshape(-1).tolist(),
				labels.detach().cpu().reshape(-1).tolist(),
			):
				probability = torch.sigmoid(torch.tensor(cls_btr)).item()
				cls_rows.append({
					"epoch": epoch,
					"split": split_name,
					"product_id": product_id,
					"cls_btr": cls_btr,
					"probability": probability,
					"bought": int(label),
				})

		if is_train:
			loss.backward()
			optimizer.step()

		total_loss += loss.item() * labels.size(0)
		all_logits.append(logits.detach().cpu())
		all_labels.append(labels.detach().cpu())
		all_row_ids.append(batch["row_id"].detach().cpu())

	if cls_output_path is not None and cls_rows:
		cls_output_path.parent.mkdir(parents=True, exist_ok=True)
		write_header = not cls_output_path.exists()
		with cls_output_path.open("a", newline="", encoding="utf-8") as file:
			writer = csv.DictWriter(file, fieldnames=cls_rows[0].keys())
			if write_header:
				writer.writeheader()
			writer.writerows(cls_rows)

	logits_np = torch.cat(all_logits).numpy().reshape(-1)
	labels_np = torch.cat(all_labels).numpy().reshape(-1)
	probs = torch.sigmoid(torch.tensor(logits_np)).numpy()
	has_both_classes = len(np.unique(labels_np)) > 1

	return {
		"loss": total_loss / len(loader.dataset),
		"row_ids": torch.cat(all_row_ids).numpy().reshape(-1),
		"roc_auc": roc_auc_score(labels_np.astype(int), probs) if has_both_classes else float("nan"),
		"pr_auc": pr_auc_score(labels_np.astype(int), probs) if has_both_classes else float("nan"),
		"labels": labels_np,
		"probs": probs,
	}


def train_one_experiment(
	name: str,
	config: dict[str, Any],
	device: torch.device,
	forced_epochs: int | None = None,
	save_plots: bool = True,
	print_cls_btr: bool = False,
	config_file: str | None = None,
	save_weights: bool = False,
) -> RunResults:
	"""Train one configured experiment and return everything it produced."""
	started_at = time.perf_counter()
	set_seed(int(config.get("seed", config.get("split_seed", 42))))
	splits = get_data_processed(config)
	loaders: ProductDataLoaders = create_data_loaders(splits, config)
	cardinalities = categorical_cardinalities(splits["train"])

	model = EncoderOnlyModel(cardinalities=cardinalities, config=config).to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=float(config.get("learning_rate", 3e-4)),
		weight_decay=float(config.get("weight_decay", 0.01)),
	)

	history = {"train": [], "val": []}
	# Keep every validation metric: the configured primary metric picks the
	# checkpoint, while loss remains useful for diagnosing overfitting.
	epoch_metrics: list[dict[str, float]] = []
	epochs = forced_epochs if forced_epochs is not None else int(config.get("epochs", 6))
	early_stopping_patience = int(config.get("early_stopping_patience", 3))
	early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
	early_stopping_metric = str(config.get("early_stopping_metric", "val_pr_auc"))
	metric_direction = {
		"val_loss": "min",
		"val_roc_auc": "max",
		"val_pr_auc": "max",
	}.get(early_stopping_metric)
	if metric_direction is None:
		raise ValueError(
			"early_stopping_metric must be one of val_loss, val_roc_auc or val_pr_auc; "
			f"got {early_stopping_metric!r}"
		)
	best_metric = float("inf") if metric_direction == "min" else float("-inf")
	best_epoch = None
	best_model_state = None
	epochs_without_improvement = 0
	# Validation and test probabilities at every epoch. Test is scored here but
	# never read here: selection below uses validation only. Storing it is what
	# lets scripts/reselect.py apply another checkpoint rule without retraining.
	# The eval loaders do not shuffle, so row order is the same at every epoch.
	epoch_val_probs: list[np.ndarray] = []
	epoch_test_probs: list[np.ndarray] = []
	eval_reference: dict[str, np.ndarray] = {}
	cls_output_path = run_dir(name) / "cls_btr_comparison.csv"
	if print_cls_btr and cls_output_path.exists():
		cls_output_path.unlink()

	for epoch in range(1, epochs + 1):
		train_metrics = run_epoch(
			model, loaders.train, criterion, device, optimizer,
			print_cls_btr=print_cls_btr, split_name="train",
			cls_output_path=cls_output_path if print_cls_btr else None, epoch=epoch,
		)
		val_metrics = run_epoch(
			model, loaders.validation, criterion, device,
			print_cls_btr=print_cls_btr, split_name="validation",
			cls_output_path=cls_output_path if print_cls_btr else None, epoch=epoch,
		)
		test_epoch = run_epoch(model, loaders.test, criterion, device)
		# CHANGED (2026-09-02): the reported train curve comes from a second,
		# gradient-free pass in eval mode, so train and val are measured under the
		# same conditions (dropout off, end-of-epoch weights). The in-training
		# loss (dropout on, averaged while the weights move) sits ~0.006 above the
		# fair number and made the curves cross on arms that barely overfit; it is
		# kept as train_loss_optim because it is what the optimizer actually sees
		# (the lr divergence diagnosis reads it). To go back: drop this pass and
		# read train_metrics for history/epoch_metrics again.
		train_eval = run_epoch(model, loaders.train, criterion, device)
		epoch_val_probs.append(np.asarray(val_metrics["probs"], dtype=np.float32))
		epoch_test_probs.append(np.asarray(test_epoch["probs"], dtype=np.float32))
		if not eval_reference:
			eval_reference = {
				"val_labels": np.asarray(val_metrics["labels"], dtype=np.int8),
				"val_row_ids": np.asarray(val_metrics["row_ids"], dtype=np.int64),
				"test_labels": np.asarray(test_epoch["labels"], dtype=np.int8),
				"test_row_ids": np.asarray(test_epoch["row_ids"], dtype=np.int64),
			}
		history["train"].append(float(train_eval["loss"]))
		history["val"].append(float(val_metrics["loss"]))
		epoch_metrics.append({
			"epoch": epoch,
			"train_loss": float(train_eval["loss"]),
			"train_loss_optim": float(train_metrics["loss"]),
			"train_roc_auc": float(train_eval["roc_auc"]),
			"train_pr_auc": float(train_eval["pr_auc"]),
			"val_loss": float(val_metrics["loss"]),
			"val_roc_auc": float(val_metrics["roc_auc"]),
			"val_pr_auc": float(val_metrics["pr_auc"]),
		})

		print(
			f"[{name}] Epoch {epoch:02d}/{epochs} | "
			f"train_loss={train_eval['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
			f"val_roc_auc={val_metrics['roc_auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
		)

		current_metric = float({
			"val_loss": val_metrics["loss"],
			"val_roc_auc": val_metrics["roc_auc"],
			"val_pr_auc": val_metrics["pr_auc"],
		}[early_stopping_metric])
		improved = (
			current_metric < best_metric - early_stopping_min_delta
			if metric_direction == "min"
			else current_metric > best_metric + early_stopping_min_delta
		)
		if improved:
			best_metric = current_metric
			best_epoch = epoch
			best_model_state = copy.deepcopy(model.state_dict())
			epochs_without_improvement = 0
		elif early_stopping_patience > 0:
			epochs_without_improvement += 1
			if epochs_without_improvement >= early_stopping_patience:
				print(f"[{name}] Early stopping at epoch {epoch}.")
				break

	completed_epochs = len(history["val"])
	# best_epoch is the epoch whose weights the test row reports (best
	# early_stopping_metric). The val-loss minimum is recorded next to it so a
	# checkpoint picked on a rank metric can be checked against a proper scoring
	# rule: a large gap between the two means PR-AUC kept drifting up on a small
	# validation set while the model was already overconfident.
	min_val_loss_epoch = (
		min(range(len(history["val"])), key=history["val"].__getitem__) + 1
		if history["val"]
		else None
	)
	if best_model_state is not None:
		model.load_state_dict(best_model_state)
	selected = epoch_metrics[best_epoch - 1] if best_epoch else {}
	selection = {
		"metric": early_stopping_metric,
		"epoch": best_epoch,
		"val_loss": selected.get("val_loss"),
		"val_pr_auc": selected.get("val_pr_auc"),
		"val_roc_auc": selected.get("val_roc_auc"),
		"min_val_loss_epoch": min_val_loss_epoch,
		"min_val_loss": min(history["val"]) if history["val"] else None,
	}
	if best_epoch and min_val_loss_epoch:
		by_metric = (
			f"{early_stopping_metric}={selected.get(early_stopping_metric):.4f}, "
			if early_stopping_metric != "val_loss"
			else ""
		)
		print(
			f"[{name}] checkpoint: epoch {best_epoch} ({by_metric}val_loss={selected['val_loss']:.4f}); "
			f"min val_loss={selection['min_val_loss']:.4f} at epoch {min_val_loss_epoch}"
		)

	test_metrics = run_epoch(
		model, loaders.test, criterion, device,
		print_cls_btr=print_cls_btr, split_name="test",
		cls_output_path=cls_output_path if print_cls_btr else None, epoch=completed_epochs,
	)
	# Score train and validation with the same selected weights the test row uses.
	# Only the scores make a reliability diagram or a validation threshold sweep
	# possible later, and a finished run cannot produce them after the fact.
	split_predictions = {
		"test": (
			np.asarray(test_metrics["labels"], dtype=int),
			np.asarray(test_metrics["probs"], dtype=float),
		)
	}
	split_row_ids = {"test": np.asarray(test_metrics["row_ids"], dtype=int)}
	for split_name, loader in (("train", loaders.train), ("validation", loaders.validation)):
		split_metrics = run_epoch(model, loader, criterion, device)
		split_predictions[split_name] = (
			np.asarray(split_metrics["labels"], dtype=int),
			np.asarray(split_metrics["probs"], dtype=float),
		)
		split_row_ids[split_name] = np.asarray(split_metrics["row_ids"], dtype=int)

	# The test rows as a person can read them, worst prediction first. This is the
	# only artifact that answers "what kind of product does it keep getting wrong",
	# which is a question about rows and cannot be read off an AUC.
	test_row_ids = split_row_ids["test"]
	errors = build_error_analysis(
		splits["test"], test_row_ids,
		test_metrics["labels"], test_metrics["probs"],
	)
	print(f"[{name}] Error analysis written to: {write_error_analysis(run_dir(name), errors)}")
	for column in ("title_tag", "category"):
		summary = error_summary(errors, column)
		if not summary.empty:
			worst = summary.head(3)
			print(f"[{name}]   peor error medio por {column}: " + ", ".join(
				f"{index} ({row.error_medio:.3f}, n={int(row.filas)})"
				for index, row in worst.iterrows()
			))
	query_ids = (
		splits["test"].loc[test_row_ids, "query_id"].to_numpy()
		if "query_id" in splits["test"].columns
		else np.empty(0, dtype=object)
	)

	if print_cls_btr:
		print(f"CLS comparison written to: {cls_output_path}")
	print(
		f"[{name}] TEST | loss={test_metrics['loss']:.4f} "
		f"roc_auc={test_metrics['roc_auc']:.4f} pr_auc={test_metrics['pr_auc']:.4f}"
	)

	summary_row = {
		"name": name,
		"epochs": completed_epochs,
		"d_model": int(config.get("d_model")),
		"n_heads": int(config.get("n_heads")),
		"num_layers": int(config.get("num_layers")),
		"learning_rate": float(config.get("learning_rate")),
		"batch_size": int(config.get("batch_size", 64)),
		# How the row reached the encoder, so two rows of this file can be told
		# apart when the ablation is representation rather than architecture.
		"n_text_cols": len(splits["train"].attrs.get("text_columns", [])),
		"max_title_len": int(config.get("max_title_len")),
		"n_numeric_cols": len(config.get("numeric_columns", [])),
		"n_categorical_cols": len(config.get("categorical_columns", [])),
		# Which weights the test columns describe, and by what rule they were picked.
		"selection_metric": early_stopping_metric,
		"selected_epoch": best_epoch,
		"min_val_loss_epoch": min_val_loss_epoch,
		"test_loss": float(test_metrics["loss"]),
		"test_roc_auc": float(test_metrics["roc_auc"]),
		"test_pr_auc": float(test_metrics["pr_auc"]),
	}

	results = RunResults(
		name=name,
		config=config,
		history=history,
		epoch_metrics=epoch_metrics,
		selection=selection,
		test={
			"loss": float(test_metrics["loss"]),
			"roc_auc": float(test_metrics["roc_auc"]),
			"pr_auc": float(test_metrics["pr_auc"]),
		},
		summary=summary_row,
		labels=np.asarray(test_metrics["labels"], dtype=int),
		probs=np.asarray(test_metrics["probs"], dtype=float),
		config_file=config_file,
		split_predictions=split_predictions,
		split_row_ids=split_row_ids,
		row_ids=test_row_ids,
		query_ids=query_ids,
		epoch_predictions={
			"epochs": np.arange(1, completed_epochs + 1, dtype=np.int64),
			"val_probs": np.stack(epoch_val_probs),
			"test_probs": np.stack(epoch_test_probs),
			**eval_reference,
		},
		duration_seconds=time.perf_counter() - started_at,
	)
	# Save before plotting: the numbers are the expensive part, the figures are
	# a pure function of them and replot.py can rebuild those at any time.
	print(f"[{name}] Results written to: {save_run(results)}")
	if save_weights:
		# model already holds best_model_state: the weights the test row reports.
		checkpoint_path = save_checkpoint(
			name=name,
			model=model,
			cardinalities=cardinalities,
			config=config,
			categorical_id_mapping=splits["train"].attrs.get("categorical_id_mapping", {}),
			best_epoch=best_epoch,
			test=results.test,
		)
		print(f"[{name}] Weights written to: {checkpoint_path}")
	if save_plots:
		plot_run(results)
	return results


def write_experiment_summary(rows: list[dict[str, float | str]]) -> Path:
	"""Write experiment metrics to <output_dir>/summary.csv."""
	return write_summary_csv(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train BTR Transformer experiments.")
	parser.add_argument(
		"--config",
		help="Config file to read, relative to the repo root. Defaults to config/config.json.",
		default=None,
	)
	parser.add_argument(
		"--experiment",
		help="Run only this experiment (all of its seeds) or this exact run name (x_seed7).",
		default=None,
	)
	parser.add_argument(
		"--epochs",
		type=int,
		help="Override epochs for every selected experiment, useful for smoke tests.",
		default=None,
	)
	parser.add_argument(
		"--no-plots",
		action="store_true",
		help="Skip saving loss/PR/ROC plots.",
	)
	parser.add_argument(
		"--no-error-patterns",
		dest="error_patterns",
		action="store_false",
		help="Skip the batch-level error-pattern report (output/<batch>/error_patterns.txt). "
		     "It runs once at the end of a multi-run batch and reads test.",
	)
	parser.add_argument(
		"--save-weights",
		action="store_true",
		help="Also write <output_dir>/<name>/model.pt (~20 MB per experiment).",
	)
	parser.add_argument(
		"--print-cls-btr",
		action="store_true",
		help="Print each product's CLS BTR output, probability, and label.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_config = load_config(args.config) if args.config else load_config()
	# Before anything is written: runs, summary and combined figures all go here.
	print(f"Writing results under: {output_dir_from_config(base_config)}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	runs = []
	for name, config in experiment_configs(base_config, args.experiment):
		runs.append(
			train_one_experiment(
				name=name,
				config=config,
				device=device,
				forced_epochs=args.epochs,
				save_plots=not args.no_plots,
				print_cls_btr=args.print_cls_btr,
				config_file=args.config,
				save_weights=args.save_weights,
			)
		)

	summary_path = write_summary_csv(
		[run.summary for run in runs],
		order=experiment_names(base_config),
	)
	print(f"Experiment summary written to: {summary_path}")

	if not args.no_plots:
		for combined_path in plot_runs_combined(runs):
			print(f"Combined plot written to: {combined_path}")

	# Batch-level error analysis, once the whole batch exists: which test rows
	# every arm gets wrong and what they have in common. Needs the full batch
	# (it scores by consensus across arms), so it cannot run per experiment, and
	# it is written to a file rather than stdout because 33 runs of training
	# scroll it away. Never fatal: a batch that took an hour to train must not be
	# lost to a reporting bug.
	#
	# It reads TEST. That is fine as the closing step of a reporting run -- the
	# protocol in docs/PROTOCOL.md is frozen before this point, so nothing here
	# can feed back into a modelling decision. What it must not become is a thing
	# you read between runs while still changing the model.
	if args.error_patterns and len(runs) > 1 and args.config:
		try:
			from scripts.error_patterns import write_figure, write_report  # noqa: PLC0415

			batch_dir = experiments_dir()
			print(f"Error patterns written to: {write_report(args.config, batch_dir / 'error_patterns.txt')}")
			drawn = write_figure(args.config, batch_dir / "error_patterns.jpg")
			if drawn:
				print(f"Error patterns figure written to: {drawn}")
		except Exception as error:  # noqa: BLE001  (reporting must never kill a batch)
			print(f"Error-pattern report skipped ({type(error).__name__}: {error})")


if __name__ == "__main__":
	main()
