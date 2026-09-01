"""Persistence for what a run produced, so plots never require retraining.

Every experiment writes two files under ``output/experiments/<name>/``:

``run.json``
    the merged config, the per-epoch history, and the test metrics.
``test_predictions.csv``
    one row per test product, ``label,probability``. The ROC and PR curves are
    integrated from every distinct score (``plots/threshold_curves.py``), so
    keeping the scores keeps every curve, threshold and derived metric
    reproducible without a forward pass.

A run takes minutes and a figure takes milliseconds; the split between the two
is what ``replot.py`` uses to restyle the whole deck from disk.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.config import PROJECT_ROOT

EXPERIMENTS_DIR = PROJECT_ROOT / "output" / "experiments"
_experiments_dir = EXPERIMENTS_DIR
RUN_FILE = "run.json"
PREDICTIONS_FILE = "test_predictions.csv"
EPOCH_PREDICTIONS_FILE = "epoch_predictions.npz"
# Validation and train scores from the same selected checkpoint the test row
# reports. Test alone answers "how good is it"; these two are what a reliability
# diagram, a validation-set threshold sweep or a train-vs-test overfitting curve
# need, and none of them can be recovered from a finished run without the scores.
# Two extra inference passes at the end of a run, which is cheap next to training.
SPLIT_PREDICTION_FILES = {
	"train": "train_predictions.csv",
	"validation": "validation_predictions.csv",
	"test": PREDICTIONS_FILE,
}

# The experiment list is the same in every run.json of a batch and is by far the
# bulkiest key; the merged config already carries what this run actually used.
_CONFIG_KEYS_NOT_WORTH_STORING = ("experiments",)


@dataclass
class RunResults:
	"""One trained experiment as it was written to disk."""

	name: str
	config: dict[str, Any] = field(default_factory=dict)
	history: dict[str, list[float]] = field(default_factory=dict)
	epoch_metrics: list[dict[str, float]] = field(default_factory=list)
	selection: dict[str, Any] = field(default_factory=dict)
	# split name -> (labels, probs), for every split that was scored.
	split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
	# split name -> dataframe index of each scored row, aligned with the arrays
	# above. Needed to join a score back to the product it belongs to, and to
	# group test scores by query_id, which is how the BTR is actually defined.
	split_row_ids: dict[str, np.ndarray] = field(default_factory=dict)
	row_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
	# The query each test impression belonged to. The BTR is defined as a rate
	# over a set of impressions, and the query is that set, so this is what turns
	# per-impression probabilities back into the quantity the assignment asks for.
	query_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
	# Per-epoch scores: "epochs" (E,), "val_probs" / "test_probs" (E, N) plus the
	# matching "*_labels" and "*_row_ids" (N,). Empty for runs written before
	# these were stored; see EPOCH_PREDICTIONS_FILE.
	epoch_predictions: dict[str, np.ndarray] = field(default_factory=dict)
	# Wall clock, so "this arm costs 20x for the same score" is a number and not
	# an argument from sequence length.
	duration_seconds: float = 0.0
	test: dict[str, float] = field(default_factory=dict)
	summary: dict[str, Any] = field(default_factory=dict)
	labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
	probs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
	config_file: str | None = None
	created_at: str = ""

	@property
	def directory(self) -> Path:
		return run_dir(self.name)

	@property
	def has_curves(self) -> bool:
		"""Curves need both classes present, which a tiny test split can miss."""
		return len(self.labels) > 0 and len(np.unique(self.labels)) > 1


def experiments_dir() -> Path:
	"""The batch folder in use: output/experiments unless a config moved it."""
	return _experiments_dir


def set_experiments_dir(path: str | Path | None) -> Path:
	"""Point every reader and writer at ``path`` (repo-relative or absolute).

	``None`` restores the default. Call it once, right after loading the config
	and before anything is saved or loaded.
	"""
	global _experiments_dir
	if path is None:
		_experiments_dir = EXPERIMENTS_DIR
	else:
		path = Path(path)
		_experiments_dir = path if path.is_absolute() else PROJECT_ROOT / path
	return _experiments_dir


def output_dir_from_config(config: dict[str, Any]) -> Path:
	"""Resolve a config's ``output_dir`` and make it the batch folder."""
	return set_experiments_dir(config.get("output_dir"))


def run_dir(name: str) -> Path:
	return experiments_dir() / name


def save_run(results: RunResults) -> Path:
	"""Write one experiment's numbers and test scores. Returns the run.json path."""
	directory = run_dir(results.name)
	directory.mkdir(parents=True, exist_ok=True)

	def write_predictions(
		filename: str, labels: np.ndarray, probs: np.ndarray,
		ids: np.ndarray | None, queries: np.ndarray | None = None,
	) -> None:
		labels = np.asarray(labels, dtype=int).reshape(-1)
		probs = np.asarray(probs, dtype=float).reshape(-1)
		ids = np.asarray(ids).reshape(-1) if ids is not None and len(ids) else None
		queries = np.asarray(queries).reshape(-1) if queries is not None and len(queries) else None
		header = ["label", "probability"]
		columns = [labels.tolist(), probs.tolist()]
		if ids is not None:
			header.insert(0, "row_id"); columns.insert(0, ids.tolist())
		if queries is not None:
			header.append("query_id"); columns.append(queries.tolist())
		with (directory / filename).open("w", newline="", encoding="utf-8") as file:
			writer = csv.writer(file)
			writer.writerow(header)
			writer.writerows(zip(*columns))

	write_predictions(
		PREDICTIONS_FILE, results.labels, results.probs, results.row_ids, results.query_ids
	)
	written_splits = {}
	for split_name, (labels, probs) in results.split_predictions.items():
		filename = SPLIT_PREDICTION_FILES.get(split_name, f"{split_name}_predictions.csv")
		write_predictions(filename, labels, probs, results.split_row_ids.get(split_name))
		written_splits[split_name] = filename

	if results.epoch_predictions:
		np.savez_compressed(directory / EPOCH_PREDICTIONS_FILE, **results.epoch_predictions)

	payload = {
		"name": results.name,
		"created_at": results.created_at
		or datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"config_file": results.config_file,
		"epoch_predictions_file": EPOCH_PREDICTIONS_FILE if results.epoch_predictions else None,
		"config": {
			key: value
			for key, value in results.config.items()
			if key not in _CONFIG_KEYS_NOT_WORTH_STORING
		},
		"history": {
			key: [float(value) for value in values] for key, values in results.history.items()
		},
		"epoch_metrics": results.epoch_metrics,
		"selection": results.selection,
		"test": {key: float(value) for key, value in results.test.items()},
		"summary": results.summary,
		"predictions_file": PREDICTIONS_FILE,
		"split_prediction_files": written_splits,
		"duration_seconds": float(results.duration_seconds),
	}
	run_path = directory / RUN_FILE
	with run_path.open("w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2, default=_json_safe)
	return run_path


def load_run(name: str) -> RunResults:
	"""Read back one saved experiment."""
	directory = run_dir(name)
	run_path = directory / RUN_FILE
	if not run_path.exists():
		raise FileNotFoundError(
			f"No saved results for '{name}' ({run_path}). Run main.py for it first."
		)
	with run_path.open(encoding="utf-8") as file:
		payload = json.load(file)

	labels, probs, row_ids, query_ids = _load_predictions(
		directory / payload.get("predictions_file", PREDICTIONS_FILE)
	)
	split_predictions = {}
	split_row_ids = {}
	for split_name, filename in (payload.get("split_prediction_files") or {}).items():
		path = directory / filename
		if path.exists():
			split_labels, split_probs, split_ids, _ = _load_predictions(path)
			split_predictions[split_name] = (split_labels, split_probs)
			split_row_ids[split_name] = split_ids
	epoch_predictions: dict[str, np.ndarray] = {}
	epoch_path = directory / (payload.get("epoch_predictions_file") or EPOCH_PREDICTIONS_FILE)
	if epoch_path.exists():
		with np.load(epoch_path, allow_pickle=False) as data:
			epoch_predictions = {key: data[key] for key in data.files}
	return RunResults(
		name=payload.get("name", name),
		config=payload.get("config", {}),
		history=payload.get("history", {}),
		epoch_metrics=payload.get("epoch_metrics", []),
		selection=payload.get("selection", {}),
		test=payload.get("test", {}),
		summary=payload.get("summary", {}),
		labels=labels,
		probs=probs,
		config_file=payload.get("config_file"),
		created_at=payload.get("created_at", ""),
		split_predictions=split_predictions,
		split_row_ids=split_row_ids,
		row_ids=row_ids,
		query_ids=query_ids,
		epoch_predictions=epoch_predictions,
		duration_seconds=float(payload.get("duration_seconds", 0.0)),
	)


def write_summary_csv(rows: list[dict[str, Any]], order: list[str] | None = None) -> Path:
	"""One row per experiment at <output_dir>/summary.csv, merged by name.

	Rows already in the file survive unless a new row has the same name, so
	``main.py --experiment X`` refreshes X without wiping the rest of the batch.
	``order`` (the config's experiment names) fixes the row order; names not in
	it go last.
	"""
	experiments_dir().mkdir(parents=True, exist_ok=True)
	output_path = experiments_dir() / "summary.csv"
	merged: dict[str, dict[str, Any]] = {}
	if output_path.exists():
		with output_path.open(encoding="utf-8") as file:
			for existing in csv.DictReader(file):
				merged[existing["name"]] = dict(existing)
	for row in rows:
		merged[str(row["name"])] = dict(row)
	ranked = sorted(
		merged.values(),
		key=lambda row: ((order or []).index(row["name"]) if row["name"] in (order or []) else len(order or [])),
	)
	headers: list[str] = []
	for row in ranked:
		headers += [key for key in row if key not in headers]
	with output_path.open("w", encoding="utf-8") as file:
		file.write(",".join(headers) + "\n")
		for row in ranked:
			file.write(",".join(str(row.get(header, "")) for header in headers) + "\n")
	return output_path


def saved_run_names() -> list[str]:
	"""Every experiment directory that holds a run.json, oldest run first."""
	if not experiments_dir().exists():
		return []
	directories = [path for path in experiments_dir().iterdir() if (path / RUN_FILE).exists()]
	directories.sort(key=lambda path: (path / RUN_FILE).stat().st_mtime)
	return [path.name for path in directories]


def load_runs(names: list[str] | None = None) -> list[RunResults]:
	"""Load the named runs, or every saved run when ``names`` is None."""
	return [load_run(name) for name in (names if names is not None else saved_run_names())]


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""labels, probs, row ids and query ids. The last two come back empty for
	runs written before those columns existed."""
	empty = (np.empty(0, dtype=int), np.empty(0, dtype=float),
	         np.empty(0, dtype=int), np.empty(0, dtype=object))
	if not path.exists():
		return empty
	labels: list[int] = []
	probs: list[float] = []
	ids: list[int] = []
	queries: list[str] = []
	with path.open(encoding="utf-8") as file:
		for row in csv.DictReader(file):
			labels.append(int(float(row["label"])))
			probs.append(float(row["probability"]))
			if row.get("row_id") is not None:
				ids.append(int(float(row["row_id"])))
			if row.get("query_id") is not None:
				queries.append(row["query_id"])
	return (
		np.asarray(labels, dtype=int),
		np.asarray(probs, dtype=float),
		np.asarray(ids, dtype=int),
		np.asarray(queries, dtype=object),
	)


def _json_safe(value: Any) -> Any:
	"""numpy scalars reach here through the metrics dicts; json cannot take them."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
