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
RUN_FILE = "run.json"
PREDICTIONS_FILE = "test_predictions.csv"

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


def run_dir(name: str) -> Path:
	return EXPERIMENTS_DIR / name


def save_run(results: RunResults) -> Path:
	"""Write one experiment's numbers and test scores. Returns the run.json path."""
	directory = run_dir(results.name)
	directory.mkdir(parents=True, exist_ok=True)

	labels = np.asarray(results.labels, dtype=int).reshape(-1)
	probs = np.asarray(results.probs, dtype=float).reshape(-1)
	with (directory / PREDICTIONS_FILE).open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["label", "probability"])
		writer.writerows(zip(labels.tolist(), probs.tolist()))

	payload = {
		"name": results.name,
		"created_at": results.created_at
		or datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"config_file": results.config_file,
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

	labels, probs = _load_predictions(directory / payload.get("predictions_file", PREDICTIONS_FILE))
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
	)


def saved_run_names() -> list[str]:
	"""Every experiment directory that holds a run.json, oldest run first."""
	if not EXPERIMENTS_DIR.exists():
		return []
	directories = [path for path in EXPERIMENTS_DIR.iterdir() if (path / RUN_FILE).exists()]
	directories.sort(key=lambda path: (path / RUN_FILE).stat().st_mtime)
	return [path.name for path in directories]


def load_runs(names: list[str] | None = None) -> list[RunResults]:
	"""Load the named runs, or every saved run when ``names`` is None."""
	return [load_run(name) for name in (names if names is not None else saved_run_names())]


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
	if not path.exists():
		return np.empty(0, dtype=int), np.empty(0, dtype=float)
	labels: list[int] = []
	probs: list[float] = []
	with path.open(encoding="utf-8") as file:
		for row in csv.DictReader(file):
			labels.append(int(float(row["label"])))
			probs.append(float(row["probability"]))
	return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def _json_safe(value: Any) -> Any:
	"""numpy scalars reach here through the metrics dicts; json cannot take them."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
