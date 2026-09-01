"""The test rows as a person can read them, next to what the model said.

Every other artifact a run writes is aggregate: a score, a curve, a mean over a
split. This one is the opposite, and it is the only way to answer "what kind of
product does it keep getting wrong" -- which is a question about individual rows
and cannot be recovered from an AUC.

Two things make the file readable rather than a dump of tensors:

    categorical ids are decoded back to their names, using the mapping
    encode_categorical_ids stored on the split, so a row says "category: Frozen"
    and not "category: 4".

    rows are sorted by how wrong the model was, so the head of the file is the
    confident mistakes -- a bought product scored 0.02, or an unbought one scored
    0.97 -- which is where a pattern shows up if there is one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ERROR_ANALYSIS_FILE = "test_error_analysis.csv"


def build_error_analysis(
	split: pd.DataFrame, row_ids: np.ndarray, labels: np.ndarray, probs: np.ndarray
) -> pd.DataFrame:
	"""Join scores back onto their rows, decode ids, and sort worst first."""
	frame = split.loc[np.asarray(row_ids, dtype=int)].copy()

	# The ids are addresses into an embedding table, not quantities: decoded, the
	# column reads as the category it came from. Ids absent from the mapping are
	# UNKNOWN_ID (0), a value that only appeared outside train.
	for column, mapping in (split.attrs.get("categorical_id_mapping") or {}).items():
		if column in frame.columns:
			frame[column] = frame[column].map(mapping).fillna("(no visto en train)")

	# Raw-scale copies of the normalised columns, named <column>_raw so the value
	# the model actually saw stays in the file next to the readable one.
	raw = split.attrs.get("raw_numeric")
	if raw is not None and len(raw):
		for column in raw.columns:
			frame[f"{column}_raw"] = raw.loc[frame.index, column]

	frame.insert(0, "probability", np.asarray(probs, dtype=float))
	frame.insert(0, "bought_real", np.asarray(labels, dtype=int))
	# Distance from the truth, so one sort puts both kinds of confident mistake on
	# top: a positive scored near 0 and a negative scored near 1.
	frame.insert(0, "error", (frame["bought_real"] - frame["probability"]).abs())
	frame.insert(
		0,
		"caso",
		np.where(
			frame["bought_real"] == 1,
			np.where(frame["probability"] >= 0.5, "acierto_positivo", "FALSO_NEGATIVO"),
			np.where(frame["probability"] >= 0.5, "FALSO_POSITIVO", "acierto_negativo"),
		),
	)
	return frame.sort_values("error", ascending=False)


def write_error_analysis(directory: Path, frame: pd.DataFrame) -> Path:
	directory.mkdir(parents=True, exist_ok=True)
	path = directory / ERROR_ANALYSIS_FILE
	frame.to_csv(path, index_label="row_id")
	return path


def error_summary(frame: pd.DataFrame, by: str, min_rows: int = 5) -> pd.DataFrame:
	"""Mean error per value of one column, worst first.

	This is the reading of the file that finds a pattern without scrolling it:
	a column value whose mean error is far above the rest is a group the model
	handles badly. ``min_rows`` drops values too rare for the mean to mean much.
	"""
	if by not in frame.columns:
		return pd.DataFrame()
	grouped = frame.groupby(by, dropna=False).agg(
		filas=("error", "size"),
		error_medio=("error", "mean"),
		btr_real=("bought_real", "mean"),
		btr_predicho=("probability", "mean"),
	)
	return grouped[grouped["filas"] >= min_rows].sort_values("error_medio", ascending=False)
