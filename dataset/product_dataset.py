"""PyTorch Dataset and DataLoader helpers for product BTR training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.preprocess_dataset import config_columns
from model.tokenizer import tokenize


@dataclass(frozen=True)
class ProductDataLoaders:
	train: DataLoader
	validation: DataLoader
	test: DataLoader


class ProductDataset(Dataset):
	"""Tensor-backed product dataset used by the Transformer training loop."""

	def __init__(
		self,
		numeric: np.ndarray,
		categorical: np.ndarray,
		title_ids: np.ndarray,
		labels: np.ndarray,
		row_ids: np.ndarray,
	) -> None:
		self.numeric = torch.tensor(numeric, dtype=torch.float32)
		self.categorical = torch.tensor(categorical, dtype=torch.long)
		self.title_ids = torch.tensor(title_ids, dtype=torch.long)
		self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
		# The dataframe index of each row, so a prediction can be joined back to
		# the product it scored. Required, not optional: the train loader shuffles,
		# so position in the output is not position in the split for that split.
		self.row_ids = torch.tensor(row_ids, dtype=torch.long)

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		return {
			"numeric": self.numeric[idx],
			"categorical": self.categorical[idx],
			"title_ids": self.title_ids[idx],
			"label": self.labels[idx],
			"row_id": self.row_ids[idx],
		}


def create_data_loaders(
	splits: dict[str, pd.DataFrame], config: dict
) -> ProductDataLoaders:
	"""Build train/validation/test DataLoaders from preprocessed splits."""
	numeric_cols = config_columns("numeric_columns", config)
	categorical_cols = config_columns("categorical_columns", config)
	target_col = str(config.get("target_column", "bought"))
	max_title_len = int(config.get("max_title_len", 24))
	vocab_size = int(config.get("vocab_size", 30000))
	batch_size = int(config.get("batch_size", 64))

	train_ds = dataframe_to_product_dataset(
		splits["train"], numeric_cols, categorical_cols, target_col, max_title_len, vocab_size
	)
	val_ds = dataframe_to_product_dataset(
		splits["validation"],
		numeric_cols,
		categorical_cols,
		target_col,
		max_title_len,
		vocab_size,
	)
	test_ds = dataframe_to_product_dataset(
		splits["test"], numeric_cols, categorical_cols, target_col, max_title_len, vocab_size
	)

	return ProductDataLoaders(
		train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
		validation=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
		test=DataLoader(test_ds, batch_size=batch_size, shuffle=False),
	)


def dataframe_to_product_dataset(
	split: pd.DataFrame,
	numeric_cols: list[str],
	categorical_cols: list[str],
	target_col: str,
	max_title_len: int,
	vocab_size: int,
) -> ProductDataset:
	"""Convert one dataframe split to a tensor-backed ProductDataset."""
	numeric = pd.DataFrame(index=split.index)
	for column in numeric_cols:
		if column in split.columns:
			numeric[column] = pd.to_numeric(split[column], errors="coerce")
		else:
			numeric[column] = 0.0
	numeric_values = numeric.fillna(0.0).to_numpy(dtype=np.float32)

	categorical = np.zeros((len(split), len(categorical_cols)), dtype=np.int64)
	for index, column in enumerate(categorical_cols):
		if column in split.columns:
			categorical[:, index] = (
				pd.to_numeric(split[column], errors="coerce")
				.fillna(0)
				.astype(np.int64)
			)

	text_source = str(split.attrs.get("text_column", "product_name"))
	if text_source not in split.columns:
		text_source = "product_name" if "product_name" in split.columns else "title"
	title_ids = encode_text(
		split.get(text_source, pd.Series([""] * len(split))),
		max_len=max_title_len,
		vocab_size=vocab_size,
	)
	labels = (
		pd.to_numeric(split[target_col], errors="coerce")
		.fillna(0)
		.astype(np.float32)
		.to_numpy()
	)
	return ProductDataset(
		numeric_values, categorical, title_ids, labels,
		np.asarray(split.index, dtype=np.int64),
	)


def encode_text(texts: pd.Series, max_len: int, vocab_size: int) -> np.ndarray:
	"""Tokenize text to a fixed-width int array with zero padding."""
	arr = np.zeros((len(texts), max_len), dtype=np.int64)
	# The current tokenizer can emit IDs up to the GPT-2 vocabulary size. Modulo
	# keeps experiments with smaller vocab_size runnable until a custom vocab lands.
	token_space = max(vocab_size - 2, 1)
	for i, text in enumerate(texts.fillna("").astype(str)):
		token_ids = tokenize(text)[:max_len]
		ids = [(token_id % token_space) + 2 for token_id in token_ids]
		arr[i, : len(ids)] = ids
	return arr
