from __future__ import annotations
from typing import Dict
from config.config import load_config

import pandas as pd

def get_raw_dataset() -> pd.DataFrame:
    """Load the supermarket products dataset."""
    return pd.read_csv(load_config().get("dataset_path"))

# TODO: we should also validate that data distribution is similar across splits. 
def split_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train/validation/test using ratios from config.json."""
    config = load_config()

    train_ratio = float(config.get("train_split", 1.0))
    valid_ratio = float(config.get("validation_split", 0.0))
    test_ratio = float(config.get("test_split", 0.0))

    if not (train_ratio + valid_ratio + test_ratio > 0):
        raise ValueError("Split ratios must sum to a positive value.")

    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    valid_size = int(total_rows * valid_ratio)

    train_df = df.iloc[:train_size].copy()
    valid_df = df.iloc[train_size : train_size + valid_size].copy()
    test_df = df.iloc[train_size + valid_size :].copy()

    splits: Dict[str, pd.DataFrame] = {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }
    return splits

def get_data_splitted() -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json."""
    return split_dataset(get_raw_dataset())


if __name__ == "__main__":
    print({key: len(value) for key, value in get_data_splitted().items()})
