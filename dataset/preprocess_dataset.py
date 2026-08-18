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

    train_ratio = float(config.get("train_split"))
    valid_ratio = float(config.get("validation_split"))
    test_ratio = float(config.get("test_split"))

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

def get_data_normalized_encoded_splitted() -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json."""
    df = get_raw_dataset()
    df = drop_columns(df)
    df = normalize_data(df)
    df = one_hot_encode_data(df)
    return split_dataset(df)

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns specified in config.json."""
    columns = load_config().get("drop_columns", [])
    if isinstance(columns, str):
        columns = [column.strip() for column in columns.split(",") if column.strip()]

    return df.drop(columns=columns, errors="ignore")

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataset using min-max normalization."""
    columns = load_config().get("normalize_columns", [])
    if isinstance(columns, str):
        columns = [column.strip() for column in columns.split(",") if column.strip()]

    if columns:
        df[columns] = (df[columns] - df[columns].min()) / (
            df[columns].max() - df[columns].min()
        )
    return df

def one_hot_encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns configured in config.json."""
    columns = load_config().get("one_hot_columns", [])
    if isinstance(columns, str):
        columns = [column.strip() for column in columns.split(",") if column.strip()]

    for column in columns:
        encoded = pd.get_dummies(df[column], dtype=int)
        df[column] = encoded.to_numpy().tolist()

    return df

if __name__ == "__main__":
    # print({key: (value) for key, value in get_data_normalized_encoded_splitted().items()})
    splits = get_data_normalized_encoded_splitted()
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        for key, value in splits.items():
            print(f"{key}:\n{value.head(2).to_string(index=False)}\n")