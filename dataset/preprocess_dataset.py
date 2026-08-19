from __future__ import annotations
from typing import Dict
from pathlib import Path
from config.config import load_config

import pandas as pd
from dataset.print_processed_data import write_processed_data_report

def get_raw_dataset() -> pd.DataFrame:
    """Load the supermarket products dataset."""
    return pd.read_csv(load_config().get("dataset_path"))

# helper function for split dataset. 
def separate(group: pd.DataFrame, ratios: tuple[float, float, float]):
    total = sum(ratios)
    train_end = int(len(group) * ratios[0] / total)
    valid_end = train_end + int(len(group) * ratios[1] / total)
    return {
        "train": group.iloc[:train_end],
        "validation": group.iloc[train_end:valid_end],
        "test": group.iloc[valid_end:],
    }

def split_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train/validation/test using ratios from config.json."""
    config = load_config()

    default_ratios = (
        float(config.get("train_split")),
        float(config.get("validation_split")),
        float(config.get("test_split")),
    )
    bought_ratios = (
        float(config.get("target_column_train_perc", 0.8)),
        float(config.get("target_column_valid_perc", 0.1)),
        0.1,
    )

    bought = df[df["bought"].astype(bool)]
    not_bought = df[~df["bought"].astype(bool)]
    bought_splits = separate(bought, bought_ratios)
    not_bought_splits = separate(not_bought, default_ratios)
    splits = {
        name: pd.concat([bought_splits[name], not_bought_splits[name]]).copy()
        for name in ("train", "validation", "test")
    }
    # Keep the category-to-vector-position mapping available on each split.
    mapping = df.attrs.get("one_hot_encoding_mapping", {})
    for split in splits.values():
        split.attrs["one_hot_encoding_mapping"] = mapping
    return splits

def get_data_processed() -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json."""
    df = get_raw_dataset()
    #TODO: define wether this is necessary or not. 
    df = process_title_column(df)
    df = drop_columns(df)
    #TODO: define wether this is necessary or not. 
    # df = normalize_data(df)
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

    encoding_mapping = {}
    for column in columns:
        encoded = pd.get_dummies(df[column], dtype=int)
        encoding_mapping[column] = {
            vector_index: category
            for vector_index, category in enumerate(encoded.columns)
        }
        df[column] = encoded.to_numpy().tolist()

    # Maps each vector index to the category represented by that position.
    df.attrs["one_hot_encoding_mapping"] = encoding_mapping
    return df

def process_title_column(df: pd.DataFrame)-> pd.DataFrame:
    """ parse the title column, extracting into new columns: product name and comments.
    Example: Harvest Lane Family Pack Blueberry Muffins - 8 ct (Customer Favorite)"""
   
    # Remove the brand prefix from the title, then keep the text before the "-".
    brand = df["brand"].fillna("").astype(str).str.strip()
    title = df["title"].fillna("").astype(str).str.strip()
    df["product_name"] = (
        title.combine(
            brand,
            lambda product_title, product_brand: product_title[
                len(product_brand) :
            ].lstrip()
            if product_brand
            and product_title.casefold().startswith(product_brand.casefold())
            else product_title,
        )
        .str.split("-", n=1)
        .str[0]
        .str.strip()
    )
    df["comments"] = df["title"].str.extract(r"\((.*?)\)", expand=False)
    return df

if __name__ == "__main__":
    splits = get_data_processed()
    report_path = write_processed_data_report(splits)
    print(f"Dataset report written to: {report_path}")