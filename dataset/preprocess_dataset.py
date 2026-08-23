from __future__ import annotations
from typing import Dict
from pathlib import Path
from config.config import load_config, resolve_path

import numpy as np
import pandas as pd
from dataset.print_processed_data import write_processed_data_report

SPLIT_NAMES = ("train", "validation", "test")
# Queries buying this many products or more are stratified as one group.
MAX_BOUGHT_STRATUM = 3

def get_raw_dataset() -> pd.DataFrame:
    """Load the supermarket products dataset.

    keep_default_na=False because "None" is a real allergens category (a product
    with no allergens), and pandas' default na_values would read it as NaN.
    """
    return pd.read_csv(
        resolve_path(load_config().get("dataset_path")), keep_default_na=False
    )

# helper function for split dataset.
def separate(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.Series:
    """Assign every query_id to exactly one split, keeping the bought rate even.

    The rows with the same query_id are the product shown for a single search,
    so they should travel together (if we leave part of a query in train and the rest
    in test, model might memorize that search instead of generalizing from it.)
    
    To keep splits comparable, queries are grouped by how many of their products
    were bought and each group is dealt out in the same proportions.

    Counts above MAX_BOUGHT_STRATUM share one group: only 5 queries bought 4
    products, too few to divide 80/10/10 (test would get none of them).
    """
    queries = (
        df.groupby("query_id")["bought"]
        .sum()
        .clip(upper=MAX_BOUGHT_STRATUM)
        .rename("bought_count")
        .sample(frac=1, random_state=seed)  # shuffle, otherwise splits follow file order
        .reset_index()
    )

    # Position of each query within its group, as a fraction of that group.
    group = queries.groupby("bought_count")["query_id"]
    position = group.cumcount() / group.transform("size")

    cutoffs = np.cumsum(np.array(ratios, dtype=float) / sum(ratios))[:-1]
    split_index = np.searchsorted(cutoffs, position, side="right")
    return pd.Series(np.array(SPLIT_NAMES)[split_index], index=queries["query_id"])

def split_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train/validation/test using ratios from config.json."""
    config = load_config()

    ratios = (
        float(config.get("train_split")),
        float(config.get("validation_split")),
        float(config.get("test_split")),
    )
    split_of_query = separate(df, ratios, int(config.get("split_seed", 42)))
    split_of_row = df["query_id"].map(split_of_query)

    mapping = df.attrs.get("one_hot_encoding_mapping", {})
    splits = {}
    for name in SPLIT_NAMES:
        split = df[split_of_row == name].copy()
        # Keep the category-to-vector-position mapping available on each split.
        split.attrs["one_hot_encoding_mapping"] = mapping
        splits[name] = split
    return splits

def get_data_processed() -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json."""
    df = get_raw_dataset()
    #TODO: define whether this is necessary or not. 
    df = process_title_column(df)
    df = drop_columns(df)
    #TODO: define whether this is necessary or not. 
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