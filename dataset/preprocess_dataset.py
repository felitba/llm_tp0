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
# Sentinel for listings who title carries no parenthetical tag
NO_TAG = "No tag"
# Id reserved for a category value that never appears in train. Real values start at 1.
UNKNOWN_ID = 0


def config_columns(key: str) -> list[str]:
    """Read a column list from config.json, accepting a comma-separated string too."""
    columns = load_config().get(key, [])
    if isinstance(columns, str):
        columns = [column.strip() for column in columns.split(",") if column.strip()]
    return list(columns)

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

    return {name: df[split_of_row == name].copy() for name in SPLIT_NAMES}

def get_data_processed() -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json.

    CHANGED (2026-08-24): the encoders used to run on the whole dataframe and the
    split happened last, so validation/test values decided the encoding layout.
    Now the split comes first and every encoder is fitted on train only.
    To go back to the old order, move the encode calls above split_dataset and
    have them take/return a single dataframe again (git history has that version).
    """
    df = get_raw_dataset()
    #TODO: define whether this is necessary or not.
    df = process_title_column(df)
    df = drop_columns(df)

    splits = split_dataset(df)
    splits = encode_categorical_ids(splits)

    # DECISION (2026-08-24): price and nutrition_score stay on their raw scale for
    # now, so the first FT-Transformer run shows what the numeric tokens do without
    # a scaling choice mixed into the result.
    # When we do normalize, it belongs right here, and as three steps: fit the
    # scaler on splits["train"] only, then apply those same frozen statistics to
    # all three splits. Fitting before the split lets validation/test values set
    # the scale the model trains on. normalize_data below is the old whole-frame
    # version and would leak; rewrite it to take the splits before calling it.
    return splits

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns specified in config.json."""
    return df.drop(columns=config_columns("drop_columns"), errors="ignore")

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataset using min-max normalization.

    UNUSED (2026-08-24): kept for reference only. min()/max() over whatever frame
    it is handed means calling it before the split fits the scale on validation and
    test rows too. See the note in get_data_processed before wiring it back in.
    """
    columns = config_columns("normalize_columns")

    if columns:
        df[columns] = (df[columns] - df[columns].min()) / (
            df[columns].max() - df[columns].min()
        )
    return df

def encode_categorical_ids(splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Replace each configured categorical column with the integer id nn.Embedding wants.

    CHANGED (2026-08-24): this replaces the one-hot encoding the pipeline used to
    emit. An embedding lookup is already "select row id of a matrix", so one-hot
    followed by a Linear would compute the same thing the long way round, and the
    one-hot widths (12 for category, 8 for allergens, 20 for title_tag) do not
    match the single d_model width the encoder needs anyway.
    The id is an address into the embedding table, not a quantity: nothing
    downstream may do arithmetic on it or compare two ids by size.

    Ids run 1..N over the train values in alphabetical order, and UNKNOWN_ID (0)
    covers a value that only appears in validation or test, so each embedding table
    needs len(values) + 1 rows. Alphabetical rather than by frequency or file order
    so an id depends only on which values train holds, not on how rows were shuffled.
    """
    columns = config_columns("categorical_columns")
    categories = {
        column: pd.Index(sorted(splits["train"][column].unique())) for column in columns
    }

    for split in splits.values():
        for column in columns:
            # get_indexer returns -1 for a value missing from the index, so +1 puts
            # it on UNKNOWN_ID and every known value on 1..N, vectorised per column.
            split[column] = categories[column].get_indexer(split[column]) + 1
        # id -> value, so a report or a prediction can be read back as a category.
        split.attrs["categorical_id_mapping"] = {
            column: dict(enumerate(values, start=1))
            for column, values in categories.items()
        }
    return splits

def categorical_cardinalities(split: pd.DataFrame) -> Dict[str, int]:
    """How many rows each categorical column needs in its embedding table.

    That is the number of train values plus one, because UNKNOWN_ID (0) sits below
    them and a validation or test row is allowed to land on it.
    """
    mapping = split.attrs["categorical_id_mapping"]
    return {column: len(values) + 1 for column, values in mapping.items()}


def process_title_column(df: pd.DataFrame)-> pd.DataFrame:
    """ parse the title column, extracting into new columns: product name and title_tag.
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
    df["title_tag"] = df["title"].str.extract(r"\((.*?)\)", expand=False).fillna(NO_TAG)
    return df

if __name__ == "__main__":
    splits = get_data_processed()
    report_path = write_processed_data_report(splits)
    print(f"Dataset report written to: {report_path}")