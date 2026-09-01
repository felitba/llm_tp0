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
# Column the serialized row text lands in when text_column names more than one column.
SERIALIZED_TEXT_COLUMN = "serialized_row"
# Kept out of the serialized text under "all": the label itself, and the query id,
# which identifies a search rather than describing the product.
DEFAULT_TEXT_EXCLUDE = ("bought", "query_id")
# Separators for the serialized text. The field name travels with its value so the
# tokenizer sees "price: 8.30", not a bare 8.30 it cannot attribute to a column.
FIELD_SEPARATOR = " | "
NAME_VALUE_SEPARATOR = ": "


def config_columns(key: str, config: dict | None = None) -> list[str]:
    """Read a column list from config.json, accepting a comma-separated string too."""
    columns = (config or load_config()).get(key, [])
    if isinstance(columns, str):
        columns = [column.strip() for column in columns.split(",") if column.strip()]
    return list(columns)

def get_raw_dataset(config: dict | None = None) -> pd.DataFrame:
    """Load the supermarket products dataset.

    keep_default_na=False because "None" is a real allergens category (a product
    with no allergens), and pandas' default na_values would read it as NaN.
    """
    config_data = config or load_config()
    return pd.read_csv(
        resolve_path(config_data.get("dataset_path")), keep_default_na=False
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

def split_dataset(df: pd.DataFrame, config: dict | None = None) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train/validation/test using ratios from config.json."""
    config_data = config or load_config()

    ratios = (
        float(config_data.get("train_split")),
        float(config_data.get("validation_split")),
        float(config_data.get("test_split")),
    )
    split_of_query = separate(df, ratios, int(config_data.get("split_seed", 42)))
    split_of_row = df["query_id"].map(split_of_query)

    return {name: df[split_of_row == name].copy() for name in SPLIT_NAMES}


def separate_bought_true(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> pd.Series:
    """Assign query_id to train/validation/test, splitting only bought=true queries.

    Queries with at least one bought=True are distributed by the supplied ratios,
    while all other queries stay in training so the labeled positive cases remain
    the only ones that get split across validation/test.
    """
    positive_queries = (
        df.loc[df["bought"] == 1, "query_id"]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    if positive_queries.empty:
        return pd.Series(index=df["query_id"].drop_duplicates(), dtype=object)

    split_of_query = pd.Series("train", index=df["query_id"].drop_duplicates())
    cutoffs = np.cumsum(np.array(ratios, dtype=float) / sum(ratios))[:-1]
    position = np.arange(len(positive_queries), dtype=float) / len(positive_queries)
    split_index = np.searchsorted(cutoffs, position, side="right")
    split_by_query = pd.Series(np.array(SPLIT_NAMES)[split_index], index=positive_queries)
    split_of_query.loc[positive_queries] = split_by_query
    return split_of_query


def split_dataset_by_bought_true(
    df: pd.DataFrame,
    config: dict | None = None,
) -> Dict[str, pd.DataFrame]:
    """Split the dataset with bought=true queries split at 80/10/10.

    This mirrors split_dataset, but only query_ids with at least one bought=True
    are distributed across train/validation/test according to the specified
    ratios. Queries without a positive label remain in train.
    """
    config_data = config or load_config()

    ratios = (
        float(config_data.get("train_split", 0.8)),
        float(config_data.get("validation_split", 0.1)),
        float(config_data.get("test_split", 0.1)),
    )
    split_of_query = separate_bought_true(df, ratios, int(config_data.get("split_seed", 42)))
    split_of_row = df["query_id"].map(split_of_query)

    return {name: df[split_of_row == name].copy() for name in SPLIT_NAMES}


def get_data_processed(config: dict | None = None) -> Dict[str, pd.DataFrame]:
    """Load data and return train/validation/test splits according to config.json.

    CHANGED (2026-08-24): the encoders used to run on the whole dataframe and the
    split happened last, so validation/test values decided the encoding layout.
    Now the split comes first and every encoder is fitted on train only.
    To go back to the old order, move the encode calls above split_dataset and
    have them take/return a single dataframe again (git history has that version).
    """
    config_data = config or load_config()
    df = get_raw_dataset(config_data)
    #TODO: define whether this is necessary or not.
    df = process_title_column(df)
    df = drop_columns(df, config_data)
    # Before the split so every row is serialized the same way, and before
    # encode_categorical_ids so the text still holds the category names.
    source_columns = text_columns(df, config_data)
    df, text_column = build_text_column(df, source_columns)

    # splits = split_dataset_by_bought_true(df, config_data)
    splits = split_dataset(df, config_data)
    # Before normalize_splits, which overwrites the raw values the bins are cut from.
    splits = bin_numeric_splits(splits, config_data)
    # Keep the pre-scaling values around for error analysis. A z-scored price of
    # -0.99 is the right input for the numeric tokenizer and the wrong thing to
    # read when eyeballing which products the model gets wrong, and the scaling
    # is not invertible from the split alone.
    scaled_columns = config_columns("normalize_columns", config_data)
    raw_numeric = {
        name: split[[c for c in scaled_columns if c in split.columns]].copy()
        for name, split in splits.items()
    }
    splits = normalize_splits(splits, config_data)
    for name, split in splits.items():
        split.attrs["raw_numeric"] = raw_numeric[name]
    splits = encode_categorical_ids(splits, config_data)
    for split in splits.values():
        # text_column is the one to read; text_columns is what went into it, which
        # is the same thing only when a single column feeds the tokenizer.
        split.attrs["text_column"] = text_column
        split.attrs["text_columns"] = source_columns

    return splits

def drop_columns(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Drop the columns specified in config.json."""
    return df.drop(columns=config_columns("drop_columns", config), errors="ignore")

def bin_numeric_splits(
    splits: Dict[str, pd.DataFrame], config: dict | None = None
) -> Dict[str, pd.DataFrame]:
    """Add quantile-bin categorical columns, with the bin edges fitted on train only.

    config.json::

        "bin_columns": [
            {"column": "price", "bins": 4, "within": "category", "name": "price_bin"}
        ]

    produces a string column ``price_bin`` with values ``Q1..Q4``: the quartile of
    the row's price among the TRAIN prices of its own category. List the new name
    in ``categorical_columns`` and it reaches the model as one embedding token,
    like any other categorical.

    Why this exists: the numeric tokenizer is ``x * w + b``, a linear function of
    the value, and the EDA found price's effect is an inverted U within category
    (cheapest and most expensive buy less). Step 1 measured the consequence: on
    the top price quartile of the high tier the linear token predicts 0.74 for a
    real rate of 0.46. A bin per quartile gives the model one free vector per
    price band, which can take any shape.

    ``within`` is optional; without it the edges are global. Groups too small to
    cut (fewer rows than bins) and groups unseen in train fall back to the global
    edges. Fitting on train and freezing is the same leakage rule normalize_splits
    follows: validation and test values never move an edge.
    """
    config_data = config or load_config()
    specs = config_data.get("bin_columns") or []
    if not specs:
        return splits

    train = splits["train"]
    edges_used: dict[str, dict] = {}
    for spec in specs:
        column = spec["column"]
        bins = int(spec.get("bins", 4))
        within = spec.get("within")
        name = spec.get("name", f"{column}_bin")
        cut_points = np.linspace(0.0, 1.0, bins + 1)[1:-1]

        def fit(values: pd.Series) -> np.ndarray:
            return np.quantile(pd.to_numeric(values, errors="coerce").dropna(), cut_points)

        global_edges = fit(train[column])
        group_edges = {
            group: fit(values)
            for group, values in (train.groupby(within)[column] if within else [])
            if len(values) >= bins
        }
        edges_used[name] = {
            "column": column, "within": within,
            "global": global_edges.tolist(),
            "groups": {str(group): edges.tolist() for group, edges in group_edges.items()},
        }

        for split in splits.values():
            values = pd.to_numeric(split[column], errors="coerce").to_numpy(dtype=float)
            index = np.zeros(len(split), dtype=int)
            if within:
                for group, rows in split.groupby(within).indices.items():
                    edges = group_edges.get(group, global_edges)
                    index[rows] = np.searchsorted(edges, values[rows], side="right")
            else:
                index = np.searchsorted(global_edges, values, side="right")
            split[name] = [f"Q{position + 1}" for position in index]

    for split in splits.values():
        split.attrs["bin_edges"] = edges_used
    return splits


def normalize_splits(
    splits: Dict[str, pd.DataFrame], config: dict | None = None
) -> Dict[str, pd.DataFrame]:
    """Impute and standardize configured numeric columns without leaking splits.

    The median, mean and standard deviation are fitted on ``train`` only. They are
    then frozen for validation and test, which is essential: fitting a scaler on
    the complete dataframe would let held-out rows influence the representation the
    model learns. Missing values receive the train median before scaling; a
    constant column is mapped to zero instead of dividing by zero.

    Text serialization has deliberately already happened at this point, while
    category names were still available. Therefore this function scales numeric
    *feature tokens* only; all-text experiments retain their literal serialized
    values as intended.
    """
    columns = [
        column
        for column in config_columns("normalize_columns", config)
        if column in splits["train"].columns
    ]
    statistics: dict[str, dict[str, float]] = {}

    for column in columns:
        train_values = pd.to_numeric(splits["train"][column], errors="coerce")
        train_values = train_values.replace([np.inf, -np.inf], np.nan)
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        filled_train = train_values.fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0

        statistics[column] = {"median": median, "mean": mean, "std": std}
        for split in splits.values():
            values = pd.to_numeric(split[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).fillna(median)
            split[column] = ((values - mean) / std).astype(np.float32)

    for split in splits.values():
        split.attrs["numeric_normalization"] = statistics
    return splits

def encode_categorical_ids(
    splits: Dict[str, pd.DataFrame], config: dict | None = None
) -> Dict[str, pd.DataFrame]:
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
    columns = config_columns("categorical_columns", config)
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


def text_columns(df: pd.DataFrame, config: dict | None = None) -> list[str]:
    """Which columns feed the text tokenizer.

    config.json's text_column accepts three forms:
        "product_name"            one column, the FT-Transformer setup
        ["product_name", "brand"] those columns, serialized into one string
        "all"                     every surviving column except DEFAULT_TEXT_EXCLUDE
                                  (override with text_exclude_columns)

    "all" reads df.columns, so it means "everything drop_columns left", and it must
    be called after drop_columns for that to hold.
    """
    config_data = config or load_config()
    setting = config_data.get("text_column", "product_name")

    if isinstance(setting, str):
        if setting.strip().casefold() != "all":
            return [setting]
        excluded = set(
            config_columns("text_exclude_columns", config_data) or DEFAULT_TEXT_EXCLUDE
        )
        return [column for column in df.columns if column not in excluded]

    columns = list(setting)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"text_column names columns not in the dataset: {missing}")
    return columns


def build_text_column(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, str]:
    """Render the given columns into one string per row.

    Returns the frame and the name of the column the text tokenizer should read,
    which is the column itself when only one takes part and SERIALIZED_TEXT_COLUMN
    when several do.

    Call this before encode_categorical_ids: afterwards a categorical column holds
    its integer id, and serializing that would hand the tokenizer "category: 3"
    instead of "category: Frozen".
    """
    if not columns:
        raise ValueError("text_column resolved to no columns")
    if len(columns) == 1:
        return df, columns[0]

    serialized = None
    for column in columns:
        field = column + NAME_VALUE_SEPARATOR + df[column].fillna("").astype(str).str.strip()
        serialized = field if serialized is None else serialized + FIELD_SEPARATOR + field
    df[SERIALIZED_TEXT_COLUMN] = serialized
    return df, SERIALIZED_TEXT_COLUMN


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
