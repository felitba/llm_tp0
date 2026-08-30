"""Turn one dataset row into the token sequence the encoder consumes.

The FT-Transformer rule is that every feature becomes exactly one vector of width
d_model, so self-attention can relate the features of a row to each other. Three
kinds of feature get there three different ways:

    numeric      price         -> x * weight + bias    one token per column
    categorical  category      -> embedding lookup     one token per column
    text         product_name  -> lookup + position    one token per subword

which assembles into

    [CLS] title_0 .. title_9 price nutrition_score category .. storage_type

[CLS] carries no input of its own: it is a free vector the encoder fills in from
whatever the rest of the row says, and the classification head reads only that.
Once assembled the encoder cannot tell the three kinds apart, which is the point.

Which columns take part comes from config.json (categorical_columns,
numeric_columns, text_column), so an ablation is a config edit, not a code change.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from config.config import load_config
from model.positional_encoding import PositionalEncoding

# Token id filling titles shorter than max_title_len. Attention never reads these
# positions, so the id only has to be one the vocabulary can index.
PAD_ID = 0


def uniform_init(tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """Fill in place with U(-1/sqrt(d_model), 1/sqrt(d_model)).

    Random rather than zeros because identical rows would receive identical
    gradients and stay identical forever; the spread is what lets them specialise.

    The bound is 1/sqrt(d_model) because a vector of d components drawn from
    U(-b, b) has length b * sqrt(d / 3), so that choice cancels the d and leaves
    every token about 0.58 long whatever d_model is set to. A fixed bound would
    make tokens grow with d_model, and attention scores are dot products of these
    vectors, so the logits would grow with it and saturate the softmax.

    Using one spread for all three tokenizers also means no family of tokens starts
    out dominating attention before training has said anything.
    """
    bound = d_model ** -0.5
    return nn.init.uniform_(tensor, -bound, bound)


class NumericTokenizer(nn.Module):
    """One token per numeric column: token_j = x_j * weight_j + bias_j.

    This is an nn.Linear(1, d_model) per column, written as a broadcast so every
    column is computed in one operation instead of n small matmuls.

    The distinction that matters is against nn.Linear(n_columns, d_model), which
    emits a single token for all the numerics together: that mixes price and
    nutrition_score before the encoder sees either, so attention can only treat
    them as one block. Separate tokens let it weigh price against category without
    dragging nutrition_score along.

    The bias is load-bearing rather than decorative. nutrition_score can be 0, and
    0 * weight_j is the zero vector for every column alike, so without a bias a
    zero-valued feature would be indistinguishable from any other zero-valued one.

    Note the token length is |x| * |weight|, so it inherits the column's raw scale:
    unnormalised nutrition_score (up to 99) produces tokens roughly 100x longer than
    the categorical ones. See get_data_processed in dataset/preprocess_dataset.
    """

    def __init__(self, n_columns: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(uniform_init(torch.empty(n_columns, d_model), d_model))
        self.bias = nn.Parameter(uniform_init(torch.empty(n_columns, d_model), d_model))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """(batch, n_columns) floats -> (batch, n_columns, d_model)."""
        return values.unsqueeze(-1) * self.weight + self.bias


class CategoricalTokenizer(nn.Module):
    """One token per categorical column, each column with its own embedding table.

    Two levels of "one per category" are both in play here:

        per column  -> its own nn.Embedding. nn.ModuleDict is only a container;
                       the five tables are five independent weight matrices.
        per value   -> its own row of that column's table. A row is already an
                       independent vector, updated only by the rows of the dataset
                       carrying that value, so separate nn.Parameters would buy
                       nothing and lose the lookup.

    Per column matters because the same string can mean different things:
    category=Frozen (the aisle) and storage_type=Frozen (the handling requirement)
    are unrelated features, and separate tables leave them free to diverge.

    Ids arrive from the pipeline as 0..cardinality-1, where 0 means "value not seen
    in train" (dataset/preprocess_dataset.encode_categorical_ids), so each table
    needs cardinality rows and row 0 is that column's unknown-value vector.

    Args:
        cardinalities: rows per column. Iteration order fixes which column each
            position of the input tensor refers to, so it must match the order the
            batch was built in.
    """

    def __init__(self, cardinalities: Mapping[str, int], d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.columns = list(cardinalities)
        self.embeddings = nn.ModuleDict(
            {column: nn.Embedding(rows, d_model) for column, rows in cardinalities.items()}
        )
        for embedding in self.embeddings.values():
            uniform_init(embedding.weight, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(batch, n_columns) ids -> (batch, n_columns, d_model)."""
        # An all-text ablation configures no categorical columns at all, and
        # torch.stack has no empty case; cat wants the (batch, 0, d_model) slice.
        if not self.columns:
            return ids.new_zeros((ids.shape[0], 0, self.d_model), dtype=torch.float32)
        tokens = [
            self.embeddings[column](ids[:, position])
            for position, column in enumerate(self.columns)
        ]
        # Each lookup is (batch, d_model); stack puts them back on a column axis.
        return torch.stack(tokens, dim=1)


class TextTokenizer(nn.Module):
    """One token per subword of the text column, plus its position.

    Unlike the tabular tokenizers this one emits several tokens per row, and the
    same table row is reused wherever a word reappears -- that sharing is how the
    model carries what it learned about "Heat" from one product to the next.
    Position is what separates two occurrences of the same word.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        # padding_idx pins row PAD_ID to zeros and keeps it out of the gradient, so
        # the padding slots never learn anything.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        uniform_init(self.embedding.weight, d_model)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=0.0, batch_first=True
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(batch, max_title_len) ids -> (batch, max_title_len, d_model)."""
        # The sinusoids are O(1) per component while the table is O(1/sqrt(d_model)),
        # so adding them raw would let position drown out which word this is. Scaling
        # up before the add and back down after keeps the ratio the original
        # Transformer intends and still leaves text tokens on the same scale as the
        # tabular ones.
        scale = math.sqrt(self.d_model)
        return self.positional_encoding(self.embedding(ids) * scale) / scale


class FeatureTokenizer(nn.Module):
    """Assemble the per-row sequence and the padding mask that goes with it.

    Args:
        cardinalities: rows needed per categorical column, from
            dataset.preprocess_dataset.categorical_cardinalities.
    """

    def __init__(
        self, cardinalities: Mapping[str, int], config: dict | None = None
    ) -> None:
        super().__init__()
        config_data = config or load_config()

        self.d_model = int(config_data.get("d_model"))
        self.categorical_columns = list(config_data.get("categorical_columns", []))
        self.numeric_columns = list(config_data.get("numeric_columns", []))
        self.max_title_len = int(config_data.get("max_title_len"))

        missing = [c for c in self.categorical_columns if c not in cardinalities]
        if missing:
            raise KeyError(f"no cardinality given for categorical columns {missing}")

        # A learned vector rather than a lookup: it is the same for every row, so
        # there is nothing to index by. Shaped to broadcast over the batch.
        self.cls_token = nn.Parameter(
            uniform_init(torch.empty(1, 1, self.d_model), self.d_model)
        )
        self.text_tokenizer = TextTokenizer(
            vocab_size=int(config_data.get("vocab_size")),
            d_model=self.d_model,
            max_len=self.max_title_len,
        )
        self.numeric_tokenizer = NumericTokenizer(len(self.numeric_columns), self.d_model)
        # Rebuilt in config order so the tokenizer's column order matches the order
        # the batch tensors are built in.
        self.categorical_tokenizer = CategoricalTokenizer(
            {column: cardinalities[column] for column in self.categorical_columns},
            self.d_model,
        )

    @property
    def sequence_length(self) -> int:
        return (
            1
            + self.max_title_len
            + len(self.numeric_columns)
            + len(self.categorical_columns)
        )

    def token_names(self) -> list[str]:
        """What each sequence position holds, for reading attention maps later."""
        return (
            ["[CLS]"]
            + [f"title_{position}" for position in range(self.max_title_len)]
            + list(self.numeric_columns)
            + list(self.categorical_columns)
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        title_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the encoder input.

        Args:
            numeric: (batch, len(numeric_columns)) float
            categorical: (batch, len(categorical_columns)) long
            title_ids: (batch, max_title_len) long, PAD_ID where the title ran out

        Returns:
            tokens: (batch, sequence_length, d_model)
            padding_mask: (batch, sequence_length) bool, True where attention must
                ignore the position. That is the layout nn.TransformerEncoder
                expects for src_key_padding_mask.
        """
        batch_size = numeric.shape[0]

        tokens = torch.cat(
            [
                self.cls_token.expand(batch_size, -1, -1),
                self.text_tokenizer(title_ids),
                self.numeric_tokenizer(numeric),
                self.categorical_tokenizer(categorical),
            ],
            dim=1,
        )

        def always_real(n_positions: int) -> torch.Tensor:
            return torch.zeros(
                batch_size, n_positions, dtype=torch.bool, device=tokens.device
            )

        # Only the title can be padded: [CLS] and every tabular token always carry a
        # real value, so their entries stay False.
        padding_mask = torch.cat(
            [
                always_real(1),
                title_ids == PAD_ID,
                always_real(len(self.numeric_columns) + len(self.categorical_columns)),
            ],
            dim=1,
        )

        return tokens, padding_mask
