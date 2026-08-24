from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


from config.config import load_config

# Following the transformer implementation from the class' colab: "clase2-step-by-step transformer.ipynb" https://colab.research.google.com/drive/1SocIaMKKFqRYFeLUUDmiSaw9OkRXjf0S#scrollTo=mVQkgrZpxm9f&line=4&uniqifier=1

class Head(nn.Module):
    """Single head of self-attention.
    
    """
    def __init__(
        self,
        head_size,
        n_embd, # this is the d_model in the paper
        dropout
    ):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = (batch_size, context_length, n_embd)
        k = self.key(x) # (batch_size, context_length, head_size)
        q = self.query(x) # (batch_size, context_length, head_size)
        
        # (BS, CL, HS) @ (BS, HS, CL) -> (BS, CL, CL)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v # (BS, CL, CL) @ (BS, CL, HS) -> (BS, CL, HS)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dim_feedforward, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dim_feedforward, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dim_feedforward, dropout)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # "x + ..." is the residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class EncodingBlock(nn.Module):
    """Stack of `num_layers` encoder blocks configured via config.json."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        config = load_config()

        self.embedding_dim = int(config.get("d_model"))
        self.num_heads = int(config.get("n_heads"))
        self.num_layers = int(config.get("num_layers"))
        self.dropout = float(config.get("dropout"))
        self.dim_feedforward = int(config.get("dim_feedforward"))
        # self.activation = config.get("activation")
        
        self.blocks = nn.Sequential(*[
            Block(
                self.embedding_dim,
                self.num_heads,
                self.dim_feedforward,
                self.dropout
            )
            for _ in range(self.num_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_f(self.blocks(x))


if __name__ == "__main__":
    # Attention in 3 dimensions, with hand-written weights.
    #
    # Three words, three dimensions, two heads whose weight matrices are
    # written by hand instead of learned. The point: see one head do something
    # you chose, so you know what a *trained* head would be discovering on
    # its own.
    #
    # Run from the repo root:  python -m model.encoding_block
    from pathlib import Path

    import matplotlib.pyplot as plt

    torch.set_grad_enabled(False)  # a demo, not training

    # ------------------------------------------------------------------
    # The vocabulary. Three dimensions, each given a meaning by hand.
    # ------------------------------------------------------------------
    WORDS = ["the", "black", "dog"]
    MEANINGS = ["animal", "colour", "det"]

    x = torch.tensor([
        [0.0, 0.0, 1.0],   # the    -> determiner
        [0.0, 1.0, 0.0],   # black  -> colour
        [1.0, 0.0, 0.0],   # dog    -> animal
    ])

    def show(title, matrix, rows=WORDS, cols=None):
        print(f"\n{title}")
        if cols:
            print(" " * 8 + "".join(f"{c:>8}" for c in cols))
        for name, row in zip(rows, matrix):
            print(f"  {name:<7}" + "".join(f"{value:>8.2f}" for value in row))

    def run_head(x, w_query, w_key, w_value, label):
        """Attention by hand, printing every intermediate step."""
        q, k, v = x @ w_query, x @ w_key, x @ w_value
        scores = q @ k.T / (x.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        print("\n" + "=" * 62)
        print(label)
        print("=" * 62)
        show("q = x @ Wq", q)
        show("k = x @ Wk", k)
        show("scores = q @ k.T / sqrt(3)", scores, cols=WORDS)
        show("weights = softmax(scores)   <- rows sum to 1", weights, cols=WORDS)
        show("out = weights @ v", out, cols=MEANINGS)
        return weights, out

    show("x  (the input embeddings)", x, cols=MEANINGS)

    # ------------------------------------------------------------------
    # HEAD 1 - the "colour" head.
    # Query fires on the animal dimension: "which colour describes me?"
    # Key fires on the colour dimension:   "I am a colour."
    # ------------------------------------------------------------------
    Wq1 = torch.tensor([[4., 0, 0], [0, 0, 0], [0, 0, 0]])  # animal only
    Wk1 = torch.tensor([[0., 0, 0], [4, 0, 0], [0, 0, 0]])  # colour only
    Wv1 = torch.eye(3)                                      # pass the embedding through
    weights1, out1 = run_head(x, Wq1, Wk1, Wv1, "HEAD 1 - colour head")

    # ------------------------------------------------------------------
    # HEAD 2 - the "determiner" head. Same query, different key.
    # ------------------------------------------------------------------
    Wq2 = torch.tensor([[4., 0, 0], [0, 0, 0], [0, 0, 0]])
    Wk2 = torch.tensor([[0., 0, 0], [0, 0, 0], [4, 0, 0]])  # determiner only
    Wv2 = torch.eye(3)
    weights2, out2 = run_head(x, Wq2, Wk2, Wv2, "HEAD 2 - determiner head")

    # ------------------------------------------------------------------
    # What moved
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("WHERE 'dog' ENDED UP")
    print("=" * 62)
    print(f"  before        : {[round(f, 2) for f in x[2].tolist()]}")
    print(f"  after head 1  : {[round(f, 2) for f in out1[2].tolist()]}"
          f"   <- picked up colour")
    print(f"  after head 2  : {[round(f, 2) for f in out2[2].tolist()]}"
          f"   <- picked up 'the'")
    print(f"\n  head 1 sent {weights1[2, 1]:.0%} of dog's attention to 'black'")
    print(f"  head 2 sent {weights2[2, 0]:.0%} of dog's attention to 'the'")
    print("\n  Same input, same query matrix, different key matrix.")
    print("  That alone makes two heads look at completely different things.")

    # ------------------------------------------------------------------
    # Why Block adds x back
    # ------------------------------------------------------------------
    # Read head 1's output literally and 'dog' has stopped being an animal:
    # it came out as pure colour. Attention returns a blend of OTHER tokens,
    # so on its own it overwrites. Block never uses it on its own - it does
    # `x = x + self.sa(self.ln1(x))`, which keeps the original meaning and
    # adds the borrowed one on top.
    print("\n" + "=" * 62)
    print("WHY Block DOES  x = x + sa(x)  INSTEAD OF  x = sa(x)")
    print("=" * 62)
    show("out1 alone: 'dog' is no longer an animal", out1, cols=MEANINGS)
    show("x + out1: animal AND colour, which is what we wanted",
         x + out1, cols=MEANINGS)
    print("\n  That is the residual connection, and this is what it is for.")

    # ------------------------------------------------------------------
    # The same weights, through the real Head class in this file
    # ------------------------------------------------------------------
    # nn.Linear computes x @ W.T, so transpose to load our matrices in.
    # If this prints True, everything above describes THIS file's code and
    # not a parallel implementation of attention.
    real_head = Head(head_size=3, n_embd=3, dropout=0.0).eval()
    real_head.query.weight.data = Wq1.T
    real_head.key.weight.data = Wk1.T
    real_head.value.weight.data = Wv1.T
    matches = torch.allclose(real_head(x.unsqueeze(0))[0], out1, atol=1e-6)

    print("\n" + "=" * 62)
    print("THE SAME THING, THROUGH THIS FILE'S Head CLASS")
    print("=" * 62)
    print(f"  Head(head_size=3, n_embd=3) with our matrices loaded in")
    print(f"  reproduces head 1 exactly: {matches}")

    # ------------------------------------------------------------------
    # Concatenate, as MultiHeadAttention does
    # ------------------------------------------------------------------
    concatenated = torch.cat([out1, out2], dim=-1)
    multi_head = MultiHeadAttention(num_heads=2, head_size=3, n_embd=3, dropout=0.0)
    print(f"\n  concat: {tuple(out1.shape)} + {tuple(out2.shape)}"
          f" -> {tuple(concatenated.shape)}")
    print(f"  MultiHeadAttention.proj is {multi_head.proj}, which mixes the two")
    print("  heads back down to d_model=3 so the next Block sees the same width.")

    # ------------------------------------------------------------------
    # What attention still cannot do
    # ------------------------------------------------------------------
    # Reorder the words and every output row simply moves with its word.
    # Nothing about a token's output depends on WHERE it sits, because
    # nothing in Head ever looks at position. Fixing that is what
    # model/positional_encoding.py is for, and encoder_only_model.py does
    # not use it yet.
    order = [2, 0, 1]  # dog, the, black
    shuffled = x[order]
    q, k, v = shuffled @ Wq1, shuffled @ Wk1, shuffled @ Wv1
    out_shuffled = F.softmax(q @ k.T / 3**0.5, dim=-1) @ v

    print("\n" + "=" * 62)
    print("WHAT ATTENTION STILL CANNOT DO")
    print("=" * 62)
    print(f"  reordered the sentence to: {[WORDS[i] for i in order]}")
    print(f"  outputs are the same rows, just reordered: "
          f"{torch.allclose(out_shuffled, out1[order], atol=1e-6)}")
    print("\n  Attention has no notion of position, so word order is invisible")
    print("  to it. model/positional_encoding.py fixes this, and")
    print("  model/encoder_only_model.py does not use it yet.")

    # ------------------------------------------------------------------
    # Draw both heads
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, weights, name in (
        (axes[0], weights1, "head 1 - colour"),
        (axes[1], weights2, "head 2 - determiner"),
    ):
        axis.imshow(weights, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(name)
        axis.set_xticks(range(3), WORDS)
        axis.set_yticks(range(3), WORDS)
        axis.set_xlabel("attends to")
        axis.set_ylabel("word")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{weights[row, column]:.2f}",
                          ha="center", va="center", color="white", fontsize=9)

    figure.suptitle("Hand-written attention: 'dog' looks where we told it to")
    figure.tight_layout()
    destination = Path(__file__).with_name("attention_heads.jpg")
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    print(f"\n  saved {destination}")
    plt.show()
