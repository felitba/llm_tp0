"""
Encoder-only transformer for BTR prediction
B = batch_size, T = sequence_length, C = d_model

token ids (B,T) int64
hidden states (B,T,C) (or feature vectors) float32
padding mask (B,T) bool
output (B,) float32 logits <- this is what we want for our result to tell us if bought=true or not
"""


from __future__ import annotations

import torch
import torch.nn as nn
from config.config import load_config
from model.embedding import TokenEmbedding
from model.encoding_block import EncodingBlock
from model.pooling import build_pooler


class EncoderOnlyModel(nn.Module):
    """Encoder-only Transformer model for BTR prediction."""

    def __init__(self, pooling:str | None = None) -> None:
        super().__init__()

        config = load_config()
        self.embedding_dim = int(config.get("d_model"))
        self.vocab_size = int(config.get("vocab_size"))
        self.dropout = float(config.get("dropout"))

        # Embedding layer
        self.embedding = TokenEmbedding()
        
        # [CLS]: one learned vector preprended to every sequence
        # shape = (1,1,C)
        # OBS: torch.randn(1, 1, self.embedding_dim)*0.02
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.embedding_dim))
        
        # Pooling: (B, T+1, C) -> (B,C). Set per run for ablation
        self.pooling = pooling or config.get("pooling")
        self.pooler = build_pooler(self.pooling)
        
        
        # TODO: PositionalEncoding

        # Encoder block
        self.encoder_block = EncodingBlock()

        # Classification head: pooled (B,C) -> one logit per row.
        # Deliberately linear. A non-linear head can partly compensate for a
        # weaker pooled representation, which is exactly what the pooling
        # ablation is trying to measure.
        # No Sigmoid: this returns a LOGIT. Train with nn.BCEWithLogitsLoss,
        # which is exact where BCELoss(sigmoid(x)) returns zero gradient once
        # |logit| >= 17 in float32.
        self.classification_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, 1),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Token ids -> contextual hidden states / features, [CLS] plus one per token 
            Args:
                x: (B,T) int64 token ids.
            Returns:
                (B,T+1,C) flaot32 hidden states. position 0 is [CLS]
        """
        assert x.ndim == 2, f"expected (B,T) token ids, got {tuple(x.shape)}"
        assert x.dtype == torch.long, f"expected int64 token ids, got {x.dtype}"
        
        batch_size, sequence_length = x.shape
        
        embedded = self.embedding(x) # (B,T,C)
        expected = (batch_size, sequence_length, self.embedding_dim)
        assert embedded.shape == expected, (f"embedding returned {tuple(embedded.shape)}, expected {expected}")
        
        cls_tokens = self.cls_token.expand(batch_size, 1, self.embedding_dim) # (B, 1, C)
        embedded = torch.cat([cls_tokens, embedded], dim = 1) # (B, T+1, C)
        
        # TODO: PositionEncoding goes here, after the prepend, so that
        # [CLS] takes position 0 and the real tokens shift to position 1..T
        
        encoded = self.encoder_block(embedded) # (B,T+1,C)
        expected = (batch_size, sequence_length + 1, self.embedding_dim)
        assert encoded.shape == expected, (
            f"encoder returned {tuple(encoded.shape)}, expected {expected}"
            )
        return encoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len) with token ids

        Returns:
            Output predictions of shape (batch_size,) float32 logits (not probabilities - apply torch.sigmoid for a probability) for BTR
        """
        encoded = self.encode(x) # (B,T+1,C)

        # Pooling strategy is chosen once in __init__, so this line never
        # changes when the ablation swaps cls / mean / max.
        pooled = self.pooler(encoded)  # (B, C)

        # (B, 1) -> (B,) to match the label shape. squeeze(-1), not squeeze():
        # bare squeeze() turns a batch of one into a 0-d scalar.
        return self.classification_head(pooled).squeeze(-1)

if __name__== "__main__":
    torch.manual_seed(0)
    
    model = EncoderOnlyModel().eval() # dropout off
    
    B,T = 3,7
    x = torch.randint(0, model.vocab_size, (B,T))
    
    with torch.no_grad():
        h = model.encode(x)
        print(f"    encode : {tuple(x.shape)} -> {tuple(h.shape)}")
        assert h.shape == (B,T+1,model.embedding_dim)
        
        y = model(x)
        print(f"    forward: {tuple(x.shape)} -> {tuple(y.shape)}")
        assert y.shape == (B,)
        
        assert y.abs().max() > 0, "output is identically zero"

        # Batch of one must stay 1-dimensional. Catches bare .squeeze().
        assert model(x[:1]).shape == (1,), f"got {tuple(model(x[:1]).shape)}"
        print(" head shapes: ok")

        
        perm = torch.tensor([2,0,1])
        assert torch.allclose(model.encode(x)[perm], model.encode(x[perm]),
                              atol=1e-6), "row permutation changed the outputs"
        assert torch.allclose(model.encode(x[:1]), model.encode(x)[:1],
                              atol=1e-6), "batch-of-one differs from the batch"
        
        print(" batch independence: ok")
        print(" shape contract: ok")
        
        # Every strategy returns (B, C), and they are genuinely different functions.
        pooled = {name: build_pooler(name)(h) for name in ("cls", "mean", "max")}
        for name, p in pooled.items():
            assert p.shape == (B, model.embedding_dim), f"{name} -> {tuple(p.shape)}"
        assert not torch.allclose(pooled["cls"], pooled["mean"])
        assert not torch.allclose(pooled["mean"], pooled["max"])

        # Unmasked mean must equal the naive mean over the non-[CLS] tokens.
        assert torch.allclose(pooled["mean"], h[:, 1:].mean(dim=1), atol=1e-6)
        print(" pooling shapes: ok")
        
    # [CLS] must be a registered, trainable param
    assert any(p is model.cls_token for p in model.parameters()), "cls_token is not registered as a param"
    
    model.zero_grad()
    model(x).sum().backward()
    assert model.cls_token.grad is not None, "cls_token received no gradient"
    assert model.cls_token.grad.abs().sum() > 0, "cls_token gradient is all zero"
    
    print( "cls_token trains: ok")
    
    # Masked pooling must ignore padding completely. Take a clean batch, glue
    # large garbage vectors onto the end, mark them as padding, and the pooled
    # result must not move at all. This fails loudly if the mask is ignored,
    # inverted, off by one, or broadcast wrong.
    clean = torch.randn(2, 1 + 5, 4)                        # [CLS] + 5 real tokens
    padded = torch.cat([clean, torch.randn(2, 3, 4) * 100], dim=1)
    pad_mask = torch.zeros(2, 1 + 5 + 3, dtype=torch.bool)
    pad_mask[:, 1 + 5:] = True                              # last 3 are padding

    for name in ("mean", "max"):
        pooler = build_pooler(name)
        assert torch.allclose(pooler(clean), pooler(padded, pad_mask), atol=1e-6), \
            f"{name} pooling is not ignoring padding"

    print(" padding invariance: ok")

    # The head must emit logits, not probabilities. Scale the final weight up
    # so that any un-squashed head produces values far outside (0, 1), while a
    # Sigmoid would pin them inside it no matter what. Last check in the file:
    # it deliberately corrupts the model.
    with torch.no_grad():
        model.classification_head[-1].weight.mul_(1000)
    assert model(x).abs().max() > 1.0, "output is bounded - is a Sigmoid still attached?"

    print(" head emits logits: ok")