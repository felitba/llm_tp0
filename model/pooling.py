""" Pooling strategies: (B, T+1, C) hidden states -> (B,C) one vector per row
(B, T+1) bool with True marking padding. Position 0 is always [CLS].
"""

from __future__ import annotations

import torch
import torch.nn as nn

class ClsPooling(nn.Module):
    """ Read [CLS] slot: index 0"""
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h[:, 0]


class MeanPooling(nn.Module):
    """Average real tokens, exclusing [CLS] and padding"""
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = h[:, 1:] # drop [CLS]
        if mask is None:
            return h.mean(dim=1)
        keep = (~mask[:, 1:]).unsqueeze(-1).to(h.dtype) # (B,T,1) 1.0 = real
        counts = keep.sum(dim=1) # (B,1)
        assert (counts > 0).all(), "a row is entirely padding"
        return (h*keep).sum(dim=1)/counts.clamp(min=1)
    
class MaxPooling(nn.Module):
    """Per dimension max over real tokens, excluding [CLS] and padding"""
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = h[:, 1:]
        if mask is None:
            return h.max(dim=1).values
        
        pad = mask[:, 1:] # (B,T)
        assert (~pad).any(dim=1).all(), "a row is entirely padding"
        return h.masked_fill(pad.unsqueeze(-1), torch.finfo(h.dtype).min).max(dim=1).values
    
POOLERS = {"cls":ClsPooling, "mean":MeanPooling, "max": MaxPooling}

def build_pooler(name:str) -> nn.Module:
    if name not in POOLERS:
        raise ValueError(f"unknown pooling {name!r}; expected one of {sorted(POOLERS)}")
    return POOLERS[name]()
