"""PatchNet: one compact architecture scaled to three size tiers (phone / 8 GB / 16 GB).

ModernTCN-flavoured: per-window instance normalisation, a strided patch embedding, then blocks of
large-kernel depthwise convolution + pointwise MLP. Convolutions only, so it exports cleanly to ONNX
and runs on a phone CPU. The output is deliberately tiny: H horizons x 2 targets x Q quantiles.
No symbol embeddings: the model is scale-free and works on tickers it never saw.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d: int, k: int, ffn: int, drop: float):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.norm = nn.GroupNorm(1, d)
        self.pw1 = nn.Conv1d(d, d * ffn, 1)
        self.pw2 = nn.Conv1d(d * ffn, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.dw(x))
        z = self.pw2(self.drop(F.gelu(self.pw1(z))))
        return x + self.drop(z)


class PatchNet(nn.Module):
    def __init__(self, n_channels: int, n_tab: int, n_horizons: int, n_quantiles: int, lookback: int = 64,
                 patch: int = 4, stride: int = 2, d_model: int = 32, n_blocks: int = 2, kernel: int = 9,
                 ffn: int = 2, dropout: float = 0.1):
        super().__init__()
        if n_quantiles % 2 == 0:
            raise ValueError("use an odd number of quantiles so the median anchors the head")
        self.cfg = dict(n_channels=n_channels, n_tab=n_tab, n_horizons=n_horizons, n_quantiles=n_quantiles,
                        lookback=lookback, patch=patch, stride=stride, d_model=d_model, n_blocks=n_blocks,
                        kernel=kernel, ffn=ffn, dropout=dropout)
        self.lookback, self.H, self.Q = lookback, n_horizons, n_quantiles
        self.embed = nn.Conv1d(n_channels, d_model, patch, stride)
        self.blocks = nn.Sequential(*[Block(d_model, kernel | 1, ffn, dropout) for _ in range(n_blocks)])
        self.side = nn.Linear(2 * n_channels + n_tab, d_model)
        self.head = nn.Linear(3 * d_model, n_horizons * 2 * n_quantiles)

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """x [B, C, L>=lookback] (most recent bar last), tab [B, T] -> [B, H, 2, Q] sorted along Q."""
        x = x[..., -self.lookback:]
        mu = x.mean(-1, keepdim=True)
        sd = ((x - mu) ** 2).mean(-1, keepdim=True).sqrt() + 1e-5
        z = self.blocks(self.embed((x - mu) / sd))
        side = self.side(torch.cat([mu.squeeze(-1) / sd.squeeze(-1), torch.log(sd.squeeze(-1)), tab], dim=1))
        feat = torch.cat([z.mean(-1), z[..., -1], F.gelu(side)], dim=1)
        raw = self.head(feat).view(-1, self.H, 2, self.Q)
        m = self.Q // 2
        mid = raw[..., m:m + 1]
        up = mid + torch.cumsum(F.softplus(raw[..., m + 1:]), dim=-1)
        lo = mid - torch.flip(torch.cumsum(F.softplus(torch.flip(raw[..., :m], dims=[-1])), dim=-1), dims=[-1])
        return torch.cat([lo, mid, up], dim=-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
