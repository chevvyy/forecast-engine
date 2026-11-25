"""Model compression: global magnitude pruning. Distillation lives in trainer.fit(teacher_pred=...).

Unstructured pruning does not make a dense ONNX file smaller by itself; here it acts as a regulariser
and as a signal for how much capacity a tier really needs. The size wins for the phone tier come from
distilling into a narrower PatchNet.
"""
from __future__ import annotations

import torch


def magnitude_masks(model: torch.nn.Module, sparsity: float) -> dict[str, torch.Tensor]:
    """Zero the smallest ``sparsity`` share of all multi-dimensional weights (biases and norms kept)."""
    weights = {n: p for n, p in model.named_parameters() if p.dim() > 1}
    if not weights or sparsity <= 0:
        return {}
    allw = torch.cat([p.detach().abs().flatten() for p in weights.values()])
    k = int(sparsity * allw.numel())
    if k < 1:
        return {}
    thr = torch.kthvalue(allw.cpu(), k).values.to(allw.device)
    return {n: (p.detach().abs() > thr).to(p.dtype) for n, p in weights.items()}


def nonzero_params(model: torch.nn.Module) -> int:
    return int(sum((p != 0).sum().item() for p in model.parameters()))
