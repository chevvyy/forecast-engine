"""Training loop shared by plain training, distillation and prune-then-finetune.

Pinball loss on monotone quantiles, AdamW + cosine schedule, bf16 autocast on CUDA, early stopping on
a purged slice of the training timeline (never on the validation block), and a wall-clock deadline so
an unattended worker always finishes its job.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from forecast.models.patchnet import PatchNet


def pinball(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    d = y.unsqueeze(-1) - pred
    return torch.maximum(q * d, (q - 1.0) * d).mean()


@torch.no_grad()
def predict(model: PatchNet, X: np.ndarray, tab: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        tb = torch.from_numpy(tab[i:i + batch]).to(device)
        out.append(model(xb, tb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.H, 2, model.Q), np.float32)


def apply_masks(model: torch.nn.Module, masks: dict[str, torch.Tensor] | None) -> None:
    if not masks:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name])


def fit(model: PatchNet, train: dict, es: dict, cfg: dict, quantiles: list[float], device: str,
        teacher_pred: np.ndarray | None = None, masks: dict[str, torch.Tensor] | None = None,
        deadline: float | None = None) -> dict:
    """train/es: dicts with X, tab, y numpy arrays. ``teacher_pred`` [n_train, H, 2, Q] turns this into
    distillation. ``masks`` keeps pruned weights at zero. Returns training stats; best weights are loaded
    back into ``model``."""
    model.to(device)
    q = torch.tensor(quantiles, dtype=torch.float32, device=device)
    bs, epochs = int(cfg.get("batch_size", 512)), int(cfg.get("max_epochs", 30))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    n = len(train["X"])
    steps = max(epochs * ((n + bs - 1) // bs), 1)
    warm = max(min(int(cfg.get("warmup_steps", 200)), steps // 10), 1)
    def lr_at(i: int) -> float:  # linear warmup then cosine decay to 5% of peak
        return (i + 1) / warm if i < warm else 0.05 + 0.475 * (1 + math.cos(math.pi * (i - warm) / max(steps - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    alpha = float(cfg.get("distill_alpha", 0.5))
    on_gpu = device == "cuda"
    # Keep tensors on the device when they fit comfortably; otherwise stream batches from host RAM.
    keep = on_gpu and train["X"].nbytes < 0.35 * torch.cuda.mem_get_info(0)[0]
    def to(a):
        return torch.from_numpy(a).to(device) if keep or not on_gpu else torch.from_numpy(a)
    Xt, Tt, Yt = to(train["X"]), to(train["tab"]), to(train["y"])
    Pt = to(teacher_pred) if teacher_pred is not None else None
    apply_masks(model, masks)
    best, best_state, bad, step, ep_done = float("inf"), None, 0, 0, 0
    patience = int(cfg.get("patience", 5))
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xt.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, tb, yb = Xt[idx].to(device), Tt[idx].to(device), Yt[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu):
                pred = model(xb, tb)
            pred = pred.float()
            loss = pinball(pred, yb, q)
            if Pt is not None:
                loss = (1 - alpha) * loss + alpha * torch.nn.functional.smooth_l1_loss(pred, Pt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            sched.step()
            step += 1
            apply_masks(model, masks)
            if deadline and time.time() > deadline:
                break
        ep_done += 1
        pe = predict(model, es["X"], es["tab"], device)
        d = es["y"][..., None] - pe
        qa = np.asarray(quantiles, np.float32)
        val = float(np.maximum(qa * d, (qa - 1) * d).mean()) if len(pe) else float(loss)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience or (deadline and time.time() > deadline):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    apply_masks(model, masks)
    peak = torch.cuda.max_memory_allocated() / 2**30 if on_gpu else 0.0
    return {"epochs": ep_done, "es_pinball": best, "peak_vram_gb": round(peak, 3)}
