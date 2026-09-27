"""Warmup-Stable-Decay learning-rate schedule (pure function)."""
from __future__ import annotations


def lr_at(step: int, *, max_lr: float, min_lr: float, warmup: int, total: int, decay_frac: float = 0.2) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    decay_start = int(total * (1.0 - decay_frac))
    if step < decay_start:
        return max_lr
    if step >= total:
        return min_lr
    frac = (step - decay_start) / max(1, total - decay_start)
    return max_lr + (min_lr - max_lr) * frac
