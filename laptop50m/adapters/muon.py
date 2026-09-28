"""Muon optimizer (momentum + Newton-Schulz orthogonalisation) for 2-D hidden matrices.

Follows Keller Jordan's public Muon (github.com/KellerJordan/Muon): quintic Newton-Schulz with coefficients
(3.4445, -4.7750, 2.0315), Nesterov momentum 0.95, update scaled by sqrt(max(1, rows/cols)).
The fused qkv matrix (3d x d) is orthogonalised as three (d x d) slices.
"""
from __future__ import annotations

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X


def split_params(model):
    """(muon_params, adamw_params): 2-D weights inside transformer blocks -> Muon; embedding, norms -> AdamW."""
    muon, adam, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        if p.dim() == 2 and ".blocks." in f".{name}":
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                u = g.lerp(buf, mom) if group["nesterov"] else buf
                rows, cols = p.shape
                if rows == 3 * cols:  # fused qkv
                    o = newton_schulz(u.view(3, cols, cols), group["ns_steps"]).reshape(rows, cols)
                    scale = 1.0
                else:
                    o = newton_schulz(u, group["ns_steps"])
                    scale = max(1.0, rows / cols) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-lr * scale)
        return loss
