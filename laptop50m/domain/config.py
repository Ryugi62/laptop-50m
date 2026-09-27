"""Pure-python model configuration and parameter budget (no torch)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

PARAM_BUDGET = 50_000_000  # GIBC V2 Track 01: total trainable params incl. embeddings + head


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    d_model: int = 512
    n_layer: int = 12
    n_head: int = 8
    ffn_hidden: int = 1_536
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def analytic_param_count(c: ModelConfig) -> int:
    """Closed-form count: tied embedding + per-layer (attn 4d^2, SwiGLU 3*d*h, 2 RMSNorm) + final norm."""
    emb = c.vocab_size * c.d_model
    head = 0 if c.tie_embeddings else c.vocab_size * c.d_model
    per_layer = 4 * c.d_model * c.d_model + 3 * c.d_model * c.ffn_hidden + 2 * c.d_model
    return emb + head + c.n_layer * per_layer + c.d_model


def check_budget(c: ModelConfig) -> int:
    n = analytic_param_count(c)
    if n > PARAM_BUDGET:
        raise ValueError(f"{n:,} params > budget {PARAM_BUDGET:,}")
    return n
