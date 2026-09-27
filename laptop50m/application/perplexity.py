"""Held-out perplexity over a token stream with a strided sliding window (each token scored exactly once)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _spans(n: int, window: int, stride: int):
    """Yield (begin, end, new): inputs ids[begin:end-1], targets ids[begin+1:end]; the last `new` targets are scored."""
    begin, prev_end = 0, 1  # position 0 has no context -> never scored
    while prev_end < n:
        end = min(begin + window + 1, n)
        yield begin, end, end - prev_end
        prev_end = end
        begin += stride


@torch.no_grad()
def sliding_window_nll(model, ids, window: int, stride: int, device: str = "cpu",
                       batch_size: int = 1) -> tuple[float, int]:
    """Return (sum of NLL in nats, number of scored tokens). Token t>0 is predicted once, with up to
    `window` tokens of left context; windows advance by `stride` (stride <= window).
    Windows of equal length are batched together."""
    ids = np.asarray(ids, dtype=np.int64)
    assert 0 < stride <= window
    spans = list(_spans(len(ids), window, stride))
    groups: dict[int, list] = {}
    for s in spans:
        groups.setdefault(s[1] - s[0], []).append(s)
    total, count = 0.0, 0
    for _, group in groups.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            x = torch.from_numpy(np.stack([ids[b:e - 1] for b, e, _ in chunk])).to(device)
            y = torch.from_numpy(np.stack([ids[b + 1:e] for b, e, _ in chunk])).to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")  # (B, T)
            for j, (_, _, new) in enumerate(chunk):
                total += float(nll[j, -new:].sum())
                count += new
    return total, count
