"""uint16 memmap token shards -> (x, y) batches."""
from __future__ import annotations

import numpy as np
import torch


class TokenShard:
    def __init__(self, path: str):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int, seed=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng(seed)
        ix = rng.integers(0, len(self.data) - seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)
