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


class MultiShard:
    """Several token files; each window's file is drawn in proportion to file length."""

    def __init__(self, paths):
        self.shards = [TokenShard(p) for p in paths]
        n = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.p = n / n.sum()

    @classmethod
    def from_path(cls, spec):
        if isinstance(spec, (list, tuple)):
            return cls(list(spec))
        return cls([s for s in str(spec).split(",") if s])

    def __len__(self) -> int:
        return int(sum(len(s) for s in self.shards))

    def batch(self, batch_size: int, seq_len: int, seed=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng(seed)
        which = rng.choice(len(self.shards), size=batch_size, p=self.p)
        xs, ys = [], []
        for i, sh in enumerate(self.shards):
            k = int((which == i).sum())
            if k:
                x, y = sh.batch(k, seq_len, rng=rng)
                xs.append(x); ys.append(y)
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.from_numpy(rng.permutation(batch_size))
        return x[perm], y[perm]
