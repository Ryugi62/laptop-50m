"""Print the total number of trainable parameters of Laptop-50M (includes token embedding and output head).

    python count_params.py --config configs/l50m-v1.json          # from the config (builds the PyTorch module)
    python count_params.py --ckpt laptop50m-v2.pt                  # from a released weights file (counts stored tensors)

A tied embedding/output head is one tensor stored under two keys. Summing `numel()` over the state_dict keys
therefore counts it twice (57,684,480 for this model); `--ckpt` counts each stored tensor once and prints both.
"""
import argparse
import json

import torch

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig, PARAM_BUDGET, analytic_param_count


def count_checkpoint_params(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    by_storage: dict = {}
    for k, v in sd.items():
        by_storage.setdefault(v.untyped_storage().data_ptr(), []).append((k, v.numel()))
    unique = sum(entries[0][1] for entries in by_storage.values())
    naive = sum(v.numel() for v in sd.values())
    shared = [tuple(k for k, _ in e) for e in by_storage.values() if len(e) > 1]
    cfg = ck.get("model_config") if isinstance(ck, dict) else None
    analytic = analytic_param_count(ModelConfig(**cfg)) if cfg else None
    return {"unique": unique, "naive": naive, "shared": shared, "analytic": analytic, "tensors": len(sd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/l50m-v1.json")
    ap.add_argument("--ckpt", default="", help="weights file to count instead of building the module")
    a = ap.parse_args()
    cfg = ModelConfig(**json.load(open(a.config))["model"])
    if a.ckpt:
        r = count_checkpoint_params(a.ckpt)
        expected = r["analytic"] if r["analytic"] is not None else analytic_param_count(cfg)
        n = r["unique"]
        print(f"checkpoint: {a.ckpt} ({r['tensors']} state_dict keys)")
        print(f"tensors stored under two keys (tied, counted once): {r['shared']}")
        print(f"naive sum over keys (counts the tied matrix twice): {r['naive']:,}")
        print(f"unique parameters in the file == config formula ({expected:,}): {n == expected}")
        print(f"TOTAL trainable parameters: {n:,} <= {PARAM_BUDGET:,}: {n <= PARAM_BUDGET}")
        if n != expected or n > PARAM_BUDGET:
            raise SystemExit(1)
        return
    m = GPT(cfg)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)  # tied weights are counted once
    emb = m.tok_emb.weight.numel()
    assert n == analytic_param_count(cfg)
    print(f"config: {cfg.to_dict()}")
    print(f"token embedding (tied with output head): {emb:,}")
    print(f"transformer blocks + final norm: {n - emb:,}")
    print(f"TOTAL trainable parameters: {n:,} <= {PARAM_BUDGET:,}: {n <= PARAM_BUDGET}")
    if n > PARAM_BUDGET:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
