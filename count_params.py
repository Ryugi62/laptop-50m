"""Print the total number of trainable parameters of Laptop-50M (includes token embedding and output head).

    python count_params.py --config runs/l50m-v1/config.json
"""
import argparse
import json

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig, PARAM_BUDGET, analytic_param_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/l50m-v1.json")
    a = ap.parse_args()
    cfg = ModelConfig(**json.load(open(a.config))["model"])
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
