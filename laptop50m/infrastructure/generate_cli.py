"""Generate text from a Laptop-50M weights file (CPU is enough) and print the speed and memory it used.

    python -m laptop50m.infrastructure.generate_cli --ckpt laptop50m-v2.pt --tokenizer tokenizer.json \
        --prompt "The water cycle is" --new-tokens 60
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import torch
from tokenizers import Tokenizer

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig


@torch.no_grad()
def sample(model, ids: list, new_tokens: int, temperature: float = 0.7, top_k: int = 40, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    window = model.cfg.max_seq_len
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(new_tokens):
        logits, _ = model(x[:, -window:])
        logits = logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return x[0].tolist()


def load(ckpt: str, config: str) -> GPT:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    cfg = ck.get("model_config") or json.load(open(config))["model"]
    m = GPT(ModelConfig(**cfg))
    m.load_state_dict(sd)
    return m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="_data/tok/tokenizer.json")
    ap.add_argument("--config", default="configs/l50m-v1.json")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    model = load(a.ckpt, a.config)
    tok = Tokenizer.from_file(a.tokenizer)
    for p in a.prompt or ["The water cycle is"]:
        t0 = time.time()
        out = sample(model, tok.encode(p).ids, a.new_tokens, a.temperature, a.top_k, a.seed)
        dt = time.time() - t0
        print(f"--- prompt: {p!r}\n{tok.decode(out)}")
        print(f"[{a.new_tokens} new tokens in {dt:.2f} s on CPU ({a.threads} threads) = {a.new_tokens / dt:.1f} tok/s, "
              f"no KV cache]")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    print(f"[peak process memory {rss_mb:.0f} MB, weights fp32 {sum(p.numel() for p in model.parameters()) * 4 / 1e6:.0f} MB]")


if __name__ == "__main__":
    main()
