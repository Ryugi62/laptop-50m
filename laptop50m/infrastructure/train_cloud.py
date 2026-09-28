"""v2 training on a CUDA GPU (Colab / Kaggle T4, single or torchrun multi-GPU): Muon + QK-norm, WSD, anneal mix.

    python -m laptop50m.infrastructure.train_cloud --data _data/tok2 --val-dir _data/tok --out runs/l50m-v2 \
        --tokens 1.2e9
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

from laptop50m.adapters.data import MultiShard
from laptop50m.adapters.model_torch import GPT
from laptop50m.application.train import TrainConfig, train
from laptop50m.domain.config import ModelConfig, check_budget


def plan(tokens: float, global_batch: int, micro: int, seq: int, world: int) -> dict:
    """Steps and grad-accumulation for a token target (pure)."""
    per_micro = micro * seq * world
    accum = max(1, round(global_batch / per_micro))
    gb = accum * per_micro
    return {"accum": accum, "global_batch": gb, "max_steps": max(1, int(tokens // gb))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="_data/tok2")
    ap.add_argument("--val-dir", default="_data/tok")
    ap.add_argument("--out", default="runs/l50m-v2")
    ap.add_argument("--tokens", type=float, default=1.2e9)
    ap.add_argument("--global-batch", type=int, default=262_144)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--decay-frac", type=float, default=0.35)
    ap.add_argument("--anneal-frac", type=float, default=0.3)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--ckpt-every", type=int, default=900)
    ap.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    ap.add_argument("--init-from", default="", help="weights-only checkpoint to continue from (e.g. v1 step 9000)")
    ap.add_argument("--no-qk-norm", action="store_true", help="needed when continuing a v1 (no QK-norm) model")
    ap.add_argument("--pace", action="store_true", help="local Mac: idle gate + 25%% duty while the user is active")
    a = ap.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = a.dtype
    if dtype == "auto":
        if device == "cuda":
            dtype = "bf16" if (torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8) else "fp16"
        else:
            dtype = "bf16" if device == "mps" else "fp32"
    mc = ModelConfig(max_seq_len=a.seq, qk_norm=not a.no_qk_norm)
    n = check_budget(mc)
    p = plan(a.tokens, a.global_batch, a.micro, a.seq, world)
    train_bins = sorted(glob.glob(os.path.join(a.data, "train_*.bin")))
    hq_bins = sorted(glob.glob(os.path.join(a.data, "hq_*.bin")))
    assert train_bins, f"no train_*.bin in {a.data}"
    tc = TrainConfig(train_bin=",".join(train_bins), val_bin=os.path.join(a.val_dir, "fineweb_val.bin"),
                     out_dir=a.out, batch_size=a.micro, seq_len=a.seq, grad_accum=p["accum"],
                     max_steps=p["max_steps"], warmup=max(1, int(p["max_steps"] * a.warmup_frac)),
                     decay_frac=a.decay_frac, max_lr=a.adam_lr, min_lr=0.0, eval_every=250, eval_batches=20,
                     log_every=10, ckpt_every_sec=a.ckpt_every, max_hours=a.max_hours,
                     device=device, dtype=dtype,
                     extra_val={"wikitext103_val": os.path.join(a.val_dir, "wikitext103_validation.bin")},
                     optimizer="muon", muon_lr=a.muon_lr, anneal_bins=hq_bins or None,
                     anneal_frac=a.anneal_frac if hq_bins else 0.0, compile=a.compile)
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(a.out, exist_ok=True)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        meta = {"model": mc.to_dict(), "params": n, "plan": p, "dtype": dtype, "gpu": gpu, "device": device, "init_from": a.init_from, "world": world,
                "train_bins": train_bins, "hq_bins": hq_bins, "torch": torch.__version__}
        json.dump(meta, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        print(f"[train_cloud] {json.dumps(meta)}", flush=True)
    factory = GPT
    if a.init_from:
        def factory(c, _src=a.init_from):
            m = GPT(c)
            ck = torch.load(_src, map_location="cpu", weights_only=False)
            m.load_state_dict(ck["model"] if "model" in ck else ck)
            return m
    pacer = None
    if a.pace:
        from laptop50m.infrastructure.idle_gate import hid_idle_seconds, wait_until_idle
        from laptop50m.infrastructure.pacer import IdlePacer
        wait_until_idle(120, log=lambda m: print(m, flush=True))
        release = torch.mps.empty_cache if device == "mps" else None
        pacer = IdlePacer(hid_idle_seconds, threshold=120, on_active=release)
    res = train(mc, tc, factory, MultiShard.from_path, log=lambda m: print(m, flush=True), pace=pacer)
    print(f"[train_cloud] finished {res}", flush=True)
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
