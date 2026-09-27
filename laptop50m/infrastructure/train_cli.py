"""Train Laptop-50M on MPS. Waits for the idle gate, then paces itself (25% duty while the user is active)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from laptop50m.adapters.data import TokenShard
from laptop50m.adapters.model_torch import GPT
from laptop50m.application.train import TrainConfig, train
from laptop50m.domain.config import ModelConfig, check_budget
from laptop50m.infrastructure.idle_gate import hid_idle_seconds, wait_until_idle
from laptop50m.infrastructure.pacer import IdlePacer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", default="_data/tok")
    ap.add_argument("--out", default="runs/l50m-v1")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-lr", type=float, default=1e-3)
    ap.add_argument("--max-hours", type=float, default=40.0)
    ap.add_argument("--idle", type=int, default=120)
    ap.add_argument("--no-pace", action="store_true")
    a = ap.parse_args()
    mc = ModelConfig(max_seq_len=a.seq)
    n = check_budget(mc)
    stats = json.load(open(os.path.join(a.tok_dir, "stats.json")))
    print(f"[train_cli] params={n:,} config={mc.to_dict()} data={stats}", flush=True)
    tc = TrainConfig(train_bin=os.path.join(a.tok_dir, "fineweb_train.bin"),
                     val_bin=os.path.join(a.tok_dir, "fineweb_val.bin"), out_dir=a.out,
                     batch_size=a.batch, seq_len=a.seq, grad_accum=a.accum, max_steps=a.max_steps,
                     warmup=300, max_lr=a.max_lr, min_lr=a.max_lr / 10, eval_every=250, eval_batches=25,
                     log_every=10, ckpt_every_sec=900, max_hours=a.max_hours, device="mps", dtype="bf16",
                     extra_val={"wikitext103_val": os.path.join(a.tok_dir, "wikitext103_validation.bin")})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": mc.to_dict(), "params": n}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    wait_until_idle(a.idle, log=lambda m: print(m, flush=True))
    release = torch.mps.empty_cache if torch.backends.mps.is_available() else None
    pacer = None if a.no_pace else IdlePacer(hid_idle_seconds, threshold=a.idle, on_active=release)
    res = train(mc, tc, GPT, TokenShard, log=lambda m: print(m, flush=True), pace=pacer)
    print(f"[train_cli] finished {res} paced_sleep_s={getattr(pacer, 'slept', 0):.0f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
