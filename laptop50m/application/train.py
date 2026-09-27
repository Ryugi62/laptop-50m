"""Training use case: WSD schedule, grad accumulation, time-based checkpoints, resume.

Ports (injected): model_factory(ModelConfig) -> nn.Module with forward(x, y) -> (logits, loss);
shard_factory(path) -> object with .batch(B, T, rng=...) and __len__.
"""
from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import torch

from laptop50m.domain.config import ModelConfig, analytic_param_count
from laptop50m.domain.schedule import lr_at


@dataclass
class TrainConfig:
    train_bin: str
    val_bin: str
    out_dir: str
    batch_size: int = 16
    seq_len: int = 1024
    grad_accum: int = 4
    max_steps: int = 5000
    warmup: int = 200
    decay_frac: float = 0.2
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 10
    ckpt_every_sec: int = 900
    max_hours: float = 14.0
    device: str = "mps"
    dtype: str = "bf16"
    seed: int = 1337
    extra_val: Optional[dict] = None  # name -> bin path


def _autocast(device: str, dtype: str):
    if dtype == "fp32" or device == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device, dtype=dt)


def _optimizer(model, tc: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=tc.max_lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, shard, tc: TrainConfig, n_batches: int, seed: int = 0) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = shard.batch(tc.batch_size, tc.seq_len, rng=rng)
        x, y = x.to(tc.device), y.to(tc.device)
        with _autocast(tc.device, tc.dtype):
            _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train(mc: ModelConfig, tc: TrainConfig, model_factory: Callable, shard_factory: Callable,
          stop_after: Optional[int] = None, log: Callable = print, pace: Optional[Callable] = None) -> dict:
    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    log_path = os.path.join(tc.out_dir, "train_log.jsonl")
    torch.manual_seed(tc.seed)
    model = model_factory(mc).to(tc.device)
    opt = _optimizer(model, tc)
    step, resumed_from, elapsed_prev, tokens_seen = 0, None, 0.0, 0
    rng = np.random.default_rng(tc.seed)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=tc.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = resumed_from = ck["step"]
        elapsed_prev = ck.get("elapsed_sec", 0.0)
        tokens_seen = ck.get("tokens_seen", 0)
        rng = np.random.default_rng(tc.seed + step)
        log(f"[train] resumed from step {step}")

    train_shard, val_shard = shard_factory(tc.train_bin), shard_factory(tc.val_bin)
    extra = {k: shard_factory(v) for k, v in (tc.extra_val or {}).items()}
    n_params = analytic_param_count(mc)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    old = signal.signal(signal.SIGTERM, _sig)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "model_config": mc.to_dict(), "train_config": asdict(tc),
                    "elapsed_sec": elapsed_prev + (time.time() - t0), "tokens_seen": tokens_seen}, tmp)
        os.replace(tmp, ckpt_path)

    t0 = time.time()
    last_ckpt = time.time()
    t_log, tok_log = time.time(), 0
    model.train()
    try:
        while step < tc.max_steps:
            if stop_after is not None and resumed_from is None and step >= stop_after:
                break
            if stop_after is not None and resumed_from is not None and step >= resumed_from + stop_after:
                break
            if stop["flag"]:
                log("[train] SIGTERM -> checkpoint and exit")
                break
            if (elapsed_prev + time.time() - t0) / 3600.0 >= tc.max_hours:
                log("[train] max_hours reached")
                break
            t_step = time.time()
            lr = lr_at(step, max_lr=tc.max_lr, min_lr=tc.min_lr, warmup=tc.warmup, total=tc.max_steps,
                       decay_frac=tc.decay_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            for _ in range(tc.grad_accum):
                x, y = train_shard.batch(tc.batch_size, tc.seq_len, rng=rng)
                x, y = x.to(tc.device), y.to(tc.device)
                with _autocast(tc.device, tc.dtype):
                    _, loss = model(x, y)
                (loss / tc.grad_accum).backward()
                loss_acc += float(loss) / tc.grad_accum
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ntok = tc.batch_size * tc.seq_len * tc.grad_accum
            tokens_seen += ntok
            tok_log += ntok
            rec = None
            if step % tc.log_every == 0 or step == 1:
                dt = time.time() - t_log
                rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr, "grad_norm": round(float(gn), 3),
                       "tok_per_s": round(tok_log / max(dt, 1e-9)), "tokens_seen": tokens_seen,
                       "elapsed_h": round((elapsed_prev + time.time() - t0) / 3600, 3)}
                t_log, tok_log = time.time(), 0
            if step % tc.eval_every == 0 or step == tc.max_steps:
                rec = rec or {"step": step}
                rec["val_loss"] = round(evaluate(model, val_shard, tc, tc.eval_batches), 4)
                for k, sh in extra.items():
                    vl = evaluate(model, sh, tc, tc.eval_batches)
                    rec[f"{k}_loss"] = round(vl, 4)
                    rec[f"{k}_ppl_tok"] = round(math.exp(min(vl, 20)), 2)
            if rec:
                rec["n_params"] = n_params
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(json.dumps(rec))
            if time.time() - last_ckpt >= tc.ckpt_every_sec:
                save()
                last_ckpt = time.time()
            if pace is not None:
                pace(time.time() - t_step)
    finally:
        save()
        signal.signal(signal.SIGTERM, old)
    return {"step": step, "resumed_from": resumed_from, "tokens_seen": tokens_seen}
