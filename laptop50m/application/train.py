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
    optimizer: str = "adamw"  # "adamw" | "muon" (Muon on block matrices + AdamW on the rest)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    anneal_bins: Optional[list] = None  # high-quality shards mixed in during the decay phase
    anneal_frac: float = 0.0
    compile: bool = False


def _autocast(device: str, dtype: str):
    device = device.split(":")[0]
    if dtype == "fp32" or device == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device, dtype=dt)


def draw_batch(main, hq, frac: float, in_decay: bool, B: int, T: int, rng):
    """B rows; round(B*frac) of them from `hq` when in the decay phase (anneal mix)."""
    k = int(round(B * frac)) if (in_decay and hq is not None and frac > 0) else 0
    if k == 0:
        return main.batch(B, T, rng=rng)
    if k >= B:
        return hq.batch(B, T, rng=rng)
    x1, y1 = main.batch(B - k, T, rng=rng)
    x2, y2 = hq.batch(k, T, rng=rng)
    return torch.cat([x1, x2]), torch.cat([y1, y2])


def _optimizers(model, tc: TrainConfig):
    """List of optimizers. Each param group carries base_lr; the schedule multiplies it."""
    if tc.optimizer == "muon":
        from laptop50m.adapters.muon import Muon, split_params
        muon_p, adam_p = split_params(model)
        o1 = Muon(muon_p, lr=tc.muon_lr, momentum=tc.muon_momentum)
        o2 = torch.optim.AdamW(adam_p, lr=tc.max_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        for g in o1.param_groups:
            g["base_lr"] = tc.muon_lr
        for g in o2.param_groups:
            g["base_lr"] = tc.max_lr
        return [o2, o1]
    o = _optimizer(model, tc)
    for g in o.param_groups:
        g["base_lr"] = tc.max_lr
    return [o]


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


def _dist():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 1, 0, 0
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return ws, dist.get_rank(), int(os.environ.get("LOCAL_RANK", "0"))


def train(mc: ModelConfig, tc: TrainConfig, model_factory: Callable, shard_factory: Callable,
          stop_after: Optional[int] = None, log: Callable = print, pace: Optional[Callable] = None) -> dict:
    world, rank, local_rank = _dist()
    if world > 1:
        tc.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    master = rank == 0
    dev_type = tc.device.split(":")[0]
    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    log_path = os.path.join(tc.out_dir, "train_log.jsonl")
    torch.manual_seed(tc.seed)
    raw = model_factory(mc).to(tc.device)
    opts = _optimizers(raw, tc)
    scaler = torch.amp.GradScaler("cuda") if (tc.dtype == "fp16" and dev_type == "cuda") else None
    step, resumed_from, elapsed_prev, tokens_seen = 0, None, 0.0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=tc.device, weights_only=False)
        raw.load_state_dict(ck["model"])
        opts[0].load_state_dict(ck["optimizer"])
        if len(opts) > 1 and "optimizer_muon" in ck:
            opts[1].load_state_dict(ck["optimizer_muon"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step = resumed_from = ck["step"]
        elapsed_prev = ck.get("elapsed_sec", 0.0)
        tokens_seen = ck.get("tokens_seen", 0)
        if master:
            log(f"[train] resumed from step {step}")
    rng = np.random.default_rng(tc.seed + step + 100_003 * rank)
    model = raw
    if tc.compile:
        model = torch.compile(model)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    train_shard, val_shard = shard_factory(tc.train_bin), shard_factory(tc.val_bin)
    hq_shard = shard_factory(tc.anneal_bins) if tc.anneal_bins else None
    extra = {k: shard_factory(v) for k, v in (tc.extra_val or {}).items()}
    n_params = analytic_param_count(mc)
    stop = {"flag": False}
    decay_start = int(tc.max_steps * (1.0 - tc.decay_frac))

    def _sig(*_):
        stop["flag"] = True
    old = signal.signal(signal.SIGTERM, _sig)

    def save():
        if not master:
            return
        tmp = ckpt_path + ".tmp"
        state = {"model": raw.state_dict(), "optimizer": opts[0].state_dict(), "step": step,
                 "model_config": mc.to_dict(), "train_config": asdict(tc),
                 "elapsed_sec": elapsed_prev + (time.time() - t0), "tokens_seen": tokens_seen}
        if len(opts) > 1:
            state["optimizer_muon"] = opts[1].state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)

    def _should_stop(local: bool) -> bool:
        if world <= 1:
            return local
        import torch.distributed as dist
        t = torch.tensor([1.0 if local else 0.0], device=tc.device)
        dist.all_reduce(t)
        return bool(t.item() > 0)

    t0 = time.time()
    last_ckpt = time.time()
    t_log, tok_log = time.time(), 0
    nb = dev_type == "cuda"
    model.train()
    try:
        while step < tc.max_steps:
            local_stop = False
            if stop_after is not None and step >= (resumed_from or 0) + stop_after and \
                    (resumed_from is not None or step >= stop_after):
                break
            if stop["flag"]:
                local_stop = True
            if (elapsed_prev + time.time() - t0) / 3600.0 >= tc.max_hours:
                local_stop = True
            if _should_stop(local_stop):
                if master:
                    log("[train] stop (SIGTERM or max_hours) -> checkpoint and exit")
                break
            t_step = time.time()
            ratio = lr_at(step, max_lr=tc.max_lr, min_lr=tc.min_lr, warmup=tc.warmup, total=tc.max_steps,
                          decay_frac=tc.decay_frac) / tc.max_lr
            for o in opts:
                for g in o.param_groups:
                    g["lr"] = g["base_lr"] * ratio
            lr = tc.max_lr * ratio
            in_decay = step >= decay_start
            loss_acc = 0.0
            for micro in range(tc.grad_accum):
                x, y = draw_batch(train_shard, hq_shard, tc.anneal_frac, in_decay, tc.batch_size, tc.seq_len, rng)
                x, y = x.to(tc.device, non_blocking=nb), y.to(tc.device, non_blocking=nb)
                sync_ctx = model.no_sync() if (world > 1 and micro < tc.grad_accum - 1) else _null()
                with sync_ctx:
                    with _autocast(dev_type, tc.dtype):
                        _, loss = model(x, y)
                    l = loss / tc.grad_accum
                    (scaler.scale(l) if scaler is not None else l).backward()
                loss_acc += float(loss.detach()) / tc.grad_accum
            if scaler is not None:
                for o in opts:
                    scaler.unscale_(o)
            gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.grad_clip)
            for o in opts:
                if scaler is not None:
                    scaler.step(o)
                else:
                    o.step()
            if scaler is not None:
                scaler.update()
            for o in opts:
                o.zero_grad(set_to_none=True)
            step += 1
            ntok = tc.batch_size * tc.seq_len * tc.grad_accum * world
            tokens_seen += ntok
            tok_log += ntok
            rec = None
            if master and (step % tc.log_every == 0 or step == 1):
                dt = time.time() - t_log
                rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr, "grad_norm": round(float(gn), 3),
                       "tok_per_s": round(tok_log / max(dt, 1e-9)), "tokens_seen": tokens_seen,
                       "elapsed_h": round((elapsed_prev + time.time() - t0) / 3600, 3), "anneal": in_decay}
                t_log, tok_log = time.time(), 0
            if master and (step % tc.eval_every == 0 or step == tc.max_steps):
                rec = rec or {"step": step}
                tc_eval = tc
                rec["val_loss"] = round(evaluate(raw, val_shard, tc_eval, tc.eval_batches), 4)
                for k, sh in extra.items():
                    vl = evaluate(raw, sh, tc_eval, tc.eval_batches)
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


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
