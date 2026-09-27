"""Evaluate a Laptop-50M checkpoint: lm-evaluation-harness (0-shot HellaSwag, ARC-Easy, PIQA, WinoGrande,
WikiText-103 word perplexity) + token-level sliding-window perplexity on WikiText-103 / FineWeb-Edu held-out.

    python -m laptop50m.infrastructure.eval_cli --ckpt runs/l50m-v1/ckpt.pt --out results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from laptop50m.adapters.lm_eval_adapter import Laptop50MLM
from laptop50m.adapters.model_torch import GPT
from laptop50m.application.perplexity import sliding_window_nll
from laptop50m.domain.config import ModelConfig

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]
HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "eval_tasks"))


def load_model(ckpt_path: str, device: str, seq_len: int):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    cfg = dict(ck.get("model_config", {}))
    cfg["max_seq_len"] = seq_len
    model = GPT(ModelConfig(**cfg))
    model.load_state_dict(sd)
    return model.to(device).eval(), ck.get("step"), ck.get("tokens_seen")


def token_ppl(model, bin_path: str, window: int, stride: int, device: str, bs: int = 16) -> dict:
    ids = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
    nll, n = sliding_window_nll(model, ids, window=window, stride=stride, device=device, batch_size=bs)
    return {"tokens": n, "nll_per_token": nll / n, "ppl_per_token": math.exp(nll / n),
            "window": window, "stride": stride}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/l50m-v1/ckpt.pt")
    ap.add_argument("--tokenizer", default="_data/tok/tokenizer.json")
    ap.add_argument("--tok-dir", default="_data/tok")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--limit", type=float, default=None, help="subset per task (debug only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model, step, tokens_seen = load_model(a.ckpt, a.device, a.seq_len)
    tok = Tokenizer.from_file(a.tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[eval] ckpt step={step} tokens_seen={tokens_seen} params={n_params:,} device={a.device}", flush=True)

    ppl = {}
    for name in ("wikitext103_test", "wikitext103_validation"):
        p = os.path.join(a.tok_dir, f"{name}.bin")
        if os.path.exists(p):
            ppl[name] = token_ppl(model, p, window=a.seq_len, stride=a.seq_len // 2, device=a.device, bs=a.batch_size)
            print(f"[eval] token ppl {name}: {ppl[name]}", flush=True)

    import lm_eval
    from lm_eval.tasks import TaskManager
    lm = Laptop50MLM(model=model, tokenizer=tok, device=a.device, batch_size=a.batch_size, max_length=a.seq_len)
    res = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","), num_fewshot=0, limit=a.limit,
                                  task_manager=TaskManager(include_path=TASK_DIR), log_samples=False,
                                  bootstrap_iters=1000)
    summary = {
        "checkpoint": {"path": a.ckpt, "step": step, "tokens_seen": tokens_seen, "params": n_params},
        "lm_eval": {"version": lm_eval.__version__ if hasattr(lm_eval, "__version__") else None,
                    "num_fewshot": 0, "limit": a.limit, "results": res["results"],
                    "n_samples": res.get("n-samples"), "versions": res.get("versions")},
        "token_ppl": ppl,
        "env": {"device": a.device, "torch": torch.__version__, "python": platform.python_version(),
                "machine": platform.machine(), "eval_wall_s": round(time.time() - t0, 1)},
    }
    out = os.path.join(a.out, "eval_l50m-v1.json")
    json.dump(summary, open(out, "w"), indent=1, default=str)
    print(f"[eval] wrote {out} in {summary['env']['eval_wall_s']} s", flush=True)
    for k, v in res["results"].items():
        print(k, {m: v[m] for m in v if not m.endswith("_stderr,none") and m != "alias"}, flush=True)


if __name__ == "__main__":
    sys.exit(main())
