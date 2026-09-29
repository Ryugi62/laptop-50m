"""Vocabulary-size ablation under a parameter cap that counts the embedding (small-scale proxy, CPU).

Same text, same parameter budget, same width, same number of training *bytes*; only the vocabulary changes, and the
depth fills whatever the embedding leaves. Compared on held-out bits per byte (tokenizer-independent).

    python -m laptop50m.infrastructure.ablate_vocab --vocabs 4096,8192,16384,32768 --budget 8e6 --out results/ablation_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from laptop50m.adapters.model_torch import GPT
from laptop50m.application.perplexity import sliding_window_nll
from laptop50m.domain.config import analytic_param_count, bits_per_byte, layers_for_budget
from laptop50m.domain.schedule import lr_at
from laptop50m.infrastructure.prepare_data import EOT, train_tokenizer


def take_texts(pf, groups, max_bytes):
    out, n = [], 0
    for g in groups:
        for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
            out.append(t)
            n += len(t.encode("utf-8"))
            if n >= max_bytes:
                return out, n
    return out, n


def encode(tok, texts):
    eot = tok.token_to_id(EOT)
    ids = []
    for e in tok.encode_batch(texts, add_special_tokens=False):
        ids.extend(e.ids)
        ids.append(eot)
    return np.asarray(ids, dtype=np.int64)


def run_one(vocab, train_texts, held_texts, held_bytes, a, work):
    tok_path = os.path.join(work, f"tok{vocab}.json")
    t0 = time.time()
    tok = train_tokenizer(iter(train_texts), vocab, tok_path)
    tr, ho = encode(tok, train_texts), encode(tok, held_texts)
    t_tok = time.time() - t0
    base = layers_for_budget(vocab, a.d_model, a.n_head, a.ffn, int(a.budget))
    cfg = type(base)(**{**base.to_dict(), "max_seq_len": a.seq})
    torch.manual_seed(a.seed)
    model = GPT(cfg)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    tokens_per_step = a.batch * a.seq
    n_chunks = (len(tr) - 1) // a.seq
    steps = n_chunks // a.batch
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(n_chunks)[: steps * a.batch]
    warmup = max(1, steps // 20)
    t1 = time.time()
    losses = []
    model.train()
    for s in range(steps):
        lr = lr_at(s, max_lr=a.lr, min_lr=a.lr / 10, warmup=warmup, total=steps, decay_frac=0.2)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = order[s * a.batch:(s + 1) * a.batch]
        xb = np.stack([tr[i * a.seq:(i + 1) * a.seq + 1] for i in idx])
        x, y = torch.from_numpy(xb[:, :-1]), torch.from_numpy(xb[:, 1:])
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if s % 100 == 0:
            print(f"[ablate v={vocab}] step {s}/{steps} loss {float(loss):.3f} {time.time() - t1:.0f}s", flush=True)
    t_train = time.time() - t1
    model.eval()
    nll, n_tok = sliding_window_nll(model, ho, window=a.seq, stride=a.seq // 2, batch_size=16)
    bpb = bits_per_byte(nll, held_bytes)
    n = analytic_param_count(cfg)
    return {"vocab": vocab, "seed": a.seed, "n_layer": cfg.n_layer, "params": n, "embedding_share": round(vocab * a.d_model / n, 3),
            "train_bytes_per_token": round(sum(len(t.encode()) + 1 for t in train_texts) / len(tr), 3),
            "train_tokens": int(steps * tokens_per_step), "steps": steps,
            "final_train_loss_last50": round(float(np.mean(losses[-50:])), 4),
            "heldout_tokens": n_tok, "heldout_bits_per_byte": round(bpb, 4),
            "tokenize_sec": round(t_tok, 1), "train_sec": round(t_train, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="_data/raw/fineweb_edu_000.parquet")
    ap.add_argument("--vocabs", default="4096,8192,16384,32768")
    ap.add_argument("--budget", type=float, default=8e6)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-bytes", type=float, default=20e6)
    ap.add_argument("--heldout-bytes", type=float, default=1e6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="model init + batch order (text and tokenizers are fixed)")
    ap.add_argument("--work", default="_data/ablation")
    ap.add_argument("--out", default="results/ablation_vocab.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.work, exist_ok=True)
    pf = pq.ParquetFile(a.parquet)
    last = pf.num_row_groups - 1
    train_texts, train_bytes = take_texts(pf, range(0, last), a.train_bytes)
    held_texts, _ = take_texts(pf, [last], a.heldout_bytes)
    held_bytes = sum(len(t.encode("utf-8")) + 1 for t in held_texts)  # +1 = end-of-text separator
    rows = []
    for v in [int(x) for x in a.vocabs.split(",")]:
        rows.append(run_one(v, train_texts, held_texts, held_bytes, a, a.work))
        print(json.dumps(rows[-1]), flush=True)
        json.dump({"setup": {k: v for k, v in vars(a).items()}, "train_text_bytes": train_bytes,
                   "heldout_bytes": held_bytes, "heldout_docs": len(held_texts), "runs": rows,
                   "device": "cpu", "torch": torch.__version__}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
