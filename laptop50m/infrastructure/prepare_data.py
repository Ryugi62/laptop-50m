"""Train a 16k byte-level BPE tokenizer and tokenize FineWeb-Edu / WikiText-103 into uint16 shards.

Heavy stages wait for the idle gate (HIDIdleTime >= 120 s). Run under `nice -n 20`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from laptop50m.infrastructure.idle_gate import wait_until_idle

EOT = "<|endoftext|>"


def log(msg):
    print(f"[prepare {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_texts(pf: pq.ParquetFile, groups):
    for g in groups:
        for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
            yield t


def train_tokenizer(texts, vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=tr)
    tok.save(out_path)
    return tok


def encode_to_bin(tok: Tokenizer, texts, out_path: str, batch: int = 2048) -> int:
    eot = tok.token_to_id(EOT)
    n = 0
    buf = []
    with open(out_path + ".tmp", "wb") as f:
        for t in texts:
            buf.append(t)
            if len(buf) >= batch:
                n += _flush(tok, buf, eot, f)
                buf = []
        if buf:
            n += _flush(tok, buf, eot, f)
    os.replace(out_path + ".tmp", out_path)
    return n


def _flush(tok, buf, eot, f) -> int:
    encs = tok.encode_batch(buf, add_special_tokens=False)
    arr = []
    for e in encs:
        arr.extend(e.ids)
        arr.append(eot)
    a = np.asarray(arr, dtype=np.uint16)
    a.tofile(f)
    return len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="_data/raw")
    ap.add_argument("--out", default="_data/tok")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tok-groups", type=int, default=60, help="row groups used to train the tokenizer")
    ap.add_argument("--train-groups", type=int, default=400, help="row groups tokenized for training")
    ap.add_argument("--val-groups", type=int, default=4, help="last row groups held out for validation")
    ap.add_argument("--idle", type=int, default=120)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pf = pq.ParquetFile(os.path.join(a.raw, "fineweb_edu_000.parquet"))
    G = pf.num_row_groups
    tok_path = os.path.join(a.out, "tokenizer.json")
    if not os.path.exists(tok_path):
        wait_until_idle(a.idle, log=log)
        log(f"tokenizer training on {a.tok_groups} row groups")
        train_tokenizer(iter_texts(pf, range(a.tok_groups)), a.vocab, tok_path)
    tok = Tokenizer.from_file(tok_path)
    log(f"tokenizer vocab={tok.get_vocab_size()}")
    stats = {}
    val_bin = os.path.join(a.out, "fineweb_val.bin")
    if not os.path.exists(val_bin):
        stats["fineweb_val"] = encode_to_bin(tok, iter_texts(pf, range(G - a.val_groups, G)), val_bin)
    for split in ("validation", "test"):
        p = os.path.join(a.out, f"wikitext103_{split}.bin")
        if not os.path.exists(p):
            texts = pq.read_table(os.path.join(a.raw, f"wikitext103_{split}.parquet")).column("text").to_pylist()
            stats[f"wikitext103_{split}"] = encode_to_bin(tok, ["".join(texts)], p)
    train_bin = os.path.join(a.out, "fineweb_train.bin")
    if not os.path.exists(train_bin):
        wait_until_idle(a.idle, log=log)
        log(f"tokenizing {a.train_groups} row groups for training")
        stats["fineweb_train"] = encode_to_bin(tok, iter_texts(pf, range(a.train_groups)), train_bin)
    stats.update({k: os.path.getsize(os.path.join(a.out, k + ".bin")) // 2 for k in
                  ("fineweb_train", "fineweb_val", "wikitext103_validation", "wikitext103_test")})
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    log(f"done {stats}")


if __name__ == "__main__":
    sys.exit(main())
