"""v2 data: tokenize FineWeb-Edu parquet files into uint16 shards with WikiText-103 decontamination.

main files  -> train_<id>.bin (all docs, minus the last `val_groups` row groups of file 000 = v1 validation)
hq files    -> hq_<id>.bin    (only docs with int_score >= 4; disjoint files from main)
Every doc sharing a word 13-gram with WikiText-103 validation/test is dropped (counted in manifest.json).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from laptop50m.domain.decontam import NgramIndex

EOT = "<|endoftext|>"
_G = {}


def _init(tok_path, refs):
    _G["tok"] = Tokenizer.from_file(tok_path)
    _G["eot"] = _G["tok"].token_to_id(EOT)
    _G["idx"] = NgramIndex(refs, n=13)


def encode_group(args):
    path, g, min_score = args
    t = pq.ParquetFile(path).read_row_group(g, columns=["text", "int_score"])
    texts, scores = t.column("text").to_pylist(), t.column("int_score").to_pylist()
    keep, dropped, low = [], 0, 0
    for tx, sc in zip(texts, scores):
        if sc is not None and sc < min_score:
            low += 1
            continue
        if _G["idx"].is_contaminated(tx):
            dropped += 1
            continue
        keep.append(tx)
    ids = []
    for e in _G["tok"].encode_batch(keep, add_special_tokens=False):
        ids.extend(e.ids)
        ids.append(_G["eot"])
    return np.asarray(ids, dtype=np.uint16), len(keep), dropped, low


def build(path, out_bin, groups, min_score, pool) -> dict:
    n_tok = n_doc = n_drop = n_low = 0
    with open(out_bin + ".tmp", "wb") as f:
        for arr, k, d, lo in pool.imap(encode_group, [(path, g, min_score) for g in groups], chunksize=2):
            arr.tofile(f)
            n_tok += len(arr); n_doc += k; n_drop += d; n_low += lo
    os.replace(out_bin + ".tmp", out_bin)
    return {"tokens": n_tok, "docs": n_doc, "dropped_ngram13": n_drop, "skipped_low_score": n_low}


def wikitext_refs(raw_dir):
    refs = []
    for split in ("validation", "test"):
        refs.append("".join(pq.read_table(os.path.join(raw_dir, f"wikitext103_{split}.parquet")).column("text").to_pylist()))
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="_data/raw")
    ap.add_argument("--out", default="_data/tok2")
    ap.add_argument("--tokenizer", default="_data/tok/tokenizer.json")
    ap.add_argument("--main", default="000,001")
    ap.add_argument("--hq", default="002,003")
    ap.add_argument("--val-groups", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=0, help="debug: limit row groups per file")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    refs = wikitext_refs(a.raw)
    jobs = [(i, "train", 0) for i in a.main.split(",") if i] + [(i, "hq", 4) for i in a.hq.split(",") if i]
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.tokenizer, refs)) as pool:
        for fid, kind, min_score in jobs:
            out_bin = os.path.join(a.out, f"{kind}_{fid}.bin")
            if os.path.exists(out_bin) and f"{kind}_{fid}" in man:
                continue
            path = os.path.join(a.raw, f"fineweb_edu_{fid}.parquet")
            G = pq.ParquetFile(path).num_row_groups
            groups = list(range(G - a.val_groups if fid == "000" else G))
            if a.max_groups:
                groups = groups[:a.max_groups]
            t0 = time.time()
            st = build(path, out_bin, groups, min_score, pool)
            st["sec"] = round(time.time() - t0, 1)
            st["row_groups"] = len(groups)
            man[f"{kind}_{fid}"] = st
            json.dump(man, open(man_path, "w"), indent=1)
            print(f"[prepare_v2 {time.strftime('%H:%M:%S')}] {kind}_{fid} {st}", flush=True)
    return man


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
