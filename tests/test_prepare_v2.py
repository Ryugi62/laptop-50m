import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from laptop50m.infrastructure.prepare_v2 import main

WIKI = "the quick brown fox jumps over the lazy dog near the old river bank today said the reporter again"


def test_prepare_v2_decontam_and_hq(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    for split in ("validation", "test"):
        pq.write_table(pa.table({"text": [WIKI]}), raw / f"wikitext103_{split}.parquet")
    docs = ["Physics lesson about energy and motion in simple words for students.",
            "Copied: The quick brown fox jumps over the lazy dog near the old river bank today said the reporter.",
            "Chemistry notes on atoms and bonds explained step by step for beginners."]
    for fid in ("000", "002"):
        pq.write_table(pa.table({"text": docs * 2, "int_score": [3, 4, 5] * 2}), raw / f"fineweb_edu_{fid}.parquet",
                       row_group_size=3)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(docs, trainer=trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.save(str(tmp_path / "tok.json"))
    man = main(["--raw", str(raw), "--out", str(out), "--tokenizer", str(tmp_path / "tok.json"),
                "--main", "000", "--hq", "002", "--val-groups", "1", "--workers", "1"])
    assert man["train_000"]["row_groups"] == 1  # last group held out
    assert man["train_000"]["dropped_ngram13"] == 1 and man["train_000"]["docs"] == 2
    # hq file: both row groups used; per group docs scored 3 (skip), 4 (contaminated, drop), 5 (keep)
    assert man["hq_002"]["docs"] == 2 and man["hq_002"]["skipped_low_score"] == 2
    assert man["hq_002"]["dropped_ngram13"] == 2
    arr = np.fromfile(out / "train_000.bin", dtype=np.uint16)
    assert (arr == 0).sum() == 2  # two EOTs
    assert json.load(open(out / "manifest.json"))["hq_002"]["tokens"] > 0
