import numpy as np
from laptop50m.infrastructure.prepare_data import train_tokenizer, encode_to_bin, EOT


def test_tokenizer_roundtrip_and_bin(tmp_path):
    texts = ["the quick brown fox jumps over the lazy dog. " * 20, "hello world " * 50] * 20
    tok = train_tokenizer(texts, 300, str(tmp_path / "tok.json"))
    ids = tok.encode("the quick brown fox").ids
    assert tok.decode(ids) == "the quick brown fox"
    n = encode_to_bin(tok, texts[:3], str(tmp_path / "x.bin"))
    arr = np.fromfile(tmp_path / "x.bin", dtype=np.uint16)
    assert len(arr) == n and (arr == tok.token_to_id(EOT)).sum() == 3
