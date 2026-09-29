import torch

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig
from laptop50m.infrastructure.generate_cli import sample


def small():
    return ModelConfig(vocab_size=256, d_model=64, n_layer=2, n_head=4, ffn_hidden=176, max_seq_len=32)


def test_ac24_sample_is_seeded_and_bounded():
    m = GPT(small()).eval()
    a = sample(m, [1, 2, 3], new_tokens=40, temperature=0.7, top_k=10, seed=0)
    b = sample(m, [1, 2, 3], new_tokens=40, temperature=0.7, top_k=10, seed=0)
    assert a == b and len(a) == 43 and a[:3] == [1, 2, 3]
    assert all(0 <= t < 256 for t in a)


def test_ac24_greedy_matches_argmax():
    m = GPT(small()).eval()
    out = sample(m, [5, 6], new_tokens=1, temperature=0.0, top_k=0, seed=0)
    with torch.no_grad():
        logits, _ = m(torch.tensor([[5, 6]]))
    assert out[-1] == int(logits[0, -1].argmax())
