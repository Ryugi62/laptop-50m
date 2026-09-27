import torch
from laptop50m.domain.config import ModelConfig, analytic_param_count
from laptop50m.adapters.model_torch import GPT


def small():
    return ModelConfig(vocab_size=256, d_model=64, n_layer=2, n_head=4, ffn_hidden=176, max_seq_len=32)


def test_ac3_param_count_matches_analytic():
    cfg = small()
    m = GPT(cfg)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == analytic_param_count(cfg)


def test_ac3_default_config_count_matches():
    cfg = ModelConfig()
    m = GPT(cfg)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == analytic_param_count(cfg)


def test_ac4_tied_embeddings():
    m = GPT(small())
    assert m.lm_head.weight is m.tok_emb.weight


def test_ac5_forward_shapes_and_loss():
    cfg = small()
    m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
