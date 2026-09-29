import torch

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig, analytic_param_count
from count_params import count_checkpoint_params


def small():
    return ModelConfig(vocab_size=256, d_model=64, n_layer=2, n_head=4, ffn_hidden=176, max_seq_len=32)


def test_ac23_bare_state_dict_counts_tied_once(tmp_path):
    cfg = small()
    p = tmp_path / "w.pt"
    torch.save(GPT(cfg).state_dict(), p)
    r = count_checkpoint_params(str(p))
    assert r["unique"] == analytic_param_count(cfg)
    assert r["naive"] == analytic_param_count(cfg) + cfg.vocab_size * cfg.d_model
    assert ("tok_emb.weight", "lm_head.weight") in r["shared"]


def test_ac23_training_checkpoint_with_config(tmp_path):
    cfg = small()
    p = tmp_path / "ck.pt"
    torch.save({"model": GPT(cfg).state_dict(), "model_config": cfg.to_dict(), "step": 3}, p)
    r = count_checkpoint_params(str(p))
    assert r["unique"] == analytic_param_count(cfg) == r["analytic"]
