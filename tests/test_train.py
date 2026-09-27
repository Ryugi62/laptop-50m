import numpy as np
import torch
from laptop50m.domain.config import ModelConfig
from laptop50m.application.train import TrainConfig, train
from laptop50m.adapters.model_torch import GPT
from laptop50m.adapters.data import TokenShard


def test_ac9_resume_continues_from_checkpoint(tmp_path):
    p = tmp_path / "t.bin"
    (np.arange(5000) % 200).astype(np.uint16).tofile(p)
    mc = ModelConfig(vocab_size=256, d_model=32, n_layer=1, n_head=2, ffn_hidden=64, max_seq_len=16)
    tc = TrainConfig(train_bin=str(p), val_bin=str(p), out_dir=str(tmp_path / "run"), batch_size=2, seq_len=16,
                     max_steps=6, warmup=2, eval_every=3, eval_batches=1, ckpt_every_sec=0, device="cpu",
                     max_lr=1e-3, log_every=1, dtype="fp32")
    s1 = train(mc, tc, GPT, TokenShard, stop_after=3)
    assert s1["step"] == 3
    s2 = train(mc, tc, GPT, TokenShard)
    assert s2["resumed_from"] == 3
    assert s2["step"] == 6
    ck = torch.load(tmp_path / "run" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and "optimizer" in ck
