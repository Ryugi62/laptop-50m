import numpy as np
import pytest
import torch

from laptop50m.adapters.data import MultiShard
from laptop50m.adapters.model_torch import GPT
from laptop50m.adapters.muon import Muon, newton_schulz, split_params
from laptop50m.application.train import TrainConfig, train
from laptop50m.domain.config import ModelConfig, analytic_param_count
from laptop50m.domain.decontam import NgramIndex


def tiny(**kw):
    return ModelConfig(vocab_size=256, d_model=32, n_layer=2, n_head=2, ffn_hidden=64, max_seq_len=16, **kw)


def test_ac16_qk_norm_no_params_and_finite():
    cfg = tiny(qk_norm=True)
    m = GPT(cfg)
    assert sum(p.numel() for p in m.parameters()) == analytic_param_count(cfg)
    x = torch.randint(0, 256, (2, 16))
    _, loss = m(x, x)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (3, 32, 32)])
def test_ac17_newton_schulz_orthogonalises(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    X = newton_schulz(G, steps=5).float()
    mats = X if X.dim() == 3 else X.unsqueeze(0)
    for M in mats:
        s = torch.linalg.svdvals(M)
        assert float(((s > 0.5) & (s < 1.5)).float().mean()) >= 0.9 and s.max() < 1.5


def test_ac18_hybrid_muon_adamw_trains():
    torch.manual_seed(0)
    cfg = tiny()
    m = GPT(cfg)
    muon_p, adam_p = split_params(m)
    assert all(p.dim() >= 2 for p in muon_p)
    assert m.tok_emb.weight in adam_p and all(p is not m.tok_emb.weight for p in muon_p)
    assert len(muon_p) + len(adam_p) == len(list(m.parameters()))
    opt1 = Muon(muon_p, lr=0.02)
    opt2 = torch.optim.AdamW(adam_p, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    x = torch.randint(0, 256, (8, 16))
    first = None
    for _ in range(50):
        _, loss = m(x, x.roll(-1, dims=1))
        first = first if first is not None else float(loss.detach())
        loss.backward()
        opt1.step(); opt2.step(); opt1.zero_grad(); opt2.zero_grad()
    assert float(loss) < 0.5 * first


def test_ac19_multishard_proportional(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    np.full(1000, 1, dtype=np.uint16).tofile(a)
    np.full(3000, 2, dtype=np.uint16).tofile(b)
    ms = MultiShard([str(a), str(b)])
    rng = np.random.default_rng(0)
    x, y = ms.batch(4000, 8, rng=rng)
    frac_a = float((x[:, 0] == 1).float().mean())
    assert abs(frac_a - 0.25) < 0.05
    c = tmp_path / "c.bin"
    np.arange(500, dtype=np.uint16).tofile(c)
    x2, y2 = MultiShard([str(c)]).batch(4, 8, rng=rng)
    assert (y2[:, :-1] == x2[:, 1:]).all()


def test_ac20_decontam_ngram():
    ref = "The quick brown fox jumps over the lazy dog near the old river bank today , said the reporter"
    idx = NgramIndex([ref], n=13)
    assert idx.is_contaminated("Blah blah. THE QUICK brown fox jumps over the lazy-dog near the old river bank TODAY!")
    assert not idx.is_contaminated("A completely different sentence about physics and chemistry lessons at school today.")


def _bins(tmp_path):
    p = tmp_path / "t.bin"
    (np.arange(5000) % 200).astype(np.uint16).tofile(p)
    q = tmp_path / "hq.bin"
    np.full(3000, 250, dtype=np.uint16).tofile(q)
    return p, q


def test_ac21_anneal_fraction(tmp_path):
    from laptop50m.application.train import draw_batch
    p, q = _bins(tmp_path)
    main, hq = MultiShard([str(p)]), MultiShard([str(q)])
    rng = np.random.default_rng(0)
    x, _ = draw_batch(main, hq, 0.5, True, 400, 8, rng)
    assert abs(float((x[:, 0] == 250).float().mean()) - 0.5) < 0.01
    x, _ = draw_batch(main, hq, 0.5, False, 400, 8, rng)
    assert float((x[:, 0] == 250).float().mean()) == 0.0


def test_ac22_muon_resume(tmp_path):
    p, q = _bins(tmp_path)
    mc = tiny(qk_norm=True)
    tc = TrainConfig(train_bin=str(p), val_bin=str(p), out_dir=str(tmp_path / "run"), batch_size=2, seq_len=16,
                     max_steps=6, warmup=2, eval_every=3, eval_batches=1, ckpt_every_sec=0, device="cpu",
                     max_lr=3e-3, log_every=1, dtype="fp32", optimizer="muon", muon_lr=0.02,
                     anneal_bins=[str(q)], anneal_frac=0.5, decay_frac=0.5)
    s1 = train(mc, tc, GPT, MultiShard.from_path, stop_after=3)
    assert s1["step"] == 3
    s2 = train(mc, tc, GPT, MultiShard.from_path)
    assert s2["resumed_from"] == 3 and s2["step"] == 6
    ck = torch.load(tmp_path / "run" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert "optimizer" in ck and "optimizer_muon" in ck


def test_train_cloud_plan():
    from laptop50m.infrastructure.train_cloud import plan
    p = plan(1.2e9, 262_144, 16, 1024, 1)
    assert p == {"accum": 16, "global_batch": 262_144, "max_steps": 4577}
    p2 = plan(1.2e9, 262_144, 16, 1024, 2)
    assert p2["accum"] == 8 and p2["global_batch"] == 262_144


def test_train_cloud_cpu_smoke(tmp_path):
    from laptop50m.infrastructure.train_cloud import main
    d, v = tmp_path / "d", tmp_path / "v"
    d.mkdir(); v.mkdir()
    (np.arange(20000) % 16000).astype(np.uint16).tofile(d / "train_000.bin")
    np.full(5000, 7, dtype=np.uint16).tofile(d / "hq_002.bin")
    for n in ("fineweb_val", "wikitext103_validation"):
        (np.arange(3000) % 300).astype(np.uint16).tofile(v / f"{n}.bin")
    import laptop50m.infrastructure.train_cloud as tcmod
    orig = tcmod.ModelConfig
    tcmod.ModelConfig = lambda **kw: orig(vocab_size=16384, d_model=32, n_layer=1, n_head=2, ffn_hidden=64, **kw)
    try:
        res = main(["--data", str(d), "--val-dir", str(v), "--out", str(tmp_path / "o"), "--tokens", "4096",
                    "--global-batch", "1024", "--micro", "2", "--seq", "64", "--dtype", "fp32"])
    finally:
        tcmod.ModelConfig = orig
    assert res["step"] == 4
    assert (tmp_path / "o" / "ckpt.pt").exists()
