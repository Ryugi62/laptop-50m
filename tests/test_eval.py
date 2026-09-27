"""AC-10..AC-15: parameter-count script, lm-eval adapter, sliding-window perplexity."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from laptop50m.adapters.model_torch import GPT
from laptop50m.domain.config import ModelConfig, PARAM_BUDGET
from laptop50m.infrastructure.prepare_data import train_tokenizer, EOT

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["the quick brown fox jumps over the lazy dog. " * 20, "hello world, small models can reason. " * 30] * 10


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return train_tokenizer(TEXTS, 300, str(tmp_path_factory.mktemp("tok") / "tok.json"))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=320, d_model=32, n_layer=2, n_head=2, ffn_hidden=64, max_seq_len=16))
    return m.eval()


def _lm(model, tok, batch_size=4):
    from laptop50m.adapters.lm_eval_adapter import Laptop50MLM
    return Laptop50MLM(model=model, tokenizer=tok, device="cpu", batch_size=batch_size, max_length=16)


def _manual_logprob(model, ctx, cont):
    full = torch.tensor([ctx + cont])
    with torch.no_grad():
        lp = F.log_softmax(model(full[:, :-1])[0].float(), -1)[0]
    idx = torch.arange(len(ctx) - 1, len(ctx) + len(cont) - 1)
    tgt = torch.tensor(cont)
    return float(lp[idx, tgt].sum()), bool((lp[idx].argmax(-1) == tgt).all())


def test_ac10_count_params_script(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model": ModelConfig().to_dict()}))
    out = subprocess.run([sys.executable, str(ROOT / "count_params.py"), "--config", str(cfg)],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout
    assert "49,295,872" in out and "<= 50,000,000" in out


def test_ac11_loglikelihood_matches_manual(model, tok):
    lm = _lm(model, tok)
    ctx, cont = tok.encode("the quick brown").ids, tok.encode(" fox jumps").ids
    (ll, greedy), = lm._loglikelihood_tokens([(("a", "b"), ctx, cont)])
    m_ll, m_greedy = _manual_logprob(model, ctx, cont)
    assert ll == pytest.approx(m_ll, abs=1e-4) and greedy == m_greedy


def test_ac11_empty_context_uses_eot_prefix(model, tok):
    from lm_eval.api.instance import Instance
    lm = _lm(model, tok)
    req = Instance(request_type="loglikelihood", doc={}, arguments=("", "hello world"), idx=0)
    (ll, _), = lm.loglikelihood([req], disable_tqdm=True)
    cont = tok.encode("hello world").ids
    m_ll, _ = _manual_logprob(model, [tok.token_to_id(EOT)], cont)
    assert ll == pytest.approx(m_ll, abs=1e-4)


def test_ac12_batching_invariance(model, tok):
    reqs = [(("", ""), tok.encode(c).ids, tok.encode(x).ids) for c, x in
            [("the quick", " brown fox"), ("hello", " world, small models"), ("the lazy dog.", " the")]]
    a = _lm(model, tok, batch_size=1)._loglikelihood_tokens(reqs)
    b = _lm(model, tok, batch_size=3)._loglikelihood_tokens(reqs)
    for (la, ga), (lb, gb) in zip(a, b):
        assert la == pytest.approx(lb, abs=1e-4) and ga == gb


def test_ac13_long_context_is_left_truncated(model, tok):
    lm = _lm(model, tok)
    ctx = tok.encode(TEXTS[0]).ids
    assert len(ctx) > 16
    cont = tok.encode(" fox").ids
    (ll, _), = lm._loglikelihood_tokens([(("", ""), ctx, cont)])
    keep = (ctx + cont)[-17:]
    m_ll, _ = _manual_logprob(model, keep[:len(keep) - len(cont)], cont)
    assert math.isfinite(ll) and ll == pytest.approx(m_ll, abs=1e-4)


@pytest.mark.parametrize("text", ["hello world", TEXTS[1][:300]])
def test_ac14_rolling_scores_every_token_once(model, tok, text):
    from lm_eval.api.instance import Instance
    lm = _lm(model, tok)
    req = Instance(request_type="loglikelihood_rolling", doc={}, arguments=(text,), idx=0)
    got, = lm.loglikelihood_rolling([req], disable_tqdm=True)
    ids = [tok.token_to_id(EOT)] + tok.encode(text).ids
    # reference: disjoint windows of 16 predicted tokens, each window conditioned only on tokens inside it
    ref, pos = 0.0, 1
    while pos < len(ids):
        end = min(pos + 16, len(ids))
        start = max(0, end - 17)
        seg = ids[start:end]
        with torch.no_grad():
            lp = F.log_softmax(model(torch.tensor([seg[:-1]]))[0].float(), -1)[0]
        for p in range(pos, end):
            ref += float(lp[p - start - 1, ids[p]])
        pos = end
    assert got == pytest.approx(ref, abs=1e-3)


def test_ac15_sliding_window_perplexity(model):
    from laptop50m.application.perplexity import sliding_window_nll
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 300, size=12)
    nll, n = sliding_window_nll(model, ids, window=16, stride=8, device="cpu")
    with torch.no_grad():
        full = F.cross_entropy(model(torch.tensor([ids[:-1]]))[0][0].float(), torch.tensor(ids[1:]), reduction="sum")
    assert n == 11 and nll == pytest.approx(float(full), abs=1e-3)
    long_ids = rng.integers(0, 300, size=100)
    nll2, n2 = sliding_window_nll(model, long_ids, window=16, stride=8, device="cpu")
    assert n2 == 99 and math.isfinite(nll2)


def test_ac15_sliding_window_batching_invariance(model):
    from laptop50m.application.perplexity import sliding_window_nll
    ids = np.random.default_rng(1).integers(0, 300, size=203)
    a = sliding_window_nll(model, ids, window=16, stride=8, device="cpu", batch_size=1)
    b = sliding_window_nll(model, ids, window=16, stride=8, device="cpu", batch_size=5)
    assert a[1] == b[1] == 202 and a[0] == pytest.approx(b[0], abs=1e-3)
