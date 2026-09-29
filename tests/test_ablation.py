import math

import pytest

from laptop50m.domain.config import ModelConfig, analytic_param_count, bits_per_byte, layers_for_budget


def test_ac18_layers_fill_budget_and_bigger_vocab_means_fewer_layers():
    budget = 8_000_000
    cfgs = [layers_for_budget(v, 192, 4, 512, budget) for v in (4096, 16384, 32768)]
    for c in cfgs:
        assert analytic_param_count(c) <= budget
        deeper = ModelConfig(c.vocab_size, c.d_model, c.n_layer + 1, c.n_head, c.ffn_hidden)
        assert analytic_param_count(deeper) > budget
    assert cfgs[0].n_layer > cfgs[1].n_layer > cfgs[2].n_layer >= 1


def test_ac18_vocab_that_eats_the_budget_is_rejected():
    with pytest.raises(ValueError):
        layers_for_budget(65536, 192, 4, 512, 8_000_000)


def test_ac18_bits_per_byte():
    assert bits_per_byte(100 * math.log(2), 100) == pytest.approx(1.0)
