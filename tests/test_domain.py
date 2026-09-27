import pytest
from laptop50m.domain.config import ModelConfig, PARAM_BUDGET, analytic_param_count, check_budget
from laptop50m.domain.schedule import lr_at


def test_ac1_default_config_within_budget():
    assert analytic_param_count(ModelConfig()) <= PARAM_BUDGET


def test_ac2_over_budget_raises():
    with pytest.raises(ValueError):
        check_budget(ModelConfig(d_model=1024, n_layer=24))


def test_ac6_wsd_schedule():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup=10, total=100, decay_frac=0.2)
    assert lr_at(0, **kw) == pytest.approx(1e-4)  # (0+1)/10 * max
    assert lr_at(9, **kw) == pytest.approx(1e-3)
    assert lr_at(50, **kw) == pytest.approx(1e-3)
    assert lr_at(79, **kw) == pytest.approx(1e-3)
    assert lr_at(100, **kw) == pytest.approx(1e-4)
    mid = lr_at(90, **kw)
    assert 1e-4 < mid < 1e-3
