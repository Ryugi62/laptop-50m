import numpy as np
from laptop50m.adapters.data import TokenShard


def test_ac7_batches_are_shifted(tmp_path):
    p = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(p)
    sh = TokenShard(str(p))
    x, y = sh.batch(4, 16, seed=1)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert str(x.dtype) == "torch.int64"
    assert (y[:, :-1] == x[:, 1:]).all()
    assert (y[:, -1] == x[:, -1] + 1).all()
