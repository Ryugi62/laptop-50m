from laptop50m.infrastructure.pacer import IdlePacer


def test_pacer_sleeps_only_when_user_active():
    slept = []
    idle = {"v": 10}
    clock = {"t": 0.0}
    p = IdlePacer(lambda: idle["v"], threshold=120, active_ratio=3.0, check_every=0,
                  sleep=slept.append, clock=lambda: clock["t"])
    p(0.5)
    assert slept == [1.5]
    idle["v"] = 300
    p(0.5)
    assert slept == [1.5]


def test_pacer_caches_idle_checks():
    calls = []
    clock = {"t": 0.0}
    p = IdlePacer(lambda: calls.append(1) or 500, check_every=20, sleep=lambda d: None, clock=lambda: clock["t"])
    p(0.1); p(0.1)
    clock["t"] = 25
    p(0.1)
    assert len(calls) == 2
