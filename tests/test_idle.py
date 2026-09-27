from laptop50m.infrastructure.idle_gate import parse_hid_idle_seconds

SAMPLE = '''    | |   "HIDIdleTime" = 125000000000
    | |   "HIDIdleTime" = 999
'''


def test_ac8_parse_idle():
    assert parse_hid_idle_seconds(SAMPLE) == 125
    assert parse_hid_idle_seconds("nothing") is None
