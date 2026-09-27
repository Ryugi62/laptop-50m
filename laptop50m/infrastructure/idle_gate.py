"""Wait until the user has been idle (HIDIdleTime >= threshold seconds) before a heavy stage."""
from __future__ import annotations

import re
import subprocess
import sys
import time

_PAT = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


def parse_hid_idle_seconds(text: str):
    m = _PAT.search(text)
    return int(m.group(1)) // 1_000_000_000 if m else None


def hid_idle_seconds():
    out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True).stdout
    return parse_hid_idle_seconds(out)


def wait_until_idle(threshold: int = 120, poll: int = 30, log=print) -> int:
    while True:
        s = hid_idle_seconds()
        if s is None or s >= threshold:
            log(f"[idle_gate] idle={s}s >= {threshold}s -> start")
            return s or 0
        log(f"[idle_gate] idle={s}s < {threshold}s -> hold")
        time.sleep(poll)


if __name__ == "__main__":
    wait_until_idle(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
