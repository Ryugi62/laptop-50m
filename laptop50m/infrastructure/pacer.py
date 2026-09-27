"""Duty-cycle pacer: full speed when the user is idle, ~25% GPU duty when the user is active."""
from __future__ import annotations

import time
from typing import Callable, Optional


class IdlePacer:
    def __init__(self, idle_fn: Callable[[], Optional[int]], threshold: int = 120, active_ratio: float = 3.0,
                 check_every: float = 20.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, on_active: Optional[Callable] = None):
        self.idle_fn, self.threshold, self.ratio = idle_fn, threshold, active_ratio
        self.check_every, self.sleep, self.clock, self.on_active = check_every, sleep, clock, on_active
        self._last_check = None
        self._active = False
        self.slept = 0.0

    def active(self) -> bool:
        now = self.clock()
        if self._last_check is None or now - self._last_check >= self.check_every:
            s = self.idle_fn()
            self._active = s is not None and s < self.threshold
            self._last_check = now
        return self._active

    def __call__(self, step_seconds: float) -> None:
        if self.active():
            if self.on_active:
                self.on_active()
            d = step_seconds * self.ratio
            self.slept += d
            self.sleep(d)
