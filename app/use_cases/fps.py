from __future__ import annotations

from collections.abc import Callable
from time import perf_counter


class PerfCounterFpsMeter:
    def __init__(self, timer: Callable[[], float] = perf_counter) -> None:
        self._timer = timer
        self._previous_time: float | None = None

    def tick(self) -> float:
        current_time = self._timer()

        if self._previous_time is None:
            self._previous_time = current_time
            return 0.0

        elapsed = max(current_time - self._previous_time, 1e-6)
        self._previous_time = current_time
        return 1.0 / elapsed

    def reset(self) -> None:
        self._previous_time = None

