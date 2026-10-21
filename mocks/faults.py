"""Fault injection for the mock backends (deterministic and seeded-random)."""
from __future__ import annotations

import random
import threading
from collections import defaultdict, deque
from dataclasses import dataclass

from tools.base import ErrorCode, ToolFailure

_MESSAGES = {
    ErrorCode.SERVER_ERROR: ("Service returned 500 Internal Server Error.", "Temporary server problem; retrying may work."),
    ErrorCode.TIMEOUT: ("The service timed out.", "Temporary; retrying may work."),
    ErrorCode.RATE_LIMITED: ("Too many requests (429).", "Wait briefly and retry."),
}


@dataclass
class _Planned:
    code: ErrorCode
    after_commit: bool = False


class FaultInjector:
    """Raises ``ToolFailure`` on demand.

    * ``fail_next(tool, code, n)`` - the next ``n`` calls to ``tool`` fail.
      ``after_commit=True`` makes a write succeed in the store and *then* fail, which
      exercises idempotent retries.
    * ``failure_rate`` - seeded random SERVER_ERRORs for demos.
    """

    def __init__(self, failure_rate: float = 0.0, seed: int = 7) -> None:
        self.failure_rate = failure_rate
        self._rng = random.Random(seed)
        self._planned: dict[str, deque[_Planned]] = defaultdict(deque)
        self._lock = threading.Lock()

    def fail_next(
        self,
        tool: str,
        code: str | ErrorCode = ErrorCode.SERVER_ERROR,
        n: int = 1,
        *,
        after_commit: bool = False,
    ) -> None:
        code = ErrorCode(code) if not isinstance(code, ErrorCode) else code
        with self._lock:
            for _ in range(n):
                self._planned[tool].append(_Planned(code, after_commit))

    def clear(self) -> None:
        with self._lock:
            self._planned.clear()

    def pending(self, tool: str) -> int:
        return len(self._planned.get(tool, ()))

    def maybe_fail(self, tool: str, phase: str = "before") -> None:
        """Called by stores at the start (``before``) and end (``after``) of a call."""
        planned: _Planned | None = None
        with self._lock:
            queue = self._planned.get(tool) or self._planned.get("*")
            if queue:
                head = queue[0]
                if (phase == "after") == head.after_commit:
                    planned = queue.popleft()
            elif phase == "before" and self.failure_rate > 0 and self._rng.random() < self.failure_rate:
                planned = _Planned(ErrorCode.SERVER_ERROR)
        if planned is None:
            return
        message, hint = _MESSAGES.get(planned.code, (f"Injected failure: {planned.code.value}", None))
        raise ToolFailure(
            planned.code,
            message,
            hint=hint,
            retry_after_s=0.1 if planned.code is ErrorCode.RATE_LIMITED else None,
        )
