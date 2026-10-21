"""Context variables identifying the current run / trace / span."""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)


def new_run_id() -> str:
    """UTC timestamp plus a short random suffix, e.g. 20261020T184900Z-a1b2c3."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def new_span_id() -> str:
    return uuid.uuid4().hex[:8]


def root_span_id(trace_id: str) -> str:
    return f"{trace_id}-root"
