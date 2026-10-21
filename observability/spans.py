"""Lightweight span helper: times a block, nests automatically, records failures."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from observability.context import new_span_id, span_id_var


@dataclass
class Span:
    name: str
    component: str
    parent_id: str | None
    span_id: str = field(default_factory=new_span_id)
    t0: float = field(default_factory=time.perf_counter)
    ok: bool = True
    error_code: str | None = None
    failure_reason: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def fail(self, code: str | None = None, reason: str | None = None) -> None:
        self.ok = False
        self.error_code = code
        self.failure_reason = reason


@contextmanager
def span(name: str, component: str, *, parent_id: str | None = None) -> Iterator[Span]:
    """Open a span. Parent defaults to the enclosing span (contextvar)."""
    parent = parent_id if parent_id is not None else span_id_var.get()
    sp = Span(name=name, component=component, parent_id=parent)
    token = span_id_var.set(sp.span_id)
    try:
        yield sp
    except BaseException as exc:  # recorded, then re-raised
        sp.fail(type(exc).__name__, str(exc)[:300])
        raise
    finally:
        span_id_var.reset(token)
