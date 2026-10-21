"""Typed agent events and the event bus.

A single ``EventBus.emit`` call does two things: it hands the event to every
registered sink (the structured-log ``RunRecorder``) and, when running inside a
LangGraph node, pushes it to the graph's custom stream so the live UI sees it
immediately. What the user sees and what is logged therefore cannot drift apart.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping

from pydantic import BaseModel, ConfigDict

from observability.context import root_span_id, span_id_var
from observability.spans import Span, span

EVENT_TYPES = (
    "turn_start",
    "confirmation_checked",
    "route",
    "llm_start",
    "llm_end",
    "llm_fallback",
    "token",
    "tool_proposed",
    "guard_block",
    "confirmation_required",
    "tool_start",
    "tool_retry",
    "tool_result",
    "memory_update",
    "recovery",
    "node_end",
    "final",
    "turn_end",
    "turn_error",
)


class AgentEvent(BaseModel):
    """One structured event. Optional fields are omitted when unset."""

    model_config = ConfigDict(extra="allow")

    # identity
    seq: int = 0
    ts: str = ""
    run_id: str = ""
    trace_id: str = ""
    turn: int = 0
    span_id: str | None = None
    parent_span_id: str | None = None
    # what happened
    event: str
    component: str = ""
    name: str | None = None
    elapsed_ms: float | None = None  # since the turn started
    # performance
    latency_ms: float | None = None
    ttft_ms: float | None = None
    # outcome
    ok: bool | None = None
    error_code: str | None = None
    retryable: bool | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    failure_reason: str | None = None
    retries_exhausted: bool | None = None
    # llm
    model: str | None = None
    fallback_used: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    # tool
    tool: str | None = None
    args: dict[str, Any] | None = None
    result_summary: str | None = None
    # flow
    step: int | None = None
    groups: list[str] | None = None
    guard_reason: str | None = None
    pending_action_state: str | None = None
    # free-form
    text: str | None = None
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


Sink = Callable[[AgentEvent], None]


class EventBus:
    """Creates events, fans them out to sinks and the LangGraph stream writer."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()
        self._sinks: list[Sink] = []
        self._turn_t0: dict[str, float] = {}

    # ----------------------------------------------------------------- sinks
    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    # ----------------------------------------------------------------- turns
    def start_turn(self, trace_id: str) -> None:
        self._turn_t0[trace_id] = time.perf_counter()

    def turn_elapsed_ms(self, trace_id: str) -> float | None:
        t0 = self._turn_t0.get(trace_id)
        return None if t0 is None else (time.perf_counter() - t0) * 1000.0

    # ---------------------------------------------------------------- events
    def make(
        self,
        event: str,
        *,
        state: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        turn: int | None = None,
        **fields: Any,
    ) -> AgentEvent:
        """Build an event (no side effects besides the sequence counter)."""
        if state is not None:
            trace_id = trace_id or state.get("trace_id")
            turn = turn if turn is not None else state.get("turn")
        trace_id = trace_id or ""
        with self._lock:
            self._seq += 1
            seq = self._seq
        if fields.get("parent_span_id") is None and trace_id:
            fields["parent_span_id"] = span_id_var.get() or root_span_id(trace_id)
        return AgentEvent(
            seq=seq,
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            run_id=self.run_id,
            trace_id=trace_id,
            turn=turn or 0,
            event=event,
            elapsed_ms=round(self.turn_elapsed_ms(trace_id) or 0.0, 2) if trace_id else None,
            **fields,
        )

    def emit(
        self,
        event: str,
        *,
        state: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        turn: int | None = None,
        **fields: Any,
    ) -> AgentEvent:
        """Record the event and push it to the live stream (when inside a graph)."""
        ev = self.make(event, state=state, trace_id=trace_id, turn=turn, **fields)
        for sink in self._sinks:
            try:
                sink(ev)
            except Exception:  # a broken sink must never break the agent
                import logging

                logging.getLogger("agent.events").warning("event sink failed", exc_info=True)
        self._write_stream(ev)
        return ev

    @staticmethod
    def _write_stream(ev: AgentEvent) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:  # not inside a graph run
            return
        try:
            writer(ev.to_dict())
        except Exception:
            pass

    # ----------------------------------------------------------------- spans
    @contextmanager
    def node(self, state: Mapping[str, Any], name: str) -> Iterator[Span]:
        """Time a graph node and emit ``node_end`` when it finishes."""
        with span(name, "node", parent_id=root_span_id(state.get("trace_id", ""))) as sp:
            try:
                yield sp
            except BaseException as exc:
                sp.fail(type(exc).__name__, str(exc)[:300])
                raise
            finally:
                self.emit(
                    "node_end",
                    state=state,
                    component="node",
                    name=name,
                    span_id=sp.span_id,
                    parent_span_id=sp.parent_id,
                    latency_ms=round(sp.elapsed_ms, 2),
                    ok=sp.ok,
                    error_code=sp.error_code,
                    failure_reason=sp.failure_reason,
                    step=state.get("steps"),
                )
