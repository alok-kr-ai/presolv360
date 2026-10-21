"""Agent facade: ``stream()`` yields live events, ``invoke()`` drains it."""
from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from config import Settings
from events import AgentEvent, EventBus
from graph.guards import message_text
from mocks import Backend
from observability.context import new_trace_id
from observability.langsmith import run_metadata
from observability.recorder import RunRecorder
from state import WorkingMemory


@dataclass
class TurnResult:
    text: str
    trace_id: str
    turn: int
    status: str
    events: list[AgentEvent] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_events(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event == "tool_result"]

    @property
    def tools_called(self) -> list[str]:
        return [e.tool for e in self.events if e.event == "tool_start" and e.tool]

    @property
    def groups(self) -> list[str]:
        route = next((e for e in self.events if e.event == "route"), None)
        return list(route.groups or []) if route else []


class Agent:
    def __init__(
        self,
        *,
        settings: Settings,
        graph: Any,
        bus: EventBus,
        recorder: RunRecorder,
        backend: Backend,
        thread_id: str,
        groups: dict[str, list],
        langsmith: bool = False,
    ) -> None:
        self.settings = settings
        self.graph = graph
        self.bus = bus
        self.recorder = recorder
        self.backend = backend
        self.thread_id = thread_id
        self.groups = groups
        self.langsmith = langsmith

    # ---------------------------------------------------------------- threads
    @property
    def run_id(self) -> str:
        return self.bus.run_id

    def new_thread(self) -> str:
        self.thread_id = f"thread-{uuid.uuid4().hex[:6]}"
        self.recorder.add_thread(self.thread_id)
        return self.thread_id

    def _config(self, thread_id: str, trace_id: str | None = None) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_steps * 4 + 20,
        }
        if trace_id:
            cfg["metadata"] = run_metadata(self.run_id, trace_id, thread_id)
            cfg["tags"] = [f"run:{self.run_id}", f"trace:{trace_id}"]
        return cfg

    def state_values(self, thread_id: str | None = None) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(thread_id or self.thread_id))
        return dict(snap.values) if snap and snap.values else {}

    def memory(self, thread_id: str | None = None) -> WorkingMemory:
        return WorkingMemory.from_state(self.state_values(thread_id))

    # ---------------------------------------------------------------- running
    def stream(self, text: str, thread_id: str | None = None) -> Iterator[AgentEvent]:
        thread_id = thread_id or self.thread_id
        turn = int(self.state_values(thread_id).get("turn", 0)) + 1
        trace_id = new_trace_id()
        self.bus.start_turn(trace_id)
        yield self.bus.emit("turn_start", trace_id=trace_id, turn=turn, component="agent", text=text,
                            data={"thread_id": thread_id})

        counts: Counter[str] = Counter()
        status = "completed"
        error: BaseException | None = None
        try:
            stream = self.graph.stream(
                {"messages": [HumanMessage(text)], "trace_id": trace_id},
                self._config(thread_id, trace_id),
                stream_mode=["custom", "messages"],
            )
            for mode, chunk in stream:
                if mode == "custom":
                    ev = AgentEvent.model_validate(chunk)
                    counts[ev.event] += 1
                    yield ev
                elif mode == "messages":
                    msg, meta = chunk
                    if (
                        meta.get("langgraph_node") == "agent"
                        and isinstance(msg, AIMessageChunk)
                        and isinstance(msg.content, str)
                        and msg.content
                    ):
                        yield self.bus.make("token", trace_id=trace_id, turn=turn, component="llm", text=msg.content)
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:  # a crash in the graph itself (bugs), not tool/LLM errors
            status, error = "crashed", exc
            yield self.bus.emit("turn_error", trace_id=trace_id, turn=turn, component="agent", ok=False,
                                error_code=type(exc).__name__, failure_reason=str(exc)[:500])

        values = self.state_values(thread_id)
        steps = int(values.get("steps", 0)) if status == "completed" else 0
        end = self.bus.emit(
            "turn_end", trace_id=trace_id, turn=turn, component="agent", step=steps,
            latency_ms=round(self.bus.turn_elapsed_ms(trace_id) or 0.0, 2), ok=status == "completed",
            data={
                "status": status,
                "tool_calls": counts["tool_start"],
                "retries": counts["tool_retry"],
                "guard_blocks": counts["guard_block"],
                "llm_calls": counts["llm_end"],
            },
        )
        yield end
        if error is not None:
            self.recorder.set_status("crashed")
            raise error

    def invoke(self, text: str, thread_id: str | None = None) -> TurnResult:
        events: list[AgentEvent] = []
        for ev in self.stream(text, thread_id):
            if ev.event != "token":
                events.append(ev)
        thread_id = thread_id or self.thread_id
        values = self.state_values(thread_id)
        final = next((e for e in reversed(events) if e.event == "final"), None)
        text_out = final.text if final and final.text else ""
        if not text_out:
            for m in reversed(values.get("messages", [])):
                if isinstance(m, AIMessage) and not m.tool_calls:
                    text_out = message_text(m)
                    break
        start = next(e for e in events if e.event == "turn_start")
        end = next((e for e in reversed(events) if e.event == "turn_end"), None)
        return TurnResult(
            text=text_out or "",
            trace_id=start.trace_id,
            turn=start.turn,
            status=(end.data or {}).get("status", "completed") if end else "completed",
            events=events,
            state=values,
        )

    def close(self, status: str = "completed") -> None:
        self.recorder.close(status)
