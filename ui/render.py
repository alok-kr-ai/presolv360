"""Rich rendering of the agent's realtime event stream.

``TurnView`` folds ``AgentEvent``s into a step timeline; it is a rich renderable so it works inside
``Live`` (animated spinners) and with ``console.print`` (tests use ``Console(record=True)``).
``render_turn`` drives a whole turn in live / plain / quiet mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.tree import Tree

from events import AgentEvent

ICONS = {"ok": ("✓", "green"), "error": ("✗", "red"), "warn": ("!", "yellow"), "skip": ("·", "dim"), "info": ("•", "blue")}


def fmt_args(args: dict[str, Any] | None, limit: int = 70) -> str:
    if not args:
        return ""
    text = ", ".join(f"{k}={v}" for k, v in args.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


@dataclass
class Step:
    key: str
    label: str
    status: str = "running"  # running | ok | error | warn | skip | info
    latency_ms: float | None = None
    detail: str = ""
    children: list[Text] = field(default_factory=list)


class TurnView:
    """Folds events into a timeline + streamed answer."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose
        self.turn = 0
        self.trace_id = ""
        self.steps: list[Step] = []
        self._by_key: dict[str, Step] = {}
        self.answer = ""
        self.final_text: str | None = None
        self.phase = "Starting"
        self.done = False
        self.status = "completed"
        self.total_ms: float | None = None
        self.llm_calls = 0
        self.tool_calls: list[tuple[str, str, float | None]] = []  # (tool, status, ms)
        self.models: list[str] = []
        self.fallback_used = False
        self.pending_summary: str | None = None
        self.groups: list[str] | None = None
        self.steps_taken = 0

    # ----------------------------------------------------------------- folding
    def _add(self, key: str, label: str, status: str = "running", detail: str = "") -> Step:
        step = Step(key=key, label=label, status=status, detail=detail)
        self.steps.append(step)
        self._by_key[key] = step
        return step

    def apply(self, ev: AgentEvent) -> None:  # noqa: C901 - one branch per event type
        kind = ev.event
        if kind == "turn_start":
            self.turn, self.trace_id = ev.turn, ev.trace_id
            self.phase = "Routing"
        elif kind == "confirmation_checked":
            decision = (ev.data or {}).get("decision", ev.name)
            status = {"yes": "ok", "no": "warn", "modify": "warn"}.get(decision, "skip")
            self._add(f"conf{ev.seq}", f"Checked your reply to the pending action: {decision}", status)
        elif kind == "route":
            self.groups = list(ev.groups or [])
            what = ", ".join(self.groups) if self.groups else "no tools needed"
            method = (ev.data or {}).get("method", "")
            suffix = f" ({method.replace('_', ' ')})" if method else ""
            self._add("route", f"Understanding request → {what}{suffix}", "ok" if method != "fail_open" else "warn")
            self.phase = "Thinking"
        elif kind == "node_end" and ev.name == "route" and "route" in self._by_key:
            self._by_key["route"].latency_ms = ev.latency_ms
        elif kind == "llm_start" and ev.name == "agent":
            self.answer = ""  # a new model call restarts the streamed answer
            self._add(f"llm{ev.span_id}", f"Thinking … {ev.model or ''}".rstrip(), "running")
            self.phase = "Thinking"
        elif kind == "llm_end" and ev.name == "agent":
            self.llm_calls += 1
            step = self._by_key.get(f"llm{ev.span_id}") or self._add(f"llm{ev.span_id}", "Thinking", "running")
            step.latency_ms = ev.latency_ms
            if ev.ok is False:
                step.status, step.detail = "error", f"{ev.error_code}: {ev.failure_reason or ''}"[:120]
            else:
                step.status = "ok"
                step.label = f"Thinking … {ev.model}"
                if ev.fallback_used:
                    step.detail = "fallback model"
                    self.fallback_used = True
                if self.verbose and (ev.prompt_tokens or ev.completion_tokens):
                    step.children.append(Text(f"tokens in={ev.prompt_tokens} out={ev.completion_tokens}", style="dim"))
            if ev.model and ev.model not in self.models:
                self.models.append(ev.model)
        elif kind == "llm_fallback":
            self._add(f"fb{ev.seq}", f"Model fallback: {ev.text}", "warn")
            self.fallback_used = True
        elif kind == "token":
            self.answer += ev.text or ""
            self.phase = "Answering"
        elif kind == "tool_start":
            step = self._add(f"tool{ev.span_id}", f"Tool {ev.tool}({fmt_args(ev.args)})", "running")
            self.phase = f"Calling {ev.tool}"
            if self.verbose and ev.args:
                step.children.append(Text(f"args: {json.dumps(ev.args, default=str)}", style="dim"))
        elif kind == "tool_retry":
            step = self._latest_tool(ev.tool)
            msg = f"retry {ev.attempt}/{ev.max_attempts} after {ev.error_code}"
            backoff = (ev.data or {}).get("backoff_s")
            if backoff is not None:
                msg += f" (waiting {backoff:.1f}s)"
            if step is not None:
                step.status, step.detail = "warn", msg
            self.phase = f"Retrying {ev.tool} {ev.attempt}/{ev.max_attempts}"
        elif kind == "tool_result":
            step = self._by_key.get(f"tool{ev.span_id}") or self._add(f"tool{ev.span_id}", f"Tool {ev.tool}", "running")
            step.latency_ms = ev.latency_ms
            if ev.ok:
                step.status, step.detail = "ok", "ok" + (f" after {ev.attempt} attempts" if (ev.attempt or 1) > 1 else "")
            else:
                step.status = "error"
                step.detail = f"failed: {ev.error_code}" + (" (retries exhausted)" if ev.retries_exhausted else "")
            if self.verbose and ev.result_summary:
                step.children.append(Text(f"result: {ev.result_summary}", style="dim"))
            self.tool_calls.append((ev.tool or "?", step.status, ev.latency_ms))
            self.phase = "Thinking"
        elif kind == "guard_block":
            self._add(f"guard{ev.seq}", f"Tool {ev.tool} blocked: {_reason_text(ev.guard_reason)}", "warn")
            if self.verbose and ev.failure_reason:
                self.steps[-1].children.append(Text(ev.failure_reason, style="dim"))
        elif kind == "confirmation_required":
            self.pending_summary = ev.text
            self._add(f"confirm{ev.seq}", f"Waiting for your confirmation: {ev.text}", "warn")
            self.phase = "Waiting for your confirmation"
        elif kind == "memory_update":
            self._add(f"mem{ev.seq}", f"Remembered {ev.name}", "skip")
            if not self.verbose:
                self.steps[-1].children = []
        elif kind == "recovery":
            self._add(f"rec{ev.seq}", f"Recovering: {ev.text}", "warn")
        elif kind == "final":
            self.final_text = ev.text or ""
            self.answer = self.final_text
            self.phase = "Done"
        elif kind == "turn_error":
            self._add(f"err{ev.seq}", f"Internal error: {ev.error_code}: {ev.failure_reason}", "error")
        elif kind == "turn_end":
            self.done = True
            self.total_ms = ev.latency_ms
            self.status = (ev.data or {}).get("status", "completed")
            self.steps_taken = ev.step or 0
            for step in self.steps:
                if step.status == "running":
                    step.status = "skip"

    def _latest_tool(self, tool: str | None) -> Step | None:
        for step in reversed(self.steps):
            if step.key.startswith("tool") and step.label.startswith(f"Tool {tool}"):
                return step
        return None

    # --------------------------------------------------------------- rendering
    def _label(self, step: Step) -> RenderableType:
        text = Text(step.label)
        if step.detail:
            text.append(f"  {step.detail}", style="yellow" if step.status == "warn" else ("red" if step.status == "error" else "dim"))
        if step.latency_ms is not None:
            text.append(f"  ({fmt_ms(step.latency_ms)})", style="cyan")
        if step.status == "running" and not self.done:
            return Spinner("dots", text=text)
        icon, color = ICONS.get(step.status, ICONS["info"])
        return Text.assemble((f"{icon} ", color), text)

    def timeline(self) -> Tree:
        root = Text.assemble(("turn ", "bold"), (str(self.turn), "bold"), ("  trace ", "dim"), (self.trace_id, "dim"))
        tree = Tree(root, guide_style="dim")
        for step in self.steps:
            node = tree.add(self._label(step))
            for child in step.children:
                node.add(child)
        return tree

    def answer_panel(self) -> Panel | None:
        if not self.answer.strip():
            return None
        return Panel(Markdown(self.answer), title="assistant", title_align="left", border_style="green")

    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = [self.timeline()]
        panel = self.answer_panel()
        if panel is not None:
            parts.append(panel)
        if not self.done:
            parts.append(Spinner("dots", text=Text(self.phase, style="cyan")))
        return Group(*parts)

    def summary_line(self) -> Text:
        line = Text()
        icon, color = ICONS["ok"] if self.status == "completed" else ICONS["warn"]
        line.append(f"{icon} ", style=color)
        line.append(f"turn {self.turn}", style="bold")
        routed = ", ".join(self.groups) if self.groups else "no tools"
        line.append(f" · {routed}", style="dim")
        if self.tool_calls:
            parts = [f"{t} {ICONS[s][0]}" for t, s, _ in self.tool_calls]
            line.append(" · " + ", ".join(parts), style="dim")
        line.append(f" · {self.llm_calls} llm call{'s' if self.llm_calls != 1 else ''}", style="dim")
        if self.models:
            line.append(f" · {', '.join(self.models)}", style="dim")
        if self.fallback_used:
            line.append(" (fallback)", style="yellow")
        if self.total_ms is not None:
            line.append(f" · {fmt_ms(self.total_ms)}", style="cyan")
        if self.status != "completed":
            line.append(f" · {self.status}", style="yellow")
        return line


def _reason_text(reason: str | None) -> str:
    return {
        "CONFIRMATION_REQUIRED": "needs your confirmation",
        "DUPLICATE_CALL": "duplicate call",
        "TOOL_NOT_ACTIVE": "tool group not enabled (enabling it)",
        "INVALID_ARGS": "invalid arguments",
        "UNKNOWN_TOOL": "unknown tool",
        "STEP_CAP": "step limit reached",
    }.get(reason or "", reason or "blocked")


def plain_line(ev: AgentEvent, verbose: bool = False) -> Text | None:
    """One line per event for non-live output (``None`` = nothing to show)."""
    k = ev.event
    if k == "turn_start":
        return Text(f"▶ turn {ev.turn} · trace {ev.trace_id}", style="bold")
    if k == "confirmation_checked":
        return Text(f"  confirmation reply: {(ev.data or {}).get('decision', ev.name)}", style="yellow")
    if k == "route":
        groups = ", ".join(ev.groups or []) or "no tools needed"
        return Text(f"  route → {groups} ({(ev.data or {}).get('method', '')})", style="dim")
    if k == "llm_end" and ev.name == "agent":
        extra = " (fallback)" if ev.fallback_used else ""
        status = "ok" if ev.ok is not False else f"failed {ev.error_code}"
        return Text(f"  llm {ev.model}{extra} {status} {fmt_ms(ev.latency_ms)}", style="dim")
    if k == "llm_fallback":
        return Text(f"  model fallback: {ev.text}", style="yellow")
    if k == "tool_start":
        return Text(f"  tool {ev.tool}({fmt_args(ev.args)}) …", style="blue")
    if k == "tool_retry":
        return Text(f"  retry {ev.tool} {ev.attempt}/{ev.max_attempts} after {ev.error_code}", style="yellow")
    if k == "tool_result":
        if ev.ok:
            line = Text(f"  tool {ev.tool} ✓ {fmt_ms(ev.latency_ms)}", style="green")
        else:
            line = Text(f"  tool {ev.tool} ✗ {ev.error_code} {fmt_ms(ev.latency_ms)}", style="red")
        if verbose and ev.result_summary:
            line.append(f"\n    {ev.result_summary}", style="dim")
        return line
    if k == "guard_block":
        return Text(f"  blocked {ev.tool}: {_reason_text(ev.guard_reason)}", style="yellow")
    if k == "confirmation_required":
        return Text(f"  waiting for confirmation: {ev.text}", style="yellow")
    if k == "memory_update":
        return Text(f"  remembered {ev.name}", style="dim")
    if k == "recovery":
        return Text(f"  recovering: {ev.text}", style="yellow")
    if k == "turn_error":
        return Text(f"  internal error: {ev.error_code}: {ev.failure_reason}", style="red")
    if k == "turn_end":
        return Text(f"■ done in {fmt_ms(ev.latency_ms)}", style="dim")
    return None


def pending_banner(summary: str) -> Panel:
    return Panel(
        Text.assemble(("Waiting for your confirmation: ", "bold yellow"), summary, ("\nReply yes to proceed or no to cancel.", "dim")),
        border_style="yellow",
    )


def render_turn(
    console: Console,
    events: Iterable[AgentEvent],
    *,
    mode: str = "live",  # live | plain | quiet
    verbose: bool = False,
) -> TurnView:
    """Drive one turn's events to the terminal and return the folded view."""
    view = TurnView(verbose=verbose)
    if mode == "live" and console.is_terminal:
        with Live(view, console=console, refresh_per_second=12, transient=True, vertical_overflow="visible") as live:
            for ev in events:
                view.apply(ev)
                live.update(view)
        console.print(view.summary_line())
        panel = view.answer_panel()
        if panel is not None:
            console.print(panel)
    elif mode == "quiet":
        with console.status("[cyan]Thinking…[/cyan]"):
            for ev in events:
                view.apply(ev)
        panel = view.answer_panel()
        if panel is not None:
            console.print(panel)
    else:  # plain: one line per event, also used for pipes / CI / recorded consoles
        for ev in events:
            view.apply(ev)
            line = plain_line(ev, verbose)
            if line is not None:
                console.print(line)
        panel = view.answer_panel()
        if panel is not None:
            console.print(panel)
    return view
