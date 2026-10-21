"""LangGraph nodes: intake, route, agent, guard, tools, finalize.

Nodes are created by :class:`Nodes`, which closes over the injected dependencies (settings, LLM
client, event bus, tools, clock, sleep) so the graph is fully configurable from ``build_agent``.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END

from config import Settings
from events import EventBus
from graph.guards import (
    claims_unverified_success,
    evaluate_calls,
    honest_failure_message,
    is_chitchat,
    last_human_text,
    message_text,
    parse_tool_message,
    quick_classify_reply,
    sanitize_history,
    this_turn_messages,
    write_outcomes,
)
from llm import LLMClient, LLMError
from memory import apply_tool_result
from observability.spans import span
from prompts import (
    build_confirm_system,
    build_router_system,
    build_system_prompt,
    extract_json,
)
from state import PendingAction, WorkingMemory
from tools import group_of, is_write_tool
from tools.base import ErrorCode, ToolResult

ROUTER_CONFIDENCE_MIN = 0.5


@dataclass
class Nodes:
    settings: Settings
    llm: LLMClient
    bus: EventBus
    groups: dict[str, list[BaseTool]]
    group_desc: dict[str, str]
    tools: dict[str, BaseTool]
    clock: Callable[[], datetime]
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(11))

    # ------------------------------------------------------------------ helpers
    @property
    def write_tools(self) -> set[str]:
        return {n for n, t in self.tools.items() if is_write_tool(t)}

    def _history(self, messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
        msgs = sanitize_history(list(messages))
        trimmed = trim_messages(
            msgs,
            max_tokens=max_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        last_human = next((i for i in range(len(msgs) - 1, -1, -1) if isinstance(msgs[i], HumanMessage)), 0)
        if not any(isinstance(m, HumanMessage) for m in trimmed) or msgs[last_human] not in trimmed:
            trimmed = msgs[last_human:]
        return sanitize_history(trimmed)

    def _recent_context(self, messages: list[BaseMessage], limit: int = 6) -> str:
        lines = []
        history = [m for m in messages[:-1] if isinstance(m, (HumanMessage, AIMessage))]
        for m in history[-limit:]:
            text = message_text(m).strip()
            if not text:
                continue
            who = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{who}: {text[:240]}")
        return "\n".join(lines) or "(none)"

    # ------------------------------------------------------------------- intake
    def intake(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        turn = state.get("turn", 0) + 1
        s = {**state, "turn": turn}
        pending_raw = state.get("pending_action")
        updates: dict[str, Any] = {
            "turn": turn,
            "steps": 0,
            "call_signatures": [],
            "approved_calls": [],
            "confirmation_note": None,
            "step_cap_hit": False,
            "final_corrected": False,
        }
        with self.bus.node(s, "intake"):
            if pending_raw:
                pending = PendingAction.model_validate(pending_raw)
                text = last_human_text(state.get("messages", []))
                if pending.confirmed:
                    # a confirmation is consumed in the turn it was given; a leftover one is stale
                    updates["pending_action"] = None
                else:
                    decision, method = self._classify_reply(s, pending, text, config)
                    self.bus.emit(
                        "confirmation_checked", state=s, component="intake", name=decision,
                        pending_action_state=f"{pending.tool}: {pending.summary}",
                        data={"decision": decision, "method": method},
                    )
                    if decision == "yes":
                        pending.confirmed = True
                        pending.confirmed_turn = turn
                        pending.idle_turns = 0
                        updates["pending_action"] = pending.model_dump(mode="json")
                    elif decision == "no":
                        updates["pending_action"] = None
                        updates["confirmation_note"] = (
                            f"The user declined the pending action ({pending.summary}). Nothing was changed; "
                            "acknowledge briefly."
                        )
                    elif decision == "modify":
                        updates["pending_action"] = None
                        updates["confirmation_note"] = (
                            f"The user wants to change the pending action ({pending.summary}). It was cancelled; "
                            "work out the new details and propose the updated action."
                        )
                    else:  # unrelated: keep waiting, but not forever
                        pending.idle_turns += 1
                        if pending.idle_turns > self.settings.pending_ttl_turns:
                            updates["pending_action"] = None
                            updates["confirmation_note"] = (
                                f"The earlier pending action ({pending.summary}) expired without confirmation."
                            )
                        else:
                            updates["pending_action"] = pending.model_dump(mode="json")
        return updates

    def _classify_reply(self, s: dict[str, Any], pending: PendingAction, text: str, config: Any) -> tuple[str, str]:
        quick = quick_classify_reply(text)
        if quick:
            return quick, "rule"
        try:
            ai = self.llm.call(
                s, "confirm",
                [SystemMessage(build_confirm_system(pending.summary)), HumanMessage(text)],
                config=config,
            )
        except LLMError:
            return "unrelated", "llm_error"
        parsed = extract_json(message_text(ai)) or {}
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision in ("yes", "no", "modify", "unrelated"):
            return decision, "llm"
        return "unrelated", "llm_unparsed"

    # -------------------------------------------------------------------- route
    def route(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        all_groups = list(self.groups)
        with self.bus.node(state, "route"):
            messages = state.get("messages", [])
            text = last_human_text(messages)
            pending = PendingAction.model_validate(state["pending_action"]) if state.get("pending_action") else None
            fail_open = False
            reason = ""
            if pending and pending.confirmed:
                groups = [group_of(self.tools[pending.tool])] if pending.tool in self.tools else all_groups
                method, reason = "pending_confirmation", f"executing confirmed {pending.tool}"
            elif is_chitchat(text):
                groups, method, reason = [], "fast_path", "greeting / thanks"
            else:
                groups, method, reason, fail_open = self._llm_route(state, text, config)
            self.bus.emit(
                "route", state=state, component="router", name=method, groups=groups, text=reason or None,
                data={"method": method, "reason": reason, "fail_open": fail_open},
            )
        return {"groups": groups}

    def _llm_route(self, state: dict[str, Any], text: str, config: Any) -> tuple[list[str], str, str, bool]:
        all_groups = list(self.groups)
        memory = WorkingMemory.from_state(state).prompt_json()
        prompt = (
            f"Recent conversation:\n{self._recent_context(state.get('messages', []))}\n\n"
            f"Working memory: {json.dumps(memory, default=str)}\n\n"
            f"Latest user message: {text}"
        )
        try:
            ai = self.llm.call(
                state, "route",
                [SystemMessage(build_router_system(self.group_desc)), HumanMessage(prompt)],
                config=config,
            )
        except LLMError as exc:
            return all_groups, "fail_open", f"router error: {exc.kind}", True
        parsed = extract_json(message_text(ai))
        if not parsed or not isinstance(parsed.get("groups"), list):
            return all_groups, "fail_open", "unparseable router output", True
        try:
            confidence = float(parsed.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        requested = [g for g in parsed["groups"] if isinstance(g, str)]
        valid = [g for g in dict.fromkeys(requested) if g in self.groups]
        if requested and not valid:
            return all_groups, "fail_open", "router returned unknown groups", True
        if confidence < ROUTER_CONFIDENCE_MIN:
            return all_groups, "fail_open", f"low router confidence ({confidence:.2f})", True
        return valid, "router", str(parsed.get("reason", ""))[:200], False

    # -------------------------------------------------------------------- agent
    def agent(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        steps = state.get("steps", 0) + 1
        s = {**state, "steps": steps}
        with self.bus.node(s, "agent"):
            active = [t for g in (state.get("groups") or []) for t in self.groups.get(g, [])]
            try:
                ai, s = self._call_agent(s, active, config)
            except LLMError as exc:
                text = self._llm_failure_text(exc)
                ai = AIMessage(content=text)
                self.bus.emit("final", state=s, component="agent", text=text, ok=False, error_code=exc.kind)
                return {"messages": [ai], "steps": steps}

            if not ai.tool_calls:
                ai = self._verify_final(s, ai, active, config)
                self.bus.emit("final", state=s, component="agent", text=message_text(ai), ok=True)
        return {"messages": [ai], "steps": steps}

    def _prompt(self, s: dict[str, Any], correction: str | None = None) -> str:
        return build_system_prompt(
            now=self.clock(),
            timezone=self.settings.timezone,
            group_desc=self.group_desc,
            tool_names={g: [t.name for t in ts] for g, ts in self.groups.items()},
            active_groups=list(s.get("groups") or []),
            memory=WorkingMemory.from_state(s),
            confirmation_note=s.get("confirmation_note"),
            correction_note=correction,
        )

    def _call_agent(
        self, s: dict[str, Any], tools: list[BaseTool], config: Any, correction: str | None = None
    ) -> tuple[AIMessage, dict[str, Any]]:
        budget = self.settings.max_history_tokens
        try:
            history = self._history(s["messages"], budget)
            return self.llm.call(s, "agent", [SystemMessage(self._prompt(s, correction)), *history],
                                 tools=tools, config=config), s
        except LLMError as exc:
            if exc.kind != "context_window":
                raise
            self.bus.emit("recovery", state=s, component="agent", text="context window exceeded; retrying with shorter history")
            history = self._history(s["messages"], max(budget // 3, 500))
            return self.llm.call(s, "agent", [SystemMessage(self._prompt(s, correction)), *history],
                                 tools=tools, config=config), s

    @staticmethod
    def _llm_failure_text(exc: LLMError) -> str:
        hints = {
            "auth": "the model credentials were rejected (check the API key)",
            "rate_limit": "the model provider is rate limiting requests right now",
            "model_not_found": "the configured model is not available",
            "timeout": "the model took too long to respond",
            "connection": "I could not reach the model provider",
            "context_window": "the conversation is too long for the model",
        }
        why = hints.get(exc.kind, "the language model is unavailable")
        return (f"Sorry, I couldn't process that: {why}. No changes were made. "
                "Please try again in a moment.")

    def _verify_final(self, s: dict[str, Any], ai: AIMessage, tools: list[BaseTool], config: Any) -> AIMessage:
        """Never let a reply claim a write succeeded when no write result says so."""
        outcomes = write_outcomes(s["messages"], self.write_tools)
        text = message_text(ai)
        if not claims_unverified_success(text, outcomes):
            return ai
        self.bus.emit(
            "recovery", state=s, component="agent",
            text="reply claimed a completed action without a successful tool result; re-prompting once",
        )
        note = ("Your previous reply claimed an action was completed, but no tool result confirms it (it failed, was "
                "blocked, or never ran). Rewrite the reply honestly: say what failed or that confirmation is still needed.")
        try:
            retry, _ = self._call_agent({**s, "final_corrected": True}, tools, config, correction=note)
        except LLMError:
            return AIMessage(content=honest_failure_message(outcomes))
        if retry.tool_calls or not claims_unverified_success(message_text(retry), outcomes):
            return retry
        return AIMessage(content=honest_failure_message(outcomes))

    # -------------------------------------------------------------------- guard
    def guard(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        with self.bus.node(state, "guard"):
            last = state["messages"][-1]
            calls = list(getattr(last, "tool_calls", None) or [])
            for call in calls:
                self.bus.emit("tool_proposed", state=state, component="guard", name=call["name"],
                              tool=call["name"], args=call.get("args") or {})
            outcome = evaluate_calls(calls, state=state, tools=self.tools, max_steps=self.settings.max_steps)

            blocked: list[ToolMessage] = []
            approved: list[dict[str, Any]] = []
            for d in outcome.decisions:
                call = d.call
                if d.approved:
                    approved.append({"id": call["id"], "name": call["name"], "args": call.get("args") or {}})
                    continue
                assert d.result is not None
                blocked.append(ToolMessage(content=d.result.envelope(call["name"]), tool_call_id=call["id"],
                                           name=call["name"], status="error"))
                self.bus.emit(
                    "guard_block", state=state, component="guard", name=call["name"], tool=call["name"],
                    guard_reason=d.reason, failure_reason=d.detail, ok=False, error_code=d.reason,
                )
            for note in outcome.recoveries:
                self.bus.emit("recovery", state=state, component="guard", text=note)
            if outcome.pending_event:
                p = outcome.pending_action or {}
                self.bus.emit(
                    "confirmation_required", state=state, component="guard", name=p.get("tool"),
                    tool=p.get("tool"), text=p.get("summary"), args=p.get("call_args"),
                    pending_action_state="awaiting_confirmation", data={"pending": outcome.pending_event},
                )
        return {
            "messages": blocked,
            "approved_calls": approved,
            "groups": outcome.groups,
            "pending_action": outcome.pending_action,
            "call_signatures": outcome.call_signatures,
            "step_cap_hit": outcome.step_cap_hit,
        }

    # -------------------------------------------------------------------- tools
    def tools_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        settings = self.settings
        mem = {k: state.get(k) for k in ("last_event", "last_contact", "last_calculation", "listed_events", "seen_event_ids", "seen_contact_ids")}
        updates: dict[str, Any] = {"approved_calls": []}
        messages: list[ToolMessage] = []
        pending = state.get("pending_action")
        signatures = list(state.get("call_signatures") or [])
        max_attempts = settings.max_tool_retries + 1

        with self.bus.node(state, "tools"):
            for call in state.get("approved_calls") or []:
                name, args = call["name"], call.get("args") or {}
                tool = self.tools[name]
                runner = tool.metadata["runner"]
                request_id = uuid.uuid4().hex if is_write_tool(tool) else None
                with span(f"tool:{name}", "tool") as sp:
                    self.bus.emit("tool_start", state=state, component="tool", name=name, tool=name, args=args,
                                  span_id=sp.span_id, parent_span_id=sp.parent_id, step=state.get("steps"))
                    attempt, total_ms = 0, 0.0
                    while True:
                        attempt += 1
                        result: ToolResult = runner(args, timeout=settings.tool_timeout_s, request_id=request_id)
                        total_ms += result.latency_ms
                        if result.ok or not (result.error and result.error.retryable) or attempt >= max_attempts:
                            break
                        delay = self._backoff(attempt, result)
                        self.bus.emit(
                            "tool_retry", state=state, component="tool", name=name, tool=name,
                            attempt=attempt, max_attempts=max_attempts, error_code=result.error_code, ok=False,
                            retryable=True, latency_ms=result.latency_ms, failure_reason=result.error.message,
                            step=state.get("steps"), data={"backoff_s": round(delay, 3)},
                        )
                        self.sleep(delay)
                    exhausted = bool(not result.ok and result.error and result.error.retryable)
                    if exhausted:
                        result.meta = {**(result.meta or {}), "retries_exhausted": True}
                    if not result.ok:
                        sp.fail(result.error_code, result.error.message if result.error else None)
                    self.bus.emit(
                        "tool_result", state=state, component="tool", name=name, tool=name, args=args,
                        span_id=sp.span_id, parent_span_id=sp.parent_id, step=state.get("steps"),
                        ok=result.ok, error_code=result.error_code,
                        retryable=result.error.retryable if result.error else None,
                        latency_ms=round(total_ms, 2), attempt=attempt, max_attempts=max_attempts,
                        retries_exhausted=exhausted or None,
                        result_summary=result.summary(600),
                        failure_reason=(result.debug or (result.error.message if result.error else None)),
                        data={"wall_ms": round(sp.elapsed_ms, 2)},
                    )
                messages.append(ToolMessage(content=result.envelope(name), tool_call_id=call["id"], name=name,
                                            status="success" if result.ok else "error"))

                mem_updates, changed = apply_tool_result(name, args, result, mem)
                for key, value in mem_updates.items():
                    mem[key] = value
                    updates[key] = value
                for key in changed:
                    self.bus.emit("memory_update", state=state, component="memory", name=key,
                                  text=json.dumps(mem.get(key), default=str)[:300])
                if is_write_tool(tool):
                    pending = None  # a confirmation is single-use, whatever the outcome
                    if result.ok:
                        signatures = []
            updates["pending_action"] = pending
            updates["call_signatures"] = signatures
            updates["messages"] = messages
        return updates

    def _backoff(self, attempt: int, result: ToolResult) -> float:
        base = self.settings.retry_base_delay_s * (2 ** (attempt - 1))
        delay = base * self.rng.uniform(0.5, 1.0)
        retry_after = (result.meta or {}).get("retry_after_s")
        return max(delay, float(retry_after)) if retry_after else delay

    # ----------------------------------------------------------------- finalize
    def finalize(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        with self.bus.node(state, "finalize"):
            done, failed = [], []
            for m in this_turn_messages(state["messages"]):
                if isinstance(m, ToolMessage):
                    env = parse_tool_message(m)
                    if env is None or (env.get("error") or {}).get("code") in (ErrorCode.STEP_CAP.value,):
                        continue
                    (done if env.get("ok") else failed).append(
                        m.name if env.get("ok") else f"{m.name} ({(env.get('error') or {}).get('code')})"
                    )
            text = f"I reached my limit of {self.settings.max_steps} steps for this request, so I stopped here."
            if done:
                text += " Completed: " + ", ".join(done) + "."
            if failed:
                text += " Did not complete: " + ", ".join(failed) + "."
            if not done and not failed:
                text += " Nothing was changed."
            text += " Tell me how you'd like to continue."
            self.bus.emit("final", state=state, component="agent", text=text, ok=False, error_code=ErrorCode.STEP_CAP.value)
        return {"messages": [AIMessage(content=text)]}

    # ------------------------------------------------------------------- edges
    @staticmethod
    def after_agent(state: dict[str, Any]) -> str:
        last = state["messages"][-1]
        return "guard" if getattr(last, "tool_calls", None) else END

    @staticmethod
    def after_guard(state: dict[str, Any]) -> str:
        if state.get("approved_calls"):
            return "tools"
        if state.get("step_cap_hit"):
            return "finalize"
        return "agent"
