"""Scripted fake chat model for network-free tests."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from prompts import CONFIRM_MARKER, ROUTER_MARKER


@dataclass
class Call:
    """What the fake model was asked."""

    purpose: str  # agent | route | confirm
    messages: list[BaseMessage]
    bound: tuple[str, ...]

    @property
    def system(self) -> str:
        return next((m.content for m in self.messages if isinstance(m, SystemMessage)), "")  # type: ignore[return-value]

    @property
    def last_human(self) -> str:
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                return str(m.content)
        return ""

    @property
    def latest_user_message(self) -> str:
        """For router calls the text after 'Latest user message:'."""
        text = self.last_human
        return text.split("Latest user message:", 1)[-1].strip()

    def tool_results(self) -> list[tuple[str, dict[str, Any]]]:
        """(tool, parsed envelope) for tool messages after the last human message in this turn."""
        out: list[tuple[str, dict[str, Any]]] = []
        idx = max((i for i, m in enumerate(self.messages) if isinstance(m, HumanMessage)), default=-1)
        for m in self.messages[idx + 1 :]:
            if isinstance(m, ToolMessage):
                match = re.search(r"\{.*\}", str(m.content), re.DOTALL)
                if match:
                    out.append((m.name or "", json.loads(match.group(0))))
        return out

    def result_of(self, tool: str) -> dict[str, Any] | None:
        for name, env in reversed(self.tool_results()):
            if name == tool:
                return env
        return None


def tc(name: str, /, **args: Any) -> dict[str, Any]:
    """A tool-call step."""
    return {"name": name, "args": args}


def reissue_confirmed(call: Call) -> dict[str, Any]:
    """Step helper: re-issue exactly the action the system prompt says the user confirmed."""
    m = re.search(r"`(\w+)` with exactly these arguments, then report the real result:\n(\{.*\})", call.system)
    assert m, "no CONFIRMED ACTION in the system prompt"
    return {"name": m.group(1), "args": json.loads(m.group(2))}


def event_id_of(call: Call, title_part: str) -> str:
    """Step helper: find an event id in this turn's list_calendar_events result."""
    env = call.result_of("list_calendar_events")
    assert env and env["ok"], "list_calendar_events has not succeeded yet"
    return next(e["id"] for e in env["data"]["events"] if title_part.lower() in e["title"].lower())


_CAL = re.compile(r"meeting|calendar|schedule|event|book|delete|remove|availab|free|standup|appointment|reschedul|slot|cancel", re.I)
_CRM = re.compile(r"contact|email|who is|interaction|priya|rahul|ananya|john|crm|spoke", re.I)
_CALC = re.compile(r"\d\s*[-+*/x%]\s*\d|multiply|divid|plus|minus|total|percent|calculate|sum of|times", re.I)


def default_route(call: Call) -> dict[str, Any]:
    text = call.latest_user_message
    groups = [g for g, rx in (("calendar", _CAL), ("crm", _CRM), ("calculator", _CALC)) if rx.search(text)]
    return {"groups": groups, "confidence": 0.9, "reason": "keywords"}


def default_confirm(call: Call) -> dict[str, Any]:
    return {"decision": "unrelated"}


Step = Any  # str | dict | list[dict] | AIMessage | Callable[[Call], Step]


class Script:
    """Agent steps are consumed in order; route/confirm use pluggable functions."""

    def __init__(
        self,
        steps: list[Step] | None = None,
        route: Callable[[Call], Any] | None = None,
        confirm: Callable[[Call], Any] | None = None,
    ) -> None:
        self.steps: list[Step] = list(steps or [])
        self.route_fn = route or default_route
        self.confirm_fn = confirm or default_confirm
        self.calls: list[Call] = []
        self.exhausted = False

    def add(self, *steps: Step) -> "Script":
        self.steps.extend(steps)
        return self

    def respond(self, call: Call) -> AIMessage:
        self.calls.append(call)
        if call.purpose == "route":
            out = self.route_fn(call)
            if isinstance(out, Exception):
                raise out
            return AIMessage(content=out if isinstance(out, str) else json.dumps(out))
        if call.purpose == "confirm":
            out = self.confirm_fn(call)
            return AIMessage(content=out if isinstance(out, str) else json.dumps(out))
        if not self.steps:
            self.exhausted = True
            return AIMessage(content="(script exhausted)")
        step = self.steps.pop(0)
        if callable(step):
            step = step(call)
        if isinstance(step, Exception):
            raise step
        return self._to_message(step)

    @staticmethod
    def _to_message(step: Step) -> AIMessage:
        if isinstance(step, AIMessage):
            return step
        if isinstance(step, str):
            return AIMessage(content=step)
        calls = [step] if isinstance(step, dict) else list(step)
        return AIMessage(
            content="",
            tool_calls=[{"name": c["name"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"} for c in calls],
        )

    # introspection helpers for tests
    def agent_calls(self) -> list[Call]:
        return [c for c in self.calls if c.purpose == "agent"]

    def purposes(self) -> list[str]:
        return [c.purpose for c in self.calls]


class FakeChatModel(BaseChatModel):
    script: Any
    bound: list[str] = Field(default_factory=list)
    model_label: str = "fake-model"

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"bound": names})

    def _purpose(self, messages: list[BaseMessage]) -> str:
        first = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        if first.startswith(ROUTER_MARKER):
            return "route"
        if first.startswith(CONFIRM_MARKER):
            return "confirm"
        return "agent"

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        call = Call(self._purpose(messages), list(messages), tuple(self.bound))
        ai = self.script.respond(call)
        words = sum(len(str(m.content).split()) for m in messages)
        return AIMessage(
            content=ai.content,
            tool_calls=ai.tool_calls,
            response_metadata={"model_name": self.model_label, "finish_reason": "stop"},
            usage_metadata={"input_tokens": words, "output_tokens": max(len(str(ai.content).split()), 1), "total_tokens": words + 1},
        )

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ai = self._respond(messages)
        if ai.tool_calls:
            chunks = [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                        for i, c in enumerate(ai.tool_calls)
                    ],
                )
            ]
        else:
            parts = re.findall(r"\S+\s*", str(ai.content)) or [""]
            chunks = [AIMessageChunk(content=p) for p in parts]
        chunks[-1] = chunks[-1] + AIMessageChunk(content="", response_metadata=ai.response_metadata, usage_metadata=ai.usage_metadata)
        for chunk in chunks:
            if run_manager:
                run_manager.on_llm_new_token(str(chunk.content), chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)
