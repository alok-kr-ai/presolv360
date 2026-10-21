"""Graph-level guards (pure Python, no LLM).

* write confirmation (only a write whose ``pending_action`` was confirmed on a prior user turn,
  with matching args, may run; delete matches ``event_id`` strictly)
* ``event_id`` provenance (ids must come from earlier tool output, never guessed)
* repeated identical calls
* unknown / inactive tools, schema validation, step cap
* final-answer integrity (no success claims after failed writes)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from state import PendingAction
from tools import group_of, is_write_tool
from tools.base import ErrorCode, ToolResult

# --------------------------------------------------------------------------- text helpers
_CHITCHAT = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|howdy|yo|good\s+(?:morning|afternoon|evening|night)|thanks?|"
    r"thank\s+you(?:\s+(?:so\s+much|very\s+much))?|thx|ty|bye|goodbye|see\s+you(?:\s+later)?|cheers|"
    r"ok(?:ay)?[,.]?\s+thanks?|(?:great|cool|nice|awesome|perfect)(?:[,.]?\s+thanks?)?|got\s+it)"
    r"(?:\s+(?:there|again|assistant|bot|friend))?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_YES = re.compile(
    r"^\s*(?:yes|yep|yeah|yup|y|sure|ok(?:ay)?|confirm(?:ed)?|go\s+ahead|do\s+it|proceed|please\s+do|"
    r"sounds\s+good|that'?s\s+right|correct|affirmative|please\s+go\s+ahead|"
    r"yes[,.]?\s+(?:please|go\s+ahead|do\s+it|book\s+it|delete\s+it|confirm(?:ed)?))\s*[.!]*\s*$",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^\s*(?:no|nope|nah|cancel(?:\s+(?:it|that))?|stop|never\s*mind|don'?t(?:\s+do\s+it)?|abort|forget\s+it|"
    r"not\s+now|no\s+thanks?)\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_chitchat(text: str) -> bool:
    return bool(_CHITCHAT.match(text or ""))


def quick_classify_reply(text: str) -> str | None:
    """Deterministic classification for explicit replies; ``None`` means ask the LLM."""
    if _YES.match(text or ""):
        return "yes"
    if _NO.match(text or ""):
        return "no"
    if is_chitchat(text):
        return "unrelated"
    return None


def message_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return message_text(m)
    return ""


def this_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return list(messages)


_ENVELOPE = re.compile(r"<tool_result[^>]*>\s*(\{.*\})\s*</tool_result>", re.DOTALL)


def parse_tool_message(msg: ToolMessage) -> dict[str, Any] | None:
    m = _ENVELOPE.search(message_text(msg))
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- history hygiene
def sanitize_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Make the history valid for the provider: every assistant tool call gets exactly one
    tool message right after it (synthesised if the turn was interrupted) and orphan tool
    messages are dropped."""
    out: list[BaseMessage] = []
    n, i = len(messages), 0
    while i < n:
        m = messages[i]
        i += 1
        if isinstance(m, ToolMessage):
            continue
        out.append(m)
        if isinstance(m, AIMessage) and m.tool_calls:
            wanted = [tc["id"] for tc in m.tool_calls]
            got: dict[str, ToolMessage] = {}
            j = i
            while j < n and isinstance(messages[j], ToolMessage):
                if messages[j].tool_call_id in wanted:  # type: ignore[union-attr]
                    got[messages[j].tool_call_id] = messages[j]  # type: ignore[union-attr]
                j += 1
            for tc in m.tool_calls:
                out.append(got.get(tc["id"]) or _synthetic_tool_message(tc))
            i = j
    return out


def _synthetic_tool_message(tc: dict[str, Any]) -> ToolMessage:
    res = ToolResult.failure(
        ErrorCode.INTERNAL, "This call was interrupted before it ran.", retryable=False,
        hint="Do not assume it happened. Ask the user whether to try again.",
    )
    return ToolMessage(content=res.envelope(tc["name"]), tool_call_id=tc["id"], name=tc["name"])


# --------------------------------------------------------------------------- canonical args
def canonical_args(parsed: BaseModel) -> dict[str, Any]:
    """Comparison form of validated args (instants in UTC, case/whitespace folded)."""
    out: dict[str, Any] = {}
    for key, value in parsed.model_dump(exclude_none=True).items():
        if isinstance(value, datetime):
            out[key] = value.astimezone(timezone.utc).isoformat()
        elif isinstance(value, str):
            out[key] = value if key.endswith("_id") else " ".join(value.split()).casefold()
        elif isinstance(value, list):
            out[key] = sorted(" ".join(str(v).split()).casefold() for v in value)
        else:
            out[key] = value
    return out


def call_signature(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, default=str)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%a %d %b %Y %H:%M") + " " + (dt.tzname() or "")


def describe_write(tool: str, parsed: BaseModel, memory: dict[str, Any]) -> str:
    data = parsed.model_dump(exclude_none=True)
    if tool == "create_calendar_event":
        extra = f" with {', '.join(data['attendees'])}" if data.get("attendees") else ""
        return f"Create event \"{data['title']}\" from {_fmt_dt(data['start'])} to {data['end'].strftime('%H:%M')}{extra}"
    if tool == "create_contact":
        extra = ", ".join(x for x in (data.get("title"), f"at {data['company']}" if data.get("company") else None) if x)
        phone = f", phone {data['phone']}" if data.get("phone") else ""
        return f"Add contact \"{data['name']}\" <{data['email']}>" + (f" ({extra})" if extra else "") + phone
    if tool in ("delete_contact", "update_contact"):
        known_c = next((c for c in (memory.get("last_contact"),) if c and c.get("id") == data.get("contact_id")), None) or {}
        who = f" (\"{known_c['name']}\" <{known_c['email']}>)" if known_c.get("name") and known_c.get("email") else (
            f" (\"{known_c['name']}\")" if known_c.get("name") else "")
        if tool == "delete_contact":
            return f"Delete contact {data.get('contact_id')}{who}"
        edits = [f"{k} -> {data[k] if data[k] else '(cleared)'}" for k in ("name", "email", "company", "title", "phone") if k in data]
        return f"Update contact {data.get('contact_id')}{who}: " + "; ".join(edits)
    known = {e.get("id"): e for e in [memory.get("last_event") or {}, *(memory.get("listed_events") or [])]}
    title = (known.get(data.get("event_id")) or {}).get("title")
    label = f"event {data.get('event_id')}" + (f" (\"{title}\")" if title else "")
    if tool == "delete_calendar_event":
        return f"Delete {label}"
    changes = []
    for key in ("title", "start", "end", "attendees", "location", "description"):
        if key in data:
            v = data[key]
            changes.append(f"{key} -> {_fmt_dt(v) if isinstance(v, datetime) else v}")
    return f"Update {label}: " + "; ".join(changes)


# --------------------------------------------------------------------------- the guard
@dataclass
class Decision:
    call: dict[str, Any]
    approved: bool
    result: ToolResult | None = None  # error fed back to the model when blocked
    reason: str | None = None
    detail: str | None = None


@dataclass
class GuardOutcome:
    decisions: list[Decision] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    call_signatures: list[str] = field(default_factory=list)
    step_cap_hit: bool = False
    pending_event: str | None = None  # "created" | "replaced"
    recoveries: list[str] = field(default_factory=list)


# write tool -> (id argument, state key of ids seen in tool output, tool that returns them)
_ID_PROVENANCE = {
    "delete_calendar_event": ("event_id", "seen_event_ids", "list_calendar_events"),
    "update_calendar_event": ("event_id", "seen_event_ids", "list_calendar_events"),
    "update_contact": ("contact_id", "seen_contact_ids", "get_contact"),
    "delete_contact": ("contact_id", "seen_contact_ids", "get_contact"),
}


def _block(call: dict[str, Any], code: ErrorCode, message: str, hint: str | None, detail: str | None = None) -> Decision:
    return Decision(
        call=call, approved=False, reason=code.value, detail=detail or message,
        result=ToolResult.failure(code, message, retryable=False, hint=hint),
    )


def evaluate_calls(
    tool_calls: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    tools: dict[str, BaseTool],
    max_steps: int,
) -> GuardOutcome:
    out = GuardOutcome(
        groups=list(state.get("groups") or []),
        call_signatures=list(state.get("call_signatures") or []),
    )
    pending = PendingAction.model_validate(state["pending_action"]) if state.get("pending_action") else None
    turn = state.get("turn", 0)
    seen = {"seen_event_ids": set(state.get("seen_event_ids") or []), "seen_contact_ids": set(state.get("seen_contact_ids") or [])}
    pending_created = False

    if state.get("steps", 0) >= max_steps:
        out.step_cap_hit = True
        for call in tool_calls:
            out.decisions.append(
                _block(call, ErrorCode.STEP_CAP, f"Step limit ({max_steps}) reached; the call was not run.",
                       "Stop calling tools and tell the user what was and was not done.")
            )
        out.pending_action = pending.model_dump(mode="json") if pending else None
        return out

    for call in tool_calls:
        name, raw = call["name"], call.get("args") or {}
        tool = tools.get(name)
        if tool is None:
            out.decisions.append(_block(call, ErrorCode.UNKNOWN_TOOL, f"There is no tool named '{name}'.",
                                        "Available tools: " + ", ".join(sorted(tools))))
            continue
        group = group_of(tool)
        if group not in out.groups:
            out.groups.append(group)
            out.recoveries.append(f"{name} belongs to the '{group}' group which was not enabled; enabling it for this turn")
            out.decisions.append(_block(call, ErrorCode.TOOL_NOT_ACTIVE,
                                        f"The '{group}' tool group was not enabled for this request.",
                                        f"It is enabled now. Call {name} again if it is the right tool."))
            continue
        parsed, error = tool.metadata["validate"](raw)
        if parsed is None:
            out.decisions.append(_block(call, ErrorCode.INVALID_ARGS, f"Invalid arguments for {name}: {error}",
                                        "Fix the listed fields and call the tool again."))
            continue
        id_field = _ID_PROVENANCE.get(name)
        if id_field:  # ids that act on existing records must have come out of an earlier tool result
            field_name, seen_key, source_tool = id_field
            value = getattr(parsed, field_name, None)
            if value and value not in seen[seen_key]:
                out.decisions.append(_block(call, ErrorCode.INVALID_ARGS,
                                            f"{field_name} '{value}' was not returned by any earlier tool result.",
                                            f"Call {source_tool} first and copy the {field_name} from its result; never guess ids."))
                continue
        canon = canonical_args(parsed)
        sig = call_signature(name, canon)
        if sig in out.call_signatures:
            out.decisions.append(_block(call, ErrorCode.DUPLICATE_CALL,
                                        "This exact call was already made in this turn.",
                                        "Reuse the earlier result instead of repeating the call."))
            continue

        if is_write_tool(tool):
            if pending and pending.tool == name and pending.args == canon and pending.executable_on(turn):
                out.decisions.append(Decision(call=call, approved=True, reason="confirmed"))
                out.call_signatures.append(sig)
                continue
            if pending_created:
                out.decisions.append(_block(call, ErrorCode.CONFIRMATION_REQUIRED,
                                            "Only one write action can be proposed at a time.",
                                            "Ask the user to confirm the first action; handle this one afterwards."))
                continue
            if pending and pending.tool == name and pending.args == canon and not pending.confirmed:
                same_proposal = True
            else:
                same_proposal = False
            summary = describe_write(name, parsed, state)
            if not same_proposal:
                pending = PendingAction(
                    tool=name, args=canon, call_args=parsed.model_dump(mode="json", exclude_none=True),
                    summary=summary, proposed_turn=turn,
                )
                out.pending_event = "replaced" if state.get("pending_action") else "created"
            pending_created = True
            out.decisions.append(_block(
                call, ErrorCode.CONFIRMATION_REQUIRED,
                f"Not executed: the user has not confirmed this action yet ({summary}).",
                "Restate this exact action to the user in plain words and ask them to confirm. Do not call the tool "
                "again until the user says yes in their next message.",
                detail=summary,
            ))
            continue

        out.decisions.append(Decision(call=call, approved=True))
        out.call_signatures.append(sig)

    out.pending_action = pending.model_dump(mode="json") if pending else None
    return out


# --------------------------------------------------------------------------- final-answer integrity
_CLAIM = re.compile(
    r"(?:\bi(?:'ve|\s+have)\s+(?:just\s+|now\s+|successfully\s+|gone\s+ahead\s+and\s+)?"
    r"(?:booked|scheduled|created|added|deleted|removed|cancel(?:l)?ed|updated|rescheduled|moved|changed)\b"
    r"|\b(?:has|have)\s+been\s+(?:successfully\s+)?(?:booked|scheduled|created|added|deleted|removed|cancel(?:l)?ed|"
    r"updated|rescheduled|moved)\b"
    r"|\bsuccessfully\s+(?:booked|scheduled|created|added|deleted|removed|cancel(?:l)?ed|updated|rescheduled)\b"
    r"|\ball\s+set\b|\bit'?s\s+(?:booked|done|scheduled)\b|✅)",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"(?:would\s+you\s+like|shall\s+i|should\s+i|do\s+you\s+want|want\s+me\s+to|please\s+confirm|can\s+you\s+confirm|"
    r"confirm\s+(?:this|that|if)|before\s+i)",
    re.IGNORECASE,
)


def write_outcomes(messages: list[BaseMessage], write_tools: set[str]) -> list[tuple[str, bool, dict[str, Any] | None]]:
    """(tool, ok, parsed envelope) for every write-tool result in this turn."""
    out = []
    for m in this_turn_messages(messages):
        if isinstance(m, ToolMessage) and m.name in write_tools:
            env = parse_tool_message(m)
            out.append((m.name, bool(env and env.get("ok")), env))
    return out


def claims_unverified_success(text: str, outcomes: list[tuple[str, bool, dict[str, Any] | None]]) -> bool:
    """True if ``text`` claims a write completed but no write succeeded this turn."""
    if not text or _HEDGE.search(text) or not _CLAIM.search(text):
        return False
    return not any(ok for _, ok, _ in outcomes)


def honest_failure_message(outcomes: list[tuple[str, bool, dict[str, Any] | None]]) -> str:
    failed = [env for _, ok, env in outcomes if not ok and env]
    if failed:
        err = (failed[-1].get("error") or {})
        reason = err.get("message") or "the action did not complete"
        if err.get("code") == ErrorCode.CONFIRMATION_REQUIRED.value:
            return ("I haven't made that change yet - I need your confirmation first. "
                    "Please confirm the action and I'll go ahead.")
        return (f"I wasn't able to complete that: {reason} Nothing was changed. "
                "Would you like me to try again or take a different approach?")
    return "I haven't made any change yet. Tell me to go ahead (after confirming the details) and I'll do it."
