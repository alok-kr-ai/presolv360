"""Deterministic working-memory updates from tool results.

Updates are per key, so a turn on one topic never wipes memory from another.
"""
from __future__ import annotations

from typing import Any

from tools.base import ToolResult

MAX_LISTED = 10


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "start", "end", "attendees", "location")
    return {k: event[k] for k in keys if event.get(k) not in (None, [], "")}


def _event_ids(data: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(data, dict):
        return ids
    for key in ("events", "conflicts"):
        for ev in data.get(key) or []:
            if isinstance(ev, dict) and ev.get("id"):
                ids.append(ev["id"])
    ev = data.get("event")
    if isinstance(ev, dict) and ev.get("id"):
        ids.append(ev["id"])
    if data.get("event_id"):
        ids.append(data["event_id"])
    return ids


def _contact_ids(data: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(data, dict):
        return ids
    for key in ("contact", "existing"):
        c = data.get(key)
        if isinstance(c, dict) and c.get("id"):
            ids.append(c["id"])
    for c in data.get("candidates") or []:
        if isinstance(c, dict) and c.get("id"):
            ids.append(c["id"])
    if data.get("contact_id"):
        ids.append(data["contact_id"])
    return ids


def apply_tool_result(
    tool: str, args: dict[str, Any], result: ToolResult, memory: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Return (state updates, changed memory keys)."""
    updates: dict[str, Any] = {}
    changed: list[str] = []
    data = result.data if isinstance(result.data, dict) else {}

    # ids seen in any tool output (even errors, e.g. CONFLICT lists conflicting events)
    new_ids = _event_ids(data)
    if new_ids:
        seen = list(memory.get("seen_event_ids") or [])
        for i in new_ids:
            if i not in seen:
                seen.append(i)
        updates["seen_event_ids"] = seen

    new_contact_ids = _contact_ids(data)
    if new_contact_ids:
        seen_c = list(memory.get("seen_contact_ids") or [])
        for i in new_contact_ids:
            if i not in seen_c:
                seen_c.append(i)
        updates["seen_contact_ids"] = seen_c

    listed = list(memory.get("listed_events") or [])

    if tool == "list_calendar_events" and result.ok:
        events = [compact_event(e) for e in data.get("events", [])][:MAX_LISTED]
        updates["listed_events"] = events
        changed.append("listed_events")
        if len(events) == 1:
            updates["last_event"] = events[0]
            changed.append("last_event")
    elif tool in ("create_calendar_event", "update_calendar_event") and result.ok:
        event = compact_event(data.get("event", {}))
        if event:
            updates["last_event"] = event
            updates["listed_events"] = [event if e.get("id") == event["id"] else e for e in listed]
            if all(e.get("id") != event["id"] for e in listed):
                updates["listed_events"].append(event)
            changed += ["last_event", "listed_events"]
    elif tool == "delete_calendar_event" and result.ok:
        event_id = data.get("event_id")
        known = next((e for e in [memory.get("last_event") or {}, *listed] if e.get("id") == event_id), None)
        updates["listed_events"] = [e for e in listed if e.get("id") != event_id]
        changed.append("listed_events")
        last = memory.get("last_event") or {}
        if last.get("id") == event_id or known:
            updates["last_event"] = {**(known or {"id": event_id}), "status": "deleted"}
            changed.append("last_event")
    elif tool in ("get_contact", "create_contact", "update_contact") and result.ok:
        contact = data.get("contact") or {}
        keep = {k: contact[k] for k in ("id", "name", "email", "company", "title") if contact.get(k)}
        if keep:
            updates["last_contact"] = keep
            changed.append("last_contact")
    elif tool == "delete_contact" and result.ok:
        contact_id = data.get("contact_id")
        last = memory.get("last_contact") or {}
        if last.get("id") == contact_id:
            updates["last_contact"] = {**last, "status": "deleted"}
            changed.append("last_contact")
        elif contact_id:
            updates["last_contact"] = {"id": contact_id, "name": data.get("name"), "status": "deleted"}
            changed.append("last_contact")
    elif tool == "calculate" and result.ok:
        updates["last_calculation"] = {"expression": data.get("expression"), "result": data.get("result")}
        changed.append("last_calculation")
    return updates, changed
