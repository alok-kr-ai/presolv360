"""Unit tests for working-memory updates and the pure-Python guard helpers."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph.guards import (
    claims_unverified_success,
    is_chitchat,
    quick_classify_reply,
    sanitize_history,
)
from memory import apply_tool_result
from tools.base import ErrorCode, ToolResult


@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "thanks", "Thank you!", "good morning", "bye", "ok thanks"])
def test_chitchat_fast_path(text):
    assert is_chitchat(text)


@pytest.mark.parametrize("text", ["hi, what's on my calendar?", "thanks, now delete it", "what is 2+2", "hello Priya's email please"])
def test_not_chitchat_when_there_is_a_request(text):
    assert not is_chitchat(text)


@pytest.mark.parametrize(
    "text,decision",
    [("yes", "yes"), ("Yes, go ahead", "yes"), ("sure, do it", None), ("confirm", "yes"), ("no", "no"), ("No thanks", "no"),
     ("cancel that", "no"), ("thanks", "unrelated"), ("hi", "unrelated"), ("what is the weather", None), ("yes but make it 4pm", None)],
)
def test_quick_reply_classification(text, decision):
    assert quick_classify_reply(text) == decision


def test_apply_tool_result_updates_only_relevant_keys():
    mem = {"last_event": None, "last_contact": None, "last_calculation": None, "listed_events": None, "seen_event_ids": None}
    ok = ToolResult.success({"result": 6, "expression": "2*3"})
    updates, changed = apply_tool_result("calculate", {"expression": "2*3"}, ok, mem)
    assert changed == ["last_calculation"] and updates["last_calculation"]["result"] == 6 and "last_event" not in updates

    failed = ToolResult.failure(ErrorCode.NOT_FOUND, "no such contact", retryable=False)
    updates, changed = apply_tool_result("get_contact", {"name": "x"}, failed, mem)
    assert updates == {} and changed == []  # failures never overwrite memory


def test_listing_events_sets_provenance_and_single_result_becomes_last_event():
    events = [{"id": "evt_002", "title": "Team standup", "start": "2026-10-22T10:00:00+05:30", "end": "2026-10-22T10:30:00+05:30"}]
    updates, changed = apply_tool_result("list_calendar_events", {}, ToolResult.success({"events": events, "count": 1}), {})
    assert "evt_002" in updates["seen_event_ids"] and updates["last_event"]["id"] == "evt_002"
    assert {"listed_events", "last_event"} <= set(changed)


def test_sanitize_history_repairs_interrupted_tool_calls():
    ai = AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "c1", "type": "tool_call"}])
    broken = [HumanMessage("1+1?"), ai, HumanMessage("hello?")]  # the tool message is missing
    fixed = sanitize_history(broken)
    assert [type(m).__name__ for m in fixed] == ["HumanMessage", "AIMessage", "ToolMessage", "HumanMessage"]
    assert fixed[2].tool_call_id == "c1" and "interrupted" in fixed[2].content

    orphan = [HumanMessage("hi"), ToolMessage(content="x", tool_call_id="zzz", name="calculate")]
    assert [type(m).__name__ for m in sanitize_history(orphan)] == ["HumanMessage"]


def test_claim_detection():
    none_ok: list = []
    assert claims_unverified_success("I've booked your meeting for 3pm.", none_ok)
    assert claims_unverified_success("Done! ✅", none_ok)
    assert not claims_unverified_success("Shall I book it for 3pm?", none_ok)
    assert not claims_unverified_success("Would you like me to schedule it? I haven't booked anything.", none_ok)
    assert not claims_unverified_success("I've booked your meeting.", [("create_calendar_event", True, {"ok": True})])
    assert not claims_unverified_success("The weather is nice.", none_ok)
