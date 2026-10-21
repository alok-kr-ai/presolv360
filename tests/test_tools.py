import time

from pydantic import BaseModel

from mocks import Backend
from tests.conftest import NOW, tomorrow
from tools import ALL_TOOLS, GROUP_DESC, TOOL_GROUPS, build_tool_groups, tools_by_name
from tools.base import ErrorCode, StrictArgs, ToolFailure, execute_tool


def run(backend, tool_name, /, **args):
    tools = tools_by_name(build_tool_groups(backend))
    return tools[tool_name].metadata["runner"](args)


def test_registry_is_derived_from_groups():
    assert set(TOOL_GROUPS) == set(GROUP_DESC) == {"calendar", "crm", "calculator"}
    assert [t.name for t in ALL_TOOLS] == [t.name for g in TOOL_GROUPS.values() for t in g]
    names = {t.name for t in ALL_TOOLS}
    assert names == {
        "list_calendar_events", "check_availability", "create_calendar_event", "update_calendar_event",
        "delete_calendar_event", "get_contact", "list_interactions", "create_contact", "update_contact", "delete_contact", "calculate",
    }
    for tool in ALL_TOOLS:  # verb_noun names with explicit when-to-use / when-not-to guidance
        assert tool.name.islower() and "DO NOT" in tool.description


def test_uniform_result_shape(settings, clock):
    b = Backend(settings, clock=clock)
    ok = run(b, "calculate", expression="2*3").model_dump(mode="json")
    assert set(ok) >= {"ok", "data", "error"} and ok["ok"] is True and ok["error"] is None
    bad = run(b, "get_contact").model_dump(mode="json")
    assert bad["ok"] is False and bad["data"] is None
    assert set(bad["error"]) == {"code", "message", "retryable", "hint"}


def test_wrapper_converts_unexpected_exceptions_to_internal():
    class Args(StrictArgs):
        x: int = 1

    def boom(args, ctx):
        raise RuntimeError("secret internals /home/x/y.py line 3")

    res = execute_tool("boom", Args, boom, {})
    assert not res.ok and res.error.code is ErrorCode.INTERNAL and not res.error.retryable
    assert "secret internals" not in res.to_json() and "Traceback" not in res.to_json()
    assert "RuntimeError" in (res.debug or "")  # kept for the log only


def test_wrapper_timeout():
    class Args(StrictArgs):
        pass

    res = execute_tool("slow", Args, lambda a, c: time.sleep(0.5), {}, timeout=0.05)
    assert res.error.code is ErrorCode.TIMEOUT and res.error.retryable


def test_wrapper_truncates_oversized_results():
    class Args(StrictArgs):
        pass

    big = {"events": [{"id": f"evt_{i}", "title": "x" * 80} for i in range(500)]}
    res = execute_tool("big", Args, lambda a, c: big, {}, max_chars=2000)
    assert res.ok and res.meta == {"truncated": True}
    assert 0 < len(res.data["events"]) < 500
    assert '"truncated": true' in res.to_json()


def test_wrapper_maps_tool_failure_and_extra_args():
    class Args(StrictArgs):
        a: int

    def fail(args, ctx):
        raise ToolFailure(ErrorCode.RATE_LIMITED, "slow down", retry_after_s=2)

    res = execute_tool("t", Args, fail, {"a": 1})
    assert res.error.code is ErrorCode.RATE_LIMITED and res.error.retryable and res.meta == {"retry_after_s": 2}
    res = execute_tool("t", Args, fail, {"a": 1, "zzz": 2})  # unknown field rejected (strict schema)
    assert res.error.code is ErrorCode.INVALID_ARGS


def test_invalid_datetimes_and_ranges(settings, clock):
    b = Backend(settings, clock=clock)
    naive = run(b, "list_calendar_events", start="2026-10-22T10:00:00", end="2026-10-22T11:00:00")
    assert naive.error.code is ErrorCode.INVALID_ARGS and "ISO 8601" in naive.error.hint
    inverted = run(b, "check_availability", start=tomorrow(12), end=tomorrow(11))
    assert inverted.error.code is ErrorCode.INVALID_ARGS
    bad_create = run(b, "create_calendar_event", title="x", start=tomorrow(12), end=tomorrow(12))
    assert bad_create.error.code is ErrorCode.INVALID_ARGS


def test_create_conflict_and_idempotent_request_id(settings, clock):
    b = Backend(settings, clock=clock)
    clash = run(b, "create_calendar_event", title="Clash", start=tomorrow(10), end=tomorrow(11))
    assert clash.error.code is ErrorCode.CONFLICT and clash.data["conflicts"][0]["title"] == "Team standup"

    tools = tools_by_name(build_tool_groups(b))
    create = tools["create_calendar_event"].metadata["runner"]
    args = {"title": "Focus time", "start": tomorrow(12), "end": tomorrow(13)}
    first = create(args, request_id="req-1")
    again = create(args, request_id="req-1")
    assert first.ok and again.ok and again.data.get("idempotent_replay")
    assert first.data["event"]["id"] == again.data["event"]["id"]
    assert sum(1 for e in b.calendar.all_events() if e.title == "Focus time") == 1


def test_delete_really_deletes_and_is_consistent(settings, clock):
    b = Backend(settings, clock=clock)
    rng = {"start": tomorrow(10), "end": tomorrow(10, 30)}
    assert run(b, "check_availability", **rng).data["available"] is False
    res = run(b, "delete_calendar_event", event_id="evt_002")
    assert res.ok and res.data["status"] == "deleted"
    assert run(b, "check_availability", **rng).data["available"] is True
    listed = run(b, "list_calendar_events", start=tomorrow(0), end=tomorrow(23, 59)).data["events"]
    assert "evt_002" not in [e["id"] for e in listed]
    again = run(b, "delete_calendar_event", event_id="evt_002")
    assert again.ok and again.data["status"] == "already_gone"


def test_delete_error_codes(settings, clock):
    b = Backend(settings, clock=clock)
    nf = run(b, "delete_calendar_event", event_id="evt_999")
    assert nf.error.code is ErrorCode.NOT_FOUND and not nf.error.retryable
    bad = run(b, "delete_calendar_event", event_id="not-an-id")
    assert bad.error.code is ErrorCode.INVALID_ARGS
    extra = run(b, "delete_calendar_event", event_id="evt_002", title="x")
    assert extra.error.code is ErrorCode.INVALID_ARGS  # delete only takes event_id
    b.fail_next("delete_calendar_event", "SERVER_ERROR")
    boom = run(b, "delete_calendar_event", event_id="evt_002")
    assert boom.error.code is ErrorCode.SERVER_ERROR and boom.error.retryable
    assert b.calendar.get("evt_002") is not None  # a failed delete deletes nothing


def test_update_event(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "update_calendar_event", event_id="evt_002", title="Daily sync")
    assert res.ok and res.data["event"]["title"] == "Daily sync"
    assert run(b, "update_calendar_event", event_id="evt_002").error.code is ErrorCode.INVALID_ARGS
    clash = run(b, "update_calendar_event", event_id="evt_002", start=tomorrow(14), end=tomorrow(15))
    assert clash.error.code is ErrorCode.CONFLICT
    assert run(b, "update_calendar_event", event_id="evt_404", title="x").error.code is ErrorCode.NOT_FOUND


def test_crm_contract(settings, clock):
    b = Backend(settings, clock=clock)
    assert run(b, "get_contact").error.code is ErrorCode.INVALID_ARGS
    assert run(b, "get_contact", name="").error.code is ErrorCode.INVALID_ARGS
    found = run(b, "get_contact", email="Rahul.Verma@globex.com")
    assert found.ok and found.data["contact"]["id"] == "con_003"
    assert run(b, "get_contact", name="Priya Sharma").data["contact"]["id"] == "con_001"
    amb = run(b, "get_contact", name="priya")
    assert amb.error.code is ErrorCode.AMBIGUOUS and len(amb.data["candidates"]) == 2
    nf = run(b, "get_contact", name="Zed Nobody")
    assert nf.error.code is ErrorCode.NOT_FOUND and "spelling" in nf.error.hint


def test_list_interactions(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "list_interactions", contact_id="con_001", limit=2)
    assert res.ok and res.data["count"] == 2
    dates = [i["date"] for i in res.data["interactions"]]
    assert dates == sorted(dates, reverse=True)
    empty = run(b, "list_interactions", contact_id="con_004")  # known contact, no history: ok, not an error
    assert empty.ok and empty.data["interactions"] == [] and "note" in empty.data
    assert run(b, "list_interactions", contact_id="con_999").error.code is ErrorCode.NOT_FOUND
    assert run(b, "list_interactions", contact_id="bogus").error.code is ErrorCode.INVALID_ARGS


def test_fault_injection_codes_and_after_commit(settings, clock):
    b = Backend(settings, clock=clock)
    b.fail_next("get_contact", "TIMEOUT", 2)
    assert run(b, "get_contact", name="John").error.code is ErrorCode.TIMEOUT
    assert run(b, "get_contact", name="John").error.code is ErrorCode.TIMEOUT
    assert run(b, "get_contact", name="John").ok
    b.fail_next("create_calendar_event", "SERVER_ERROR", after_commit=True)
    res = run(b, "create_calendar_event", title="Ghost", start=tomorrow(12), end=tomorrow(13))
    assert res.error.code is ErrorCode.SERVER_ERROR
    assert any(e.title == "Ghost" for e in b.calendar.all_events())  # committed, then the response "failed"


def test_seeded_failure_rate_is_deterministic(settings, clock):
    def codes(seed):
        b = Backend(settings.model_copy(update={"mock_failure_rate": 0.5, "mock_seed": seed}), clock=clock)
        return [run(b, "get_contact", name="John").ok for _ in range(12)]

    assert codes(3) == codes(3) and not all(codes(3)) and any(codes(3))


def test_base_schema_models():
    assert issubclass(StrictArgs, BaseModel)
    assert NOW.tzinfo is not None
