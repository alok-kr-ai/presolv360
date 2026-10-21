"""Per-run structured logs: layout, trace ids, metrics, redaction and degradation."""
import json
from pathlib import Path

import pytest
from rich.console import Console

from graph.nodes import Nodes
from observability import report
from observability.metrics import percentile
from tests.conftest import tomorrow
from tests.fakes import Script, tc

LIST_TOMORROW = dict(start=tomorrow(0), end=tomorrow(23, 59))


def read_events(agent) -> list[dict]:
    path = agent.recorder.run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_script_turns(make_agent, **overrides):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW), "You have two events.",
        tc("get_contact", name="Zed Nobody"), "No such contact.",
        "Hello!",
    ])
    agent, script = make_agent(script, **overrides)
    results = [agent.invoke(t) for t in ("what's on my calendar tomorrow?", "who is Zed Nobody?", "hi")]
    return agent, results


def test_run_folder_layout(make_agent):
    agent, results = run_script_turns(make_agent)
    agent.close()
    run_dir = agent.recorder.run_dir
    assert run_dir.name == agent.run_id and run_dir.parent == agent.settings.log_dir
    assert {p.name for p in run_dir.iterdir()} == {"run.json", "events.jsonl", "summary.json", "turns"}
    turn_files = sorted(p.name for p in (run_dir / "turns").iterdir())
    assert len(turn_files) == 3 and turn_files[0].startswith("turn-001-") and turn_files[0].endswith(f"{results[0].trace_id}.json")
    run = json.loads((run_dir / "run.json").read_text())
    assert run["run_id"] == agent.run_id and run["status"] == "completed" and run["model"] == "fake-model"
    assert agent.thread_id in run["threads"] and run["ended_at"]


def test_two_runs_get_distinct_folders(make_agent):
    a, _ = make_agent(Script(["hi"]))
    b, _ = make_agent(Script(["hi"]))
    assert a.run_id != b.run_id and a.recorder.run_dir != b.recorder.run_dir
    assert a.recorder.run_dir.exists() and b.recorder.run_dir.exists()


def test_events_are_valid_json_with_ids_and_one_trace_per_turn(make_agent):
    agent, results = run_script_turns(make_agent)
    events = read_events(agent)
    assert events and all({"run_id", "trace_id", "turn", "event", "ts"} <= set(e) for e in events)
    assert {e["run_id"] for e in events} == {agent.run_id}
    for r in results:
        turn_events = [e for e in events if e["turn"] == r.turn]
        assert {e["trace_id"] for e in turn_events} == {r.trace_id}
        assert turn_events[0]["event"] == "turn_start" and turn_events[-1]["event"] == "turn_end"
    assert len({r.trace_id for r in results}) == 3
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


def test_tool_and_llm_events_carry_performance_fields(make_agent):
    agent, _ = run_script_turns(make_agent)
    events = read_events(agent)
    tool = next(e for e in events if e["event"] == "tool_result" and e["tool"] == "list_calendar_events")
    assert tool["ok"] is True and tool["latency_ms"] >= 0 and tool["attempt"] == 1
    failed = next(e for e in events if e["event"] == "tool_result" and e["tool"] == "get_contact")
    assert failed["ok"] is False and failed["error_code"] == "NOT_FOUND" and failed["failure_reason"]
    llm = next(e for e in events if e["event"] == "llm_end" and e["name"] == "agent")
    assert llm["model"] == "fake-model" and llm["fallback_used"] is False and llm["latency_ms"] >= 0
    assert llm["prompt_tokens"] > 0 and llm["completion_tokens"] > 0


def test_retries_and_guard_blocks_are_logged_with_reasons(make_agent):
    script = Script([
        tc("delete_calendar_event", event_id="evt_002"),  # unseen id -> blocked
        tc("list_calendar_events", **LIST_TOMORROW), "Done.",
    ])
    agent, _ = make_agent(script)
    agent.backend.fail_next("list_calendar_events", "SERVER_ERROR", 1)
    agent.invoke("delete my standup from the calendar")
    events = read_events(agent)
    retry = next(e for e in events if e["event"] == "tool_retry")
    assert retry["tool"] == "list_calendar_events" and retry["attempt"] == 1 and retry["error_code"] == "SERVER_ERROR"
    assert "backoff_s" in retry["data"]
    block = next(e for e in events if e["event"] == "guard_block")
    assert block["guard_reason"] == "INVALID_ARGS" and "was not returned" in block["failure_reason"]
    assert {e["event"] for e in events} >= {"route", "memory_update", "final"}


def test_turn_file_has_span_tree_flow_and_metrics(make_agent):
    agent, results = run_script_turns(make_agent)
    trace = json.loads(next((agent.recorder.run_dir / "turns").glob("turn-002-*.json")).read_text())
    assert trace["trace_id"] == results[1].trace_id and trace["user_text"] == "who is Zed Nobody?"
    assert trace["flow"][:3] == ["intake", "route", "agent"] and "tools(get_contact err:NOT_FOUND)" in trace["flow"]
    assert trace["flow"][-1] == "final"
    root = trace["spans"]
    assert root["name"] == "turn" and root["children"]
    names = []

    def walk(sp):
        names.append((sp["component"], sp["name"]))
        for child in sp["children"]:
            walk(child)

    walk(root)
    assert ("tool", "get_contact") in names and any(c == "llm" for c, _ in names)
    assert trace["metrics"]["tool_calls"] == 1 and trace["metrics"]["errors"] == 1
    assert trace["tool_calls"][0]["error_code"] == "NOT_FOUND" and trace["events"]


def test_summary_has_per_tool_and_model_stats(make_agent):
    agent, _ = run_script_turns(make_agent)
    summary = json.loads((agent.recorder.run_dir / "summary.json").read_text())
    assert summary["totals"]["turns"] == 3 and summary["totals"]["tool_calls"] == 2
    assert summary["tools"]["get_contact"]["errors"] == {"NOT_FOUND": 1}
    assert summary["tools"]["list_calendar_events"]["ok"] == 1
    assert {"avg_ms", "p50_ms", "p95_ms"} <= set(summary["tools"]["list_calendar_events"])
    assert summary["llm"]["fake-model"]["calls"] >= 5
    assert summary["turn_mix"]["no_tool_turns"] == 1 and summary["turn_mix"]["tool_turns"] == 2
    assert summary["router"]["methods"]["fast_path"] == 1


def test_percentile():
    assert percentile([], 50) == 0.0 and percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([1, 2, 3, 4], 100) == 4


def test_a_crash_leaves_a_readable_log_and_a_non_running_status(make_agent, monkeypatch):
    def boom(self, state, config):
        raise RuntimeError("bug in route")

    monkeypatch.setattr(Nodes, "route", boom)
    agent, _ = make_agent(Script(["never reached"]))
    with pytest.raises(RuntimeError, match="bug in route"):
        agent.invoke("what is 2 + 2")
    lines = read_events(agent)
    kinds = [e["event"] for e in lines]
    assert kinds[0] == "turn_start" and "turn_error" in kinds and kinds[-1] == "turn_end"
    run = json.loads((agent.recorder.run_dir / "run.json").read_text())
    assert run["status"] == "crashed"
    assert next(e for e in lines if e["event"] == "turn_error")["failure_reason"] == "bug in route"
    assert list((agent.recorder.run_dir / "turns").glob("turn-001-*.json"))


def test_unclosed_run_is_marked_interrupted_at_exit(make_agent):
    agent, _ = make_agent(Script(["hi"]))
    agent.invoke("hi")
    agent.recorder._atexit()
    assert json.loads((agent.recorder.run_dir / "run.json").read_text())["status"] == "interrupted"


def test_secrets_never_reach_the_logs(make_agent, monkeypatch):
    secret = "sk-test-SECRET-1234567890abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    script = Script([f"Your key {secret} is not something I store."])
    agent, _ = make_agent(script)
    agent.invoke(f"my api key is {secret}, remember it")
    agent.close()
    for path in agent.recorder.run_dir.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8"), path
    assert "***" in (agent.recorder.run_dir / "events.jsonl").read_text()


def test_unwritable_log_dir_degrades_with_a_warning(make_agent, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    agent, _ = make_agent(Script(["Hello!"]), log_dir=blocker / "logs")
    assert agent.recorder.enabled is False and "disabled" in (agent.recorder.warning or "")
    res = agent.invoke("hi")
    assert res.text == "Hello!" and res.status == "completed"
    assert agent.recorder.turn_events(res.trace_id)  # still kept in memory for the /trace command


def test_log_content_false_removes_message_text(make_agent):
    script = Script([tc("get_contact", name="Rahul Verma"), "Rahul works at Globex."])
    agent, _ = make_agent(script, log_content=False)
    agent.invoke("who is Rahul Verma?")
    agent.close()
    blob = "".join(p.read_text() for p in agent.recorder.run_dir.rglob("*.json*"))
    assert "Rahul" not in blob and "Globex" not in blob
    events = read_events(agent)
    assert any(e["event"] == "tool_result" and e["tool"] == "get_contact" for e in events)  # structure is kept


def test_report_renderers_work_on_a_recorded_run(make_agent):
    agent, results = run_script_turns(make_agent)
    agent.close()
    log_dir: Path = agent.settings.log_dir
    console = Console(record=True, width=140, force_terminal=False)
    report.show_runs(console, log_dir)
    assert agent.run_id in console.export_text()
    console = Console(record=True, width=140, force_terminal=False)
    assert report.show_run(console, log_dir, agent.run_id)
    text = console.export_text()
    assert "get_contact" in text and "NOT_FOUND" in text
    console = Console(record=True, width=140, force_terminal=False)
    assert report.show_turn(console, log_dir, agent.run_id, 2)
    assert "get_contact" in console.export_text()
    assert not report.show_run(Console(record=True), log_dir, "no-such-run")
    assert not report.show_turn(Console(record=True), log_dir, agent.run_id, 99)
