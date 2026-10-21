"""Realtime event stream and the rich renderers that consume it."""
from rich.console import Console

from tests.conftest import tomorrow
from tests.fakes import Script, tc
from ui.render import TurnView, render_turn

LIST_TOMORROW = dict(start=tomorrow(0), end=tomorrow(23, 59))
TOOL_SCRIPT = lambda: Script([tc("list_calendar_events", **LIST_TOMORROW), "You have two events tomorrow."])  # noqa: E731


def kinds(events):
    return [e.event for e in events]


def test_stream_event_order_for_a_tool_turn(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    events = list(agent.stream("what's on my calendar tomorrow?"))
    order = kinds(events)
    expected = ["turn_start", "node_end", "route", "tool_proposed", "tool_start", "tool_result", "final", "turn_end"]
    positions = [order.index(k) for k in expected]
    assert positions == sorted(positions)
    assert order[0] == "turn_start" and order[-1] == "turn_end"
    assert order.count("llm_start") == order.count("llm_end") >= 3  # route + two agent steps


def test_answer_tokens_are_streamed_and_match_the_final_text(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    events = list(agent.stream("what's on my calendar tomorrow?"))
    tokens = "".join(e.text for e in events if e.event == "token")
    assert tokens == "You have two events tomorrow."
    assert next(e for e in events if e.event == "final").text == tokens


def test_every_event_has_the_trace_id_and_turn(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    events = list(agent.stream("what's on my calendar tomorrow?"))
    assert len({e.trace_id for e in events}) == 1 and all(e.trace_id for e in events)
    assert {e.turn for e in events} == {1} and {e.run_id for e in events} == {agent.run_id}


def test_retried_tool_emits_tool_retry_events(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    agent.backend.fail_next("list_calendar_events", "SERVER_ERROR", 2)
    order = kinds(agent.invoke("what's on my calendar tomorrow?").events)
    assert order.count("tool_retry") == 2 and order.count("tool_start") == 1 and order.count("tool_result") == 1
    assert order.index("tool_retry") < order.index("tool_result")


def test_no_tool_turn_emits_no_tool_events(make_agent):
    agent, _ = make_agent(Script(["Hello!"]))
    order = kinds(agent.invoke("hello").events)
    assert not any(k.startswith("tool_") or k == "guard_block" for k in order)
    assert "route" in order and "final" in order


def test_events_streamed_live_are_also_logged(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    live = [e for e in agent.stream("what's on my calendar tomorrow?") if e.event != "token"]
    logged = agent.recorder.turn_events()
    assert [e.event for e in live] == [e["event"] for e in logged]


def test_invoke_and_stream_reach_the_same_state(make_agent):
    a1, _ = make_agent(TOOL_SCRIPT())
    a2, _ = make_agent(TOOL_SCRIPT())
    res = a1.invoke("what's on my calendar tomorrow?")
    list(a2.stream("what's on my calendar tomorrow?"))
    s1, s2 = a1.state_values(), a2.state_values()
    assert [m.content for m in s1["messages"] if m.type == "ai"][-1] == [m.content for m in s2["messages"] if m.type == "ai"][-1]
    assert res.text == "You have two events tomorrow."
    for key in ("turn", "steps", "groups", "listed_events"):
        assert s1[key] == s2[key]


def record(make_agent, mode, verbose=False, script=None, text="what's on my calendar tomorrow?"):
    agent, _ = make_agent(script or TOOL_SCRIPT())
    console = Console(record=True, width=120, force_terminal=False)
    view = render_turn(console, agent.stream(text), mode=mode, verbose=verbose)
    return console.export_text(), view


def test_plain_render_shows_the_flow_and_the_answer(make_agent):
    out, view = record(make_agent, "plain")
    assert "list_calendar_events" in out and "route" in out.lower()
    assert "You have two events tomorrow." in out
    assert view.answer_panel() is not None


def test_plain_verbose_shows_tool_arguments_and_llm_details(make_agent):
    out, _ = record(make_agent, "plain", verbose=True)
    assert "start" in out and "fake-model" in out


def test_quiet_render_only_prints_the_answer(make_agent):
    out, _ = record(make_agent, "quiet")
    assert "You have two events tomorrow." in out and "list_calendar_events" not in out


def test_live_mode_falls_back_to_plain_off_a_terminal(make_agent):
    out, _ = record(make_agent, "live")
    assert "list_calendar_events" in out and "You have two events tomorrow." in out


def test_render_shows_retries_blocks_and_pending_confirmation(make_agent):
    create = dict(title="Sync", start=tomorrow(15), end=tomorrow(16))
    script = Script([tc("create_calendar_event", **create), "Please confirm the booking."])
    out, view = record(make_agent, "plain", script=script, text="schedule a sync tomorrow at 3pm")
    assert "CONFIRMATION_REQUIRED" in out or "awaiting confirmation" in out.lower() or "confirm" in out.lower()
    assert "Create event" in out  # the pending action summary is shown, not just the model's wording


def test_turn_view_folds_a_full_turn_into_a_tree_and_summary(make_agent):
    agent, _ = make_agent(TOOL_SCRIPT())
    view = TurnView()
    for ev in agent.stream("what's on my calendar tomorrow?"):
        view.apply(ev)
    console = Console(record=True, width=120, force_terminal=False)
    console.print(view.timeline())
    console.print(view.summary_line())
    out = console.export_text()
    assert "list_calendar_events" in out and "calendar" in out and "turn 1" in out
    assert not any(s.status == "running" for s in view.steps)  # nothing left spinning
