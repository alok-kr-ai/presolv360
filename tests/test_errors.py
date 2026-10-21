"""Tool errors, retries, guards against loops and the final-answer integrity check."""
from tests.conftest import tomorrow
from tests.fakes import Script, event_id_of, reissue_confirmed, tc

LIST_TOMORROW = dict(start=tomorrow(0), end=tomorrow(23, 59))


def final_text_ok(call):  # step: answer from whatever the last tool said
    return "ok" if all(env["ok"] for _, env in call.tool_results()) else "problem"


def test_retryable_error_is_retried_then_succeeds(make_agent):
    script = Script([tc("list_calendar_events", **LIST_TOMORROW), "You have 2 events tomorrow."])
    agent, _ = make_agent(script)
    agent.backend.fail_next("list_calendar_events", "SERVER_ERROR", 1)
    res = agent.invoke("what's on my calendar tomorrow?")
    retries = [e for e in res.events if e.event == "tool_retry"]
    assert len(retries) == 1 and retries[0].error_code == "SERVER_ERROR" and retries[0].attempt == 1
    assert res.tool_events[0].ok and res.tool_events[0].attempt == 2
    assert len(agent.sleeps) == 1  # backoff was requested (and skipped in tests)


def test_retries_are_bounded_and_failure_is_reported_honestly(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW),
        lambda c: "The calendar service is failing right now, so I couldn't check. Try again shortly."
        if c.result_of("list_calendar_events")["error"]["code"] == "SERVER_ERROR" else "?",
    ])
    agent, script = make_agent(script)
    agent.backend.fail_next("list_calendar_events", "SERVER_ERROR", 10)
    res = agent.invoke("what's on my calendar tomorrow?")
    assert len([e for e in res.events if e.event == "tool_retry"]) == 2  # max_tool_retries=2 -> 3 attempts
    result = res.tool_events[0]
    assert not result.ok and result.retries_exhausted and result.attempt == 3
    assert "couldn't check" in res.text
    assert res.status == "completed"


def test_non_retryable_errors_are_not_retried(make_agent):
    script = Script([tc("get_contact", name="Zed Nobody"), "I couldn't find anyone called Zed Nobody. Check the spelling?"])
    agent, _ = make_agent(script)
    res = agent.invoke("who is Zed Nobody?")
    assert not any(e.event == "tool_retry" for e in res.events) and agent.sleeps == []
    assert res.tool_events[0].error_code == "NOT_FOUND" and res.tool_events[0].attempt == 1
    assert "couldn't find" in res.text


def test_retried_write_creates_exactly_one_event(make_agent):
    create = dict(title="Focus time", start=tomorrow(12), end=tomorrow(13))
    script = Script([
        tc("create_calendar_event", **create), "Please confirm.",
        lambda c: reissue_confirmed(c), "Created 'Focus time' tomorrow 12:00.",
    ])
    agent, _ = make_agent(script)
    agent.invoke("schedule focus time tomorrow at noon")
    # the first attempt commits but the response is lost; the retry must not duplicate it
    agent.backend.fail_next("create_calendar_event", "SERVER_ERROR", 1, after_commit=True)
    res = agent.invoke("yes")
    assert res.tool_events[0].ok and res.tool_events[0].attempt == 2
    assert [e.title for e in agent.backend.calendar.all_events()].count("Focus time") == 1


def test_failed_write_never_reported_as_success(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW),
        lambda c: tc("delete_calendar_event", event_id=event_id_of(c, "standup")),
        "Delete the standup? Please confirm.",
        lambda c: reissue_confirmed(c),
        "I've deleted the Team standup!",  # false: the delete failed
        "Sorry - the calendar failed and the standup is still there. Want me to retry?",
    ])
    agent, _ = make_agent(script)
    agent.invoke("delete my standup tomorrow")
    agent.backend.fail_next("delete_calendar_event", "SERVER_ERROR", 10)
    res = agent.invoke("yes")
    assert "I've deleted" not in res.text and "still there" in res.text
    assert any(e.event == "recovery" and "claimed" in (e.text or "") for e in res.events)
    assert agent.backend.calendar.get("evt_002") is not None


def test_repeated_false_claim_gets_a_deterministic_honest_message(make_agent):
    script = Script([tc("get_contact", name="x"), "I've booked it!", "I have successfully booked it!"])
    agent, _ = make_agent(script)
    res = agent.invoke("book the meeting")
    assert "booked" not in res.text.lower() or "haven't" in res.text.lower()
    assert "haven't made" in res.text


def test_claim_with_no_tool_call_at_all_is_corrected(make_agent):
    script = Script(["Done, I've scheduled the meeting!", "I haven't scheduled anything yet - what time works?"])
    agent, _ = make_agent(script)
    res = agent.invoke("schedule a meeting")
    assert "haven't scheduled" in res.text


def test_wrong_tool_is_recovered_from(make_agent):
    script = Script([
        tc("get_contact", name="Design review"),  # a meeting, not a person
        lambda c: tc("list_calendar_events", **LIST_TOMORROW) if c.result_of("get_contact")["error"]["code"] == "NOT_FOUND" else "?",
        "Your Design review is tomorrow at 14:00.",
    ], route=lambda c: {"groups": ["crm", "calendar"], "confidence": 0.9})
    agent, _ = make_agent(script)
    res = agent.invoke("when is the design review?")
    assert res.tools_called == ["get_contact", "list_calendar_events"]
    assert [e.ok for e in res.tool_events] == [False, True]
    assert "14:00" in res.text


def test_validation_errors_are_fed_back_to_the_model(make_agent):
    script = Script([
        tc("list_calendar_events", start="2026-10-22T00:00:00", end="2026-10-22T23:59:00"),  # naive datetimes
        lambda c: tc("list_calendar_events", **LIST_TOMORROW) if c.result_of("list_calendar_events")["error"]["code"] == "INVALID_ARGS" else "?",
        "Here are tomorrow's events.",
    ])
    agent, script = make_agent(script)
    res = agent.invoke("what's on my calendar tomorrow?")
    block = next(e for e in res.events if e.event == "guard_block")
    assert block.guard_reason == "INVALID_ARGS"
    assert res.tool_events and res.tool_events[0].ok
    assert "Fix the listed fields" in str(script.agent_calls()[1].messages[-1].content)  # the model saw the hint


def test_duplicate_calls_are_blocked(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW),
        tc("list_calendar_events", **LIST_TOMORROW),
        "You have two events.",
    ])
    agent, _ = make_agent(script)
    res = agent.invoke("what's on my calendar tomorrow?")
    assert len(res.tool_events) == 1
    assert [e.guard_reason for e in res.events if e.event == "guard_block"] == ["DUPLICATE_CALL"]


def test_step_cap_stops_runaway_loops(make_agent):
    counter = iter(range(100))
    runaway = lambda c: tc("calculate", expression=f"{next(counter)} + 1")  # a new call each time  # noqa: E731
    script = Script([runaway] * 12)
    agent, _ = make_agent(script, max_steps=3)
    res = agent.invoke("calculate 1+1 forever")
    assert len(res.tool_events) <= 3
    assert "limit of 3 steps" in res.text
    assert res.status == "completed" and agent.invoke("2*2").status == "completed"  # the thread is still usable


def test_unknown_tool_is_rejected(make_agent):
    script = Script([tc("send_email", to="x"), "I can't send emails."])
    agent, _ = make_agent(script)
    res = agent.invoke("email Priya")
    block = next(e for e in res.events if e.event == "guard_block")
    assert block.guard_reason == "UNKNOWN_TOOL" and res.tool_events == []


def test_inactive_group_is_enabled_on_demand(make_agent):
    script = Script(
        [
            tc("list_calendar_events", **LIST_TOMORROW),  # router said "no tools", model disagrees
            tc("list_calendar_events", **LIST_TOMORROW),
            "Two events tomorrow.",
        ],
        route=lambda c: {"groups": [], "confidence": 0.9},
    )
    agent, script = make_agent(script)
    res = agent.invoke("anything on my calendar tomorrow?")
    block = next(e for e in res.events if e.event == "guard_block")
    assert block.guard_reason == "TOOL_NOT_ACTIVE"
    assert any(e.event == "recovery" for e in res.events)
    assert res.tool_events and res.tool_events[0].ok
    assert "list_calendar_events" in script.agent_calls()[1].bound


def test_llm_failure_is_reported_not_raised(make_agent):
    agent, _ = make_agent(Script([RuntimeError("401 invalid api key"), "Hello!"]))
    res = agent.invoke("what is 2+2?")
    assert res.status == "completed" and "Sorry" in res.text and "No changes were made" in res.text
    assert agent.invoke("hi").text  # the next turn still works
