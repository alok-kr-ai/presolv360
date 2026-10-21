"""Write confirmations: the model asks, and the graph guard independently enforces it."""
from tests.conftest import tomorrow
from tests.fakes import Script, event_id_of, reissue_confirmed, tc

LIST_TOMORROW = dict(start=tomorrow(0), end=tomorrow(23, 59))
CREATE = dict(title="Design sync", start=tomorrow(15), end=tomorrow(16), attendees=["ananya@initech.io"])


def events(agent):
    return [e.title for e in agent.backend.calendar.all_events()]


def test_write_is_blocked_until_the_user_confirms(make_agent):
    script = Script([
        tc("create_calendar_event", **CREATE),
        "Shall I create 'Design sync' tomorrow 15:00-16:00 with Ananya? Please confirm.",
    ])
    agent, _ = make_agent(script)
    res = agent.invoke("book a design sync with Ananya tomorrow at 3pm")

    assert "Design sync" not in events(agent)
    assert res.tool_events == []  # the tool never ran
    block = next(e for e in res.events if e.event == "guard_block")
    assert block.guard_reason == "CONFIRMATION_REQUIRED"
    pending = agent.memory().pending_action
    assert pending and pending.tool == "create_calendar_event" and not pending.confirmed
    assert any(e.event == "confirmation_required" for e in res.events)


def test_write_runs_after_yes_on_a_later_turn(make_agent):
    script = Script([
        tc("create_calendar_event", **CREATE),
        "Shall I create it? Please confirm.",
        lambda call: reissue_confirmed(call),
        lambda call: "Created 'Design sync' tomorrow 15:00." if call.result_of("create_calendar_event")["ok"] else "failed",
    ])
    agent, script = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    res = agent.invoke("yes, go ahead")

    assert "Design sync" in events(agent)
    assert res.tools_called == ["create_calendar_event"]
    assert agent.memory().pending_action is None  # single use
    assert "CONFIRMED ACTION" in script.agent_calls()[2].system
    assert any(e.event == "confirmation_checked" and e.data["decision"] == "yes" for e in res.events)


def test_confirmation_is_not_valid_on_the_same_turn(make_agent):
    """The model cannot confirm for the user by calling the write tool twice in one turn."""
    script = Script([
        tc("create_calendar_event", **CREATE),
        tc("create_calendar_event", **CREATE),
        "Please confirm the booking.",
    ])
    agent, _ = make_agent(script)
    res = agent.invoke("book a design sync with Ananya tomorrow at 3pm, yes I confirm")
    assert "Design sync" not in events(agent) and res.tool_events == []


def test_changed_args_after_yes_are_blocked(make_agent):
    script = Script([
        tc("create_calendar_event", **CREATE),
        "Confirm?",
        tc("create_calendar_event", **{**CREATE, "start": tomorrow(17), "end": tomorrow(18)}),  # not what was approved
        "That is a different time; do you want 5pm instead?",
    ])
    agent, _ = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    res = agent.invoke("yes")
    assert "Design sync" not in events(agent)
    assert [e.guard_reason for e in res.events if e.event == "guard_block"] == ["CONFIRMATION_REQUIRED"]
    pending = agent.memory().pending_action
    assert pending and not pending.confirmed and "17:00" in pending.summary  # a fresh proposal replaced it


def test_delete_requires_strict_event_id_match(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW),
        lambda c: tc("delete_calendar_event", event_id=event_id_of(c, "standup")),
        "Delete 'Team standup'? Please confirm.",
        tc("delete_calendar_event", event_id="evt_003"),  # a different event than the one approved
        "I can only delete the event you approved.",
    ])
    agent, _ = make_agent(script)
    agent.invoke("delete my standup tomorrow")
    res = agent.invoke("yes")
    assert agent.backend.calendar.get("evt_002") is not None and agent.backend.calendar.get("evt_003") is not None
    assert res.tool_events == []
    assert any(e.event == "guard_block" and e.tool == "delete_calendar_event" for e in res.events)


def test_delete_runs_when_the_same_event_id_is_confirmed(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW),
        lambda c: tc("delete_calendar_event", event_id=event_id_of(c, "standup")),
        "Delete 'Team standup'? Please confirm.",
        lambda c: reissue_confirmed(c),
        "Deleted the Team standup.",
    ])
    agent, _ = make_agent(script)
    agent.invoke("delete my standup tomorrow")
    agent.invoke("yes please")
    assert agent.backend.calendar.get("evt_002") is None


def test_unseen_event_id_is_rejected_and_creates_no_pending(make_agent):
    script = Script([
        tc("delete_calendar_event", event_id="evt_002"),  # guessed, never listed
        tc("list_calendar_events", **LIST_TOMORROW),
        "I found it; shall I delete it?",
    ])
    agent, _ = make_agent(script)
    res = agent.invoke("delete my standup")
    block = next(e for e in res.events if e.event == "guard_block")
    assert block.guard_reason == "INVALID_ARGS" and "was not returned" in (block.failure_reason or "")
    assert agent.backend.calendar.get("evt_002") is not None
    assert agent.memory().pending_action is None


def test_thanks_or_hi_never_confirms_a_pending_action(make_agent):
    script = Script([tc("create_calendar_event", **CREATE), "Please confirm.", "You're welcome!", "Hello again!"])
    agent, script = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    for text in ("thanks", "hi"):
        res = agent.invoke(text)
        assert res.groups == [] and res.tools_called == []
    pending = agent.memory().pending_action
    assert pending and not pending.confirmed
    assert "Design sync" not in events(agent)


def test_no_declines_and_clears_the_pending_action(make_agent):
    script = Script([tc("create_calendar_event", **CREATE), "Please confirm.", "Okay, I won't book it."])
    agent, script = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    res = agent.invoke("no")
    assert agent.memory().pending_action is None and "Design sync" not in events(agent)
    assert "declined" in script.agent_calls()[-1].system.lower()
    assert any(e.event == "confirmation_checked" and e.data["decision"] == "no" for e in res.events)


def test_modification_cancels_the_pending_action(make_agent):
    script = Script(
        [tc("create_calendar_event", **CREATE), "Please confirm.", "Sure, 4pm instead - shall I book it?"],
        confirm=lambda call: {"decision": "modify"},
    )
    agent, script = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    res = agent.invoke("actually make it a bit later than that")
    assert agent.memory().pending_action is None
    assert any(e.event == "confirmation_checked" and e.data["method"] == "llm" for e in res.events)
    assert "change" in script.agent_calls()[-1].system.lower()


def test_pending_action_expires(make_agent):
    script = Script([tc("create_calendar_event", **CREATE), "Please confirm.", "a", "b", "c"])
    agent, _ = make_agent(script, pending_ttl_turns=1)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    agent.invoke("what is the capital of France?")
    assert agent.memory().pending_action is not None
    agent.invoke("and of Spain?")
    assert agent.memory().pending_action is None


def test_unconfirmed_replacement_proposal(make_agent):
    script = Script([
        tc("create_calendar_event", **CREATE), "Please confirm.",
        tc("create_calendar_event", **{**CREATE, "title": "Other"}), "Confirm the other one?",
    ])
    agent, _ = make_agent(script)
    agent.invoke("book a design sync with Ananya tomorrow at 3pm")
    res = agent.invoke("no wait, book 'Other' instead at the same time")
    pending = agent.memory().pending_action
    assert pending and "Other" in pending.summary
    assert any(e.event == "confirmation_required" for e in res.events)
