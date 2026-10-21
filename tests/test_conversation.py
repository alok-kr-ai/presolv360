"""No-tool turns, topic shifts and multi-turn context resolution."""
from tests.conftest import tomorrow
from tests.fakes import Script, reissue_confirmed, tc

LIST_TOMORROW = dict(start=tomorrow(0), end=tomorrow(23, 59))


def tool_events(res):
    return [e for e in res.events if e.event.startswith("tool_")]


# ------------------------------------------------------------------------------ no-tool turns
def test_greetings_and_thanks_use_no_tools_and_no_router_call(make_agent):
    script = Script(["Hello! How can I help?", "You're welcome!"])
    agent, script = make_agent(script)
    for text in ("Hi!", "thanks"):
        res = agent.invoke(text)
        assert res.groups == [] and tool_events(res) == []
    assert script.purposes() == ["agent", "agent"]  # fast path: no router LLM call
    assert script.agent_calls()[0].bound == ()


def test_general_knowledge_question_is_answered_directly(make_agent):
    script = Script(["The capital of France is Paris."])
    agent, script = make_agent(script)
    before = agent.memory()
    res = agent.invoke("What is the capital of France?")
    assert res.text == "The capital of France is Paris."
    assert script.purposes() == ["route", "agent"] and script.agent_calls()[0].bound == ()
    assert res.groups == [] and tool_events(res) == []
    assert agent.memory() == before  # memory and pending action untouched
    assert not any(e.event in ("guard_block", "memory_update") for e in res.events)


def test_capability_question_is_answered_from_the_prompt(make_agent):
    script = Script(["I can manage your calendar, look up contacts and do maths."])
    agent, script = make_agent(script)
    res = agent.invoke("What can you do?")
    system = script.agent_calls()[0].system
    for needle in ("calendar", "crm", "calculator", "list_calendar_events", "get_contact", "calculate"):
        assert needle in system
    assert tool_events(res) == []


def test_live_data_questions_get_an_honest_limitation(make_agent):
    script = Script(["I can't look up live weather - I only have calendar, contact and calculator tools."])
    agent, script = make_agent(script)
    res = agent.invoke("What's the weather in Mumbai right now?")
    assert "can't" in res.text and tool_events(res) == []
    assert "live" in script.agent_calls()[0].system.lower() or "real-time" in script.agent_calls()[0].system.lower()


def test_stray_tool_call_on_a_no_tool_turn_is_recovered(make_agent):
    script = Script([tc("calculate", expression="2+2"), "4", "The answer is 4."])
    agent, _ = make_agent(script)
    res = agent.invoke("What is the capital of Spain?")  # router: no tools
    assert any(e.event == "guard_block" and e.guard_reason == "TOOL_NOT_ACTIVE" for e in res.events)


# ------------------------------------------------------------------------------- topic shifts
def test_calculator_then_crm_then_calendar_in_one_thread(make_agent):
    script = Script([
        tc("calculate", expression="2 * 3"), "2 times 3 is 6.",
        tc("get_contact", name="Rahul Verma"), "Rahul Verma is at Globex.",
        tc("list_calendar_events", **LIST_TOMORROW), "You have the Team standup tomorrow.",
    ])
    agent, script = make_agent(script)
    results = [agent.invoke(t) for t in ("multiply 2 and 3", "who is Rahul Verma?", "what's on my calendar tomorrow?")]
    assert [r.groups for r in results] == [["calculator"], ["crm"], ["calendar"]]
    assert [r.tools_called for r in results] == [["calculate"], ["get_contact"], ["list_calendar_events"]]
    assert [c.bound for c in script.agent_calls() if c.bound][::2] == [
        ("calculate",), ("get_contact", "list_interactions", "create_contact", "update_contact", "delete_contact"),
        ("list_calendar_events", "check_availability", "create_calendar_event", "update_calendar_event", "delete_calendar_event"),
    ]
    mem = agent.memory()
    assert mem.last_calculation["result"] == 6 and mem.last_contact["id"] == "con_003"


def test_calculation_result_is_available_to_a_later_calendar_action(make_agent):
    def propose(call):
        assert '"result": 60' in call.system  # the earlier calculation is in working memory
        return tc("create_calendar_event", title="Deep work", start=tomorrow(12), end=tomorrow(13))

    script = Script([tc("calculate", expression="2 * 30"), "That's 60 minutes.", propose, "Book 'Deep work' 12:00-13:00?"])
    agent, _ = make_agent(script)
    agent.invoke("calculate 2 * 30 for me")
    res = agent.invoke("schedule a meeting for that many minutes tomorrow at noon")
    pending = agent.memory().pending_action
    assert pending and pending.tool == "create_calendar_event"
    assert any(e.event == "confirmation_required" for e in res.events)


def test_calendar_then_calculator_then_back_to_the_meeting(make_agent):
    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW, query="standup"), "Your Team standup is at 10:00.",
        tc("calculate", expression="15 * 4"), "That's 60.",
        "The standup you asked about earlier is at 10:00 tomorrow.",
    ])
    agent, script = make_agent(script)
    agent.invoke("find my standup on my calendar tomorrow")
    agent.invoke("what is 15 times 4?")
    res = agent.invoke("when was that meeting again?")
    system = script.agent_calls()[-1].system
    assert "Team standup" in system and '"result": 60' in system  # both memories survive the topic shift
    assert agent.memory().last_event["title"] == "Team standup"
    assert tool_events(res) == []


def test_mixed_intent_turn_enables_multiple_groups(make_agent):
    script = Script([
        [tc("get_contact", name="Priya Sharma"), tc("list_calendar_events", **LIST_TOMORROW)],
        "Priya Sharma is your contact and you're free after 11:00.",
    ])
    agent, _ = make_agent(script)
    res = agent.invoke("look up Priya Sharma's email and tell me when I'm free tomorrow")
    assert set(res.groups) == {"calendar", "crm"}
    assert sorted(res.tools_called) == ["get_contact", "list_calendar_events"]


def test_router_failures_fail_open_to_all_groups(make_agent):
    for route in (RuntimeError("router down"), "this is not json", {"groups": ["nope"], "confidence": 0.9},
                  {"groups": ["crm"], "confidence": 0.1}):
        script = Script(["Fine."], route=lambda c, r=route: r)
        agent, script = make_agent(script)
        res = agent.invoke("do the thing")
        assert set(res.groups) == {"calendar", "crm", "calculator"}, route
        assert next(e for e in res.events if e.event == "route").data["fail_open"] is True


# ------------------------------------------------------------------------- multi-turn context
def test_book_that_for_me_at_3pm_resolves_from_working_memory(make_agent):
    def propose(call):
        assert "Design review" in call.system  # last_event is in the prompt
        return tc("create_calendar_event", title="Design review follow-up", start=tomorrow(15), end=tomorrow(16))

    script = Script([
        tc("list_calendar_events", **LIST_TOMORROW, query="design review"), "Your Design review is tomorrow 14:00.",
        propose, "Shall I book 'Design review follow-up' tomorrow 15:00-16:00?",
        lambda c: reissue_confirmed(c), "Booked 'Design review follow-up' tomorrow at 15:00.",
    ])
    agent, script = make_agent(script)
    agent.invoke("when is my design review on my calendar tomorrow?")
    assert agent.memory().last_event["title"] == "Design review"
    res = agent.invoke("book that for me at 3pm")
    assert res.tools_called == [] and agent.memory().pending_action is not None
    done = agent.invoke("yes")
    assert done.tools_called == ["create_calendar_event"] and "Booked" in done.text
    assert "Design review follow-up" in [e.title for e in agent.backend.calendar.all_events()]
    assert agent.memory().last_event["title"] == "Design review follow-up"
    # the checkpointer carried the whole conversation across the three turns
    assert len(agent.state_values()["messages"]) >= 8


def test_threads_are_isolated(make_agent):
    script = Script([tc("get_contact", name="Rahul Verma"), "Rahul is at Globex.", "Hello!"])
    agent, _ = make_agent(script)
    agent.invoke("who is Rahul Verma?", thread_id="a")
    agent.invoke("hi", thread_id="b")
    assert agent.memory("a").last_contact and not agent.memory("b").last_contact
