from tests.conftest import tomorrow
from tests.fakes import Script, tc


def test_no_tool_greeting(make_agent):
    agent, script = make_agent(Script(["Hello! How can I help?"]))
    res = agent.invoke("hi")
    assert res.text == "Hello! How can I help?"
    assert res.tools_called == []
    assert res.groups == []


def test_calendar_list(make_agent):
    script = Script([
        tc("list_calendar_events", start=tomorrow(0), end=tomorrow(23, 59)),
        "You have a few things tomorrow.",
    ])
    agent, script = make_agent(script)
    res = agent.invoke("what's on my calendar tomorrow?")
    assert res.tools_called == ["list_calendar_events"]
    assert res.groups == ["calendar"]
    assert "tomorrow" in res.text
