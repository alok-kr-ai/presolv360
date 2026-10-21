"""Model fallback chain: every call goes through litellm models, one at a time, in order."""
import pytest

import llm as llm_module
from config import Settings
from graph.builder import build_agent
from llm import LLMClient, LLMError, ModelEntry
from mocks import Backend
from tests.conftest import NOW, TEST_DATA
from tests.fakes import FakeChatModel, Script
from ui.render import TurnView


def fake_models(monkeypatch, scripts: dict[str, Script]) -> None:
    """Make ``build_chat_model(settings, name)`` return a scripted fake per model name."""
    monkeypatch.setattr(
        llm_module, "build_chat_model",
        lambda settings, model=None, *, streaming=None: FakeChatModel(script=scripts[model or settings.model], model_label=model),
    )


def agent_with(tmp_path, scripts, monkeypatch, **overrides):
    fake_models(monkeypatch, scripts)
    cfg = Settings(model="primary", fallback_models=["backup", "last"], context_fallback_model="big",
                   log_dir=tmp_path / "logs", mock_data_dir=TEST_DATA, retry_base_delay_s=0, **overrides)
    agent = build_agent(cfg, backend=Backend(cfg, clock=lambda: NOW), clock=lambda: NOW, sleep=lambda s: None)
    return agent


def test_fallback_model_answers_when_the_primary_fails(tmp_path, monkeypatch):
    scripts = {
        "primary": Script([RuntimeError("503 service unavailable")], route=lambda c: RuntimeError("down")),
        "backup": Script(["Hello from the backup model."]),
        "last": Script(["unused"]), "big": Script(["unused"]),
    }
    agent = agent_with(tmp_path, scripts, monkeypatch)
    res = agent.invoke("what is the capital of France?")
    assert res.text == "Hello from the backup model."

    ends = [e for e in res.events if e.event == "llm_end"]
    failed = [e for e in ends if e.ok is False]
    assert {e.name for e in failed} == {"route", "agent"} and {e.model for e in failed} == {"primary"}
    answered = [e for e in ends if e.ok]
    assert {e.model for e in answered} == {"backup"} and all(e.fallback_used for e in answered)
    hops = [e for e in res.events if e.event == "llm_fallback"]
    assert [(e.data["from"], e.data["to"]) for e in hops] == [("primary", "backup")] * 2
    assert scripts["last"].calls == []  # later models are never touched

    summary = agent.recorder.metrics.snapshot()["llm"]
    assert summary["backup"]["fallbacks_used"] == 2 and summary["primary"]["errors"] == 2
    view = TurnView()
    for ev in res.events:
        view.apply(ev)
    assert view.fallback_used and any("fallback" in s.label.lower() for s in view.steps)
    agent.close()


def test_context_window_errors_jump_to_the_context_fallback_model(tmp_path, monkeypatch):
    scripts = {
        "primary": Script([RuntimeError("This model's maximum context length is 8192 tokens")]),
        "backup": Script(["from backup"]), "last": Script([]), "big": Script(["Answered with the big window."]),
    }
    agent = agent_with(tmp_path, scripts, monkeypatch)
    res = agent.invoke("hi")
    assert res.text == "Answered with the big window."
    assert [(e.data["from"], e.data["to"]) for e in res.events if e.event == "llm_fallback"] == [("primary", "big")]
    assert scripts["backup"].calls == []
    agent.close()


def test_all_models_failing_gives_a_friendly_answer_and_a_failed_turn_log(tmp_path, monkeypatch):
    boom = lambda: Script([RuntimeError("Incorrect API key provided")] * 4, route=lambda c: RuntimeError("Incorrect API key"))  # noqa: E731
    agent = agent_with(tmp_path, {n: boom() for n in ("primary", "backup", "last", "big")}, monkeypatch)
    res = agent.invoke("what is 2+2?")
    assert res.status == "completed" and "credentials were rejected" in res.text and "No changes were made" in res.text
    hops = [e for e in res.events if e.event == "llm_fallback"]
    assert [e.data["to"] for e in hops if e.name == "agent"] == ["backup", "last"]
    final = next(e for e in res.events if e.event == "final")
    assert final.ok is False and final.error_code == "auth"
    agent.close()


def test_fallback_works_with_streamed_tokens(tmp_path, monkeypatch):
    scripts = {
        "primary": Script([RuntimeError("boom")]), "backup": Script(["streamed from the backup"]),
        "last": Script([]), "big": Script([]),
    }
    agent = agent_with(tmp_path, scripts, monkeypatch)
    events = list(agent.stream("tell me a fun fact"))
    assert "".join(e.text for e in events if e.event == "token") == "streamed from the backup"
    agent.close()


def test_llm_client_requires_a_model_and_reports_the_first_meaningful_error():
    with pytest.raises(ValueError):
        LLMClient([], None, Settings(), bus=None)  # type: ignore[arg-type]

    from events import EventBus
    from langchain_core.messages import HumanMessage

    bad_key = FakeChatModel(script=Script([RuntimeError("Incorrect API key provided")]))
    weird = FakeChatModel(script=Script([RuntimeError("???")]))
    client = LLMClient([ModelEntry("a", lambda: bad_key, bad_key), ModelEntry("b", lambda: weird, weird)], None,
                       Settings(model="a"), EventBus("run-x"))
    with pytest.raises(LLMError) as exc:
        client.call({"trace_id": "t", "turn": 1}, "agent", [HumanMessage("hi")])
    assert exc.value.kind == "auth"  # the specific cause wins over the last, vaguer one


def test_permanently_failing_model_is_skipped_on_later_calls(tmp_path, monkeypatch):
    scripts = {
        "primary": Script([RuntimeError("The model `primary` does not exist")], route=lambda c: RuntimeError("does not exist")),
        "backup": Script(["first answer", "second answer"]), "last": Script([]), "big": Script([]),
    }
    agent = agent_with(tmp_path, scripts, monkeypatch)
    first = agent.invoke("what is the capital of Spain?")
    second = agent.invoke("and of Italy?")
    assert (first.text, second.text) == ("first answer", "second answer")
    assert [e.event for e in second.events if e.event in ("llm_fallback",)] == []  # no wasted hop on turn 2
    assert {e.model for e in second.events if e.event == "llm_end"} == {"backup"}
    agent.close()
