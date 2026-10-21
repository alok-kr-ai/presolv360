"""LLM routing config, settings, prompts and the "rich only" output rule."""
import ast
from pathlib import Path

import pytest

from config import Settings
from llm import build_chain, build_chat_model, build_context_entry, classify_llm_error, model_chain
from prompts import build_system_prompt, extract_json
from tests.fakes import Script

ROOT = Path(__file__).resolve().parent.parent


def test_model_chain_is_primary_then_fallbacks_without_duplicates():
    s = Settings(model="openai/a", fallback_models=["openai/b", "openai/a", "anthropic/c", "openai/b"])
    assert model_chain(s) == ["openai/a", "openai/b", "anthropic/c"]
    assert model_chain(Settings(model="m", fallback_models=[])) == ["m"]


def test_chat_models_are_litellm_and_built_lazily_per_model():
    s = Settings(model="openai/a", fallback_models=["openai/b"], context_fallback_model="openai/big")
    chain = build_chain(s, streaming=False)
    assert [e.name for e in chain] == ["openai/a", "openai/b"]
    assert all(e._chat is None for e in chain)  # nothing is built until needed
    model = chain[1].chat
    assert type(model).__name__ == "ChatLiteLLM" and model.model == "openai/b" and model.streaming is False
    assert build_context_entry(s, streaming=True).name == "openai/big"
    assert build_context_entry(Settings(context_fallback_model=None), streaming=True) is None


def test_litellm_is_quieted_and_drops_unsupported_params():
    import litellm

    build_chat_model(Settings(model="openai/a"))
    assert litellm.drop_params is True and litellm.suppress_debug_info is True


def test_every_llm_call_goes_through_the_llm_client():
    """No node talks to a chat model directly: only llm.py invokes models."""
    offenders = []
    for path in list(ROOT.glob("*.py")) + list(ROOT.glob("graph/*.py")) + list(ROOT.glob("ui/*.py")):
        if path.name == "llm.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("completion", "acompletion") or (
                isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("openai", "anthropic"))
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize(
    "exc,kind",
    [
        (type("ContextWindowExceededError", (Exception,), {})("too long"), "context_window"),
        (Exception("maximum context length is 8192 tokens"), "context_window"),
        (type("RateLimitError", (Exception,), {})("slow"), "rate_limit"),
        (Exception("Incorrect API key provided"), "auth"),
        (Exception("The model `gpt-6` does not exist"), "model_not_found"),
        (type("Timeout", (Exception,), {})("x"), "timeout"),
        (Exception("boom"), "unknown"),
    ],
)
def test_llm_error_classification(exc, kind):
    assert classify_llm_error(exc) == kind


def test_settings_from_env(monkeypatch, tmp_path):
    for k in ("LLM_MODEL", "FALLBACK_MODELS", "CONTEXT_FALLBACK_MODEL", "LOG_DIR", "MAX_STEPS", "LOG_CONTENT", "TIMEZONE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/primary")
    monkeypatch.setenv("FALLBACK_MODELS", "openai/primary, openai/second ,anthropic/third")
    monkeypatch.setenv("MAX_STEPS", "5")
    monkeypatch.setenv("LOG_CONTENT", "false")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "L"))
    s = Settings.from_env(env_file=tmp_path / "missing.env")
    assert s.model == "openai/primary"
    assert s.fallback_models == ["openai/second", "anthropic/third"]  # primary never falls back to itself
    assert s.max_steps == 5 and s.log_content is False and s.log_dir == tmp_path / "L"
    assert s.timezone == "Asia/Kolkata"


def test_settings_redaction_and_bad_numbers(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_STEPS", "not-a-number")
    assert Settings.from_env(env_file=tmp_path / "missing.env").max_steps == 8
    assert "secret" not in str(Settings().redacted()).lower()


def test_system_prompt_has_current_datetime_and_timezone(make_agent):
    agent, script = make_agent(Script(["Hi!"]))
    agent.invoke("hello")
    system = script.agent_calls()[0].system
    assert "Asia/Kolkata" in system and "2026-10-21" in system and "Wednesday" in system and "+05:30" in system


def test_system_prompt_lists_rules_and_pending_sections():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from state import PendingAction, WorkingMemory

    pending = PendingAction(tool="delete_calendar_event", args={"event_id": "evt_1"}, call_args={"event_id": "evt_1"},
                            summary="Delete event evt_1", proposed_turn=1)
    prompt = build_system_prompt(
        now=datetime(2026, 10, 21, 9, tzinfo=ZoneInfo("Asia/Kolkata")), timezone="Asia/Kolkata",
        group_desc={"calendar": "events"}, tool_names={"calendar": ["list_calendar_events"]},
        active_groups=["calendar"], memory=WorkingMemory(pending_action=pending),
        confirmation_note=None, correction_note="fix it",
    )
    assert "PENDING ACTION" in prompt and "Delete event evt_1" in prompt and "fix it" in prompt
    assert "untrusted" in prompt.lower()


def test_extract_json_is_forgiving():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("no json here") is None and extract_json("") is None


def test_all_terminal_output_goes_through_rich():
    """No bare print()/input() anywhere in the app (tests and the optional Streamlit UI excluded)."""
    skip_dirs = {".venv", "tests", "logs", "__pycache__", ".git"}
    offenders = []
    for path in ROOT.rglob("*.py"):
        if skip_dirs & set(path.relative_to(ROOT).parts) or path.name == "streamlit_app.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("print", "input"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []
