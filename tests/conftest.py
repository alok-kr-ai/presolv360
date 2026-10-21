from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from config import Settings
from graph.builder import build_agent
from mocks import Backend
from tests.fakes import FakeChatModel, Script

TEST_DATA = Path(__file__).parent / "data"  # pinned seed CSVs: tests never depend on the live mocks/data files
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 21, 9, 0, tzinfo=IST)  # a Wednesday morning


def tomorrow(hour: int, minute: int = 0, day_offset: int = 1) -> str:
    """ISO string for NOW's date + day_offset at hour:minute IST."""
    base = NOW.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return (base + timedelta(days=day_offset)).isoformat()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        model="fake-model",
        fallback_models=[],
        context_fallback_model=None,
        log_dir=tmp_path / "logs",
        mock_data_dir=TEST_DATA,
        max_tool_retries=2,
        retry_base_delay_s=0.0,
        max_steps=8,
    )


@pytest.fixture
def clock():
    return lambda: NOW


@pytest.fixture
def make_agent(settings, clock):
    """Factory: make_agent(Script(...), **settings_overrides) -> (agent, script)."""
    agents = []

    def factory(script: Script | None = None, *, backend: Backend | None = None, **overrides):
        script = script or Script()
        cfg = settings.model_copy(update=overrides) if overrides else settings
        sleeps: list[float] = []
        agent = build_agent(
            cfg, llm=FakeChatModel(script=script), backend=backend or Backend(cfg, clock=clock),
            clock=clock, sleep=sleeps.append,
        )
        agent.sleeps = sleeps  # type: ignore[attr-defined]
        agents.append(agent)
        return agent, script

    yield factory
    for a in agents:
        a.close()
