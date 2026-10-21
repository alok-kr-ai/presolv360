"""LangSmith tracing: switched on only when LANGSMITH_API_KEY is set."""
from __future__ import annotations

import os
from typing import Any


def langsmith_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY", "").strip())


def configure_langsmith() -> bool:
    """Enable tracing if (and only if) a key is present. Returns whether enabled."""
    if not langsmith_enabled():
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "presolv-agent")
    return True


def run_metadata(run_id: str, trace_id: str, thread_id: str) -> dict[str, Any]:
    """Tag LangSmith runs so they can be cross-referenced with local logs."""
    # NB: the key must not be ``run_id`` - LangGraph reserves it for its own run identity.
    return {"agent_run_id": run_id, "agent_trace_id": trace_id, "thread_id": thread_id}
