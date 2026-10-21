"""Application settings.

``Settings`` is a plain pydantic v2 model. ``Settings.from_env()`` loads a
``.env`` file (python-dotenv) and reads the process environment, so the whole
agent is wired from one object (see ``graph.builder.build_agent``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "openai/gpt-6-luna"
DEFAULT_FALLBACKS = ["openai/gpt-4o-mini", "openai/gpt-4.1-mini"]
DEFAULT_CONTEXT_FALLBACK = "openai/gpt-4.1-mini"
DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parent / "mocks" / "data"

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_num(name: str, default: float, cast: type = float) -> Any:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


class Settings(BaseModel):
    """All tunables for the agent. Defaults are safe for local use."""

    model_config = ConfigDict(extra="forbid")

    # --- LLM (always routed through litellm) ---
    model: str = DEFAULT_MODEL
    fallback_models: list[str] = Field(default_factory=lambda: list(DEFAULT_FALLBACKS))
    context_fallback_model: str | None = DEFAULT_CONTEXT_FALLBACK
    temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 1
    streaming: bool = True
    max_history_tokens: int = 6000

    # --- Agent behaviour ---
    timezone: str = "Asia/Kolkata"
    max_steps: int = 8
    max_tool_retries: int = 2
    tool_timeout_s: float = 10.0
    retry_base_delay_s: float = 0.5
    pending_ttl_turns: int = 3

    # --- Observability ---
    log_dir: Path = Path("logs")
    log_content: bool = True

    # --- Mock backends ---
    mock_data_dir: Path = DEFAULT_MOCK_DATA_DIR  # events.csv, contacts.csv, interactions.csv
    mock_persist: bool = False  # write calendar changes back to events.csv (from_env turns it on)
    mock_failure_rate: float = 0.0
    mock_seed: int = 7

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = None) -> "Settings":
        """Build settings from ``.env`` + environment variables."""
        load_dotenv(env_file, override=False)
        model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL

        fallbacks = _split_csv(os.environ.get("FALLBACK_MODELS")) or list(DEFAULT_FALLBACKS)
        fallbacks = [m for m in dict.fromkeys(fallbacks) if m != model]
        ctx_fallback = os.environ.get("CONTEXT_FALLBACK_MODEL", "").strip() or DEFAULT_CONTEXT_FALLBACK
        if ctx_fallback == model:
            ctx_fallback = None

        return cls(
            model=model,
            fallback_models=fallbacks,
            context_fallback_model=ctx_fallback,
            temperature=_env_num("LLM_TEMPERATURE", 0.0),
            llm_timeout_s=_env_num("LLM_TIMEOUT_S", 60.0),
            llm_max_retries=_env_num("LLM_MAX_RETRIES", 1, int),
            streaming=_env_bool("LLM_STREAMING", True),
            max_history_tokens=_env_num("MAX_HISTORY_TOKENS", 6000, int),
            timezone=os.environ.get("TIMEZONE", "").strip() or "Asia/Kolkata",
            max_steps=_env_num("MAX_STEPS", 8, int),
            max_tool_retries=_env_num("MAX_TOOL_RETRIES", 2, int),
            tool_timeout_s=_env_num("TOOL_TIMEOUT_S", 10.0),
            retry_base_delay_s=_env_num("RETRY_BASE_DELAY_S", 0.5),
            pending_ttl_turns=_env_num("PENDING_TTL_TURNS", 3, int),
            log_dir=Path(os.environ.get("LOG_DIR", "").strip() or "logs"),
            log_content=_env_bool("LOG_CONTENT", True),
            mock_data_dir=Path(os.environ.get("MOCK_DATA_DIR", "").strip() or DEFAULT_MOCK_DATA_DIR),
            mock_persist=_env_bool("MOCK_PERSIST", True),
            mock_failure_rate=_env_num("MOCK_FAILURE_RATE", 0.0),
            mock_seed=_env_num("MOCK_SEED", 7, int),
        )

    def redacted(self) -> dict[str, Any]:
        """JSON-safe snapshot with anything secret-looking masked."""
        data = self.model_dump(mode="json")
        for key in list(data):
            if _SECRET_NAME.search(key):
                data[key] = "***"
        return data


def collect_secret_values() -> list[str]:
    """Values of env vars that look like credentials (for log redaction)."""
    values = []
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and value and len(value) >= 8:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)
