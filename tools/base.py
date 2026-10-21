"""Tool core: uniform result type, error taxonomy and the ``safe_tool`` wrapper.

Every tool call goes through :func:`execute_tool`, which guarantees a tool never
raises and always returns a :class:`ToolResult`:

    {ok, data, error{code, message, retryable, hint}}

Tool bodies just return data or raise :class:`ToolFailure`.
"""
from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger("agent.tools")

MAX_DATA_CHARS = 6000


class ErrorCode(str, Enum):
    # --- raised by tools ---
    INVALID_ARGS = "INVALID_ARGS"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    AMBIGUOUS = "AMBIGUOUS"
    SERVER_ERROR = "SERVER_ERROR"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UNSUPPORTED = "UNSUPPORTED"
    MATH_ERROR = "MATH_ERROR"
    INTERNAL = "INTERNAL"
    # --- raised by the graph guard (the call never reached the tool) ---
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    DUPLICATE_CALL = "DUPLICATE_CALL"
    TOOL_NOT_ACTIVE = "TOOL_NOT_ACTIVE"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    STEP_CAP = "STEP_CAP"


RETRYABLE_CODES = {ErrorCode.SERVER_ERROR, ErrorCode.TIMEOUT, ErrorCode.RATE_LIMITED}


class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False
    hint: str | None = None


class ToolResult(BaseModel):
    """Uniform tool result. ``meta`` is only present when something notable happened
    (``retries_exhausted``, ``truncated``, ``retry_after_s``)."""

    ok: bool
    data: Any = None
    error: ToolError | None = None
    meta: dict[str, Any] | None = None
    # never shown to the model, used for logging/metrics
    latency_ms: float = Field(default=0.0, exclude=True)
    debug: str | None = Field(default=None, exclude=True)

    @classmethod
    def success(cls, data: Any = None, **kw: Any) -> "ToolResult":
        return cls(ok=True, data=data, **kw)

    @classmethod
    def failure(
        cls,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool | None = None,
        hint: str | None = None,
        data: Any = None,
        meta: dict[str, Any] | None = None,
        **kw: Any,
    ) -> "ToolResult":
        if retryable is None:
            retryable = code in RETRYABLE_CODES
        return cls(
            ok=False,
            data=data,
            error=ToolError(code=code, message=message, retryable=retryable, hint=hint),
            meta=meta,
            **kw,
        )

    def to_json(self) -> str:
        payload = self.model_dump(mode="json")
        if not payload.get("meta"):
            payload.pop("meta", None)
        return json.dumps(payload, ensure_ascii=False, default=str)

    def envelope(self, tool: str) -> str:
        """Model-facing string. Tool output is *data*, never instructions."""
        return f'<tool_result tool="{tool}">\n{self.to_json()}\n</tool_result>'

    @property
    def error_code(self) -> str | None:
        return self.error.code.value if self.error else None

    def summary(self, limit: int = 300) -> str:
        text = self.to_json()
        return text if len(text) <= limit else text[:limit] + "..."


class ToolFailure(Exception):
    """Typed failure raised from a tool body."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool | None = None,
        hint: str | None = None,
        data: Any = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = code in RETRYABLE_CODES if retryable is None else retryable
        self.hint = hint
        self.data = data
        self.retry_after_s = retry_after_s


@dataclass
class ToolContext:
    """Per-call context injected by the graph (not part of the model-facing schema)."""

    request_id: str | None = None


class StrictArgs(BaseModel):
    """Base for tool argument schemas: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


INVALID_ARGS_HINT = "Fix the listed fields and call the tool again with valid arguments."
_DATETIME_HINT = " Datetimes must be ISO 8601 with a timezone offset, e.g. 2026-10-21T15:00:00+05:30."


def invalid_args_hint(message: str) -> str:
    low = message.lower()
    return INVALID_ARGS_HINT + (_DATETIME_HINT if ("datetime" in low or "timezone" in low) else "")


def truncate_data(data: Any, max_chars: int = MAX_DATA_CHARS) -> tuple[Any, bool]:
    """Trim the largest list in ``data`` until its JSON fits; flag if trimmed."""

    def size(obj: Any) -> int:
        return len(json.dumps(obj, default=str))

    if size(data) <= max_chars:
        return data, False
    if isinstance(data, list):
        trimmed = list(data)
        while trimmed and size(trimmed) > max_chars:
            trimmed = trimmed[: max(1, len(trimmed) // 2)] if len(trimmed) > 1 else []
        return trimmed, True
    if isinstance(data, dict):
        data = dict(data)
        biggest = max((k for k, v in data.items() if isinstance(v, list)), key=lambda k: size(data[k]), default=None)
        if biggest is not None:
            data[biggest], _ = truncate_data(data[biggest], max_chars // 2)
            if size(data) > max_chars:
                data = {k: (v if not isinstance(v, str) else v[:500]) for k, v in data.items()}
            return data, True
    text = json.dumps(data, default=str)
    return text[:max_chars], True


Handler = Callable[[Any, ToolContext], Any]


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)


def execute_tool(
    name: str,
    schema: type[BaseModel],
    handler: Handler,
    raw_args: dict[str, Any] | None,
    *,
    timeout: float = 10.0,
    request_id: str | None = None,
    max_chars: int = MAX_DATA_CHARS,
) -> ToolResult:
    """The safe-tool wrapper. Never raises."""
    t0 = perf_counter()

    def done(result: ToolResult) -> ToolResult:
        result.latency_ms = round((perf_counter() - t0) * 1000.0, 2)
        return result

    # 1. validate with the strict schema
    try:
        args = schema.model_validate(raw_args or {})
    except ValidationError as exc:
        detail = format_validation_error(exc)
        return done(
            ToolResult.failure(
                ErrorCode.INVALID_ARGS,
                f"Invalid arguments for {name}: {detail}",
                hint=invalid_args_hint(detail),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        return done(ToolResult.failure(ErrorCode.INVALID_ARGS, f"Could not parse arguments: {exc}"))

    # 2. run the body with a timeout and a catch-all
    ctx = ToolContext(request_id=request_id)
    try:
        data = _run_with_timeout(lambda: handler(args, ctx), timeout)
    except FutureTimeout:
        return done(
            ToolResult.failure(
                ErrorCode.TIMEOUT,
                f"{name} did not respond within {timeout:g}s.",
                hint="This is usually temporary; the call can be retried.",
            )
        )
    except ToolFailure as fail:
        meta = {"retry_after_s": fail.retry_after_s} if fail.retry_after_s else None
        return done(
            ToolResult.failure(
                fail.code, fail.message, retryable=fail.retryable, hint=fail.hint, data=fail.data, meta=meta
            )
        )
    except Exception as exc:
        trace = traceback.format_exc()
        log.error("tool %s crashed: %s", name, exc, exc_info=False)
        res = ToolResult.failure(
            ErrorCode.INTERNAL,
            f"{name} failed with an internal error.",
            retryable=False,
            hint="Do not retry. Tell the user the action could not be completed.",
        )
        res.debug = trace[-1500:]
        return done(res)

    # 3. size-cap the payload
    data, truncated = truncate_data(data, max_chars)
    meta = {"truncated": True} if truncated else None
    return done(ToolResult.success(data, meta=meta))


def make_tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    handler: Handler,
    *,
    group: str,
    write: bool = False,
) -> StructuredTool:
    """Build a LangChain ``StructuredTool`` whose metadata carries the safe runner.

    The graph executes tools through ``tool.metadata["runner"]`` (never raises) and
    validates through ``tool.metadata["validate"]``. Calling ``tool.invoke`` directly
    also works and returns the JSON envelope.
    """

    def runner(
        raw_args: dict[str, Any] | None,
        *,
        timeout: float = 10.0,
        request_id: str | None = None,
    ) -> ToolResult:
        return execute_tool(name, args_schema, handler, raw_args, timeout=timeout, request_id=request_id)

    def validate(raw_args: dict[str, Any] | None) -> tuple[BaseModel | None, str | None]:
        try:
            return args_schema.model_validate(raw_args or {}), None
        except ValidationError as exc:
            return None, format_validation_error(exc)

    def func(**kwargs: Any) -> str:
        return runner(kwargs).envelope(name)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        func=func,
        metadata={"group": group, "write": write, "runner": runner, "validate": validate},
    )
