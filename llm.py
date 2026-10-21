"""LLM access. Every LLM call in the app goes through litellm (via ``ChatLiteLLM``).

Fallbacks are handled here, one litellm call per model, instead of litellm's built-in ``fallbacks``
kwarg. Reasons: the built-in one cannot stream (it fails with ``'AsyncStream' object is not an
iterator`` once a fallback is needed), and ``ChatLiteLLM`` reports the *requested* model name even
when a fallback answered, which would make the logs lie. With an explicit chain we know exactly
which model answered, we can stream tokens from any of them, and each hop is an observable
``llm_fallback`` event.

* ``model_chain(settings)`` - ``[primary, *fallbacks]`` (model / provider errors).
* ``context_fallback_model`` - tried first when the error is a context-window overflow.
* ``LLMClient.call`` - the single call path used by every graph node (agent, router, confirmation
  classifier). Emits ``llm_start`` / ``llm_end`` per attempt, ``llm_fallback`` between attempts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from config import Settings
from events import EventBus
from observability.spans import span


class LLMError(RuntimeError):
    """The LLM could not produce a response (after trying every model in the chain)."""

    def __init__(self, message: str, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind


def quiet_litellm() -> None:
    """Keep litellm from writing to the terminal itself (all output goes through rich)."""
    import warnings

    import litellm

    # litellm builds responses whose usage block pydantic re-serialises with a noisy (harmless) warning
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
    litellm.suppress_debug_info = True
    litellm.drop_params = True  # e.g. reasoning models reject temperature=0; drop instead of failing
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def model_chain(settings: Settings) -> list[str]:
    """Primary model followed by its fallbacks (duplicates removed, order kept)."""
    return list(dict.fromkeys([settings.model, *settings.fallback_models]))


def build_chat_model(settings: Settings, model: str | None = None, *, streaming: bool | None = None) -> BaseChatModel:
    """One ``ChatLiteLLM`` for ``model`` (default: the primary model)."""
    from langchain_litellm import ChatLiteLLM

    quiet_litellm()
    return ChatLiteLLM(
        model=model or settings.model,
        temperature=settings.temperature,
        request_timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        streaming=settings.streaming if streaming is None else streaming,
    )


def classify_llm_error(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "contextwindow" in name or "context length" in text or "maximum context" in text or "context_length" in text:
        return "context_window"
    if "ratelimit" in name or "rate limit" in text:
        return "rate_limit"
    if "authentication" in name or "api key" in text or "incorrect api key" in text:
        return "auth"
    if "notfound" in name or "does not exist" in text or "model_not_found" in text:
        return "model_not_found"
    if "timeout" in name or "timed out" in text:
        return "timeout"
    if "connection" in name or "serviceunavailable" in name or "apiconnection" in name:
        return "connection"
    return "unknown"


class _TTFT(BaseCallbackHandler):
    """Captures time to first streamed token."""

    def __init__(self) -> None:
        self.first: float | None = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # noqa: D401
        if self.first is None:
            self.first = time.perf_counter()


@dataclass
class ModelEntry:
    """A model in the chain; the chat model is built lazily so unused fallbacks cost nothing."""

    name: str
    factory: Callable[[], BaseChatModel]
    _chat: BaseChatModel | None = field(default=None, repr=False)

    @property
    def chat(self) -> BaseChatModel:
        if self._chat is None:
            self._chat = self.factory()
        return self._chat


def build_chain(settings: Settings, *, streaming: bool) -> list[ModelEntry]:
    return [
        ModelEntry(name, lambda n=name: build_chat_model(settings, n, streaming=streaming))
        for name in model_chain(settings)
    ]


def build_context_entry(settings: Settings, *, streaming: bool) -> ModelEntry | None:
    name = settings.context_fallback_model
    if not name:
        return None
    return ModelEntry(name, lambda: build_chat_model(settings, name, streaming=streaming))


class LLMClient:
    def __init__(
        self,
        chain: Sequence[ModelEntry],
        classifier_chain: Sequence[ModelEntry] | None,
        settings: Settings,
        bus: EventBus,
        context_entry: ModelEntry | None = None,
        classifier_context_entry: ModelEntry | None = None,
    ) -> None:
        if not chain:
            raise ValueError("LLMClient needs at least one model")
        self.chain = list(chain)
        self.classifier_chain = list(classifier_chain or chain)
        self.context_entry = context_entry
        self.classifier_context_entry = classifier_context_entry if classifier_chain else context_entry
        self.settings = settings
        self.bus = bus
        self._dead: set[str] = set()  # models that failed permanently (bad name / key): skipped for the rest of the run

    @classmethod
    def from_model(cls, model: BaseChatModel, classifier: BaseChatModel | None, settings: Settings, bus: EventBus) -> "LLMClient":
        """Single injected model (tests): a chain of one, labelled with the configured model name."""
        primary = ModelEntry(settings.model, lambda: model, model)
        other = ModelEntry(settings.model, lambda: classifier, classifier) if classifier is not None else None
        return cls([primary], [other] if other else None, settings, bus)

    # ----------------------------------------------------------------------- call
    def call(
        self,
        state: dict[str, Any],
        purpose: str,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[Any] | None = None,
        config: Any = None,
    ) -> AIMessage:
        """Invoke the first model that works. Raises ``LLMError`` when every model failed."""
        is_agent = purpose == "agent"
        full = list(self.chain if is_agent else self.classifier_chain)
        queue = [e for e in full if e.name not in self._dead] or full  # never skip every model
        context_entry = self.context_entry if is_agent else self.classifier_context_entry
        primary = full[0].name
        tried: set[str] = set()
        first_error: LLMError | None = None

        while queue:
            entry = queue.pop(0)
            if entry.name in tried:
                continue
            tried.add(entry.name)
            try:
                return self._attempt(entry, primary, state, purpose, messages, tools, config)
            except LLMError as exc:
                first_error = first_error or exc
                if exc.kind in ("model_not_found", "auth") and len(full) > 1:
                    self._dead.add(entry.name)
                if exc.kind == "context_window" and context_entry and context_entry.name not in tried:
                    queue.insert(0, context_entry)  # a bigger window is the right fix for this error
                nxt = next((e for e in queue if e.name not in tried), None)
                if nxt is not None:
                    self.bus.emit(
                        "llm_fallback", state=state, component="llm", name=purpose, model=nxt.name,
                        error_code=exc.kind, failure_reason=str(exc)[:300], ok=False,
                        text=f"{entry.name} failed ({exc.kind}); trying {nxt.name}",
                        data={"from": entry.name, "to": nxt.name},
                    )
                else:
                    raise first_error if first_error.kind != "unknown" else exc
        raise first_error or LLMError("no model available")

    def _attempt(
        self,
        entry: ModelEntry,
        primary: str,
        state: dict[str, Any],
        purpose: str,
        messages: Sequence[BaseMessage],
        tools: Sequence[Any] | None,
        config: Any,
    ) -> AIMessage:
        try:
            base = entry.chat
            model = base.bind_tools(list(tools)) if tools else base
        except Exception as exc:  # misconfigured model string, missing provider package, ...
            raise LLMError(f"{type(exc).__name__}: {str(exc)[:300]}", classify_llm_error(exc)) from exc
        ttft = _TTFT()
        model = model.with_config({"callbacks": [ttft]})

        with span(f"llm:{purpose}", "llm") as sp:
            common = dict(state=state, component="llm", name=purpose, span_id=sp.span_id,
                          parent_span_id=sp.parent_id, step=state.get("steps"))
            self.bus.emit("llm_start", model=entry.name, fallback_used=entry.name != primary, **common)
            t0 = time.perf_counter()
            try:
                ai = model.invoke(list(messages), config=config)
            except Exception as exc:
                kind = classify_llm_error(exc)
                reason = f"{type(exc).__name__}: {str(exc)[:300]}"
                sp.fail(kind, reason)
                self.bus.emit(
                    "llm_end", model=entry.name, fallback_used=entry.name != primary,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2), ok=False,
                    error_code=kind, failure_reason=reason, **common,
                )
                raise LLMError(reason, kind) from exc

            latency = round((time.perf_counter() - t0) * 1000, 2)
            meta = getattr(ai, "response_metadata", None) or {}
            usage = getattr(ai, "usage_metadata", None) or {}
            if not isinstance(ai, AIMessage):  # defensive: some wrappers return chunks
                ai = AIMessage(content=getattr(ai, "content", ""), tool_calls=getattr(ai, "tool_calls", []) or [])
            self.bus.emit(
                "llm_end", model=entry.name, fallback_used=entry.name != primary,
                latency_ms=latency,
                ttft_ms=round((ttft.first - t0) * 1000, 2) if ttft.first else None,
                ok=True,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                finish_reason=meta.get("finish_reason"),
                **common,
            )
            return ai
