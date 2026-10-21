"""RunRecorder: owns ``logs/<run_id>/`` for one run (CLI process / Streamlit session).

Layout::

    logs/<run_id>/
      run.json          metadata + status (running -> completed|interrupted|crashed)
      events.jsonl      every event, one JSON object per line, flushed immediately
      turns/turn-001-<trace_id>.json   full trace of one turn
      summary.json      aggregate metrics, rewritten after every turn and at exit
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import platform
import threading
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from config import Settings, collect_secret_values
from events import AgentEvent
from observability.context import root_span_id
from observability.metrics import RunMetrics

log = logging.getLogger("agent.observability")

MAX_TEXT = 4000
MAX_SUMMARY = 2000
SKIP_EVENTS = {"token"}  # UI-only, far too noisy for the log


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("langgraph", "litellm", "langchain-litellm", "pydantic", "rich"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _cap(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + f"...[+{len(value) - limit} chars]"


def _fingerprint(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class RunRecorder:
    def __init__(
        self,
        run_id: str,
        settings: Settings,
        *,
        thread_id: str | None = None,
        log_dir: Path | str | None = None,
        secrets: list[str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.settings = settings
        self.log_content = settings.log_content
        self.secrets = secrets if secrets is not None else collect_secret_values()
        self.metrics = RunMetrics()
        self.enabled = True
        self.warning: str | None = None
        self.status = "running"
        self.threads: list[str] = [thread_id] if thread_id else []
        self.started_at = _now()
        self._lock = threading.Lock()
        self._events_fh = None
        self._turn_events: dict[str, list[dict[str, Any]]] = {}
        self._turn_order: list[str] = []
        self.last_trace_id: str | None = None

        base = Path(log_dir) if log_dir is not None else settings.log_dir
        self.run_dir = Path(base) / run_id
        self.turns_dir = self.run_dir / "turns"
        try:
            self.turns_dir.mkdir(parents=True, exist_ok=True)
            self._events_fh = open(self.run_dir / "events.jsonl", "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            self.enabled = False
            self.warning = f"Run logging disabled: cannot write to {self.run_dir} ({exc})"
            log.warning(self.warning)
        else:
            self._write_run_json()
            self._write_summary()
        atexit.register(self._atexit)

    # ---------------------------------------------------------------- sanitising
    def _redact(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                text = text.replace(secret, "***")
        return text

    def _sanitize(self, event: AgentEvent) -> dict[str, Any]:
        d = event.to_dict()
        text = d.get("text")
        if text is not None:
            d["text"] = _cap(text, MAX_TEXT) if self.log_content else f"<redacted len={len(text)} sha={_fingerprint(text)}>"
        args = d.get("args")
        if args is not None and not self.log_content:
            d["args"] = {"_redacted": _fingerprint(args), "keys": sorted(args)}
        summary = d.get("result_summary")
        if summary is not None:
            d["result_summary"] = (
                _cap(summary, MAX_SUMMARY)
                if self.log_content
                else f"<redacted len={len(summary)} sha={_fingerprint(summary)}>"
            )
        if not self.log_content and isinstance(d.get("data"), dict):
            d["data"] = {k: v for k, v in d["data"].items() if k in _SAFE_DATA_KEYS}
        return d

    # ------------------------------------------------------------------- record
    def record(self, event: AgentEvent) -> None:
        """EventBus sink."""
        if event.event in SKIP_EVENTS:
            return
        raw = event.to_dict()
        d = self._sanitize(event)
        with self._lock:
            self.metrics.update(raw)
            buf = self._turn_events.setdefault(event.trace_id, [])
            if event.trace_id and event.trace_id not in self._turn_order:
                self._turn_order.append(event.trace_id)
            buf.append(d)
            self.last_trace_id = event.trace_id or self.last_trace_id
            if self.enabled and self._events_fh is not None:
                line = self._redact(json.dumps(d, default=str, ensure_ascii=False))
                try:
                    self._events_fh.write(line + "\n")
                    self._events_fh.flush()
                except OSError as exc:
                    self.enabled = False
                    self.warning = f"Run logging disabled: write failed ({exc})"
                    log.warning(self.warning)
            if event.event == "turn_end":
                self._finalize_turn(event.trace_id, event.turn)
                self._write_summary()

    # --------------------------------------------------------------- turn files
    def turn_events(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        trace_id = trace_id or self.last_trace_id
        return list(self._turn_events.get(trace_id or "", []))

    def _finalize_turn(self, trace_id: str, turn: int) -> None:
        events = self._turn_events.get(trace_id, [])
        trace = build_turn_trace(self.run_id, trace_id, turn, events)
        if not self.enabled:
            return
        path = self.turns_dir / f"turn-{turn:03d}-{trace_id}.json"
        try:
            self._atomic_write(path, trace)
        except OSError as exc:
            log.warning("could not write turn trace %s: %s", path, exc)

    # ---------------------------------------------------------------- run files
    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        text = self._redact(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_run_json(self) -> None:
        if not self.enabled:
            return
        payload = {
            "run_id": self.run_id,
            "threads": self.threads,
            "started_at": self.started_at,
            "ended_at": None if self.status == "running" else _now(),
            "status": self.status,
            "model": self.settings.model,
            "fallback_models": self.settings.fallback_models,
            "context_fallback_model": self.settings.context_fallback_model,
            "settings": self.settings.redacted(),
            "versions": _versions(),
        }
        try:
            self._atomic_write(self.run_dir / "run.json", payload)
        except OSError as exc:
            log.warning("could not write run.json: %s", exc)

    def _write_summary(self) -> None:
        if not self.enabled:
            return
        payload = {"run_id": self.run_id, "status": self.status, **self.metrics.snapshot()}
        try:
            self._atomic_write(self.run_dir / "summary.json", payload)
        except OSError as exc:
            log.warning("could not write summary.json: %s", exc)

    def add_thread(self, thread_id: str) -> None:
        with self._lock:
            if thread_id not in self.threads:
                self.threads.append(thread_id)
                self._write_run_json()

    def set_status(self, status: str) -> None:
        """Change the run status (e.g. ``crashed``) without closing the log files."""
        with self._lock:
            self.status = status
            self._write_run_json()
            self._write_summary()

    def close(self, status: str = "completed") -> None:
        """Flip run.json out of ``running`` and flush summary. Idempotent."""
        with self._lock:
            if self.status != "running":
                return
            self.status = status
            self._write_run_json()
            self._write_summary()
            if self._events_fh is not None:
                try:
                    self._events_fh.flush()
                    self._events_fh.close()
                except OSError:
                    pass
                self._events_fh = None

    def _atexit(self) -> None:
        if self.status == "running":
            self.close("interrupted")


_SAFE_DATA_KEYS = {"method", "decision", "tool_calls", "retries", "guard_blocks", "llm_calls", "status", "fail_open", "keys"}


# ---------------------------------------------------------------------------
# Per-turn trace construction (also used by tests and the `logs` reports)
# ---------------------------------------------------------------------------
def build_turn_trace(run_id: str, trace_id: str, turn: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    user_text = next((e.get("text") for e in events if e["event"] == "turn_start"), None)
    final_text = next((e.get("text") for e in reversed(events) if e["event"] == "final"), None)
    end = next((e for e in reversed(events) if e["event"] == "turn_end"), {})
    started = next((e["ts"] for e in events if e["event"] == "turn_start"), None)

    # flow
    flow: list[str] = []
    pending_tools: list[str] = []
    pending_blocks: list[str] = []
    for e in events:
        kind = e["event"]
        if kind == "tool_result":
            status = "ok" if e.get("ok") else f"err:{e.get('error_code', '?')}"
            pending_tools.append(f"{e.get('tool') or e.get('name')} {status}")
        elif kind == "guard_block":
            pending_blocks.append(e.get("guard_reason", "blocked"))
        elif kind == "node_end":
            name = e.get("name", "?")
            if name == "tools":
                flow.append(f"tools({', '.join(pending_tools)})" if pending_tools else "tools")
                pending_tools = []
            elif name == "guard":
                flow.append(f"guard(blocked: {', '.join(pending_blocks)})" if pending_blocks else "guard")
                pending_blocks = []
            else:
                flow.append(name)
    if any(e["event"] == "final" for e in events):
        flow.append("final")

    # spans
    root = {
        "span_id": root_span_id(trace_id),
        "name": "turn",
        "component": "turn",
        "start_ms": 0.0,
        "duration_ms": end.get("latency_ms") or end.get("elapsed_ms") or 0.0,
        "ok": (end.get("data") or {}).get("status", "ok") in ("ok", "completed"),
        "children": [],
    }
    nodes: dict[str, dict[str, Any]] = {root["span_id"]: root}
    ordered: list[dict[str, Any]] = []
    for e in events:
        if e["event"] in ("node_end", "llm_end", "tool_result") and e.get("span_id"):
            latency = e.get("latency_ms") or 0.0
            sp = {
                "span_id": e["span_id"],
                "parent_span_id": e.get("parent_span_id"),
                "name": e.get("tool") or e.get("model") or e.get("name"),
                "component": e.get("component") or e["event"],
                "start_ms": round(max((e.get("elapsed_ms") or 0.0) - latency, 0.0), 2),
                "duration_ms": latency,
                "ok": e.get("ok"),
                "error_code": e.get("error_code"),
                "children": [],
            }
            nodes[sp["span_id"]] = sp
            ordered.append(sp)
    for sp in ordered:
        parent = nodes.get(sp.get("parent_span_id") or "", root)
        parent["children"].append(sp)
    for sp in nodes.values():
        sp["children"].sort(key=lambda s: s["start_ms"])

    tool_calls = [
        {
            "tool": e.get("tool"),
            "args": e.get("args"),
            "ok": e.get("ok"),
            "error_code": e.get("error_code"),
            "latency_ms": e.get("latency_ms"),
            "attempts": e.get("attempt"),
            "retries_exhausted": e.get("retries_exhausted"),
            "result_summary": e.get("result_summary"),
        }
        for e in events
        if e["event"] == "tool_result"
    ]
    data = end.get("data") or {}
    counts = {
        "steps": end.get("step", 0),
        "tool_calls": len(tool_calls),
        "retries": sum(1 for e in events if e["event"] == "tool_retry"),
        "guard_blocks": sum(1 for e in events if e["event"] == "guard_block"),
        "llm_calls": sum(1 for e in events if e["event"] == "llm_end"),
        "errors": sum(1 for e in events if e.get("ok") is False and e["event"] in ("tool_result", "llm_end")),
    }
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "turn": turn,
        "started_at": started,
        "status": data.get("status", "completed"),
        "user_text": user_text,
        "final_text": final_text,
        "metrics": {**counts, "latency_ms": end.get("latency_ms")},
        "flow": flow,
        "spans": root,
        "tool_calls": tool_calls,
        "events": events,
    }
