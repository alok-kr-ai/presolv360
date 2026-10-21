"""Streaming aggregation of per-run performance metrics."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100)."""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return data[int(k)]
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def _lat_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "max_ms": round(max(values), 2),
    }


class RunMetrics:
    """Updated per event; ``snapshot()`` yields the summary.json payload."""

    def __init__(self) -> None:
        self.turns = 0
        self.steps = 0
        self.tool_calls = 0
        self.llm_calls = 0
        self.errors = 0
        self.turn_latencies: list[float] = []
        self.no_tool_turns = 0
        self.tool_turns = 0
        self._tool_lat: dict[str, list[float]] = defaultdict(list)
        self._tool_ok: Counter[str] = Counter()
        self._tool_calls: Counter[str] = Counter()
        self._tool_errors: dict[str, Counter[str]] = defaultdict(Counter)
        self._tool_retries: Counter[str] = Counter()
        self._llm_lat: dict[str, list[float]] = defaultdict(list)
        self._llm_calls: Counter[str] = Counter()
        self._llm_tokens_in: Counter[str] = Counter()
        self._llm_tokens_out: Counter[str] = Counter()
        self._llm_fallbacks: Counter[str] = Counter()
        self._llm_errors: Counter[str] = Counter()
        self._node_lat: dict[str, list[float]] = defaultdict(list)
        self.guard_blocks: Counter[str] = Counter()
        self.route_groups: Counter[str] = Counter()
        self.route_methods: Counter[str] = Counter()

    # ------------------------------------------------------------------ update
    def update(self, e: dict[str, Any]) -> None:
        kind = e.get("event")
        lat = e.get("latency_ms")
        if kind == "tool_result":
            tool = e.get("tool") or e.get("name") or "?"
            self.tool_calls += 1
            self._tool_calls[tool] += 1
            if lat is not None:
                self._tool_lat[tool].append(float(lat))
            if e.get("ok"):
                self._tool_ok[tool] += 1
            else:
                self.errors += 1
                self._tool_errors[tool][e.get("error_code") or "UNKNOWN"] += 1
        elif kind == "tool_retry":
            self._tool_retries[e.get("tool") or "?"] += 1
        elif kind == "llm_end":
            model = e.get("model") or "?"
            self.llm_calls += 1
            self._llm_calls[model] += 1
            if lat is not None:
                self._llm_lat[model].append(float(lat))
            self._llm_tokens_in[model] += int(e.get("prompt_tokens") or 0)
            self._llm_tokens_out[model] += int(e.get("completion_tokens") or 0)
            if e.get("fallback_used"):
                self._llm_fallbacks[model] += 1
            if e.get("ok") is False:
                self.errors += 1
                self._llm_errors[model] += 1
        elif kind == "node_end":
            if lat is not None:
                self._node_lat[e.get("name") or "?"].append(float(lat))
        elif kind == "guard_block":
            self.guard_blocks[e.get("guard_reason") or "UNKNOWN"] += 1
        elif kind == "route":
            for group in e.get("groups") or []:
                self.route_groups[group] += 1
            if not e.get("groups"):
                self.route_groups["(none)"] += 1
            self.route_methods[e.get("data", {}).get("method", "?") if isinstance(e.get("data"), dict) else "?"] += 1
        elif kind == "turn_end":
            self.turns += 1
            self.steps += int(e.get("step") or 0)
            tool_calls = (e.get("data") or {}).get("tool_calls", 0)
            if tool_calls:
                self.tool_turns += 1
            else:
                self.no_tool_turns += 1
            if lat is not None:
                self.turn_latencies.append(float(lat))

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        tools = {}
        for tool, calls in self._tool_calls.items():
            tools[tool] = {
                "calls": calls,
                "ok": self._tool_ok[tool],
                "errors": dict(self._tool_errors[tool]),
                "retries": self._tool_retries[tool],
                **_lat_stats(self._tool_lat[tool]),
            }
        models = {}
        for model, calls in self._llm_calls.items():
            models[model] = {
                "calls": calls,
                "prompt_tokens": self._llm_tokens_in[model],
                "completion_tokens": self._llm_tokens_out[model],
                "fallbacks_used": self._llm_fallbacks[model],
                "errors": self._llm_errors[model],
                **_lat_stats(self._llm_lat[model]),
            }
        nodes = {name: {"calls": len(vals), **_lat_stats(vals)} for name, vals in self._node_lat.items()}
        total_turns = max(self.turns, 1)
        return {
            "totals": {
                "turns": self.turns,
                "steps": self.steps,
                "tool_calls": self.tool_calls,
                "llm_calls": self.llm_calls,
                "errors": self.errors,
                "total_turn_latency_ms": round(sum(self.turn_latencies), 2),
                "avg_turn_latency_ms": round(sum(self.turn_latencies) / len(self.turn_latencies), 2)
                if self.turn_latencies
                else 0.0,
            },
            "tools": tools,
            "llm": models,
            "nodes": nodes,
            "guard_blocks": dict(self.guard_blocks),
            "router": {
                "group_choices": dict(self.route_groups),
                "methods": dict(self.route_methods),
            },
            "turn_mix": {
                "no_tool_turns": self.no_tool_turns,
                "tool_turns": self.tool_turns,
                "no_tool_share": round(self.no_tool_turns / total_turns, 3) if self.turns else 0.0,
                "tool_share": round(self.tool_turns / total_turns, 3) if self.turns else 0.0,
            },
        }
