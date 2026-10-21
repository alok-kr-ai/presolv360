"""Rich views over recorded run folders (`main.py logs ...`, `/runs`, `/trace`)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_run_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.is_dir():
        return []
    return sorted((p for p in log_dir.iterdir() if p.is_dir() and (p / "run.json").exists()), reverse=True)


def _duration(run: dict[str, Any]) -> str:
    try:
        start = datetime.fromisoformat(run["started_at"])
        end = datetime.fromisoformat(run["ended_at"]) if run.get("ended_at") else datetime.now(start.tzinfo)
        return f"{(end - start).total_seconds():.1f}s"
    except (KeyError, ValueError, TypeError):
        return "-"


def show_runs(console: Console, log_dir: Path) -> None:
    dirs = list_run_dirs(log_dir)
    if not dirs:
        console.print(f"[yellow]No runs found in {escape(str(log_dir))}[/yellow]")
        return
    table = Table(title="Runs", header_style="bold cyan")
    # the run id already encodes the UTC start time; the model is shown by `logs show`
    for col in ("run id", "status", "turns", "tools", "errors", "time"):
        # the run id is the key everything else is looked up by: never truncate it
        table.add_column(col, no_wrap=True, min_width=24 if col == "run id" else len(col),
                         overflow="fold" if col == "run id" else "ellipsis")
    for d in dirs:
        run = _load(d / "run.json") or {}
        summary = _load(d / "summary.json") or {}
        totals = summary.get("totals", {})
        status = run.get("status", "?")
        style = {"completed": "green", "running": "yellow"}.get(status, "red")
        table.add_row(
            d.name,
            f"[{style}]{status}[/{style}]",
            str(totals.get("turns", 0)),
            str(totals.get("tool_calls", 0)),
            str(totals.get("errors", 0)),
            _duration(run),
        )
    console.print(table)


def show_run(console: Console, log_dir: Path, run_id: str) -> bool:
    d = log_dir / run_id
    run, summary = _load(d / "run.json"), _load(d / "summary.json")
    if run is None or summary is None:
        console.print(f"[red]Run not found:[/red] {escape(run_id)}")
        return False
    totals = summary.get("totals", {})
    console.print(
        f"[bold]Run {escape(run_id)}[/bold]  status={run.get('status')}  "
        f"model={escape(str(run.get('model')))}  duration={_duration(run)}"
    )
    t = Table(title="Totals", show_header=False)
    for k, v in totals.items():
        t.add_row(k, str(v))
    mix = summary.get("turn_mix", {})
    t.add_row("no-tool turn share", str(mix.get("no_tool_share", 0)))
    t.add_row("tool turn share", str(mix.get("tool_share", 0)))
    console.print(t)

    tools = summary.get("tools", {})
    if tools:
        tt = Table(title="Tool performance", header_style="bold cyan")
        for col in ("tool", "calls", "ok", "errors", "retries", "avg ms", "p50", "p95", "max"):
            tt.add_column(col, justify="right" if col != "tool" else "left")
        for name, s in sorted(tools.items()):
            tt.add_row(
                name, str(s["calls"]), str(s["ok"]), escape(", ".join(f"{k}×{v}" for k, v in s["errors"].items())) if s["errors"] else "-",
                str(s["retries"]), str(s["avg_ms"]), str(s["p50_ms"]), str(s["p95_ms"]), str(s["max_ms"]),
            )
        console.print(tt)
    models = summary.get("llm", {})
    if models:
        mt = Table(title="LLM performance", header_style="bold cyan")
        for col in ("model", "calls", "tokens in", "tokens out", "fallbacks", "errors", "avg ms", "p95"):
            mt.add_column(col, justify="right" if col != "model" else "left")
        for name, s in sorted(models.items()):
            mt.add_row(
                escape(name), str(s["calls"]), str(s["prompt_tokens"]), str(s["completion_tokens"]),
                str(s["fallbacks_used"]), str(s["errors"]), str(s["avg_ms"]), str(s["p95_ms"]),
            )
        console.print(mt)
    nodes = summary.get("nodes", {})
    if nodes:
        nt = Table(title="Graph nodes", header_style="bold cyan")
        for col in ("node", "calls", "avg ms", "p95"):
            nt.add_column(col, justify="right" if col != "node" else "left")
        for name, s in nodes.items():
            nt.add_row(name, str(s["calls"]), str(s["avg_ms"]), str(s["p95_ms"]))
        console.print(nt)
    if summary.get("guard_blocks"):
        blocks = ", ".join(f"{k}×{v}" for k, v in summary["guard_blocks"].items())
        console.print(f"Guard blocks: {escape(blocks)}")
    router = summary.get("router", {})
    if router:
        groups = ", ".join(f"{k}×{v}" for k, v in router.get("group_choices", {}).items())
        methods = ", ".join(f"{k}×{v}" for k, v in router.get("methods", {}).items())
        console.print(f"Router groups: {escape(groups)}  |  methods: {escape(methods)}")
    return True


def _add_span(tree: Tree, span: dict[str, Any]) -> None:
    ok = span.get("ok")
    icon = "[green]✓[/green]" if ok in (True, None) else "[red]✗[/red]"
    err = f" [red]{escape(str(span['error_code']))}[/red]" if span.get("error_code") else ""
    label = (
        f"{icon} {escape(str(span.get('name')))} [dim]({escape(str(span.get('component')))})[/dim]"
        f"{err}  [cyan]{span.get('duration_ms', 0):.1f} ms[/cyan] [dim]@+{span.get('start_ms', 0):.0f}[/dim]"
    )
    node = tree.add(label)
    for child in span.get("children", []):
        _add_span(node, child)


def render_turn_trace(console: Console, trace: dict[str, Any]) -> None:
    metrics = trace.get("metrics", {})
    console.print(
        f"[bold]Turn {trace.get('turn')}[/bold] trace=[cyan]{escape(str(trace.get('trace_id')))}[/cyan] "
        f"status={trace.get('status')}  latency={metrics.get('latency_ms')} ms"
    )
    if trace.get("user_text"):
        console.print(f"[dim]user:[/dim] {escape(str(trace['user_text']))}")
    console.print(f"[dim]flow:[/dim] {escape(' -> '.join(trace.get('flow', [])))}")
    tree = Tree("[bold]spans[/bold]")
    root = trace.get("spans", {})
    _add_span(tree, root)
    console.print(tree)
    console.print(
        "[dim]steps={steps} tool_calls={tool_calls} retries={retries} guard_blocks={guard_blocks} "
        "llm_calls={llm_calls} errors={errors}[/dim]".format(**{k: metrics.get(k, 0) for k in
            ("steps", "tool_calls", "retries", "guard_blocks", "llm_calls", "errors")})
    )
    if trace.get("final_text"):
        console.print(f"[dim]answer:[/dim] {escape(str(trace['final_text']))}")


def show_turn(console: Console, log_dir: Path, run_id: str, turn: int) -> bool:
    turns_dir = log_dir / run_id / "turns"
    matches = sorted(turns_dir.glob(f"turn-{turn:03d}-*.json")) if turns_dir.is_dir() else []
    if not matches:
        console.print(f"[red]Turn {turn} not found in run {escape(run_id)}[/red]")
        return False
    trace = _load(matches[0])
    if trace is None:
        console.print("[red]Could not read turn trace[/red]")
        return False
    render_turn_trace(console, trace)
    return True
