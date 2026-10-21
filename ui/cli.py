"""Rich REPL for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from config import Settings
from graph.agent import Agent
from observability import report
from observability.recorder import build_turn_trace
from ui.console import error_panel, warn_panel
from ui.render import pending_banner, render_turn

HELP_COMMANDS = [
    ("/help", "show this help"),
    ("/new", "start a new conversation thread (fresh memory)"),
    ("/memory", "show working memory and any pending action"),
    ("/trace", "show the span tree of the last turn"),
    ("/runs", "list recorded runs in the log folder"),
    ("/quit", "exit"),
]


@dataclass
class CliOptions:
    verbose: bool = False
    quiet: bool = False
    live: bool = True

    @property
    def mode(self) -> str:
        if self.quiet:
            return "quiet"
        return "live" if self.live else "plain"


def banner(console: Console, agent: Agent, settings: Settings) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("model", escape(settings.model))
    table.add_row("fallbacks", escape(", ".join(settings.fallback_models) or "-"))
    table.add_row("timezone", escape(settings.timezone))
    table.add_row("thread", escape(agent.thread_id))
    table.add_row("run id", escape(agent.run_id))
    table.add_row("logs", escape(str(agent.recorder.run_dir)) if agent.recorder.enabled else "[yellow]disabled[/yellow]")
    if agent.langsmith:
        table.add_row("langsmith", "tracing on")
    console.print(
        Panel(table, title="[bold cyan]Presolv agent[/bold cyan]  [dim]calendar · crm · calculator[/dim]", border_style="cyan")
    )
    console.print("[dim]Type a message, or /help. Ctrl+C cancels a turn, Ctrl+D exits.[/dim]")


def check_credentials(console: Console, settings: Settings) -> None:
    try:
        import litellm

        info = litellm.validate_environment(model=settings.model)
    except Exception:
        return
    if not info.get("keys_in_environment", True):
        missing = ", ".join(info.get("missing_keys", [])) or "provider credentials"
        console.print(
            warn_panel(
                "Model credentials not found",
                f"No credentials found for [bold]{escape(settings.model)}[/bold] (missing: {escape(missing)}).\n"
                "Set them in .env (e.g. OPENAI_API_KEY=...) or choose another model with LLM_MODEL. "
                "Fallback models are still tried if their credentials exist.",
            )
        )


def show_memory(console: Console, agent: Agent) -> None:
    mem = agent.memory()
    table = Table(title="Working memory", show_header=True, header_style="bold cyan")
    table.add_column("key")
    table.add_column("value")
    data = mem.model_dump(mode="json")
    for key in ("last_event", "last_contact", "last_calculation", "listed_events"):
        value = data.get(key)
        table.add_row(key, escape(str(value)) if value else "[dim]-[/dim]")
    pending = mem.pending_action
    if pending:
        state = "confirmed" if pending.confirmed else "awaiting confirmation"
        table.add_row("pending_action", escape(f"{pending.summary} [{state}]"))
    else:
        table.add_row("pending_action", "[dim]-[/dim]")
    console.print(table)


def show_last_trace(console: Console, agent: Agent) -> None:
    events = agent.recorder.turn_events()
    if not events:
        console.print("[yellow]No turn recorded yet.[/yellow]")
        return
    turn = next((e["turn"] for e in events if e["event"] == "turn_start"), 0)
    report.render_turn_trace(console, build_turn_trace(agent.run_id, events[0]["trace_id"], turn, events))


def handle_command(console: Console, agent: Agent, settings: Settings, line: str, print_help: Any) -> bool:
    """Returns True if the REPL should exit."""
    cmd = line.split()[0]
    if cmd in ("/quit", "/exit", "/q"):
        return True
    if cmd == "/help":
        print_help(console)
    elif cmd == "/new":
        console.print(f"[green]New thread:[/green] {escape(agent.new_thread())}")
    elif cmd == "/memory":
        show_memory(console, agent)
    elif cmd == "/trace":
        show_last_trace(console, agent)
    elif cmd == "/runs":
        report.show_runs(console, settings.log_dir)
    else:
        console.print(f"[yellow]Unknown command {escape(cmd)}.[/yellow] Try /help")
    return False


def run_repl(console: Console, agent: Agent, settings: Settings, opts: CliOptions, print_help: Any) -> int:
    banner(console, agent, settings)
    check_credentials(console, settings)
    status = "completed"
    while True:
        try:
            line = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except EOFError:
            console.print()
            break
        except KeyboardInterrupt:
            console.print("\n[dim]bye[/dim]")
            status = "interrupted"
            break
        if not line:
            continue
        if line.startswith("/"):
            if handle_command(console, agent, settings, line, print_help):
                break
            continue
        try:
            view = render_turn(console, agent.stream(line), mode=opts.mode, verbose=opts.verbose)
        except KeyboardInterrupt:
            console.print("[yellow]Turn cancelled.[/yellow]")
            continue
        except Exception:
            console.print(error_panel("Internal error", "The agent crashed while handling that message; the error was logged."))
            console.print_exception(show_locals=False)
            continue
        if view.status == "interrupted":
            console.print("[yellow]Turn cancelled.[/yellow]")
        pending = agent.memory().pending_action
        if pending and not pending.confirmed:
            console.print(pending_banner(pending.summary))
    agent.close(status)
    if agent.recorder.enabled:
        console.print(f"[dim]Run log: {escape(str(agent.recorder.run_dir))}[/dim]")
    return 0
