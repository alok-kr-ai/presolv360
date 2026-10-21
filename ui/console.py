"""The single shared rich Console used for every terminal write.

``Live`` displays, panels, tables, rich logging and tracebacks all use this
console so they never fight over the terminal.
"""
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.traceback import install as install_traceback

console = Console(highlight=False)

_configured = False


def setup_console(*, traceback: bool = True) -> Console:
    """Install rich tracebacks and route WARNING+ stdlib logs to the console."""
    global _configured
    if _configured:
        return console
    if traceback:
        install_traceback(console=console, show_locals=False)
    handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        level=logging.WARNING,
    )
    root = logging.getLogger("agent")
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    root.propagate = False
    _configured = True
    return console


def error_panel(title: str, message: str, hint: str | None = None) -> Panel:
    body = message if not hint else f"{message}\n\n[dim]{hint}[/dim]"
    return Panel(body, title=f"[bold red]{title}[/bold red]", border_style="red")


def warn_panel(title: str, message: str) -> Panel:
    return Panel(message, title=f"[bold yellow]{title}[/bold yellow]", border_style="yellow")
