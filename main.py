"""Entry point.

    python main.py                       interactive rich CLI
    python main.py logs                  list recorded runs
    python main.py logs show <run_id>    performance summary of a run
    python main.py logs turn <run_id> N  span tree of turn N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ui.console import console, error_panel, setup_console

OPTIONS = [
    ("-v, --verbose", "show tool arguments, result JSON and guard details"),
    ("-q, --quiet", "show only the final answer"),
    ("--no-live", "plain one-line-per-event output (also automatic when not a terminal)"),
    ("--model MODEL", "override LLM_MODEL for this run"),
    ("--thread ID", "conversation thread id to use"),
    ("--log-dir DIR", "where per-run logs are written (default: logs)"),
    ("-h, --help", "show this help"),
]
COMMANDS = [
    ("python main.py", "start the interactive agent"),
    ("python main.py logs", "list recorded runs"),
    ("python main.py logs show <run_id>", "run summary with tool and LLM performance"),
    ("python main.py logs turn <run_id> <n>", "span tree for turn n of a run"),
]


def print_help(c: Console) -> None:
    from ui.cli import HELP_COMMANDS

    usage = Table(title="Usage", show_header=False, box=None)
    for cmd, desc in COMMANDS:
        usage.add_row(f"[cyan]{escape(cmd)}[/cyan]", escape(desc))
    opts = Table(title="Options", show_header=False, box=None)
    for opt, desc in OPTIONS:
        opts.add_row(f"[cyan]{escape(opt)}[/cyan]", escape(desc))
    repl = Table(title="In the REPL", show_header=False, box=None)
    for cmd, desc in HELP_COMMANDS:
        repl.add_row(f"[cyan]{escape(cmd)}[/cyan]", escape(desc))
    c.print(usage)
    c.print(opts)
    c.print(repl)


class RichParser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        console.print(error_panel("Invalid arguments", message, "Run `python main.py --help` for usage."))
        raise SystemExit(2)

    def print_help(self, file=None):  # noqa: D401 - rendered with rich instead of argparse
        print_help(console)


def build_parser() -> argparse.ArgumentParser:
    p = RichParser(prog="main.py", add_help=False)
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--no-live", action="store_true")
    p.add_argument("--model")
    p.add_argument("--thread")
    p.add_argument("--log-dir")
    p.add_argument("command", nargs="*", help="logs [show RUN_ID | turn RUN_ID N]")
    return p


def run_logs(args: list[str], log_dir: Path) -> int:
    from observability import report

    if not args:
        report.show_runs(console, log_dir)
        return 0
    sub, rest = args[0], args[1:]
    if sub == "show" and len(rest) == 1:
        return 0 if report.show_run(console, log_dir, rest[0]) else 1
    if sub == "turn" and len(rest) == 2 and rest[1].isdigit():
        return 0 if report.show_turn(console, log_dir, rest[0], int(rest[1])) else 1
    console.print(error_panel("Usage", "python main.py logs [show <run_id> | turn <run_id> <n>]"))
    return 2


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    if args.help:
        print_help(console)
        return 0

    from config import Settings

    settings = Settings.from_env()
    if args.model:
        settings = settings.model_copy(update={"model": args.model, "fallback_models": [m for m in settings.fallback_models if m != args.model]})
    if args.log_dir:
        settings = settings.model_copy(update={"log_dir": Path(args.log_dir)})

    if args.command and args.command[0] == "logs":
        return run_logs(args.command[1:], settings.log_dir)
    if args.command:
        console.print(error_panel("Unknown command", f"'{' '.join(args.command)}'", "Run `python main.py --help`."))
        return 2

    from graph.builder import build_agent
    from ui.cli import CliOptions, run_repl

    from mocks.loader import MockDataError

    try:
        agent = build_agent(settings, thread_id=args.thread)
    except MockDataError as exc:
        console.print(error_panel("Mock data error", str(exc), f"Fix the CSV files in {settings.mock_data_dir} (MOCK_DATA_DIR)."))
        return 1
    opts = CliOptions(verbose=args.verbose, quiet=args.quiet, live=not args.no_live)
    return run_repl(console, agent, settings, opts, print_help)


if __name__ == "__main__":
    sys.exit(main())
