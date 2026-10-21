"""Mock backends: calendar, CRM (data from CSV) and fault injection, wired together by ``Backend``."""
from __future__ import annotations

from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from config import Settings
from mocks.calendar_store import CalendarStore
from mocks.crm_store import CRMStore
from mocks.faults import FaultInjector
from mocks.loader import load_contacts, load_events, load_interactions
from tools.base import ErrorCode


def make_clock(timezone: str) -> Callable[[], datetime]:
    tz = ZoneInfo(timezone)
    return lambda: datetime.now(tz)


class Backend:
    """The mocked services plus a shared fault injector."""

    def __init__(
        self,
        settings: Settings | None = None,
        clock: Callable[[], datetime] | None = None,
        faults: FaultInjector | None = None,
    ) -> None:
        settings = settings or Settings()
        self.tz = ZoneInfo(settings.timezone)
        self.clock = clock or make_clock(settings.timezone)
        self.faults = faults or FaultInjector(settings.mock_failure_rate, settings.mock_seed)
        data_dir = settings.mock_data_dir  # events.csv / contacts.csv / interactions.csv
        self.calendar = CalendarStore(
            self.tz, self.clock, self.faults, events=load_events(data_dir),
            persist_dir=data_dir if settings.mock_persist else None,
        )
        self.crm = CRMStore(
            self.tz, self.clock, self.faults,
            contacts=load_contacts(data_dir), interactions=load_interactions(data_dir),
            persist_dir=data_dir if settings.mock_persist else None,
        )

    def fail_next(
        self, tool: str, code: str | ErrorCode = ErrorCode.SERVER_ERROR, n: int = 1, *, after_commit: bool = False
    ) -> None:
        self.faults.fail_next(tool, code, n, after_commit=after_commit)
