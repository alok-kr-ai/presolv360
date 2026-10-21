"""Mock calendar service, seeded from ``events.csv`` and (optionally) persisted back to it.

When ``persist_dir`` is set, every committed create / update / delete rewrites ``events.csv`` so the
file always reflects the calendar. Rows still at their hand-written ``day_offset`` keep it (the demo
data stays relative to "today"); anything the agent creates or moves is stored with absolute dates.

Deletes are real (the event leaves the store, so ``check_availability`` and
``list_calendar_events`` agree), deleted ids are remembered as tombstones so a repeat
delete reports ``already_gone``, and creates are idempotent per request id.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable

from mocks.faults import FaultInjector
from mocks.loader import EventSeed, save_events
from tools.base import ErrorCode, ToolFailure

log = logging.getLogger("agent.mocks")


@dataclass
class Event:
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = field(default_factory=list)
    location: str | None = None
    description: str | None = None
    seed_offset: int | None = None  # set while the event still sits at its CSV ``day_offset``

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "attendees": list(self.attendees),
            "location": self.location,
            "description": self.description,
        }


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


class CalendarStore:
    def __init__(
        self,
        tz: tzinfo,
        clock: Callable[[], datetime],
        faults: FaultInjector | None = None,
        *,
        events: list[EventSeed] | None = None,
        persist_dir: Path | str | None = None,
    ) -> None:
        self.tz = tz
        self.clock = clock
        self.faults = faults or FaultInjector()
        self._events: dict[str, Event] = {}
        self._tombstones: set[str] = set()
        self._requests: dict[str, str] = {}
        self._counter = 0
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._lock = threading.RLock()
        self._seed(events or [])

    # ------------------------------------------------------------------ seeding (from events.csv)
    def _seed(self, seeds: list[EventSeed]) -> None:
        """Create the CSV rows as events (``day_offset`` / ``date`` and times are in ``self.tz``)."""
        today = self.clock().astimezone(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        # explicit ids first, so ids generated for rows without one can never collide with them
        self._counter = max((int(s.id.split("_")[1]) for s in seeds if s.id), default=0)
        for seed in seeds:
            if seed.date is not None:
                day = today.replace(year=seed.date.year, month=seed.date.month, day=seed.date.day)
            else:
                day = today + timedelta(days=seed.day_offset or 0)
            end_day = day if seed.end_date is None else today.replace(
                year=seed.end_date.year, month=seed.end_date.month, day=seed.end_date.day
            )
            start = day.replace(hour=seed.start.hour, minute=seed.start.minute, second=seed.start.second)
            end = end_day.replace(hour=seed.end.hour, minute=seed.end.minute, second=seed.end.second)
            event = self._add(seed.title, start, end, list(seed.attendees), seed.location, seed.description, seed.id)
            if seed.date is None and seed.end_date is None:
                event.seed_offset = seed.day_offset

    def _persist(self) -> None:
        """Write the calendar back to ``events.csv`` (no-op unless persistence is on). Never raises:
        the in-memory change already happened, so a disk problem is logged, not turned into a tool error."""
        if self._persist_dir is None:
            return

        def clock_text(dt: datetime) -> str:
            return dt.strftime("%H:%M" if dt.second == 0 else "%H:%M:%S")

        rows = []
        for e in self._events.values():
            same_day = e.start.date() == e.end.date()
            relative = e.seed_offset is not None and same_day
            rows.append({
                "id": e.id,
                "title": e.title,
                "date": None if relative else e.start.date().isoformat(),
                "day_offset": e.seed_offset if relative else None,
                "start": clock_text(e.start),
                "end": clock_text(e.end),
                "end_date": None if same_day else e.end.date().isoformat(),
                "attendees": ";".join(e.attendees),
                "location": e.location,
                "description": e.description,
            })
        try:
            save_events(self._persist_dir, rows)
        except OSError as exc:
            log.warning("could not write %s/events.csv: %s", self._persist_dir, exc)

    def _add(
        self,
        title: str,
        start: datetime,
        end: datetime,
        attendees: list[str],
        location: str | None,
        description: str | None,
        event_id: str | None = None,
    ) -> Event:
        if event_id is None:
            self._counter += 1
            event_id = f"evt_{self._counter:03d}"
        event = Event(
            id=event_id,
            title=title,
            start=start.astimezone(self.tz),
            end=end.astimezone(self.tz),
            attendees=list(attendees),
            location=location,
            description=description,
        )
        self._events[event.id] = event
        return event

    # ---------------------------------------------------------------- helpers
    def _conflicts(self, start: datetime, end: datetime, ignore_id: str | None = None) -> list[Event]:
        return sorted(
            (e for e in self._events.values() if e.id != ignore_id and _overlaps(start, end, e.start, e.end)),
            key=lambda e: e.start,
        )

    def get(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def all_events(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda e: e.start)

    # -------------------------------------------------------------- operations
    def list_events(self, start: datetime, end: datetime, query: str | None = None, limit: int = 50) -> dict[str, Any]:
        self.faults.maybe_fail("list_calendar_events")
        with self._lock:
            matches = [e for e in self._events.values() if _overlaps(start, end, e.start, e.end)]
            if query:
                q = query.lower()
                matches = [
                    e
                    for e in matches
                    if q in e.title.lower()
                    or q in (e.location or "").lower()
                    or any(q in a.lower() for a in e.attendees)
                ]
            matches.sort(key=lambda e: e.start)
            events = [e.to_dict() for e in matches[:limit]]
        out: dict[str, Any] = {"events": events, "count": len(events), "range": {"start": start.isoformat(), "end": end.isoformat()}}
        if not events:
            out["note"] = "No events in this range."
        return out

    def check_availability(self, start: datetime, end: datetime) -> dict[str, Any]:
        self.faults.maybe_fail("check_availability")
        with self._lock:
            conflicts = self._conflicts(start, end)
            return {
                "available": not conflicts,
                "range": {"start": start.isoformat(), "end": end.isoformat()},
                "conflicts": [e.to_dict() for e in conflicts],
            }

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        attendees: list[str],
        location: str | None,
        description: str | None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.faults.maybe_fail("create_calendar_event")
        with self._lock:
            if request_id and request_id in self._requests:
                existing = self._events.get(self._requests[request_id])
                if existing is not None:
                    return {"event": existing.to_dict(), "idempotent_replay": True}
            conflicts = self._conflicts(start, end)
            if conflicts:
                raise ToolFailure(
                    ErrorCode.CONFLICT,
                    f"The slot overlaps {len(conflicts)} existing event(s).",
                    hint="Offer the user another time, or ask whether to double-book / move the existing event.",
                    data={"conflicts": [e.to_dict() for e in conflicts]},
                )
            event = self._add(title, start, end, attendees, location, description)
            if request_id:
                self._requests[request_id] = event.id
            self._persist()
        self.faults.maybe_fail("create_calendar_event", phase="after")
        return {"event": event.to_dict()}

    def update_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self.faults.maybe_fail("update_calendar_event")
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                gone = event_id in self._tombstones
                raise ToolFailure(
                    ErrorCode.NOT_FOUND,
                    f"No event with id {event_id}" + (" (it was deleted)." if gone else "."),
                    hint="Call list_calendar_events to find the correct event_id.",
                )
            start = (changes.get("start") or event.start).astimezone(self.tz)
            end = (changes.get("end") or event.end).astimezone(self.tz)
            if end <= start:
                raise ToolFailure(ErrorCode.INVALID_ARGS, "The event end must be after its start.")
            conflicts = self._conflicts(start, end, ignore_id=event_id)
            if conflicts:
                raise ToolFailure(
                    ErrorCode.CONFLICT,
                    f"The new time overlaps {len(conflicts)} existing event(s).",
                    hint="Offer the user another time.",
                    data={"conflicts": [e.to_dict() for e in conflicts]},
                )
            if changes.get("start") is not None or changes.get("end") is not None:
                event.seed_offset = None  # moved: no longer at its hand-written relative day
            event.start, event.end = start, end
            for key in ("title", "attendees", "location", "description"):
                if changes.get(key) is not None:
                    setattr(event, key, changes[key])
            result = {"event": event.to_dict()}
            self._persist()
        self.faults.maybe_fail("update_calendar_event", phase="after")
        return result

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self.faults.maybe_fail("delete_calendar_event")
        with self._lock:
            event = self._events.pop(event_id, None)
            if event is not None:
                self._tombstones.add(event_id)
                result = {"status": "deleted", "event_id": event_id, "title": event.title}
                self._persist()
            elif event_id in self._tombstones:
                return {"status": "already_gone", "event_id": event_id}
            else:
                raise ToolFailure(
                    ErrorCode.NOT_FOUND,
                    f"No event with id {event_id}.",
                    hint="Call list_calendar_events to get a valid event_id; never guess ids.",
                )
        self.faults.maybe_fail("delete_calendar_event", phase="after")
        return result
