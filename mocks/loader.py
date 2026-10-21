"""Load the mock data sets from CSV files.

``events.csv``        id, title, date, day_offset, start, end, end_date, attendees, location, description
                      Give each row either ``date`` (YYYY-MM-DD) or ``day_offset`` (relative to *today*:
                      0 = today, 1 = tomorrow, so hand-written demo data never goes stale). start/end are
                      HH:MM wall-clock times in the agent's timezone; ``end_date`` is only needed when an
                      event ends on a later day. ``id`` (evt_NNN) is optional. attendees are ``;``-separated.
                      Events the agent creates / changes / deletes are written back to this file
                      (``MOCK_PERSIST=false`` turns that off).
``contacts.csv``      id, name, email, company, title, phone  (new contacts are written back, like events)
``interactions.csv``  contact_id, days_ago, type, subject, summary

Point ``MOCK_DATA_DIR`` at another folder with the same three files to use your own data.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class MockDataError(ValueError):
    """A mock CSV file is missing, malformed or has an invalid value."""


@dataclass(frozen=True)
class EventSeed:
    title: str
    start: time
    end: time
    day_offset: int | None = None  # relative to today ...
    date: date | None = None  # ... or an absolute day (exactly one of the two is set)
    end_date: date | None = None
    id: str | None = None
    attendees: tuple[str, ...] = ()
    location: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class InteractionSeed:
    contact_id: str
    days_ago: int
    type: str
    subject: str
    summary: str


def _rows(path: Path, required: tuple[str, ...]) -> list[tuple[int, dict[str, str | None]]]:
    """(line number, row) pairs; blank cells become ``None`` and surrounding spaces are stripped."""
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise MockDataError(f"Cannot read mock data file {path}: {exc.strerror or exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise MockDataError(f"{path.name}: missing column(s) {', '.join(missing)} (found: {reader.fieldnames})")
        out = []
        for row in reader:
            clean = {(k or "").strip(): (v.strip() if v and v.strip() else None) for k, v in row.items() if k}
            if any(clean.values()):  # skip blank lines
                out.append((reader.line_num, clean))
        return out


def _need(path: Path, line: int, row: dict[str, Any], column: str) -> str:
    value = row.get(column)
    if value is None:
        raise MockDataError(f"{path.name} line {line}: '{column}' is required")
    return value


def _int(path: Path, line: int, row: dict[str, Any], column: str) -> int:
    raw = _need(path, line, row, column)
    try:
        return int(raw)
    except ValueError:
        raise MockDataError(f"{path.name} line {line}: '{column}' must be a whole number, got {raw!r}") from None


def _time(path: Path, line: int, row: dict[str, Any], column: str) -> time:
    raw = _need(path, line, row, column)
    try:
        return time.fromisoformat(raw)
    except ValueError:
        raise MockDataError(f"{path.name} line {line}: '{column}' must be HH:MM, got {raw!r}") from None


def _date(path: Path, line: int, row: dict[str, Any], column: str) -> date | None:
    raw = row.get(column)
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise MockDataError(f"{path.name} line {line}: '{column}' must be YYYY-MM-DD, got {raw!r}") from None


EVENT_COLUMNS = ("id", "title", "date", "day_offset", "start", "end", "end_date", "attendees", "location", "description")


def load_events(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[EventSeed]:
    path = Path(data_dir) / "events.csv"
    rows = _rows(path, ("title", "start", "end"))
    events, seen_ids = [], set()
    for line, row in rows:
        day = _date(path, line, row, "date")
        offset = _int(path, line, row, "day_offset") if row.get("day_offset") is not None else None
        if (day is None) == (offset is None):
            raise MockDataError(f"{path.name} line {line}: give exactly one of 'date' or 'day_offset'")
        event_id = row.get("id")
        if event_id is not None:
            if not re.fullmatch(r"evt_\d+", event_id):
                raise MockDataError(f"{path.name} line {line}: 'id' must look like evt_001, got {event_id!r}")
            if event_id in seen_ids:
                raise MockDataError(f"{path.name} line {line}: duplicate event id {event_id}")
            seen_ids.add(event_id)
        seed = EventSeed(
            id=event_id,
            title=_need(path, line, row, "title"),
            date=day,
            day_offset=offset,
            start=_time(path, line, row, "start"),
            end=_time(path, line, row, "end"),
            end_date=_date(path, line, row, "end_date"),
            attendees=tuple(a.strip() for a in (row.get("attendees") or "").split(";") if a.strip()),
            location=row.get("location"),
            description=row.get("description"),
        )
        if seed.end_date is None and seed.end <= seed.start:
            raise MockDataError(f"{path.name} line {line}: 'end' must be after 'start' (or give 'end_date')")
        events.append(seed)
    return events


def save_events(data_dir: Path | str, rows: list[dict[str, Any]]) -> None:
    """Rewrite ``events.csv`` atomically (temp file + rename) so a crash never leaves a half-written file."""
    path = Path(data_dir) / "events.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in EVENT_COLUMNS})
    os.replace(tmp, path)


def load_contacts(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    path = Path(data_dir) / "contacts.csv"
    contacts, seen = [], set()
    for line, row in _rows(path, ("id", "name", "email")):
        contact = {k: row.get(k) for k in ("id", "name", "email", "company", "title", "phone")}
        for column in ("id", "name", "email"):
            _need(path, line, row, column)
        if contact["id"] in seen:
            raise MockDataError(f"{path.name} line {line}: duplicate contact id {contact['id']}")
        seen.add(contact["id"])
        contacts.append(contact)
    return contacts


CONTACT_COLUMNS = ("id", "name", "email", "company", "title", "phone")


def save_contacts(data_dir: Path | str, contacts: list[dict[str, Any]]) -> None:
    """Rewrite ``contacts.csv`` atomically (temp file + rename)."""
    path = Path(data_dir) / "contacts.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for contact in contacts:
            writer.writerow({k: ("" if contact.get(k) is None else contact[k]) for k in CONTACT_COLUMNS})
    os.replace(tmp, path)


def load_interactions(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[InteractionSeed]:
    path = Path(data_dir) / "interactions.csv"
    return [
        InteractionSeed(
            contact_id=_need(path, line, row, "contact_id"),
            days_ago=_int(path, line, row, "days_ago"),
            type=_need(path, line, row, "type"),
            subject=_need(path, line, row, "subject"),
            summary=row.get("summary") or "",
        )
        for line, row in _rows(path, ("contact_id", "days_ago", "type", "subject"))
    ]
