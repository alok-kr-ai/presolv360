"""Calendar tool group (mock service behind ``mocks.calendar_store``)."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import AwareDatetime, Field, model_validator

from tools.base import StrictArgs, ToolContext, make_tool

if TYPE_CHECKING:  # pragma: no cover
    from mocks.calendar_store import CalendarStore

EVENT_ID_PATTERN = r"^evt_[A-Za-z0-9_-]+$"
_ISO_HINT = "ISO 8601 with timezone offset, e.g. 2026-10-21T15:00:00+05:30"

WRITE_TOOLS = {"create_calendar_event", "update_calendar_event", "delete_calendar_event"}


class _Range(StrictArgs):
    start: AwareDatetime = Field(description=f"Range start, {_ISO_HINT}.")
    end: AwareDatetime = Field(description=f"Range end (exclusive), {_ISO_HINT}. Must be after start.")

    @model_validator(mode="after")
    def _end_after_start(self) -> "_Range":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class ListEventsArgs(_Range):
    query: str | None = Field(default=None, max_length=100, description="Optional text to match in title, location or attendees.")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of events to return.")


class CheckAvailabilityArgs(_Range):
    pass


class CreateEventArgs(_Range):
    title: str = Field(min_length=1, max_length=200, description="Event title.")
    attendees: list[str] = Field(default_factory=list, max_length=20, description="Attendee emails or names (optional).")
    location: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class UpdateEventArgs(StrictArgs):
    event_id: str = Field(pattern=EVENT_ID_PATTERN, description="Id of the event, taken from list_calendar_events. Never guess.")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: AwareDatetime | None = Field(default=None, description=f"New start, {_ISO_HINT}.")
    end: AwareDatetime | None = Field(default=None, description=f"New end, {_ISO_HINT}.")
    attendees: list[str] | None = Field(default=None, max_length=20)
    location: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _validate(self) -> "UpdateEventArgs":
        changes = [self.title, self.start, self.end, self.attendees, self.location, self.description]
        if all(c is None for c in changes):
            raise ValueError("provide at least one field to change (title, start, end, attendees, location, description)")
        if self.start and self.end and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class DeleteEventArgs(StrictArgs):
    event_id: str = Field(
        pattern=EVENT_ID_PATTERN,
        description="Id of the event to delete, copied from a list_calendar_events result. Never guess or invent ids.",
    )


def build_calendar_tools(store: "CalendarStore") -> list[StructuredTool]:
    def list_events(args: ListEventsArgs, ctx: ToolContext) -> Any:
        return store.list_events(args.start, args.end, args.query, args.limit)

    def check_availability(args: CheckAvailabilityArgs, ctx: ToolContext) -> Any:
        return store.check_availability(args.start, args.end)

    def create_event(args: CreateEventArgs, ctx: ToolContext) -> Any:
        return store.create_event(
            args.title, args.start, args.end, args.attendees, args.location, args.description, request_id=ctx.request_id
        )

    def update_event(args: UpdateEventArgs, ctx: ToolContext) -> Any:
        changes = args.model_dump(exclude={"event_id"}, exclude_none=True)
        return store.update_event(args.event_id, changes)

    def delete_event(args: DeleteEventArgs, ctx: ToolContext) -> Any:
        return store.delete_event(args.event_id)

    return [
        make_tool(
            "list_calendar_events",
            "List calendar events overlapping a time range, with their event ids. "
            "USE to see the schedule, to find an event (e.g. before updating or deleting it - the event_id must come from here). "
            "DO NOT use to check whether a specific slot is free (use check_availability) or to look up people (use get_contact).",
            ListEventsArgs,
            list_events,
            group="calendar",
        ),
        make_tool(
            "check_availability",
            "Check whether a time range is free. Returns available true/false and any conflicting events. "
            "USE before proposing or booking a time. DO NOT use to browse the whole schedule (use list_calendar_events).",
            CheckAvailabilityArgs,
            check_availability,
            group="calendar",
        ),
        make_tool(
            "create_calendar_event",
            "Create a calendar event. WRITE action: the user must have explicitly confirmed the exact title, start and end first. "
            "USE only after check_availability shows the slot is free and the user said yes. "
            "DO NOT use to modify an existing event (use update_calendar_event).",
            CreateEventArgs,
            create_event,
            group="calendar",
            write=True,
        ),
        make_tool(
            "update_calendar_event",
            "Change the title, time, attendees, location or description of an existing event. WRITE action: needs explicit "
            "user confirmation of the exact change. The event_id must come from list_calendar_events. "
            "DO NOT use to create new events or delete events.",
            UpdateEventArgs,
            update_event,
            group="calendar",
            write=True,
        ),
        make_tool(
            "delete_calendar_event",
            "Delete one event by event_id. WRITE action: needs explicit user confirmation of that exact event. "
            "The event_id must be copied from list_calendar_events, never guessed. If the result says status 'already_gone', "
            "tell the user the event was already gone (that is not a failure). "
            "DO NOT use to reschedule (use update_calendar_event).",
            DeleteEventArgs,
            delete_event,
            group="calendar",
            write=True,
        ),
    ]


__all__ = ["build_calendar_tools", "WRITE_TOOLS", "EVENT_ID_PATTERN", "re"]
