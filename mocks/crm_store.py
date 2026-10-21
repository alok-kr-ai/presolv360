"""Mock CRM. Contacts and interactions are loaded from CSV (see ``mocks/loader.py``); contacts created
through ``create_contact`` / ``update_contact`` / ``delete_contact`` are written back to ``contacts.csv`` when ``persist_dir`` is set."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable

from mocks.faults import FaultInjector
from mocks.loader import InteractionSeed, save_contacts
from tools.base import ErrorCode, ToolFailure

log = logging.getLogger("agent.mocks")

class CRMStore:
    def __init__(
        self,
        tz: tzinfo,
        clock: Callable[[], datetime],
        faults: FaultInjector | None = None,
        *,
        contacts: list[dict[str, Any]] | None = None,
        interactions: list[InteractionSeed] | None = None,
        persist_dir: Path | str | None = None,
    ) -> None:
        self.tz = tz
        self.clock = clock
        self.faults = faults or FaultInjector()
        self._contacts = [dict(c) for c in (contacts or [])]
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._requests: dict[str, str] = {}  # request_id -> contact id (idempotent creates)
        self._tombstones: set[str] = set()  # deleted ids, so a repeat delete says "already gone"
        self._lock = threading.RLock()
        now = clock().astimezone(tz)
        self._interactions = [
            {
                "id": f"int_{i:03d}",
                "contact_id": it.contact_id,
                "type": it.type,
                "date": (now - timedelta(days=it.days_ago)).replace(hour=11, minute=0, second=0, microsecond=0).isoformat(),
                "subject": it.subject,
                "summary": it.summary,
            }
            for i, it in enumerate(interactions or [], start=1)
        ]

    def get_contact(self, name: str | None, email: str | None) -> dict[str, Any]:
        self.faults.maybe_fail("get_contact")
        if email:
            for c in self._contacts:
                if c["email"].lower() == email.lower():
                    return {"contact": dict(c)}
            raise ToolFailure(
                ErrorCode.NOT_FOUND,
                f"No contact with email {email}.",
                hint="Check the spelling or ask the user for the contact's name. If they meant a calendar event, use list_calendar_events.",
            )
        query = (name or "").strip().lower()
        exact = [c for c in self._contacts if c["name"].lower() == query]
        if len(exact) == 1:
            return {"contact": dict(exact[0])}
        tokens = query.split()
        matches = [
            c for c in self._contacts if all(any(part.startswith(t) for part in c["name"].lower().split()) for t in tokens)
        ]
        if len(matches) == 1:
            return {"contact": dict(matches[0])}
        if len(matches) > 1:
            raise ToolFailure(
                ErrorCode.AMBIGUOUS,
                f"{len(matches)} contacts match '{name}'.",
                hint="Ask the user which one they mean (offer the candidates), or ask for the email address.",
                data={"candidates": [{k: c[k] for k in ("id", "name", "email", "company")} for c in matches]},
            )
        raise ToolFailure(
            ErrorCode.NOT_FOUND,
            f"No contact found matching '{name}'.",
            hint="Ask the user to check the spelling or provide an email address. If they meant a calendar event, use list_calendar_events.",
        )

    def list_interactions(self, contact_id: str, limit: int = 10) -> dict[str, Any]:
        self.faults.maybe_fail("list_interactions")
        contact = next((c for c in self._contacts if c["id"] == contact_id), None)
        if contact is None:
            raise ToolFailure(
                ErrorCode.NOT_FOUND,
                f"No contact with id {contact_id}.",
                hint="Call get_contact first to obtain a valid contact_id.",
            )
        items = sorted(
            (i for i in self._interactions if i["contact_id"] == contact_id), key=lambda i: i["date"], reverse=True
        )[:limit]
        out: dict[str, Any] = {"contact": {"id": contact["id"], "name": contact["name"]}, "interactions": items, "count": len(items)}
        if not items:
            out["note"] = "No recorded interactions for this contact."
        return out

    def create_contact(
        self,
        name: str,
        email: str,
        company: str | None,
        title: str | None,
        phone: str | None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.faults.maybe_fail("create_contact")
        with self._lock:
            if request_id and request_id in self._requests:  # a retry of a create that already committed
                existing = next((c for c in self._contacts if c["id"] == self._requests[request_id]), None)
                if existing is not None:
                    return {"contact": dict(existing), "idempotent_replay": True}
            dup = next((c for c in self._contacts if c["email"].lower() == email.lower()), None)
            if dup is not None:
                raise ToolFailure(
                    ErrorCode.CONFLICT,
                    f"A contact with email {email} already exists: {dup['name']} ({dup['id']}).",
                    hint="Tell the user this contact already exists and show it. Editing contacts is not supported.",
                    data={"existing": {k: dup.get(k) for k in ("id", "name", "email", "company", "title")}},
                )
            number = max((int(c["id"].split("_")[1]) for c in self._contacts if c["id"].split("_")[-1].isdigit()), default=0)
            contact = {"id": f"con_{number + 1:03d}", "name": name, "email": email,
                       "company": company, "title": title, "phone": phone}
            self._contacts.append(contact)
            if request_id:
                self._requests[request_id] = contact["id"]
            self._persist()
        self.faults.maybe_fail("create_contact", phase="after")
        return {"contact": dict(contact)}

    def update_contact(self, contact_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply ``changes`` (name / email / company / title / phone). A blank company, title or phone clears it."""
        self.faults.maybe_fail("update_contact")
        with self._lock:
            contact = next((c for c in self._contacts if c["id"] == contact_id), None)
            if contact is None:
                gone = contact_id in self._tombstones
                raise ToolFailure(
                    ErrorCode.NOT_FOUND,
                    f"No contact with id {contact_id}" + (" (it was deleted)." if gone else "."),
                    hint="Call get_contact to find the contact and copy its id; never guess ids.",
                )
            new_email = changes.get("email")
            if new_email:
                clash = next((c for c in self._contacts if c["id"] != contact_id and c["email"].lower() == new_email.lower()), None)
                if clash is not None:
                    raise ToolFailure(
                        ErrorCode.CONFLICT,
                        f"Another contact already uses {new_email}: {clash['name']} ({clash['id']}).",
                        hint="Tell the user that email belongs to another contact and ask what they want to do.",
                        data={"existing": {k: clash.get(k) for k in ("id", "name", "email", "company", "title")}},
                    )
            changed = []
            for key in ("name", "email", "company", "title", "phone"):
                if key in changes:
                    value = changes[key] or None  # blank clears the optional fields
                    if value != contact.get(key):
                        contact[key] = value
                        changed.append(key)
            if changed:
                self._persist()
            result: dict[str, Any] = {"contact": dict(contact), "changed": changed}
            if not changed:
                result["note"] = "The contact already had these values; nothing was changed."
        if changed:
            self.faults.maybe_fail("update_contact", phase="after")
        return result

    def delete_contact(self, contact_id: str) -> dict[str, Any]:
        self.faults.maybe_fail("delete_contact")
        with self._lock:
            contact = next((c for c in self._contacts if c["id"] == contact_id), None)
            if contact is None:
                if contact_id in self._tombstones:
                    return {"status": "already_gone", "contact_id": contact_id}
                raise ToolFailure(
                    ErrorCode.NOT_FOUND,
                    f"No contact with id {contact_id}.",
                    hint="Call get_contact to find the contact and copy its id; never guess ids.",
                )
            self._contacts.remove(contact)
            self._interactions = [i for i in self._interactions if i["contact_id"] != contact_id]
            self._tombstones.add(contact_id)
            result = {"status": "deleted", "contact_id": contact_id, "name": contact["name"], "email": contact["email"]}
            self._persist()
        self.faults.maybe_fail("delete_contact", phase="after")
        return result

    def _persist(self) -> None:
        """Write contacts back to ``contacts.csv``; a disk problem is logged, never turned into a tool error."""
        if self._persist_dir is None:
            return
        try:
            save_contacts(self._persist_dir, self._contacts)
        except OSError as exc:
            log.warning("could not write %s/contacts.csv: %s", self._persist_dir, exc)
