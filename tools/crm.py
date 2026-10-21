"""CRM tool group (mock service behind ``mocks.crm_store``): get_contact, list_interactions, create_contact, update_contact, delete_contact."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import Field, field_validator, model_validator

from tools.base import StrictArgs, ToolContext, make_tool

if TYPE_CHECKING:  # pragma: no cover
    from mocks.crm_store import CRMStore

CONTACT_ID_PATTERN = r"^con_[A-Za-z0-9_-]+$"


class GetContactArgs(StrictArgs):
    name: str | None = Field(default=None, max_length=100, description="Full or partial contact name, e.g. 'Priya Sharma'.")
    email: str | None = Field(default=None, max_length=200, description="Exact contact email address.")

    @field_validator("name", "email")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return v or None

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str | None) -> str | None:
        if v is not None and ("@" not in v or " " in v):
            raise ValueError("email is not a valid email address")
        return v

    @model_validator(mode="after")
    def _need_one(self) -> "GetContactArgs":
        if not self.name and not self.email:
            raise ValueError("provide a name or an email (at least one is required)")
        return self


class CreateContactArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=100, description="Full name of the new contact.")
    email: str = Field(max_length=200, description="Email address of the new contact (required, must be unique).")
    company: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=100, description="Job title / role, e.g. 'Engineer'.")
    phone: str | None = Field(default=None, max_length=40)

    @field_validator("name", "email", "company", "title", "phone", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("company", "title", "phone")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return v or None

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("name must not be blank")
        return v

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("email is not a valid email address")
        return v


class UpdateContactArgs(StrictArgs):
    contact_id: str = Field(pattern=CONTACT_ID_PATTERN, description="Id of the contact, taken from get_contact. Never guess.")
    name: str | None = Field(default=None, min_length=1, max_length=100, description="New full name.")
    email: str | None = Field(default=None, max_length=200, description="New email address (must not belong to another contact).")
    company: str | None = Field(default=None, max_length=100, description="New company; an empty string clears it.")
    title: str | None = Field(default=None, max_length=100, description="New job title; an empty string clears it.")
    phone: str | None = Field(default=None, max_length=40, description="New phone number; an empty string clears it.")

    @field_validator("name", "email", "company", "title", "phone", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("name", "email")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError("must not be blank (only company, title and phone can be cleared)")
        return v

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("email is not a valid email address")
        return v

    @model_validator(mode="after")
    def _need_a_change(self) -> "UpdateContactArgs":
        if all(v is None for v in (self.name, self.email, self.company, self.title, self.phone)):
            raise ValueError("provide at least one field to change (name, email, company, title or phone)")
        return self


class DeleteContactArgs(StrictArgs):
    contact_id: str = Field(pattern=CONTACT_ID_PATTERN, description="Id of the contact, taken from get_contact. Never guess.")


class ListInteractionsArgs(StrictArgs):
    contact_id: str = Field(pattern=CONTACT_ID_PATTERN, description="Contact id from get_contact (e.g. con_001).")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of interactions, newest first.")


def build_crm_tools(store: "CRMStore") -> list[StructuredTool]:
    def get_contact(args: GetContactArgs, ctx: ToolContext) -> Any:
        return store.get_contact(args.name, args.email)

    def list_interactions(args: ListInteractionsArgs, ctx: ToolContext) -> Any:
        return store.list_interactions(args.contact_id, args.limit)

    def create_contact(args: CreateContactArgs, ctx: ToolContext) -> Any:
        return store.create_contact(
            args.name, args.email, args.company, args.title, args.phone, request_id=ctx.request_id
        )

    def update_contact(args: UpdateContactArgs, ctx: ToolContext) -> Any:
        return store.update_contact(args.contact_id, args.model_dump(exclude={"contact_id"}, exclude_none=True))

    def delete_contact(args: DeleteContactArgs, ctx: ToolContext) -> Any:
        return store.delete_contact(args.contact_id)

    return [
        make_tool(
            "get_contact",
            "Look up a CRM contact by name or email (at least one is required). Returns the contact record and its contact_id. "
            "USE when the user asks about a person, their company, email or role. "
            "DO NOT use to find calendar events or meetings (use list_calendar_events). "
            "If several people match, ask the user which one.",
            GetContactArgs,
            get_contact,
            group="crm",
        ),
        make_tool(
            "list_interactions",
            "List past interactions (emails, calls, meetings) with a contact, newest first. Needs a contact_id from get_contact. "
            "USE when the user asks about history, recent conversations or last contact with someone. "
            "DO NOT use to look up the contact itself (use get_contact).",
            ListInteractionsArgs,
            list_interactions,
            group="crm",
        ),
        make_tool(
            "create_contact",
            "Add a new person to the CRM. WRITE action: the user must have confirmed the exact name and email first. "
            "Needs a name and an email; company, title and phone are optional. "
            "USE when the user asks to add / save / create a contact. "
            "DO NOT use to look someone up (use get_contact) or to edit an existing contact (not supported). "
            "If the email is missing, ask for it instead of inventing one.",
            CreateContactArgs,
            create_contact,
            group="crm",
            write=True,
        ),
        make_tool(
            "update_contact",
            "Change a CRM contact's name, email, company, title or phone. WRITE action: needs explicit user confirmation of "
            "the exact change. The contact_id must be copied from get_contact (look the person up first), never guessed; "
            "pass only the fields that change. "
            "DO NOT use to create a new contact (use create_contact) or to delete one (use delete_contact).",
            UpdateContactArgs,
            update_contact,
            group="crm",
            write=True,
        ),
        make_tool(
            "delete_contact",
            "Delete one CRM contact by contact_id. WRITE action: needs explicit user confirmation of that exact contact. "
            "The contact_id must be copied from get_contact (look the person up first by name or email), never guessed. "
            "If the result says status 'already_gone', tell the user the contact was already gone (not a failure). "
            "DO NOT use to edit a contact or to remove calendar events (use delete_calendar_event).",
            DeleteContactArgs,
            delete_contact,
            group="crm",
            write=True,
        ),
    ]
