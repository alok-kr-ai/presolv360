"""Graph state and working memory."""
from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class PendingAction(BaseModel):
    """A write the model proposed and the user has (or has not) confirmed."""

    tool: str
    args: dict[str, Any]  # canonical, comparison form
    call_args: dict[str, Any]  # JSON args the model should re-issue
    summary: str
    confirmed: bool = False
    proposed_turn: int
    confirmed_turn: int | None = None
    idle_turns: int = 0

    def executable_on(self, turn: int) -> bool:
        """Confirmed on a *prior* user turn relative to the one that proposed it."""
        return self.confirmed and self.confirmed_turn is not None and self.confirmed_turn > self.proposed_turn


class WorkingMemory(BaseModel):
    """What the agent remembers across topics (rendered into the system prompt)."""

    last_event: dict[str, Any] | None = None
    last_contact: dict[str, Any] | None = None
    last_calculation: dict[str, Any] | None = None
    listed_events: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: PendingAction | None = None

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "WorkingMemory":
        pending = state.get("pending_action")
        return cls(
            last_event=state.get("last_event"),
            last_contact=state.get("last_contact"),
            last_calculation=state.get("last_calculation"),
            listed_events=state.get("listed_events") or [],
            pending_action=PendingAction.model_validate(pending) if pending else None,
        )

    def prompt_json(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"pending_action"})
        return {k: v for k, v in data.items() if v}


class AgentState(MessagesState):
    # tool groups enabled for *this turn only* (recomputed by the router every turn)
    groups: list[str]
    # working memory
    last_event: dict[str, Any] | None
    last_contact: dict[str, Any] | None
    last_calculation: dict[str, Any] | None
    listed_events: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    # event ids that appeared in tool output (so ids are never guessed)
    seen_event_ids: list[str]
    # contact ids that appeared in tool output (so contact ids are never guessed either)
    seen_contact_ids: list[str]
    # per-turn bookkeeping
    trace_id: str
    turn: int
    steps: int
    call_signatures: list[str]
    approved_calls: list[dict[str, Any]]
    confirmation_note: str | None
    step_cap_hit: bool
    final_corrected: bool
