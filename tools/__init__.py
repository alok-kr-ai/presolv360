"""Tool registry.

* ``TOOL_GROUPS``  group name -> list of tools
* ``GROUP_DESC``   group name -> one-line description (used by the router and prompt)
* ``ALL_TOOLS``    derived from the groups

``TOOL_GROUPS`` / ``ALL_TOOLS`` are built lazily against a default mock backend (module
``__getattr__``) to avoid an import cycle with ``mocks``. Use ``build_tool_groups(backend)``
to get tools bound to a specific backend (this is what ``build_agent`` does).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:  # pragma: no cover
    from mocks import Backend

GROUP_DESC: dict[str, str] = {
    "calendar": "Read and manage the user's calendar: list events, check availability, create, update and delete events.",
    "crm": "Look up CRM contacts (by name or email), list past interactions with them, add, update and delete contacts.",
    "calculator": "Exact arithmetic: percentages, totals, multiplication, division and other calculations.",
}


def build_tool_groups(backend: "Backend") -> dict[str, list[BaseTool]]:
    from tools.calculator import build_calculator_tools
    from tools.calendar import build_calendar_tools
    from tools.crm import build_crm_tools

    return {
        "calendar": build_calendar_tools(backend.calendar),
        "crm": build_crm_tools(backend.crm),
        "calculator": build_calculator_tools(),
    }


def derive_all_tools(groups: dict[str, list[BaseTool]]) -> list[BaseTool]:
    return [tool for tools in groups.values() for tool in tools]


def tools_by_name(groups: dict[str, list[BaseTool]]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in derive_all_tools(groups)}


def group_of(tool: BaseTool) -> str:
    return (tool.metadata or {})["group"]


def is_write_tool(tool: BaseTool) -> bool:
    return bool((tool.metadata or {}).get("write"))


_default: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name in ("TOOL_GROUPS", "ALL_TOOLS"):
        if "groups" not in _default:
            from mocks import Backend

            _default["groups"] = build_tool_groups(Backend())
        groups = _default["groups"]
        return groups if name == "TOOL_GROUPS" else derive_all_tools(groups)
    raise AttributeError(f"module 'tools' has no attribute {name!r}")


__all__ = [
    "TOOL_GROUPS",
    "GROUP_DESC",
    "ALL_TOOLS",
    "build_tool_groups",
    "derive_all_tools",
    "tools_by_name",
    "group_of",
    "is_write_tool",
]
