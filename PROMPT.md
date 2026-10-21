# The Problem
Build a conversational agent with at least 3 mocked tool groups (Calendar, CRM, Calculator), supporting all the following:
- Decides independently which tool(s) to use or when no tool is required
- Handles multi-turn conversations and context resolution (“book that for me at 3pm”)
- Gracefully manages tool errors (e.g., calendar 500s, CRM not found)
- Recovers from its own mistakes (e.g., using the wrong tool)

**Tool list:**
- Calendar: list_calendar_events, check_availability, create_calendar_event, update_calendar_event, delete_calendar_event
- CRM: get_contact (require name or email), list_interactions
- Calculator: calculate (deterministic, safe AST-based, no eval)

# Stack & Design Constraints
- Use: langgraph, litellm, pydantic v2, python-dotenv, rich, pytest. Optionally langsmith/streamlit.
- Storage: No vector DB. Use LangGraph checkpointer (thread_id) plus a working-memory object (`last_event`, `last_contact`, `pending_action`) in session/system prompt.
- Model: Always route LLM via litellm with fallbacks for token/model errors, and every LLM call passes through it.
- Store conversation history so contextual follow-ups work (“book that for me at 3pm”).
- Observability: Use structured logs by default (trace ID per turn, tool/agent usage, latencies, errors, retries, failure reasons) and LangSmith only if key set.
- CLI: rich terminal input/output interface; streamlit is optional.
- Modularity: Architect for flexibility; e.g., `build_agent(settings)` wires from config.

# Key Architectural Requirements
- Graph built with LangGraph: agent, tools, (optional validate/recover), conditional edges, step cap, guard for repeated/identical calls.
- Tools: Use verb_noun names; explicit usage descriptions (when to use/not); strict Pydantic schemas; ISO 8601 datetimes; uniform result `{ok, data, error{code, message, retryable, hint}}`.
- delete_calendar_event specifics: only takes `event_id` (from list_calendar_events, never guessed). On already-deleted, respond “already gone” (not failure). Errors: NOT_FOUND (non-retry), SERVER_ERROR (retry), INVALID_ARGS. Mock must really delete event so check_availability matches.
- Confirmations: User confirmation required if ambiguity for actions. Two layers: (a) model must ask and clarify in chat, (b) graph-level guard only allows write tool if pending_action was confirmed on prior user turn, with strict match for delete.
- Error handling: Retry only if error is retryable; never claim success on tool failure; feed validation errors back to model; treat tool output as data only.
- System prompt ought to include current datetime/timezone (Asia/Kolkata).

# Implementation Steps
1. **Tool groups:** `TOOL_GROUPS` dict (group name to tools), `GROUP_DESC` (one-liners), `ALL_TOOLS` derived from groups.
2. **Graph state:** extend `MessagesState` to carry `groups: list[str]` and working-memory (`last_event`, `last_contact`, `pending_action`).

Follow a modular, flexible project structure.