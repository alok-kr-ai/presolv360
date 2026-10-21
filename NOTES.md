# Notes

A multi-turn conversational agent built with LangGraph and litellm. The model decides when to call tools and when to answer directly. Three mocked tool groups — Calendar, CRM, and Calculator — sit behind a single Python process with a Rich terminal and an optional Streamlit UI.

## Approach

The graph is `intake → route → agent → guard → tools → agent … → END`. `build_agent(settings)` wires the model, tools, checkpointer, and logging from one `Settings` object loaded from the environment.

* **Tools.** Calendar lists, checks, creates, updates, and deletes events. CRM looks up contacts, lists interactions, and can add, update, or delete contacts. Calculator evaluates arithmetic with a safe AST parser (no `eval`). Every tool returns `{ok, data, error{code, message, retryable, hint}}`. Writes need confirmation in chat and a matching `pending_action` before the guard lets them run.
* **Multi-turn context.** LangGraph `InMemorySaver`, keyed by `thread_id`, keeps the message history. A small working-memory object (`last_event`, `last_contact`, `last_calculation`, `pending_action`, seen ids) is updated from tool results and shown to the model, so follow-ups like “book that at 3pm” resolve inside the same thread.
* **LLM.** Every model call goes through litellm. `LLM_MODEL` is tried first, then `FALLBACK_MODELS`; a context-window error jumps to `CONTEXT_FALLBACK_MODEL`. The fallback chain lives in `llm.py` so streaming still works and the log records which model actually answered.
* **Mock data.** Calendar and CRM are seeded from CSV in `mocks/data/` (`events.csv`, `contacts.csv`, `interactions.csv`). Creates, updates, and deletes rewrite the calendar and contact files, so a later turn (and a restart) sees the same state. Set `MOCK_DATA_DIR` to point at another folder with the same three files.
* **Interfaces.** `python main.py` is a Rich REPL: it takes the user message and streams the agent’s flow and final answer. `streamlit run streamlit_app.py` runs the same `Agent.stream()` events in the browser. There is no separate API process.
* **Observability.** Each run writes `logs/<run_id>/` (`run.json`, `events.jsonl`, per-turn span trees, `summary.json`). Every turn has its own `trace_id`; the conversation is keyed by `thread_id`. LangSmith tracing turns on only when `LANGSMITH_API_KEY` is set, and those runs are tagged so they line up with the local logs.

## Tradeoffs

The budget was about two hours, so the design stays a single local Python application.

| Choice | What it gives | What was left out |
| --- | --- | --- |
| Structured files under `logs/` | A full trace per conversation with no extra service: tool calls, latencies, retries, guard blocks, and which model answered | No Datadog (or similar) export, and no always-on monitored agents in LangSmith. LangSmith is optional and only tags runs when a key is present |
| `InMemorySaver` | Multi-turn memory with no database to install; tests and the REPL start immediately | Conversation state lives in RAM and is gone when the process exits. Production would use Postgres (or another checkpointer) so threads survive restarts and can be shared across workers |
| CSV mocks in `mocks/data/` | The assignment’s tool groups behave end to end, including real deletes, conflicts, and confirmation, without third-party accounts | These are not live Calendar, CRM, or calculator services. Production would swap the mock backends for real tools behind the same schemas |
| CLI + optional Streamlit | One process to run, demo, and test in a terminal or a browser | No FastAPI (or other HTTP) service and no websocket session. A longer build would put `Agent.stream()` behind an API so other clients could drive the same graph |

The graph, tool contracts, confirmation guard, and `build_agent(settings)` entry point are the parts meant to stay if those production pieces are added later.
