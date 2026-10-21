# Conversational tool agent (LangGraph + litellm)

A multi-turn agent with three mocked tool groups: **Calendar**, **CRM** and **Calculator**. It decides on its own
when to use tools, answers greetings and general questions directly, follows topic shifts mid-conversation,
recovers from tool errors and its own mistakes, and never changes your calendar or contacts without confirmation.

## Capabilities

| Group | What you can ask for | Tools | Confirmation |
| --- | --- | --- | --- |
| **Calendar** | List events, check availability, **create**, **update** and **delete** events | `list_calendar_events`, `check_availability`, `create_calendar_event`, `update_calendar_event`, `delete_calendar_event` | Create / update / delete |
| **CRM** | Look up a contact by name or email, list past interactions, **add**, **update** and **delete** contacts | `get_contact`, `list_interactions`, `create_contact`, `update_contact`, `delete_contact` | Add / update / delete |
| **Calculator** | Safe arithmetic (AST-evaluated, no `eval`) | `calculate` | None (read-only) |

Contact management in detail:

* **Add**: `add contact Zoe Park, zoe@park.io, CTO at Acme`. Needs a name and a valid email; a duplicate email is
  reported as a conflict with the existing contact instead of creating a second one.
* **Update**: `change Zoe's title to CTO`. Changes any of name, email, company, title or phone; only the fields you
  mention change, and an empty company / title / phone clears it. The agent looks the contact up first and asks you to
  confirm the exact change (e.g. `title -> CTO`). Changing to an email another contact already uses is rejected.
* **Delete**: `remove contact zoe@park.io`. Looks the contact up, asks you to confirm, then removes it; deleting
  twice reports "already gone".
* All three are saved to `mocks/data/contacts.csv`, and the id used for an update or delete must come from a lookup
  in the same conversation (never guessed).

Across all groups the agent also:

* switches tools mid-conversation (calculator, then CRM, then calendar) without losing context;
* resolves references like "that", "her" or "book that at 3pm" from working memory;
* recovers from tool errors (bounded retries, corrected arguments, another tool) and never claims a write succeeded
  when it did not;
* answers greetings and general questions directly, with no tool calls;
* streams a live flow timeline in the terminal and writes a structured log for every run (`logs/<run_id>/`).

## Setup

Linux / WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

Windows (PowerShell as Administrator):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv venv --python 3.12
uv pip install -r requirements.txt
```

Then `cp .env.example .env` and set `OPENAI_API_KEY` (and optionally `LLM_MODEL`, `FALLBACK_MODELS`, ...).
See `.env.example` for every option.

## Run

```bash
python main.py                 # interactive terminal agent (rich, live flow timeline)
python main.py -v              # also show tool arguments, results and guard details
python main.py -q              # only the final answers
python main.py --no-live       # one line per event (automatic when piped)
python main.py logs            # list recorded runs
python main.py logs show <run_id>        # tool / LLM / node performance for a run
python main.py logs turn <run_id> <n>    # span tree and flow of one turn
streamlit run streamlit_app.py           # optional web UI with the same live flow
pytest                                   # 220+ tests, no network or API key needed
```

Things to try:

```
multiply 2 and 3                          -> calculator
who is Rahul Verma?                       -> CRM (topic shift, same conversation)
add contact Zoe Park, zoe@park.io          -> asks for confirmation, then saves it to contacts.csv
change Zoe's title to CTO                  -> looks her up, asks for confirmation, then updates the row
remove contact zoe@park.io                 -> looks it up, asks for confirmation, then deletes the row
what's on my calendar tomorrow?           -> calendar
delete my standup tomorrow                -> asks for confirmation, then "yes" deletes it
book that for me at 3pm                   -> resolved from working memory
hi / what is the capital of France?       -> answered without tools
```

REPL commands: `/help`, `/new`, `/memory`, `/trace`, `/runs`, `/quit`, and `/fail <tool> <code> [n] [after]` to inject a
mock failure (e.g. `/fail list_calendar_events SERVER_ERROR 2`) and watch the retries in the live timeline.

## How it works

```
intake -> route -> agent -> guard -> tools -> agent ... -> END        (finalize when the step cap is hit)
```

| Node | Job |
| --- | --- |
| `intake` | New turn bookkeeping; if a write is pending, classifies the reply (yes / no / modify / unrelated). |
| `route` | Per turn, picks tool groups (`calendar`, `crm`, `calculator`, or none). Greetings skip the LLM; router failures fail open to all groups. |
| `agent` | The LLM bound to the active tools only. System prompt carries the current datetime in `Asia/Kolkata`, rules and working memory. |
| `guard` | Pure Python: unknown/inactive tools, argument validation, unseen `event_id`s, duplicate calls, write confirmation, step cap. |
| `tools` | Runs approved calls with timeout, bounded retries with backoff (retryable errors only) and idempotent writes. |

* **Context**: LangGraph `InMemorySaver` keyed by `thread_id` keeps the conversation; a working-memory object
  (`last_event`, `last_contact`, `last_calculation`, `pending_action`, ...) is updated deterministically from tool
  results and shown to the model, so "that", "her" and "book that at 3pm" resolve.
* **Confirmations (two layers)**: the model must ask in chat, and the guard only runs a write if a matching
  `pending_action` was confirmed on an earlier user turn (arguments must match exactly; delete matches `event_id`).
* **Tool results** are always `{ok, data, error{code, message, retryable, hint}}`. Errors are fed back to the model,
  only retryable ones are retried, and a reply that claims success without a successful write is corrected.
* **Delete** (events and contacts) takes only an id (`event_id` / `contact_id`) that a previous result returned; deleting twice reports "already gone" and the
  mock really removes the event, so `check_availability` agrees.
* **LLM calls** all go through litellm (`ChatLiteLLM`). `LLM_MODEL` is tried first, then each of `FALLBACK_MODELS`
  in order; a context-window error jumps straight to `CONTEXT_FALLBACK_MODEL`. The chain lives in `llm.py` rather than in
  litellm's own `fallbacks` kwarg because that one cannot stream and mislabels which model answered. Each hop is a
  `llm_fallback` event in the live timeline and the logs, and a model that fails permanently (bad name / key) is skipped
  for the rest of the run. Unsupported params (e.g. `temperature` on reasoning models) are dropped, not fatal.

## Mock data (CSV)

The calendar and CRM are seeded from CSV files in `mocks/data/`, not from Python code:

| File | Columns |
| --- | --- |
| `events.csv` | `id, title, date, day_offset, start, end, end_date, attendees, location, description` |
| `contacts.csv` | `id, name, email, company, title, phone` |
| `interactions.csv` | `contact_id, days_ago, type, subject, summary` |

* **Events** take either `date` (`YYYY-MM-DD`) or `day_offset` (0 = today, 1 = tomorrow), plus `HH:MM` `start`/`end` in
  the agent's timezone (`end_date` only for events that end on a later day). `id` (`evt_001`) is optional; `attendees`
  is a `;`-separated list. Hand-written rows with `day_offset` keep the demo data relative to today.
* **Calendar changes are saved to `events.csv`.** Creating, updating or deleting an event rewrites the file (atomically),
  so what the agent did is still there after a restart. Untouched `day_offset` rows stay relative; created or moved
  events are stored with absolute `date`s. Set `MOCK_PERSIST=false` to keep everything in memory, or restore the seed
  with `git checkout mocks/data/events.csv`. Contacts the agent adds, changes or deletes (`create_contact`, `update_contact`, `delete_contact`) are saved to `contacts.csv` the same way; `interactions.csv` is read-only (a deleted contact's rows are ignored).
* Set `MOCK_DATA_DIR` to a folder with the same three files to use your own data. Malformed rows fail fast with the file
  name and line number.

## Observability

Every run writes its own folder, identified by a run id:

```
logs/<run_id>/
  run.json        model, settings (secrets redacted), versions, status (running/completed/interrupted/crashed)
  events.jsonl    every event as one JSON line (trace_id, turn, latency, retries, error codes, failure reasons)
  turns/turn-001-<trace_id>.json   flow, span tree, tool calls and metrics of one turn
  summary.json    per-tool / per-model / per-node latency (avg, p50, p95), error counts, guard blocks
```

Each turn has a `trace_id` shared by all its events. Set `LOG_CONTENT=false` to drop message text and tool arguments
from logs. LangSmith tracing is enabled only when `LANGSMITH_API_KEY` is set and is tagged with the run and trace ids.

## Layout

```
main.py  streamlit_app.py  config.py  llm.py  prompts.py  state.py  memory.py  events.py
graph/          builder (build_agent), nodes, guards, Agent facade (stream / invoke)
tools/          base (uniform result + wrapper), calendar, crm, calculator (AST), TOOL_GROUPS registry
mocks/          in-memory calendar and CRM with fault injection; data/*.csv is the seed data, loader.py reads it
observability/  recorder, spans, metrics, reports, LangSmith
ui/             rich console, live renderer, CLI loop
tests/          fake scripted LLM + tests (tests/data = pinned seed CSVs, so tests never touch the live data)
```

`build_agent(settings)` wires everything from `Settings`; tests inject a fake model, clock and mock backend through its
keyword arguments.
