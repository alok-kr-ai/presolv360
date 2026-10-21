"""Optional Streamlit chat UI:  streamlit run streamlit_app.py

One browser session = one agent run (own ``logs/<run_id>/`` folder) and one conversation thread.
The same ``Agent.stream()`` events that drive the terminal timeline drive the ``st.status`` view.
"""
from __future__ import annotations

import streamlit as st

from config import Settings
from events import AgentEvent
from graph.builder import build_agent
from ui.render import TurnView, fmt_ms, plain_line

st.set_page_config(page_title="Presolv agent", page_icon="🗓️", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


def _agent():
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(_settings())
        st.session_state.history = []  # (role, text, step_lines)
        st.session_state.traces = []
    return st.session_state.agent


agent = _agent()

with st.sidebar:
    st.header("Session")
    st.caption(f"run `{agent.run_id}`")
    st.caption(f"thread `{agent.thread_id}`")
    st.caption(f"model `{agent.settings.model}`")
    if st.button("New conversation"):
        agent.new_thread()
        st.session_state.history = []
        st.rerun()
    st.subheader("Working memory")
    mem = agent.memory()
    st.json(mem.model_dump(mode="json", exclude={"pending_action"}), expanded=False)
    if mem.pending_action:
        state = "confirmed" if mem.pending_action.confirmed else "awaiting your confirmation"
        st.warning(f"Pending: {mem.pending_action.summary} ({state})")
    if st.session_state.traces:
        st.subheader("Traces")
        for trace in reversed(st.session_state.traces[-10:]):
            st.code(trace, language=None)

st.title("Presolv agent")
st.caption("Calendar, CRM and calculator tools - or just chat.")

for role, text, steps in st.session_state.history:
    with st.chat_message(role):
        if steps:
            with st.expander("Steps", expanded=False):
                st.text("\n".join(steps))
        st.markdown(text)

if prompt := st.chat_input("Ask about your calendar, a contact, a calculation..."):
    st.session_state.history.append(("user", prompt, []))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)
        answer_box = st.empty()
        view = TurnView()
        lines: list[str] = []
        for ev in agent.stream(prompt):
            view.apply(ev)
            line = plain_line(ev) if isinstance(ev, AgentEvent) else None
            if line is not None:
                lines.append(line.plain)
                status.write(line.plain)
            if ev.event in ("token", "final"):
                answer_box.markdown(view.answer)
            if ev.event == "turn_start":
                st.session_state.traces.append(f"turn {ev.turn}  trace {ev.trace_id}")
        label = f"{view.phase if not view.done else 'Done'} - {fmt_ms(view.total_ms)}"
        status.update(label=label, state="complete" if view.status == "completed" else "error", expanded=False)
        answer_box.markdown(view.answer or "_(no answer)_")
    st.session_state.history.append(("assistant", view.answer or "", lines))
    st.rerun()
