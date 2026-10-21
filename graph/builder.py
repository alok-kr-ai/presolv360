"""Graph construction and the ``build_agent(settings)`` entry point."""
from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from config import Settings
from events import EventBus
from graph.agent import Agent
from graph.nodes import Nodes
from llm import LLMClient, build_chain, build_context_entry
from mocks import Backend
from observability.context import new_run_id
from observability.langsmith import configure_langsmith
from observability.recorder import RunRecorder
from state import AgentState
from tools import GROUP_DESC, build_tool_groups, tools_by_name


def build_graph(nodes: Nodes, checkpointer: Any = None):
    """Wire the nodes: intake -> route -> agent -> (guard -> tools|agent|finalize)* -> END."""
    g = StateGraph(AgentState)
    g.add_node("intake", nodes.intake)
    g.add_node("route", nodes.route)
    g.add_node("agent", nodes.agent)
    g.add_node("guard", nodes.guard)
    g.add_node("tools", nodes.tools_node)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "intake")
    g.add_edge("intake", "route")
    g.add_edge("route", "agent")
    g.add_conditional_edges("agent", nodes.after_agent, {"guard": "guard", END: END})
    g.add_conditional_edges("guard", nodes.after_guard, {"tools": "tools", "agent": "agent", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_agent(
    settings: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
    classifier_llm: BaseChatModel | None = None,
    backend: Backend | None = None,
    tool_groups: dict[str, list[BaseTool]] | None = None,
    group_desc: dict[str, str] | None = None,
    checkpointer: Any = None,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log_dir: Path | str | None = None,
    thread_id: str | None = None,
) -> Agent:
    """Wire the whole agent from config.

    ``llm`` injects a chat model (tests use a scripted fake); by default ``ChatLiteLLM`` models routed
    through litellm, tried in order: ``LLM_MODEL`` then ``FALLBACK_MODELS``. ``backend`` / ``tool_groups`` swap the mocked
    services or the tool set.
    """
    settings = settings or Settings.from_env()
    backend = backend or Backend(settings, clock=clock)
    groups = tool_groups or build_tool_groups(backend)
    desc = group_desc or {g: GROUP_DESC.get(g, g) for g in groups}
    tools = tools_by_name(groups)

    run_id = new_run_id()
    langsmith_on = configure_langsmith()
    bus = EventBus(run_id)
    thread_id = thread_id or f"thread-{run_id.split('-')[-1]}"
    recorder = RunRecorder(run_id, settings, thread_id=thread_id, log_dir=log_dir)
    bus.add_sink(recorder.record)

    if llm is not None:  # injected model (tests): a chain of one
        client = LLMClient.from_model(llm, classifier_llm, settings, bus)
    else:  # primary + fallbacks, all through litellm; the router/classifier never need streaming
        client = LLMClient(
            build_chain(settings, streaming=settings.streaming),
            build_chain(settings, streaming=False),
            settings, bus,
            context_entry=build_context_entry(settings, streaming=settings.streaming),
            classifier_context_entry=build_context_entry(settings, streaming=False),
        )

    nodes = Nodes(
        settings=settings,
        llm=client,
        bus=bus,
        groups=groups,
        group_desc=desc,
        tools=tools,
        clock=clock or backend.clock,
        sleep=sleep,
        rng=random.Random(settings.mock_seed),
    )
    graph = build_graph(nodes, checkpointer)
    return Agent(
        settings=settings, graph=graph, bus=bus, recorder=recorder, backend=backend,
        thread_id=thread_id, groups=groups, langsmith=langsmith_on,
    )
