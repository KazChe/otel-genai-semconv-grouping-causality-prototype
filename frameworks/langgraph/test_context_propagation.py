"""LangGraph context propagation tests — follows the frameworks/ convention.

LangGraph serves as the "same-process demonstrator" for both proposals.
These tests verify whether W3C Baggage propagates through LangGraph's
internal dispatch mechanisms (node execution, conditional edges,
auto-instrumentation).

Key questions:
  1. Does baggage set before graph.invoke() reach node functions?
  2. Does baggage survive across sequential nodes in the same graph?
  3. Does baggage survive conditional edge routing?
  4. Do auto-instrumented spans (via LangChainInstrumentor) carry baggage?
"""

import pytest

langgraph = pytest.importorskip("langgraph")
langchain_core = pytest.importorskip("langchain_core")

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from typing import TypedDict

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS


_captured_baggage = {}


@pytest.fixture()
def tracing():
    """Set up an in-memory tracing pipeline with BaggageSpanProcessor."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    tracer = provider.get_tracer("test-langgraph-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage.clear()
    yield
    _captured_baggage.clear()


class TestLangGraphContextPropagation:
    """Baggage propagation through LangGraph's dispatch mechanisms."""

    def test_baggage_propagates_through_graph_nodes(self, tracing):
        """Baggage set before graph.invoke() is visible inside node
        functions. LangGraph nodes execute in the caller's context,
        so baggage propagates automatically.

        This is the baseline test, analogous to the direct function
        call tests in frameworks/ (Haystack, PydanticAI, etc.)."""
        tracer, exporter = tracing

        class SimpleState(TypedDict):
            value: str

        def check_node(state: SimpleState) -> SimpleState:
            _captured_baggage["gen_ai.group.id"] = baggage.get_baggage(
                "gen_ai.group.id"
            )
            _captured_baggage["gen_ai.group.iteration.type"] = baggage.get_baggage(
                "gen_ai.group.iteration.type"
            )
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
            return {"value": "done"}

        graph = StateGraph(SimpleState)
        graph.add_node("check", check_node)
        graph.add_edge(START, "check")
        graph.add_edge("check", END)
        app = graph.compile()

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            app.invoke({"value": "input"})
        finally:
            context.detach(token)

        assert _captured_baggage.get("gen_ai.group.id") == "round-1", (
            "Baggage should propagate into LangGraph node functions"
        )
        assert _captured_baggage.get("gen_ai.group.iteration.type") == "react"

        # Verify span attributes via BaggageSpanProcessor
        spans = exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name == "chat"]
        assert len(chat_spans) == 1
        attrs = dict(chat_spans[0].attributes)
        assert attrs.get("gen_ai.group.id") == "round-1"
        assert attrs.get("gen_ai.group.iteration.type") == "react"

    def test_baggage_across_sequential_nodes(self, tracing):
        """Baggage set in the caller context is visible in all sequential
        nodes of the graph, not just the first one. This confirms
        LangGraph does not reset context between node transitions."""
        tracer, exporter = tracing

        captured_per_node = {}

        class SeqState(TypedDict):
            step: str

        def node_a(state: SeqState) -> SeqState:
            captured_per_node["node_a"] = baggage.get_baggage("gen_ai.group.id")
            with tracer.start_as_current_span("node_a_span"):
                pass
            return {"step": "a_done"}

        def node_b(state: SeqState) -> SeqState:
            captured_per_node["node_b"] = baggage.get_baggage("gen_ai.group.id")
            with tracer.start_as_current_span("node_b_span"):
                pass
            return {"step": "b_done"}

        graph = StateGraph(SeqState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)
        try:
            app.invoke({"step": ""})
        finally:
            context.detach(token)

        assert captured_per_node.get("node_a") == "round-1", (
            "First node should see baggage"
        )
        assert captured_per_node.get("node_b") == "round-1", (
            "Second node should also see baggage (context not reset between nodes)"
        )

        # Both node spans should carry baggage attributes
        spans = exporter.get_finished_spans()
        for name in ["node_a_span", "node_b_span"]:
            matching = [s for s in spans if s.name == name]
            assert len(matching) == 1
            assert matching[0].attributes.get("gen_ai.group.id") == "round-1"

    def test_baggage_in_conditional_edges(self, tracing):
        """Baggage survives conditional edge routing. The router function
        and the target node both see the baggage. This models the
        should_continue pattern used in agent_overlapping_groups.py."""
        tracer, exporter = tracing

        captured_in_router = {}
        captured_in_target = {}

        class CondState(TypedDict):
            needs_tool: bool
            result: str

        def llm_node(state: CondState) -> CondState:
            return state

        def router(state: CondState) -> str:
            captured_in_router["group_id"] = baggage.get_baggage("gen_ai.group.id")
            if state["needs_tool"]:
                return "tool_node"
            return END

        def tool_node(state: CondState) -> CondState:
            captured_in_target["group_id"] = baggage.get_baggage("gen_ai.group.id")
            with tracer.start_as_current_span("execute_tool"):
                pass
            return {"needs_tool": False, "result": "done"}

        graph = StateGraph(CondState)
        graph.add_node("llm", llm_node)
        graph.add_node("tool_node", tool_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", router, {"tool_node": "tool_node", END: END})
        graph.add_edge("tool_node", END)
        app = graph.compile()

        ctx = baggage.set_baggage("gen_ai.group.id", "round-2")
        token = context.attach(ctx)
        try:
            app.invoke({"needs_tool": True, "result": ""})
        finally:
            context.detach(token)

        assert captured_in_router.get("group_id") == "round-2", (
            "Router function should see baggage"
        )
        assert captured_in_target.get("group_id") == "round-2", (
            "Target node after conditional edge should see baggage"
        )

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "execute_tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes.get("gen_ai.group.id") == "round-2"

    def test_baggage_with_manual_spans_in_nodes(self, tracing):
        """When nodes create spans manually (as in agent_overlapping_groups.py),
        those spans carry baggage attributes via BaggageSpanProcessor.
        This confirms the end-to-end pattern: baggage set at graph entry,
        BaggageSpanProcessor copies to span attributes in every node."""
        tracer, exporter = tracing

        class DemoState(TypedDict):
            messages: list

        def llm_call(state: DemoState) -> DemoState:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.request.model", "gpt-4o")
            return {"messages": [{"role": "assistant", "tool": "search"}]}

        def tool_call(state: DemoState) -> DemoState:
            with tracer.start_as_current_span("execute_tool") as span:
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                span.set_attribute("gen_ai.tool.name", "web_search")
            return {"messages": state["messages"]}

        graph = StateGraph(DemoState)
        graph.add_node("llm", llm_call)
        graph.add_node("tool", tool_call)
        graph.add_edge(START, "llm")
        graph.add_edge("llm", "tool")
        graph.add_edge("tool", END)
        app = graph.compile()

        # Set multiple baggage dimensions (overlapping membership)
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.id", "rag-retrieval", ctx)
        ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)
        token = context.attach(ctx)

        try:
            app.invoke({"messages": []})
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name == "chat"]
        tool_spans = [s for s in spans if s.name == "execute_tool"]

        assert len(chat_spans) == 1
        assert len(tool_spans) == 1

        # Both spans carry all baggage dimensions
        for span in [chat_spans[0], tool_spans[0]]:
            attrs = dict(span.attributes)
            assert attrs.get("gen_ai.group.id") == "round-1"
            assert attrs.get("gen_ai.group.iteration.type") == "react"
            assert attrs.get("gen_ai.group.skill.id") == "rag-retrieval"
            assert attrs.get("gen_ai.agent.id") == "main-agent"

    @pytest.mark.asyncio
    async def test_baggage_propagates_through_ainvoke(self, tracing):
        """Async dispatch path: baggage set before app.ainvoke() must
        reach node functions and propagate to manually-created spans
        within them. Mirrors test_baggage_propagates_through_graph_nodes
        but exercises LangGraph's native async entrypoint.

        Closes the action item in DISCOVERIES.md ("Test async dispatch
        (graph.ainvoke()) for baggage propagation")."""
        tracer, exporter = tracing

        class SimpleState(TypedDict):
            value: str

        async def check_node(state: SimpleState) -> SimpleState:
            _captured_baggage["gen_ai.group.id"] = baggage.get_baggage(
                "gen_ai.group.id"
            )
            _captured_baggage["gen_ai.group.iteration.type"] = baggage.get_baggage(
                "gen_ai.group.iteration.type"
            )
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
            return {"value": "done"}

        graph = StateGraph(SimpleState)
        graph.add_node("check", check_node)
        graph.add_edge(START, "check")
        graph.add_edge("check", END)
        app = graph.compile()

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            await app.ainvoke({"value": "input"})
        finally:
            context.detach(token)

        assert _captured_baggage.get("gen_ai.group.id") == "round-1", (
            "Baggage should propagate into LangGraph node functions via ainvoke"
        )
        assert _captured_baggage.get("gen_ai.group.iteration.type") == "react"

        # Verify span attributes via BaggageSpanProcessor
        spans = exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name == "chat"]
        assert len(chat_spans) == 1
        attrs = dict(chat_spans[0].attributes)
        assert attrs.get("gen_ai.group.id") == "round-1"
        assert attrs.get("gen_ai.group.iteration.type") == "react"
