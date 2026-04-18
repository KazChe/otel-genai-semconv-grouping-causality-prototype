"""LangGraph envelope shape tests — follows the frameworks/ convention.

LangGraph serves as the "same-process demonstrator" for both proposals.
Unlike the frameworks/ tests (which verify behavior against opaque third-party
dispatch models), these tests verify behavior against a dispatch model we
control.

Key questions:
  1. Does config["configurable"] work as a sidecar for carrier injection?
  2. Does _otel carrier in TypedDict state survive graph execution?
  3. Do graph nodes receive full state including extra fields?
  4. Does carrier survive checkpoint serialization (MemorySaver)?
"""

import json

import pytest

langgraph = pytest.importorskip("langgraph")

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated
from operator import add

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS


@pytest.fixture()
def tracing():
    """Set up an in-memory tracing pipeline with BaggageSpanProcessor."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    tracer = provider.get_tracer("test-langgraph-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestLangGraphEnvelopeShape:
    """LangGraph envelope and sidecar tests."""

    def test_config_configurable_as_sidecar(self, tracing):
        """config["configurable"] is a dict[str, Any] passed to every graph
        node via the second positional argument. It survives graph execution
        and can carry the _otel carrier alongside tool execution.

        This is LangGraph's native sidecar mechanism, analogous to
        Haystack's ToolCall.extra or PydanticAI's RunContext.deps."""
        tracer, exporter = tracing
        received_carrier = {}

        class SimpleState(TypedDict):
            value: str

        def node_a(state: SimpleState, config: RunnableConfig) -> SimpleState:
            # Node receives config with configurable dict
            carrier = config.get("configurable", {}).get("_otel", {})
            received_carrier.update(carrier)
            return {"value": "processed"}

        graph = StateGraph(SimpleState)
        graph.add_node("node_a", node_a)
        graph.add_edge(START, "node_a")
        graph.add_edge("node_a", END)
        app = graph.compile()

        # Inject carrier into configurable
        carrier = {}
        with tracer.start_as_current_span("chat") as span:
            inject(carrier)
            parent_ctx = span.get_span_context()

        result = app.invoke(
            {"value": "input"},
            config={"configurable": {"_otel": carrier}},
        )

        assert "traceparent" in received_carrier, (
            "config['configurable'] should carry _otel carrier to graph nodes"
        )

        # Verify we can extract and establish causality
        extracted_ctx = extract(received_carrier)
        from opentelemetry import context
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool") as tool_span:
                tool_parent = tool_span.parent
        finally:
            context.detach(token)

        assert tool_parent.span_id == parent_ctx.span_id, (
            "Carrier from configurable establishes causal parent-child link"
        )

    def test_carrier_in_tool_payload_survives_state(self, tracing):
        """_otel carrier injected into a message dict within TypedDict state
        survives graph execution. LangGraph uses TypedDict for state (not
        Pydantic), so there is no extra-field validation to strip the carrier.

        This is how agent_overlapping_groups.py passes causality context
        from llm_call to tool_call nodes."""
        tracer, exporter = tracing

        class AgentState(TypedDict):
            messages: list
            result: str

        carrier = {}
        with tracer.start_as_current_span("chat"):
            inject(carrier)

        tool_msg = {
            "role": "tool_call",
            "tool": "web_search",
            "input": "HNSW algorithm",
            "_otel": carrier,
        }

        received_otel = {}

        def tool_node(state: AgentState) -> AgentState:
            last_msg = state["messages"][-1]
            otel = last_msg.get("_otel", {})
            received_otel.update(otel)
            return {"messages": state["messages"], "result": "done"}

        graph = StateGraph(AgentState)
        graph.add_node("tool", tool_node)
        graph.add_edge(START, "tool")
        graph.add_edge("tool", END)
        app = graph.compile()

        app.invoke({"messages": [tool_msg], "result": ""})

        assert "traceparent" in received_otel, (
            "_otel carrier in message dict survives TypedDict state"
        )

    def test_state_graph_node_receives_full_state(self, tracing):
        """Graph nodes receive all state fields defined in the TypedDict.
        Unlike Pydantic-based frameworks that may strip unknown fields,
        TypedDict is a structural type hint with no runtime validation."""
        tracer, exporter = tracing

        class FullState(TypedDict):
            question: str
            messages: list
            round: int
            final_answer: str

        received_state_keys = []

        def check_node(state: FullState) -> FullState:
            received_state_keys.extend(state.keys())
            return state

        graph = StateGraph(FullState)
        graph.add_node("check", check_node)
        graph.add_edge(START, "check")
        graph.add_edge("check", END)
        app = graph.compile()

        app.invoke({
            "question": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "round": 1,
            "final_answer": "",
        })

        assert "question" in received_state_keys
        assert "messages" in received_state_keys
        assert "round" in received_state_keys
        assert "final_answer" in received_state_keys

    def test_checkpoint_round_trip(self, tracing):
        """Carrier in state survives MemorySaver checkpoint serialization.
        MemorySaver stores state in-memory (no serialization), but this
        confirms the state dict including _otel is preserved across
        graph steps that trigger checkpointing."""
        tracer, exporter = tracing

        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            pytest.skip("MemorySaver not available in this langgraph version")

        class CheckpointState(TypedDict):
            messages: list
            step: str

        carrier = {}
        with tracer.start_as_current_span("chat"):
            inject(carrier)

        recovered_carrier = {}

        def step_one(state: CheckpointState) -> CheckpointState:
            return {
                "messages": [{"_otel": carrier, "content": "tool call"}],
                "step": "one",
            }

        def step_two(state: CheckpointState) -> CheckpointState:
            msg = state["messages"][-1]
            recovered_carrier.update(msg.get("_otel", {}))
            return {"messages": state["messages"], "step": "two"}

        graph = StateGraph(CheckpointState)
        graph.add_node("step_one", step_one)
        graph.add_node("step_two", step_two)
        graph.add_edge(START, "step_one")
        graph.add_edge("step_one", "step_two")
        graph.add_edge("step_two", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        app.invoke(
            {"messages": [], "step": ""},
            config={"configurable": {"thread_id": "test-1"}},
        )

        assert "traceparent" in recovered_carrier, (
            "Carrier in state should survive checkpoint round-trip"
        )
