"""AutoGen v0.4 — verify OTel context/baggage propagation across message dispatch.

Tests whether W3C Baggage set in the caller context survives AutoGen's
SingleThreadedAgentRuntime per-message asyncio.create_task() dispatch.

Research classification: "Requires manual propagation" for in-process,
"Breaks" for GrpcWorkerAgentRuntime (cross-process).

These are integration tests that import the real framework.
"""

import asyncio
import json
import pytest

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS

autogen_core = pytest.importorskip("autogen_core")

from autogen_core import (
    SingleThreadedAgentRuntime,
    MessageContext,
    AgentId,
    DefaultTopicId,
    RoutedAgent,
    message_handler,
)
from dataclasses import dataclass


@dataclass
class BaggageTestMessage:
    content: str


# Storage for baggage values captured inside agent handlers
_captured_baggage = {}


class BaggageCapturingAgent(RoutedAgent):
    """Agent that captures baggage values from its execution context."""

    @message_handler
    async def handle_message(self, message: BaggageTestMessage, ctx: MessageContext) -> None:
        # Read baggage from the current context
        group_id = baggage.get_baggage("gen_ai.group.id")
        iteration_type = baggage.get_baggage("gen_ai.group.iteration.type")
        agent_id = baggage.get_baggage("gen_ai.agent.id")

        _captured_baggage["gen_ai.group.id"] = group_id
        _captured_baggage["gen_ai.group.iteration.type"] = iteration_type
        _captured_baggage["gen_ai.agent.id"] = agent_id


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-autogen-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage.clear()
    yield
    _captured_baggage.clear()


class TestAutoGenContextPropagation:
    """Verify baggage propagation across AutoGen's message dispatch."""

    @pytest.mark.asyncio
    async def test_baggage_at_runtime_dispatch(self, tracing):
        """Set baggage before sending a message via SingleThreadedAgentRuntime.
        Check if baggage arrives in the agent's message handler.

        This tests the real asyncio.create_task() dispatch boundary
        that AutoGen uses for per-message processing."""
        tracer, exporter = tracing

        runtime = SingleThreadedAgentRuntime()

        await BaggageCapturingAgent.register(
            runtime,
            "baggage_agent",
            lambda: BaggageCapturingAgent("Baggage capturing agent"),
        )

        runtime.start()

        # Set baggage in caller context
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)
        token = context.attach(ctx)

        try:
            await runtime.publish_message(
                BaggageTestMessage(content="test baggage propagation"),
                topic_id=DefaultTopicId(),
            )
            await runtime.stop_when_idle()
        finally:
            context.detach(token)

        # Check what baggage the agent handler captured
        print(f"\nCaptured baggage in handler: {_captured_baggage}")

        # Document the actual behavior — does baggage arrive?
        if _captured_baggage.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES across AutoGen message dispatch")
        else:
            print("RESULT: Baggage LOST at AutoGen message dispatch boundary")
            print("Classification: Requires manual propagation")

    @pytest.mark.asyncio
    async def test_baggage_in_span_attributes_via_processor(self, tracing):
        """Even if baggage doesn't propagate to the handler context,
        verify that spans created in the caller context carry baggage
        attributes via BaggageSpanProcessor."""
        tracer, exporter = tracing

        # Set baggage and create a span
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1

        chat_span = [s for s in spans if s.name == "chat"][0]
        attrs = dict(chat_span.attributes)

        # BaggageSpanProcessor should have copied baggage to span attributes
        assert attrs.get("gen_ai.group.id") == "round-1"
        assert attrs.get("gen_ai.group.iteration.type") == "react"
