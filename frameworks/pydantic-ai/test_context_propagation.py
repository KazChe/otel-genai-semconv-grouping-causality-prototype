"""PydanticAI — verify OTel context/baggage propagation across agent execution.

Tests whether W3C Baggage set before agent.run() survives into
tool function execution.

PydanticAI tools are Python functions invoked in-process, so baggage
should propagate if execution stays in the same async context. This
test verifies whether PydanticAI introduces any context isolation
between agent loop and tool execution.

These are integration tests that import the real framework.
"""

import asyncio
import pytest

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


# Storage for baggage values captured inside tool functions
_captured_baggage = {}


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-pydanticai-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage.clear()
    yield
    _captured_baggage.clear()


class TestPydanticAIContextPropagation:
    """Verify baggage propagation across PydanticAI agent execution."""

    @pytest.mark.asyncio
    async def test_baggage_arrives_in_tool_plain(self, tracing):
        """Set baggage before agent.run() and check if a @tool_plain
        function can read it. Tests whether PydanticAI's agent loop
        preserves OTel context into tool execution."""
        tracer, exporter = tracing

        agent = Agent(
            TestModel(custom_output_text="Final answer."),
            system_prompt="Test agent.",
        )

        @agent.tool_plain
        def search(query: str) -> str:
            """Search tool."""
            _captured_baggage["gen_ai.group.id"] = baggage.get_baggage("gen_ai.group.id")
            _captured_baggage["gen_ai.group.iteration.type"] = baggage.get_baggage("gen_ai.group.iteration.type")
            return f"Results for {query}"

        # Set baggage in caller context
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            result = await agent.run("Search for HNSW")
        finally:
            context.detach(token)

        print(f"\nTool captured baggage: {_captured_baggage}")

        if _captured_baggage.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES into PydanticAI tool_plain")
        elif not _captured_baggage:
            print("NOTE: Tool was not called by TestModel — cannot determine propagation")
        else:
            print("RESULT: Baggage LOST at PydanticAI agent boundary")

    @pytest.mark.asyncio
    async def test_baggage_arrives_in_tool_with_context(self, tracing):
        """Set baggage before agent.run() with a @tool (RunContext) function.
        Check if baggage is accessible alongside the deps context."""
        tracer, exporter = tracing

        from dataclasses import dataclass

        @dataclass
        class Deps:
            session_id: str

        agent = Agent(
            TestModel(custom_output_text="Done."),
            deps_type=Deps,
            system_prompt="Test agent.",
        )

        @agent.tool
        def search_with_ctx(ctx: RunContext[Deps], query: str) -> str:
            """Search with context."""
            _captured_baggage["gen_ai.group.id"] = baggage.get_baggage("gen_ai.group.id")
            _captured_baggage["deps.session_id"] = ctx.deps.session_id
            return f"Results for {query}"

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            result = await agent.run(
                "Search for HNSW",
                deps=Deps(session_id="test-session"),
            )
        finally:
            context.detach(token)

        print(f"\nTool captured: {_captured_baggage}")

        if _captured_baggage.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES into PydanticAI tool with RunContext")
            print("Both baggage AND deps are accessible in the tool")
        elif _captured_baggage.get("deps.session_id") == "test-session":
            print("RESULT: Deps arrived but baggage LOST")
        elif not _captured_baggage:
            print("NOTE: Tool was not called by TestModel — cannot determine propagation")

    @pytest.mark.asyncio
    async def test_baggage_in_spans_via_processor(self, tracing):
        """Verify that spans created within the agent context carry
        baggage attributes via BaggageSpanProcessor, regardless of
        whether baggage propagates into tools."""
        tracer, exporter = tracing

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        chat_span = [s for s in spans if s.name == "chat"][0]
        attrs = dict(chat_span.attributes)

        assert attrs.get("gen_ai.group.id") == "round-1"
        assert attrs.get("gen_ai.group.iteration.type") == "react"
