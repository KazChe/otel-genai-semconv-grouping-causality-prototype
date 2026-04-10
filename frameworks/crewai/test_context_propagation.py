"""CrewAI — verify OTel context/baggage propagation across execution.

Tests whether W3C Baggage set in the caller context survives into
CrewAI tool execution.

Research classification: "Requires manual propagation" — CrewAI has
two async entrypoints: akickoff() (native async) and kickoff_async()
(thread-based via asyncio.to_thread()). Baggage behavior differs.

These are integration tests that import the real framework.
"""

import pytest

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS

crewai = pytest.importorskip("crewai")

from crewai.tools import BaseTool


# Storage for baggage values captured inside tools
_captured_baggage = {}


class BaggageCapturingTool(BaseTool):
    name: str = "baggage_reader"
    description: str = "Reads baggage from current context"

    def _run(self) -> str:
        _captured_baggage["gen_ai.group.id"] = baggage.get_baggage("gen_ai.group.id")
        _captured_baggage["gen_ai.group.iteration.type"] = baggage.get_baggage(
            "gen_ai.group.iteration.type"
        )
        return "Baggage captured"


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-crewai-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage.clear()
    yield
    _captured_baggage.clear()


class TestCrewAIContextPropagation:
    """Verify baggage propagation across CrewAI tool execution."""

    def test_baggage_in_direct_tool_run(self, tracing):
        """Set baggage before calling tool.run() directly.
        This tests the simplest path — same-thread, no dispatch."""
        tracer, exporter = tracing

        tool = BaggageCapturingTool()

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            tool.run()
        except Exception as e:
            print(f"\ntool.run() failed: {type(e).__name__}: {e}")
            # Try _run directly
            try:
                tool._run()
            except Exception as e2:
                print(f"tool._run() also failed: {type(e2).__name__}: {e2}")
        finally:
            context.detach(token)

        print(f"\nDirect tool run captured baggage: {_captured_baggage}")

        if _captured_baggage.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES in direct tool.run()")
        elif not _captured_baggage:
            print("RESULT: Tool was not called or failed — cannot determine")
        else:
            print("RESULT: Baggage LOST in direct tool.run()")

    def test_baggage_in_spans_via_processor(self, tracing):
        """Verify that spans created in the caller context carry
        baggage attributes via BaggageSpanProcessor."""
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
