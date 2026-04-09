"""Google ADK — verify OTel context/baggage propagation across execution.

Tests whether W3C Baggage set in the caller context survives into
Google ADK tool execution.

Research classification: Unknown — no prior data on ADK's dispatch model.

These are integration tests that import the real framework.
"""

import pytest

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS

google_adk = pytest.importorskip("google.adk")

from google.adk.tools import FunctionTool


# Storage for baggage values captured inside tools
_captured_baggage = {}


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-google-adk-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage.clear()
    yield
    _captured_baggage.clear()


class TestGoogleADKContextPropagation:
    """Verify baggage propagation across Google ADK tool execution."""

    def test_baggage_in_direct_tool_function(self, tracing):
        """Set baggage before calling the tool's underlying function
        directly. This tests the simplest path — same-thread."""
        tracer, exporter = tracing

        def search(query: str) -> str:
            """Search tool."""
            _captured_baggage["gen_ai.group.id"] = baggage.get_baggage(
                "gen_ai.group.id"
            )
            _captured_baggage["gen_ai.group.iteration.type"] = baggage.get_baggage(
                "gen_ai.group.iteration.type"
            )
            return f"Results for {query}"

        tool = FunctionTool(search)

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            # Call the underlying function directly
            search(query="HNSW")
        finally:
            context.detach(token)

        print(f"\nDirect function call captured baggage: {_captured_baggage}")

        if _captured_baggage.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES in direct function call")
        else:
            print("RESULT: Baggage LOST in direct function call")

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
