"""Haystack — verify OTel context/baggage propagation across pipeline dispatch.

Tests whether W3C Baggage set before pipeline.run() survives into
component execution, and whether baggage set in one component
arrives in the next.

Research classification: "Requires manual propagation" — Haystack
explicitly uses contextvars.copy_context() for executor offload,
but component-to-component baggage mutations don't flow automatically.

These are integration tests that import the real framework.
"""

import pytest

from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS

haystack = pytest.importorskip("haystack")

from haystack import Pipeline, AsyncPipeline, component, Document


# Storage for baggage values captured inside components
_captured_baggage_comp1 = {}
_captured_baggage_comp2 = {}


@component
class BaggageReaderComponent1:
    """Component that reads baggage and stores it for inspection."""

    @component.output_types(output=str)
    def run(self, input_text: str) -> dict:
        _captured_baggage_comp1["gen_ai.group.id"] = baggage.get_baggage("gen_ai.group.id")
        _captured_baggage_comp1["gen_ai.group.iteration.type"] = baggage.get_baggage("gen_ai.group.iteration.type")
        return {"output": f"comp1: {input_text}"}


@component
class BaggageReaderComponent2:
    """Second component that reads baggage after the first."""

    @component.output_types(output=str)
    def run(self, input_text: str) -> dict:
        _captured_baggage_comp2["gen_ai.group.id"] = baggage.get_baggage("gen_ai.group.id")
        _captured_baggage_comp2["gen_ai.group.iteration.type"] = baggage.get_baggage("gen_ai.group.iteration.type")
        return {"output": f"comp2: {input_text}"}


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-haystack-propagation")
    yield tracer, exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def reset_captured_baggage():
    _captured_baggage_comp1.clear()
    _captured_baggage_comp2.clear()
    yield
    _captured_baggage_comp1.clear()
    _captured_baggage_comp2.clear()


class TestHaystackContextPropagation:
    """Verify baggage propagation across Haystack pipeline execution."""

    def test_baggage_arrives_in_first_component(self, tracing):
        """Set baggage before pipeline.run() and check if the first
        component can read it. This tests whether Haystack's pipeline
        runner preserves the caller's OTel context."""
        tracer, exporter = tracing

        pipe = Pipeline()
        pipe.add_component("reader1", BaggageReaderComponent1())

        # Set baggage in caller context
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            pipe.run({"reader1": {"input_text": "test"}})
        finally:
            context.detach(token)

        print(f"\nComponent 1 captured baggage: {_captured_baggage_comp1}")

        if _captured_baggage_comp1.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES from caller into Haystack component")
        else:
            print("RESULT: Baggage LOST at pipeline entry boundary")

    def test_baggage_between_sequential_components(self, tracing):
        """Set baggage before pipeline.run() with two sequential components.
        Check if both components see the baggage. This tests whether
        Haystack preserves context across component-to-component dispatch."""
        tracer, exporter = tracing

        pipe = Pipeline()
        pipe.add_component("reader1", BaggageReaderComponent1())
        pipe.add_component("reader2", BaggageReaderComponent2())
        pipe.connect("reader1.output", "reader2.input_text")

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            pipe.run({"reader1": {"input_text": "test"}})
        finally:
            context.detach(token)

        print(f"\nComponent 1 captured: {_captured_baggage_comp1}")
        print(f"Component 2 captured: {_captured_baggage_comp2}")

        comp1_has = _captured_baggage_comp1.get("gen_ai.group.id") == "round-1"
        comp2_has = _captured_baggage_comp2.get("gen_ai.group.id") == "round-1"

        if comp1_has and comp2_has:
            print("RESULT: Baggage PROPAGATES across both components")
        elif comp1_has and not comp2_has:
            print("RESULT: Baggage arrives in comp1 but LOST between components")
            print("Classification: Requires manual propagation")
        else:
            print("RESULT: Baggage LOST at pipeline entry")

    @pytest.mark.asyncio
    async def test_async_pipeline_baggage_in_first_component(self, tracing):
        """Set baggage before async pipeline.run() and check if the first
        component can read it. AsyncPipeline uses asyncio.create_task()
        and may offload sync components to an executor — different
        dispatch path from sync Pipeline."""
        tracer, exporter = tracing

        pipe = AsyncPipeline()
        pipe.add_component("reader1", BaggageReaderComponent1())

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            await pipe.run_async({"reader1": {"input_text": "test"}})
        finally:
            context.detach(token)

        print(f"\n[ASYNC] Component 1 captured baggage: {_captured_baggage_comp1}")

        if _captured_baggage_comp1.get("gen_ai.group.id") == "round-1":
            print("RESULT: Baggage PROPAGATES from caller into async component")
        else:
            print("RESULT: Baggage LOST at async pipeline entry boundary")

    @pytest.mark.asyncio
    async def test_async_pipeline_baggage_between_components(self, tracing):
        """Set baggage before async pipeline.run() with two sequential
        components. Check if both see the baggage. This tests whether
        AsyncPipeline's create_task() / executor offload preserves
        context across component-to-component dispatch.

        Research says Haystack uses contextvars.copy_context() for
        executor offload — this test verifies that claim."""
        tracer, exporter = tracing

        pipe = AsyncPipeline()
        pipe.add_component("reader1", BaggageReaderComponent1())
        pipe.add_component("reader2", BaggageReaderComponent2())
        pipe.connect("reader1.output", "reader2.input_text")

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            await pipe.run_async({"reader1": {"input_text": "test"}})
        finally:
            context.detach(token)

        print(f"\n[ASYNC] Component 1 captured: {_captured_baggage_comp1}")
        print(f"[ASYNC] Component 2 captured: {_captured_baggage_comp2}")

        comp1_has = _captured_baggage_comp1.get("gen_ai.group.id") == "round-1"
        comp2_has = _captured_baggage_comp2.get("gen_ai.group.id") == "round-1"

        if comp1_has and comp2_has:
            print("RESULT: Baggage PROPAGATES across both async components")
            print("Haystack's copy_context() preserves baggage in executor")
        elif comp1_has and not comp2_has:
            print("RESULT: Baggage arrives in comp1 but LOST between async components")
            print("Classification: Requires manual propagation for async pipeline")
        else:
            print("RESULT: Baggage LOST at async pipeline entry")
