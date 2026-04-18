"""Baggage-based grouping mechanism tests — pure OTel primitives, no framework imports.

These tests validate the W3C Baggage + BaggageSpanProcessor mechanism that
underpins the grouping proposal (ISSUE_GROUPING.md). They are framework-independent
and test three concerns:

Part 1 — Overlapping group membership (TestOverlappingGroupMembership):
  Namespaced baggage keys solve mutual exclusivity. Each group dimension
  gets its own key, so a span belongs to multiple groups simultaneously.

Part 2 — Baggage propagation across execution boundaries (TestBaggagePropagationBoundaries):
  Three propagation states:
    Propagates: same-task sync, in-process sequential await chains
    Requires manual propagation: asyncio.create_task(), asyncio.to_thread(), ThreadPoolExecutor
    Breaks: cross-process boundaries (contextvars don't serialize)

Part 3 — Mitigation patterns (TestBaggagePropagationMitigations):
  Capture/Re-attach: context.get_current() + context.attach()
  Copy context: contextvars.copy_context().run()
  Serialize: baggage dict to message metadata for cross-process

Tests use InMemorySpanExporter for programmatic assertions, no Docker needed.
"""

import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import trace, baggage, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS


@pytest.fixture()
def tracing():
    """Set up an in-memory tracing pipeline with BaggageSpanProcessor."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
    # Get tracer directly from provider to avoid global state conflicts
    tracer = provider.get_tracer("test-baggage-mechanism")
    yield tracer, exporter
    provider.shutdown()


class TestOverlappingGroupMembership:
    """A span can belong to multiple groups across different dimensions.

    Addresses @Cirilla-zmh's concern: "If gen_ai.group.type is of type
    StringAttributeKey, shouldn't its value be mutually exclusive?"

    Solution: namespaced baggage keys (gen_ai.group.id, gen_ai.group.iteration.type,
    gen_ai.group.skill.id, gen_ai.group.skill.type) are independent dimensions.
    """

    def test_span_carries_all_baggage_dimensions(self, tracing):
        """Core test: a single span can carry iteration + skill + agent
        dimensions simultaneously via separate baggage keys."""
        tracer, exporter = tracing

        ctx = baggage.set_baggage("gen_ai.group.id", "round-2")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.id", "rag-retrieval", ctx)
        ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.group.id"] == "round-2"
        assert attrs["gen_ai.group.iteration.type"] == "react"
        assert attrs["gen_ai.group.skill.id"] == "rag-retrieval"
        assert attrs["gen_ai.agent.id"] == "main-agent"

    def test_skill_and_iteration_coexist_on_tool_span(self, tracing):
        """A tool span invoked as part of a skill within a ReAct iteration
        carries both the skill and iteration group attributes."""
        tracer, exporter = tracing

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.id", "knowledge-lookup", ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.type", "rag", ctx)
        ctx = baggage.set_baggage("gen_ai.agent.id", "research-agent", ctx)
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("execute_tool") as span:
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                span.set_attribute("gen_ai.tool.name", "vector_search")
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["gen_ai.group.skill.id"] == "knowledge-lookup"
        assert attrs["gen_ai.group.skill.type"] == "rag"
        assert attrs["gen_ai.group.iteration.type"] == "react"
        assert attrs["gen_ai.group.id"] == "round-1"
        assert attrs["gen_ai.agent.id"] == "research-agent"

    def test_dimensions_change_between_phases(self, tracing):
        """Group dimensions can change between phases: skill ends but
        iteration continues. Proves dimensions are independent."""
        tracer, exporter = tracing

        # Phase 1: skill active within iteration
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.id", "rag-retrieval", ctx)
        token = context.attach(ctx)
        try:
            with tracer.start_as_current_span("execute_tool"):
                pass
        finally:
            context.detach(token)

        # Phase 2: same iteration, no skill (reasoning after tool result)
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)
        try:
            with tracer.start_as_current_span("chat"):
                pass
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        tool_attrs = dict(spans[0].attributes)
        chat_attrs = dict(spans[1].attributes)

        assert "gen_ai.group.skill.id" in tool_attrs
        assert tool_attrs["gen_ai.group.skill.id"] == "rag-retrieval"
        assert "gen_ai.group.skill.id" not in chat_attrs
        assert tool_attrs["gen_ai.group.id"] == "round-1"
        assert chat_attrs["gen_ai.group.id"] == "round-1"

    def test_nested_agent_delegation(self, tracing):
        """An inner agent's spans carry both the outer agent's iteration
        group and the inner agent's own identity."""
        tracer, exporter = tracing

        ctx = baggage.set_baggage("gen_ai.group.id", "round-3")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        ctx = baggage.set_baggage("gen_ai.agent.id", "orchestrator", ctx)
        token = context.attach(ctx)

        try:
            with tracer.start_as_current_span("chat") as outer_span:
                outer_span.set_attribute("gen_ai.operation.name", "chat")

                inner_ctx = baggage.set_baggage(
                    "gen_ai.group.delegated_from", "orchestrator"
                )
                inner_ctx = baggage.set_baggage(
                    "gen_ai.agent.id", "research-sub-agent", inner_ctx
                )
                inner_ctx = baggage.set_baggage(
                    "gen_ai.group.id", "round-3", inner_ctx
                )
                inner_ctx = baggage.set_baggage(
                    "gen_ai.group.iteration.type", "react", inner_ctx
                )
                inner_token = context.attach(inner_ctx)
                try:
                    with tracer.start_as_current_span("chat") as inner_span:
                        inner_span.set_attribute("gen_ai.operation.name", "chat")
                finally:
                    context.detach(inner_token)
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        inner_attrs = dict(spans[0].attributes)
        assert inner_attrs["gen_ai.agent.id"] == "research-sub-agent"
        assert inner_attrs["gen_ai.group.delegated_from"] == "orchestrator"
        assert inner_attrs["gen_ai.group.id"] == "round-3"
        assert inner_attrs["gen_ai.group.iteration.type"] == "react"

        outer_attrs = dict(spans[1].attributes)
        assert outer_attrs["gen_ai.agent.id"] == "orchestrator"
        assert outer_attrs["gen_ai.group.id"] == "round-3"

    def test_queryability_by_any_dimension(self, tracing):
        """Create multiple spans across dimensions and verify each dimension
        can be used independently for filtering."""
        tracer, exporter = tracing

        scenarios = [
            {"gen_ai.group.id": "round-1", "gen_ai.group.iteration.type": "react",
             "gen_ai.group.skill.id": "rag", "gen_ai.agent.id": "agent-A"},
            {"gen_ai.group.id": "round-1", "gen_ai.group.iteration.type": "react",
             "gen_ai.agent.id": "agent-A"},
            {"gen_ai.group.id": "round-2", "gen_ai.group.iteration.type": "react",
             "gen_ai.group.skill.id": "code-gen", "gen_ai.agent.id": "agent-B"},
        ]

        for scenario in scenarios:
            ctx = context.get_current()
            for key, value in scenario.items():
                ctx = baggage.set_baggage(key, value, ctx)
            token = context.attach(ctx)
            try:
                with tracer.start_as_current_span("chat"):
                    pass
            finally:
                context.detach(token)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        round1 = [s for s in spans if s.attributes.get("gen_ai.group.id") == "round-1"]
        assert len(round1) == 2

        rag = [s for s in spans if s.attributes.get("gen_ai.group.skill.id") == "rag"]
        assert len(rag) == 1

        agent_a = [s for s in spans if s.attributes.get("gen_ai.agent.id") == "agent-A"]
        assert len(agent_a) == 2

        combo = [s for s in spans
                 if s.attributes.get("gen_ai.group.id") == "round-2"
                 and s.attributes.get("gen_ai.agent.id") == "agent-B"]
        assert len(combo) == 1
        assert combo[0].attributes.get("gen_ai.group.skill.id") == "code-gen"


class TestBaggagePropagationBoundaries:
    """Maps which execution boundaries preserve or break baggage propagation.

    These tests model the dispatch patterns used by LLM orchestration
    frameworks without importing any framework, using asyncio, threads,
    and OTel primitives only."""

    def test_baggage_survives_same_task(self, tracing):
        """Baggage set and read within the same synchronous execution
        flow. Baseline happy path: no dispatch boundary, no context copy.
        Mirrors Instructor and ControlFlow patterns where orchestration
        stays in-process and sequential."""
        tracer, exporter = tracing

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            # Read baggage in the same sync flow
            value = baggage.get_baggage("gen_ai.group.id")
            assert value == "round-1"

            with tracer.start_as_current_span("chat"):
                # Baggage is still accessible inside the span
                inner_value = baggage.get_baggage("gen_ai.group.id")
                assert inner_value == "round-1"
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("gen_ai.group.id") == "round-1"

    @pytest.mark.asyncio
    async def test_baggage_snapshot_at_create_task(self, tracing):
        """asyncio.create_task() copies the current contextvars snapshot
        into the new task. Parent baggage is visible in the child task,
        but baggage set in the child does NOT flow back to the parent.
        This is the dispatch pattern used by LlamaIndex workflows,
        AutoGen SingleThreadedAgentRuntime, and Haystack AsyncPipeline."""
        captured = {}

        async def child_task():
            captured["child_sees"] = baggage.get_baggage("gen_ai.group.id")
            # Set new baggage in child; should NOT affect parent
            child_ctx = baggage.set_baggage("gen_ai.group.id", "child-override")
            context.attach(child_ctx)

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            task = asyncio.create_task(child_task())
            await task

            # Child saw parent's baggage
            assert captured["child_sees"] == "round-1"

            # Parent's baggage is unchanged (child mutation didn't flow back)
            assert baggage.get_baggage("gen_ai.group.id") == "round-1"
        finally:
            context.detach(token)

    def test_baggage_lost_in_thread_pool(self, tracing):
        """ThreadPoolExecutor.submit() does NOT automatically propagate
        contextvars into worker threads. Baggage set before submit is
        not visible inside the worker callable. This is the dispatch
        pattern used by DSPy's parallelizer and batch evaluation."""
        captured = {}

        def worker():
            captured["worker_sees"] = baggage.get_baggage("gen_ai.group.id")

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker)
                future.result()
        finally:
            context.detach(token)

        # Baggage is lost in the worker thread
        assert captured["worker_sees"] is None, (
            "ThreadPoolExecutor does not propagate contextvars automatically"
        )

    @pytest.mark.asyncio
    async def test_asyncio_to_thread_snapshot(self, tracing):
        """asyncio.to_thread() copies the current context into the
        worker thread, so baggage set before the call is visible inside
        the thread. But mutations inside the thread do NOT flow back.
        This is the dispatch pattern used by CrewAI's kickoff_async()
        and Flow._execute_method()."""
        captured = {}

        def thread_work():
            captured["thread_sees"] = baggage.get_baggage("gen_ai.group.id")
            # Mutate in thread; should not affect caller
            mutated_ctx = baggage.set_baggage("gen_ai.group.id", "thread-override")
            context.attach(mutated_ctx)

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            await asyncio.to_thread(thread_work)

            # Thread saw the baggage (asyncio.to_thread copies context)
            assert captured["thread_sees"] == "round-1"

            # Caller's baggage is unchanged
            assert baggage.get_baggage("gen_ai.group.id") == "round-1"
        finally:
            context.detach(token)

    def test_baggage_lost_cross_process(self, tracing):
        """Simulates a process boundary. Baggage in Python contextvars
        does not survive serialization. This models AutoGen's
        GrpcWorkerAgentRuntime for distributed agent execution.

        We simulate by serializing baggage entries to JSON (as a process
        boundary would require) and showing that the receiver has no
        baggage unless explicitly reconstructed."""
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            # "Sender" side: serialize message (no baggage in the wire format)
            message = json.dumps({"content": "hello", "role": "user"})
        finally:
            context.detach(token)

        # "Receiver" side: new context, no baggage
        received = json.loads(message)
        assert received["content"] == "hello"
        assert baggage.get_baggage("gen_ai.group.id") is None, (
            "Baggage does not survive process/serialization boundaries"
        )


class TestBaggagePropagationMitigations:
    """Mitigation patterns for execution boundaries where baggage is lost."""

    @pytest.mark.asyncio
    async def test_capture_reattach_shim(self, tracing):
        """Capture/Re-attach pattern: context.get_current() before dispatch,
        context.attach() on the receiver side. Mitigation for
        asyncio.create_task() and thread boundaries. Recommended for
        LlamaIndex, AutoGen (in-process), Semantic Kernel, and Haystack."""
        captured = {}

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            # Capture context BEFORE dispatch
            saved_context = context.get_current()
        finally:
            context.detach(token)

        # After detach, baggage is gone from current context
        assert baggage.get_baggage("gen_ai.group.id") is None

        # Re-attach on the "receiver" side
        reattach_token = context.attach(saved_context)
        try:
            captured["reattached"] = baggage.get_baggage("gen_ai.group.id")
        finally:
            context.detach(reattach_token)

        assert captured["reattached"] == "round-1", (
            "Capture/re-attach recovers baggage after dispatch boundary"
        )

    def test_copy_context_thread_pool(self, tracing):
        """contextvars.copy_context().run() preserves baggage in
        ThreadPoolExecutor workers. Mitigation for DSPy's parallelizer
        and any framework using thread pools without built-in OTel
        context propagation. This is the pattern Haystack uses."""
        captured = {}

        def worker():
            captured["worker_sees"] = baggage.get_baggage("gen_ai.group.id")

        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        token = context.attach(ctx)

        try:
            copied_ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(copied_ctx.run, worker)
                future.result()
        finally:
            context.detach(token)

        assert captured["worker_sees"] == "round-1", (
            "copy_context().run() should preserve baggage into executor thread"
        )

    def test_serialize_baggage_cross_process(self, tracing):
        """Serialize baggage entries to a dict for cross-process transmission,
        then reconstruct on the receiver side. Mitigation for AutoGen
        GrpcWorkerAgentRuntime and any framework crossing process/language
        boundaries. Analogous to the 'Out-of-Band Correlation' pattern
        in the causality tests."""
        tracer, exporter = tracing

        # "Sender" side: extract baggage entries into a serializable dict
        ctx = baggage.set_baggage("gen_ai.group.id", "round-1")
        ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
        token = context.attach(ctx)

        try:
            baggage_dict = {
                "gen_ai.group.id": baggage.get_baggage("gen_ai.group.id"),
                "gen_ai.group.iteration.type": baggage.get_baggage(
                    "gen_ai.group.iteration.type"
                ),
            }
            # Serialize as part of message metadata
            wire_message = json.dumps({
                "content": "hello",
                "_baggage": baggage_dict,
            })
        finally:
            context.detach(token)

        # "Receiver" side: reconstruct baggage from message metadata
        received = json.loads(wire_message)
        receiver_ctx = context.get_current()
        for key, value in received["_baggage"].items():
            receiver_ctx = baggage.set_baggage(key, value, receiver_ctx)
        receiver_token = context.attach(receiver_ctx)

        try:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
        finally:
            context.detach(receiver_token)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.group.id"] == "round-1"
        assert attrs["gen_ai.group.iteration.type"] == "react"
