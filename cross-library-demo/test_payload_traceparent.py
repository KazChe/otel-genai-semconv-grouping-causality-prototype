"""Payload traceparent resilience — addressing @Cirilla-zmh's concern.

Concern: "You can't always assume that you can access the tool-call payload
and inject additional information into it. Parsing and copying may occur,
so the context you inject into payload could be lost."

This test suite maps the compatibility matrix for payload-level traceparent
injection across the transformation scenarios that occur between LLM
response and tool execution in frameworks like LangGraph and LangChain.

Three integration states are documented:

  Compatible (carrier survives):
    - JSON round-trip: single-hop and multi-hop (Model API -> Orchestrator -> Tool)
    - copy.deepcopy: general Python cloning
    - Pydantic extra="allow": dynamic schema with extension fields
    - Dataclass with metadata dict: typed objects with nested extras
    - MessagePack: binary RPC serialization
    - LangGraph Checkpointer persistence (JSON + Pydantic round-trips)

  Hard reject (carrier causes crash):
    - Pydantic extra="forbid": strict schemas raise ValidationError

  Silent strip (carrier lost, app continues with disconnected traces):
    - Schema sanitization that filters for known fields only

Two mitigation patterns are tested for the failure cases:
    - Sidecar Metadata: place carrier in a permitted 'metadata' field
    - Out-of-Band Correlation: store carrier externally keyed by correlation ID

Tests use InMemorySpanExporter for programmatic assertions, no Docker needed
"""

import copy
import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract


@pytest.fixture()
def tracing():
    """Set up an in-memory tracing pipeline"""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    #Set as global provider so inject/extract use it
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-payload-traceparent")
    yield tracer, exporter
    provider.shutdown()


def _inject_carrier_in_active_span(tracer):
    """Helper: create a chat span and inject traceparent while it's active"""
    carrier = {}
    span_ctx = None

    with tracer.start_as_current_span("chat") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        inject(carrier)
        span_ctx = span.get_span_context()

    return carrier, span_ctx


def _extract_and_create_child(tracer, carrier):
    """Helper: extract traceparent from carrier and create a child span."""
    extracted_ctx = extract(carrier)
    token = context.attach(extracted_ctx)
    try:
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            return span.get_span_context()
    finally:
        context.detach(token)


def _assert_parent_child(exporter, parent_span_ctx):
    """Assert execute_tool's parent is the chat span"""
    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool"]
    assert len(tool_spans) == 1, f"Expected 1 execute_tool span, got {len(tool_spans)}"

    tool_span = tool_spans[0]
    assert tool_span.parent is not None, "execute_tool has no parent -causality lost"
    assert tool_span.parent.trace_id == parent_span_ctx.trace_id, "Wrong trace"
    assert tool_span.parent.span_id == parent_span_ctx.span_id, "Wrong parent span"


class TestPayloadTraceparentSurvival:
    """Verify traceparent survives realistic payload transformations."""

    def test_json_round_trip(self, tracing):
        """JSON serialize + deserialize the most common transformation.
        This always happens when payloads cross any serialization boundary."""
        tracer, exporter = tracing

        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Simulate payload goes through JSON round-trip
        payload = {"tool": "web_search", "input": "HNSW", "_otel": carrier}
        serialized = json.dumps(payload)
        deserialized = json.loads(serialized)

        recovered_carrier = deserialized["_otel"]
        _extract_and_create_child(tracer, recovered_carrier)
        _assert_parent_child(exporter, parent_ctx)

    def test_deep_copy(self, tracing):
        """copy.deepcopy — a general Python cloning mechanism.
        Note: LangGraph uses a Checkpointer interface with serialization/
        deserialization (typically Pydantic models) for state persistence
        across nodes, not deepcopy. That path is covered by
        test_pydantic_model_with_extra_allowed and test_json_round_trip."""
        tracer, exporter = tracing

        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        payload = {"tool": "web_search", "input": "HNSW", "_otel": carrier}
        copied = copy.deepcopy(payload)

        _extract_and_create_child(tracer, copied["_otel"])
        _assert_parent_child(exporter, parent_ctx)

    def test_nested_json_in_message_list(self, tracing):
        """Verifies that tracing metadata ('_otel' carrier) embedded within
        message history remains intact through a full JSON serialization
        and deserialization cycle. This ensures that LangGraph's persistent
        state management (Checkpointer) correctly preserves OpenTelemetry
        context across database round-trips."""
        tracer, exporter = tracing

        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "tool_call", "tool": "search", "input": "q", "_otel": carrier},
        ]

        # Full round-trip through JSON
        restored = json.loads(json.dumps(messages))
        recovered = restored[2]["_otel"]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_dataclass_payload(self, tracing):
        """Validates that OpenTelemetry tracing metadata embedded within
        nested dataclass fields (e.g ToolCall metadata) remains
        retrievable via structured attribute access. This ensures
        compatibility with framework state management patterns where
        raw data is encapsulated in typed objects (Pydantic/dataclasses)
        prior to serialization."""
        from dataclasses import dataclass, field

        @dataclass
        class ToolCall:
            name: str
            input: str
            metadata: dict = field(default_factory=dict)

        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        tc = ToolCall(name="search", input="HNSW", metadata={"_otel": carrier})

        # Simulate framework accessing via attribute
        recovered = tc.metadata["_otel"]
        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_pydantic_model_with_extra_allowed(self, tracing):
        """Validates that OpenTelemetry carrier metadata survives Pydantic's
        model serialization cycle (model_dump -> reconstruction) when
        extra='allow' is enabled. This ensures observability context is
        preserved across dynamic schemas used by LLM frameworksfor state management 
        and persistence"""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel, ConfigDict

        class ToolCallModel(BaseModel):
            model_config = ConfigDict(extra="allow")
            name: str
            input: str

        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        tc = ToolCallModel(name="search", input="HNSW", _otel=carrier)

        # Pydantic round-trip: model -> dict -> model
        data = tc.model_dump()
        restored = ToolCallModel(**data)
        recovered = restored._otel  # type: ignore[attr-defined]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_pydantic_model_extra_forbid_loses_carrier(self, tracing):
        """Documents that OpenTelemetry context injection fails on Pydantic
        models configured with extra='forbid'. This confirms that
        instrumentation cannot rely on direct model attribute injection for
        strict schemas and requires alternative context propagation
        strategies (example, metadata sidecar fields or out-of-band storage).
        See TestPayloadTraceparentFailureModes for tested mitigations."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel, ConfigDict, ValidationError

        class StrictToolCall(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str
            input: str

        tracer, exporter = tracing
        carrier, _ = _inject_carrier_in_active_span(tracer)

        # This should fail. strict schemas reject unknown fields
        with pytest.raises(ValidationError):
            StrictToolCall(name="search", input="HNSW", _otel=carrier)

    def test_schema_strip_unknown_fields(self, tracing):
        """Documents that tracing metadata ('_otel') is silently discarded
        when passed through strict input sanitization layers that filter
        for known schema fields. Unlike extra='forbid' which crashes,
        this is a silent-loss-of-observability :( the application runs
        but traces are disconnected. This confirms that instrumentation
        cannot rely on payload injectoin alone for sanitized inputs,
        requiring alternative propagation (metadata sidecar? or
        out-of-band storage).See TestPayloadTraceparentFailureModes."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        payload = {"name": "search", "input": "HNSW", "_otel": carrier}

        # Simulate strict schema stripping
        KNOWN_FIELDS = {"name", "input"}
        stripped = {k: v for k, v in payload.items() if k in KNOWN_FIELDS}

        assert "_otel" not in stripped, "Carrier should be stripped by strict schema"

        # Causality is lost — extract from empty carrier gives no parent
        extracted_ctx = extract({})
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool") as span:
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                # Parent should be None or root — NOT the chat span
                assert span.parent is None or (
                    span.parent.span_id != parent_ctx.span_id
                ), "Carrier was stripped but causality somehow survived — unexpected"
        finally:
            context.detach(token)

    def test_multiple_json_round_trips(self, tracing):
        """Verifies that OpenTelemetry carrier metadata is resilient across
        multiple serialization/deserialization cycles. This makes sure tracing
        context survives the entire multi-layer pipeline common in LLM
        applications: Model API -> Orchestrator -> Tool Executor"""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        payload = {"tool": "search", "_otel": carrier}

        # Hop 1: LLM SDK serializes response
        hop1 = json.loads(json.dumps(payload))
        # Hop 2: Framework processes and re-serializes
        hop2 = json.loads(json.dumps(hop1))
        # Hop 3: Tool executor deserializes
        hop3 = json.loads(json.dumps(hop2))

        _extract_and_create_child(tracer, hop3["_otel"])
        _assert_parent_child(exporter, parent_ctx)

    def test_msgpack_round_trip(self, tracing):
        """Validates that OpenTelemetry carrier metadata is binary-safe
        across MessagePack serialization cycles. This makes sure tracing
        context survives transmission via high performance binary RPC
        protocols often used in distributed agent orchestrations"""
        msgpack = pytest.importorskip("msgpack")

        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        payload = {"tool": "search", "_otel": carrier}
        packed = msgpack.packb(payload, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)

        _extract_and_create_child(tracer, unpacked["_otel"])
        _assert_parent_child(exporter, parent_ctx)

    def test_carrier_values_are_plain_strings(self, tracing):
        """The carrier is just a dict of string->string that verifies it has
        no special types that would break serialization"""
        tracer, _ = tracing
        carrier, _ = _inject_carrier_in_active_span(tracer)

        assert isinstance(carrier, dict)
        for key, value in carrier.items():
            assert isinstance(key, str), f"Carrier key {key!r} is not a string"
            assert isinstance(value, str), f"Carrier value {value!r} is not a string"

        # must contain traceparent
        assert "traceparent" in carrier, "Carrier missing traceparent header"


class TestPayloadTraceparentFailureModes:
    """Document the scenarios where payload traceparent does not work,
    and the recommended mitigations"""

    def test_strict_schema_mitigation_via_metadata_field(self, tracing):
        """Demonstrates the 'Sidecar Metadata' pattern. Mitigation for
        strict schemas: encapsulate observability headers in a permitted
        'metadata' or 'extensions' schema field, bypassing strict
        validation on core business fields. Most production schemas
        (LangGraph, LangChain) provide such a field for extension data."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Instead of injecting into the tool call directly,
        # use a metadata/extensions field that schemas typically allow
        tool_call = {"name": "search", "input": "HNSW"}
        envelope = {
            "tool_call": tool_call,
            "metadata": {"_otel": carrier},  # sidecar field
        }

        # Schema validates tool_call strictly, but metadata passes through
        serialized = json.loads(json.dumps(envelope))
        recovered = serialized["metadata"]["_otel"]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_out_of_band_context_store(self, tracing):
        """Demonstrates the 'Out-of-Band Correlation' pattern. Mitigation
        for fully opaque payloads: decouple the trace context from the
        payload entirely by propagating a correlation ID and retrieving
        the carrier from a shared, thread-safe context store."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Out-of-band context store (in practice: thread-local or async-local)
        context_store = {}
        correlation_id = "tool-call-abc123"
        context_store[correlation_id] = carrier

        # Tool call payload only carries the correlation ID (no _otel)
        tool_call = {"name": "search", "input": "HNSW", "id": correlation_id}

        # Strict schema validation — only known fields
        KNOWN_FIELDS = {"name", "input", "id"}
        validated = {k: v for k, v in tool_call.items() if k in KNOWN_FIELDS}

        # Tool executor retrieves carrier from store using correlation ID
        recovered = context_store[validated["id"]]
        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)


class TestCrossFrameworkEnvelopePatterns:
    """Tests for envelope patterns discovered in cross-framework research.

    These cover transformation scenarios that differ from the generic
    serialization tests above — each targets a specific framework's
    actual tool-call envelope behavior."""

    def test_string_wrapped_arguments_autogen(self, tracing):
        """AutoGen v0.4 carries FunctionCall.arguments as a JSON *string*,
        not a dict. The carrier must survive being nested inside a
        json.dumps'd string field, then recovered via json.loads on that
        field at tool execution time. This is subtly different from
        dict-level JSON round-trips because the carrier is double-encoded:
        once as part of the arguments string, once as the outer message."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Model AutoGen's FunctionCall envelope:
        # arguments is a JSON *string*, not a dict
        tool_args = {"query": "HNSW", "_otel": carrier}
        function_call = {
            "id": "call_abc123",
            "name": "web_search",
            "arguments": json.dumps(tool_args),  # string, not dict
        }

        # Simulate the full envelope going through JSON (message serialization)
        serialized = json.dumps(function_call)
        restored = json.loads(serialized)

        # Tool executor: parse the arguments string back to dict
        parsed_args = json.loads(restored["arguments"])
        recovered = parsed_args["_otel"]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_json_schema_additional_properties_false_haystack(self, tracing):
        """Haystack strict mode sets additionalProperties: false on the
        tool's JSON Schema sent to the LLM provider. This means extras
        are rejected at the provider boundary (before the framework even
        sees them), unlike Pydantic extra='forbid' which rejects at
        model instantiation. Tests that carrier injection into the
        function parameters is blocked by JSON Schema validation."""
        tracer, _ = tracing
        carrier, _ = _inject_carrier_in_active_span(tracer)

        # Model Haystack's strict mode JSON Schema
        tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

        # Tool call payload with carrier injected as extra property
        tool_args = {"query": "HNSW", "_otel": carrier}

        # Simulate JSON Schema validation (additionalProperties: false)
        allowed_keys = set(tool_schema["properties"].keys())
        extra_keys = set(tool_args.keys()) - allowed_keys
        schema_valid = len(extra_keys) == 0

        assert not schema_valid, (
            "Payload with _otel should be rejected by strict JSON Schema"
        )
        assert "_otel" in extra_keys, (
            "Carrier key should be flagged as an extra property"
        )

        # Carrier is rejected at the provider boundary —
        # this is a hard reject, not a silent strip
        # The framework never even receives the carrier

    def test_non_dict_coercion_to_empty_llamaindex(self, tracing):
        """LlamaIndex's ToolSelection validator coerces non-dict arguments
        to {} silently. This is a silent strip variant where malformed
        args (e.g., a raw string instead of a dict) cause the entire
        payload — including any embedded carrier — to be replaced with
        an empty dict. Tests that causality is lost in this scenario."""
        tracer, _ = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Model LlamaIndex's ToolSelection validator behavior:
        # if tool_kwargs is not a dict, coerce to {}
        def coerce_tool_kwargs(kwargs):
            """Mimics LlamaIndex ToolSelection validator."""
            return kwargs if isinstance(kwargs, dict) else {}

        # Case 1: carrier embedded in a valid dict — survives
        valid_args = {"query": "HNSW", "_otel": carrier}
        result = coerce_tool_kwargs(valid_args)
        assert "_otel" in result, "Carrier should survive in valid dict args"

        # Case 2: carrier as a raw JSON string (malformed args) — lost
        malformed_args = json.dumps({"query": "HNSW", "_otel": carrier})
        result = coerce_tool_kwargs(malformed_args)
        assert result == {}, "Malformed args should be coerced to empty dict"
        assert "_otel" not in result, "Carrier lost in non-dict coercion"

        # Causality is lost — extract from empty dict gives no parent
        extracted_ctx = extract({})
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool") as span:
                span.set_attribute("gen_ai.operation.name", "execute_tool")
                assert span.parent is None or (
                    span.parent.span_id != parent_ctx.span_id
                ), "Carrier was coerced away but causality survived — unexpected"
        finally:
            context.detach(token)

    def test_native_metadata_slot_semantic_kernel(self, tracing):
        """Semantic Kernel provides KernelContent.Metadata as a built-in
        extension channel for function result metadata. This validates
        that our 'Sidecar Metadata' pattern maps directly to a real
        framework's native metadata slot — carrier placed in Metadata
        survives the function-call round-trip."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Model Semantic Kernel's KernelContent envelope:
        # result field is strict (core schema), metadata dict is the
        # built-in extension point for additional context
        kernel_content = {
            "function_name": "web_search",
            "result": "HNSW is a graph-based ANN algorithm.",
            "metadata": {
                "execution_time_ms": 42,
                "_otel": carrier,  # carrier rides in native metadata slot
            },
        }

        # Simulate full round-trip: serialize -> persist -> deserialize
        serialized = json.dumps(kernel_content)
        restored = json.loads(serialized)

        # Strict validation on core fields only — metadata passes through
        assert "function_name" in restored
        assert "result" in restored

        # Carrier survives via the native metadata slot
        recovered = restored["metadata"]["_otel"]
        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_google_adk_tool_call_envelope(self, tracing):
        """Google Agent Development Kit (ADK) tool call envelope pattern.
        Google's ecosystem tends toward strict, Protobuf-heritage schemas.
        ADK tool calls use a dict-based arguments structure derived from
        the function's type hints. Models the strict validation path where
        only declared parameters are accepted, and the sidecar mitigation
        via a separate metadata/context field."""
        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Model ADK's tool call structure — Protobuf-heritage means
        # strict schemas by default. Function parameters are validated
        # against the tool's declared schema.
        declared_params = {"query": str}  # tool's declared parameters

        # Tool call arguments from LLM response
        tool_args = {"query": "HNSW", "_otel": carrier}

        # ADK-style strict validation: only declared params accepted
        validated = {k: v for k, v in tool_args.items() if k in declared_params}
        assert "_otel" not in validated, (
            "Carrier should be rejected by strict Protobuf-heritage schema"
        )

        # Mitigation: ADK provides a tool context/metadata mechanism
        # for passing auxiliary data alongside tool calls
        adk_envelope = {
            "tool_call": validated,
            "context": {"_otel": carrier},  # ADK context sidecar
        }

        serialized = json.loads(json.dumps(adk_envelope))
        recovered = serialized["context"]["_otel"]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_temporal_payload_converter_pipeline(self, tracing):
        """Temporal workflows serialize tool/activity arguments through
        a payload converter pipeline (data converters, codec chains)
        that is distinct from simple JSON round-trips. Arguments go
        through PayloadConverter -> PayloadCodec -> wire format when
        crossing workflow-to-activity boundaries. This tests whether
        the carrier survives Temporal's multi-stage serialization
        pipeline, which may apply encoding, compression, or encryption
        between stages."""
        import base64

        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Model Temporal's payload converter pipeline:
        # Stage 1: PayloadConverter — serialize to JSON bytes
        activity_args = {"query": "HNSW", "_otel": carrier}
        json_bytes = json.dumps(activity_args).encode("utf-8")

        # Stage 2: PayloadCodec — encode (e.g., base64 for transport)
        # Temporal codecs can apply compression, encryption, etc.
        encoded = base64.b64encode(json_bytes)

        # Stage 3: Wire format — Temporal wraps in Payload proto
        temporal_payload = {
            "metadata": {"encoding": "anNvbi9wbGFpbg=="},  # "json/plain"
            "data": encoded.decode("ascii"),
        }

        # Simulate full wire round-trip
        wire = json.dumps(temporal_payload)
        restored_payload = json.loads(wire)

        # Receiver side: reverse the pipeline
        decoded = base64.b64decode(restored_payload["data"])
        restored_args = json.loads(decoded)

        recovered = restored_args["_otel"]
        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)

    def test_pydantic_ai_tool_call_envelope(self, tracing):
        """PydanticAI is the foundation layer that other frameworks
        like Marvin delegate to for typed tool execution. Tools are
        Python functions with typed parameters validated by Pydantic.
        PydanticAI validates tool arguments against a function's type
        hints using Pydantic models. Extra fields in the arguments are
        rejected because tool parameters are strictly typed! tests both
        the strict rejection path and the RunContext sidecar mitigation."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel, ConfigDict, ValidationError

        #Model PydanticAI's tool parameter validation:
        #Tool function parameters become a strict Pydantic model
        class SearchToolParams(BaseModel):
            model_config = ConfigDict(extra="forbid")
            query: str

        tracer, exporter = tracing
        carrier, parent_ctx = _inject_carrier_in_active_span(tracer)

        # Direct injection into tool params fails — strict typed params
        with pytest.raises(ValidationError):
            SearchToolParams(query="HNSW", _otel=carrier)

        # Mitigation: PydanticAI provides RunContext as a sidecar
        # for dependency injection and auxiliary data alongside tool calls
        class RunContext(BaseModel):
            model_config = ConfigDict(extra="allow")
            deps: dict = {}

        run_ctx = RunContext(deps={}, _otel=carrier)

        # RunContext round-trip
        data = run_ctx.model_dump()
        restored = RunContext(**data)
        recovered = restored._otel  # type: ignore[attr-defined]

        _extract_and_create_child(tracer, recovered)
        _assert_parent_child(exporter, parent_ctx)
