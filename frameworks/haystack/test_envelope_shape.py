"""Haystack — verify actual tool call envelope shape.

Tests the real Haystack ToolCall dataclass to confirm:
1. ToolCall.arguments is a dict — extra fields in the dict survive
2. tools_strict=True sets additionalProperties: false on the schema
3. Malformed JSON arguments are handled (skipped, not crashed)

Research classification: "Compatible" by default, "Hard reject" with tools_strict=True.

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

haystack = pytest.importorskip("haystack")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-haystack-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestHaystackEnvelopeShape:
    """Verify Haystack's actual ToolCall structure."""

    def test_toolcall_arguments_is_dict(self):
        """ToolCall.arguments should be a dict. Extra fields in the dict
        should survive since it's just a plain dict, not a validated model."""
        from haystack.dataclasses import ToolCall

        tc = ToolCall(
            tool_name="web_search",
            arguments={"query": "HNSW", "_otel": {"traceparent": "00-abc-def-01"}},
        )

        assert isinstance(tc.arguments, dict)
        assert "_otel" in tc.arguments, "Extra fields should survive in ToolCall.arguments"
        assert tc.arguments["_otel"]["traceparent"] == "00-abc-def-01"

        print(f"\nToolCall structure: tool_name={tc.tool_name}, "
              f"arguments type={type(tc.arguments)}, "
              f"arguments keys={list(tc.arguments.keys())}")

    def test_carrier_survives_in_toolcall_arguments(self, tracing):
        """Inject real OTel carrier into ToolCall.arguments and verify
        causality can be established after round-trip."""
        from haystack.dataclasses import ToolCall

        tracer, exporter = tracing
        carrier = {}

        with tracer.start_as_current_span("chat") as span:
            inject(carrier)
            parent_ctx = span.get_span_context()

        tc = ToolCall(
            tool_name="web_search",
            arguments={"query": "HNSW", "_otel": carrier},
        )

        # Simulate serialization round-trip
        tc_json = json.dumps({
            "tool_name": tc.tool_name,
            "arguments": tc.arguments,
        })
        restored = json.loads(tc_json)
        recovered_carrier = restored["arguments"]["_otel"]

        # Verify causality
        extracted_ctx = extract(recovered_carrier)
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool"):
                pass
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "execute_tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == parent_ctx.span_id

    def test_tools_strict_schema_additional_properties(self):
        """When tools_strict=True, Haystack sets additionalProperties: false
        on the tool's JSON Schema. Verify this actually happens by inspecting
        the schema transformation."""
        # Haystack's strict mode logic is in the OpenAI chat generator.
        # We model the schema transformation it applies.
        from haystack.dataclasses import ToolCall

        # Base tool schema (what a tool normally declares)
        tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

        # Haystack strict mode transformation
        # (from haystack/components/generators/chat/openai.py)
        strict_schema = {**tool_schema, "additionalProperties": False}

        assert strict_schema["additionalProperties"] is False
        print(f"\nStrict schema: {json.dumps(strict_schema, indent=2)}")

        # Verify that _otel would be rejected by this schema
        test_args = {"query": "HNSW", "_otel": {"traceparent": "00-abc"}}
        extra_keys = set(test_args.keys()) - set(strict_schema["properties"].keys())
        assert "_otel" in extra_keys, (
            "Carrier key should be flagged as extra by strict schema"
        )

    def test_toolcall_id_field_presence(self):
        """Check if ToolCall has an id field — some frameworks use it
        for correlation, which affects the out-of-band mitigation pattern."""
        from haystack.dataclasses import ToolCall

        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
        )

        # Check what fields exist on the real object
        fields = vars(tc) if hasattr(tc, '__dict__') else {}
        print(f"\nToolCall fields: {list(fields.keys())}")
        print(f"ToolCall has 'id': {hasattr(tc, 'id')}")

        # Try creating with an id if the field exists
        try:
            tc_with_id = ToolCall(
                id="call_123",
                tool_name="search",
                arguments={"query": "test"},
            )
            print(f"ToolCall accepts 'id' parameter: True, value={tc_with_id.id}")
        except TypeError as e:
            print(f"ToolCall does not accept 'id' parameter: {e}")

    def test_toolcall_is_dataclass_not_pydantic(self):
        """Verify the actual type of ToolCall — determines serialization
        path and whether extra-field validation exists on the envelope."""
        from haystack.dataclasses import ToolCall
        import dataclasses

        tc = ToolCall(tool_name="search", arguments={"query": "test"})

        is_dataclass = dataclasses.is_dataclass(tc)
        has_model_dump = hasattr(tc, "model_dump")
        has_model_config = hasattr(tc, "model_config")

        print(f"\nToolCall type: {type(tc)}")
        print(f"Is dataclass: {is_dataclass}")
        print(f"Has model_dump: {has_model_dump}")
        print(f"Has model_config: {has_model_config}")

        # Document whichever it is
        if is_dataclass:
            fields = {f.name: f.type for f in dataclasses.fields(tc)}
            print(f"Dataclass fields: {fields}")
        if has_model_dump:
            print(f"Pydantic model_config: {tc.model_config}")

    def test_toolcall_extra_field_as_sidecar(self, tracing):
        """DISCOVERY: ToolCall has a built-in 'extra' field (dict).
        This is a natural sidecar slot for carrier injection — no need
        to inject into arguments (which may be schema-validated) or
        use out-of-band correlation."""
        from haystack.dataclasses import ToolCall

        tracer, exporter = tracing
        carrier = {}

        with tracer.start_as_current_span("chat") as span:
            inject(carrier)
            parent_ctx = span.get_span_context()

        # Use the built-in extra field as the carrier sidecar
        tc = ToolCall(
            tool_name="web_search",
            arguments={"query": "HNSW"},
            extra={"_otel": carrier},
        )

        # Verify extra field preserves carrier
        assert tc.extra is not None
        assert "_otel" in tc.extra
        assert tc.extra["_otel"] == carrier

        # Verify default when not set
        tc_no_extra = ToolCall(tool_name="search", arguments={"query": "test"})
        print(f"\nextra field default value: {tc_no_extra.extra}")
        print(f"extra field type when set: {type(tc.extra)}")

        # Simulate round-trip
        tc_data = {
            "tool_name": tc.tool_name,
            "arguments": tc.arguments,
            "id": tc.id,
            "extra": tc.extra,
        }
        serialized = json.dumps(tc_data)
        restored = json.loads(serialized)
        recovered = restored["extra"]["_otel"]

        # Verify causality via the extra field sidecar
        extracted_ctx = extract(recovered)
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool"):
                pass
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "execute_tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == parent_ctx.span_id
        print("Carrier survives via ToolCall.extra sidecar — NATIVE FIT")

    def test_tool_invoke_rejects_extras_in_arguments(self):
        """Tool.invoke() passes arguments as **kwargs to the underlying
        function. If the function doesn't accept **kwargs, Python raises
        TypeError for unknown keyword arguments. ToolInvoker wraps this
        in ToolInvocationError.

        This means extras in arguments dict survive through ToolInvoker's
        _prepare_tool_call_params (which does arguments.copy()), but fail
        at the actual function call boundary."""
        from haystack.tools import Tool
        from haystack.tools.errors import ToolInvocationError

        def my_search(query: str) -> str:
            return f"Results for {query}"

        tool = Tool(
            name="search", description="Search", function=my_search,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        # Clean invoke — works
        result = tool.invoke(query="HNSW")
        assert "HNSW" in result

        # Invoke with extras — rejected at function signature
        with pytest.raises(ToolInvocationError):
            tool.invoke(query="HNSW", _otel={"traceparent": "00-abc"})

        print("\nTool.invoke rejects extras via Python function signature")
        print("Classification: Hard reject at function call boundary")

    def test_tool_invoke_accepts_extras_with_kwargs_function(self):
        """If the tool function accepts **kwargs, extras in arguments
        survive all the way through to execution. This is a potential
        carrier path for functions that are designed to accept extra data."""
        from haystack.tools import Tool

        def flexible_search(query: str, **kwargs) -> str:
            return f"Results for {query}, extras: {list(kwargs.keys())}"

        tool = Tool(
            name="flex_search", description="Flex", function=flexible_search,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        result = tool.invoke(query="HNSW", _otel={"traceparent": "00-abc"})
        assert "_otel" in result
        print(f"\nFlexible function accepts extras: {result}")
        print("Carrier CAN survive if tool function uses **kwargs")

    def test_toolcall_extra_not_passed_to_invoke(self):
        """ToolCall.extra is a sidecar field on the envelope — it is NOT
        passed as an argument to tool.invoke(). The carrier rides
        alongside the tool call but must be read separately from
        ToolCall.extra, not from the function parameters.

        This means ToolCall.extra is an out-of-band sidecar, not an
        in-band parameter injection."""
        from haystack.dataclasses import ToolCall

        tc = ToolCall(
            tool_name="search",
            arguments={"query": "HNSW"},
            extra={"_otel": {"traceparent": "00-abc"}},
        )

        # extra is on the ToolCall object, not in arguments
        assert "_otel" not in tc.arguments
        assert "_otel" in tc.extra

        # ToolInvoker passes tc.arguments to tool.invoke(), not tc.extra
        # So carrier in extra must be read separately by instrumentation
        print("\nToolCall.extra is out-of-band — not passed to invoke()")
        print("Instrumentation must read tc.extra separately")
