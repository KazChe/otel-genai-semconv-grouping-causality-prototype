"""AutoGen v0.4 — verify actual tool call envelope shape.

Tests the real AutoGen FunctionCall and FunctionTool objects to confirm:
1. FunctionCall.arguments is a JSON string (not a dict)
2. Carrier injected into the arguments string survives double-encoding
3. FunctionTool schema generation — does it allow or reject extra fields?

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

autogen_core = pytest.importorskip("autogen_core")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-autogen-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestAutoGenEnvelopeShape:
    """Verify AutoGen's actual tool call envelope structure."""

    def test_function_call_is_dataclass_not_pydantic(self):
        """FunctionCall is a plain Python dataclass, NOT a Pydantic model.
        This is a key discovery — our simulated tests assumed Pydantic
        semantics (model_dump, extra='allow'/'forbid'), but the real
        envelope uses dataclasses.asdict() for serialization and has no
        Pydantic-level field validation or extra-field handling."""
        from autogen_core import FunctionCall
        import dataclasses

        fc = FunctionCall(id="call_test", name="search", arguments='{"q": "test"}')

        # It IS a dataclass
        assert dataclasses.is_dataclass(fc), "FunctionCall should be a dataclass"

        # It is NOT a Pydantic model
        assert not hasattr(fc, 'model_dump'), (
            "FunctionCall should not have model_dump (not a Pydantic model)"
        )
        assert not hasattr(fc, 'model_config'), (
            "FunctionCall should not have model_config (not a Pydantic model)"
        )
        assert not hasattr(fc, 'model_fields'), (
            "FunctionCall should not have model_fields (not a Pydantic model)"
        )

        # Serialization uses dataclasses.asdict, not model_dump
        dumped = dataclasses.asdict(fc)
        assert isinstance(dumped, dict)
        assert dumped == {"id": "call_test", "name": "search", "arguments": '{"q": "test"}'}

        # No extra-field validation — dataclass accepts any fields at construction
        # (unlike Pydantic extra="forbid" which would raise ValidationError)
        print(f"\nFunctionCall type: {type(fc)}")
        print(f"Is dataclass: True")
        print(f"Has model_dump: False")
        print(f"Implication: no Pydantic extra-field validation on the envelope itself")

    def test_function_call_arguments_is_string(self):
        """FunctionCall.arguments should be a JSON string, not a dict.
        This confirms our simulated test assumption."""
        from autogen_core import FunctionCall

        args_dict = {"query": "HNSW", "top_k": 10}
        fc = FunctionCall(
            id="call_123",
            name="web_search",
            arguments=json.dumps(args_dict),
        )

        # Verify arguments is stored as a string
        assert isinstance(fc.arguments, str), (
            f"Expected arguments to be str, got {type(fc.arguments)}"
        )

        # Verify it round-trips correctly
        parsed = json.loads(fc.arguments)
        assert parsed == args_dict

    def test_carrier_survives_in_arguments_string(self, tracing):
        """Inject _otel carrier into FunctionCall.arguments JSON string
        and verify it survives the double-encoding."""
        from autogen_core import FunctionCall

        tracer, exporter = tracing
        carrier = {}

        with tracer.start_as_current_span("chat") as span:
            inject(carrier)
            parent_ctx = span.get_span_context()

        # Embed carrier in arguments JSON string
        args_with_carrier = {"query": "HNSW", "_otel": carrier}
        fc = FunctionCall(
            id="call_456",
            name="web_search",
            arguments=json.dumps(args_with_carrier),
        )

        # Simulate wire round-trip of the entire FunctionCall
        # FunctionCall is a dataclass, not a Pydantic model
        import dataclasses
        fc_dict = dataclasses.asdict(fc)
        fc_json = json.dumps(fc_dict)
        restored_dict = json.loads(fc_json)

        # Extract carrier from the restored arguments string
        restored_args = json.loads(restored_dict["arguments"])
        recovered_carrier = restored_args["_otel"]

        # Verify carrier is intact
        assert "traceparent" in recovered_carrier
        assert recovered_carrier == carrier

        # Verify causality can be established
        extracted_ctx = extract(recovered_carrier)
        token = context.attach(extracted_ctx)
        try:
            with tracer.start_as_current_span("execute_tool") as tool_span:
                pass
        finally:
            context.detach(token)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "execute_tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == parent_ctx.span_id

    def test_function_call_asdict_preserves_arguments(self):
        """dataclasses.asdict() should preserve the arguments string
        without parsing or modifying it. FunctionCall is a dataclass,
        not a Pydantic model."""
        from autogen_core import FunctionCall
        import dataclasses

        original_args = '{"query": "test", "_otel": {"traceparent": "00-abc-def-01"}}'
        fc = FunctionCall(
            id="call_789",
            name="search",
            arguments=original_args,
        )

        dumped = dataclasses.asdict(fc)
        assert dumped["arguments"] == original_args, (
            "asdict() modified the arguments string"
        )

    def test_function_tool_schema_extra_fields(self):
        """Inspect FunctionTool's generated schema to see if extra fields
        are allowed or rejected in the tool's input model."""
        from autogen_core.tools import FunctionTool

        async def web_search(query: str, top_k: int = 10) -> str:
            return f"Results for {query}"

        tool = FunctionTool(web_search, description="Search the web")
        schema = tool.schema

        # Inspect the schema structure
        assert "parameters" in schema or "properties" in schema, (
            f"Unexpected schema structure: {schema}"
        )

        # Print schema for manual inspection
        print(f"\nFunctionTool schema: {json.dumps(schema, indent=2)}")

        # Check if additionalProperties is set
        params = schema.get("parameters", schema)
        additional_props = params.get("additionalProperties")
        print(f"additionalProperties: {additional_props}")

    def test_function_call_arguments_dict_coercion(self):
        """Test what happens when arguments is passed as a dict instead
        of a string — does AutoGen stringify it or reject it?"""
        from autogen_core import FunctionCall

        args_dict = {"query": "HNSW"}

        try:
            fc = FunctionCall(
                id="call_coerce",
                name="search",
                arguments=args_dict,
            )
            # If it accepts a dict, check what it stored
            if isinstance(fc.arguments, str):
                print(f"\nAutoGen stringified dict args: {fc.arguments}")
            else:
                print(f"\nAutoGen stored dict args as: {type(fc.arguments)}")
        except Exception as e:
            # If it rejects a dict, document the error
            print(f"\nAutoGen rejects dict arguments: {type(e).__name__}: {e}")
            pytest.skip(f"AutoGen rejects dict arguments: {e}")

    @pytest.mark.asyncio
    async def test_run_json_silently_strips_extra_fields(self):
        """DISCOVERY: tool.run_json(args) SILENTLY STRIPS extra fields.

        Despite the schema having additionalProperties: false, the
        generated Pydantic args model has empty model_config (defaults
        to extra='ignore'). model_validate() silently discards unknown
        fields rather than rejecting them.

        The additionalProperties: false in the schema is only for the
        LLM provider boundary — it tells the LLM not to generate extras.
        It is NOT enforced at runtime validation.

        This is a SILENT STRIP, not a hard reject — the most dangerous
        failure mode for carrier injection."""
        from autogen_core.tools import FunctionTool
        from autogen_core import CancellationToken

        async def search(query: str) -> str:
            return f"Results for {query}"

        tool = FunctionTool(search, description="Search")
        ct = CancellationToken()

        # Valid args
        result = await tool.run_json({"query": "HNSW"}, ct)
        assert "HNSW" in result

        # Args with _otel carrier — silently stripped, no error
        result_with_extra = await tool.run_json(
            {"query": "HNSW", "_otel": {"traceparent": "00-abc-def-01"}},
            ct,
        )
        assert "HNSW" in result_with_extra, "Tool should still execute"
        print("\nrun_json SILENTLY STRIPS extra fields (no error)")
        print("Classification: Silent strip, NOT hard reject")

    @pytest.mark.asyncio
    async def test_run_typed_silently_ignores_extra_fields(self):
        """DISCOVERY: The generated Pydantic args model has empty
        model_config — Pydantic defaults to extra='ignore', so extras
        are silently discarded at model construction, not rejected.

        This differs from our simulated test which assumed extra='forbid'."""
        from autogen_core.tools import FunctionTool

        async def search(query: str) -> str:
            return f"Results for {query}"

        tool = FunctionTool(search, description="Search")
        args_type = tool.args_type()

        # model_config is empty — no extra='forbid'
        print(f"\nargs_type model_config: {args_type.model_config}")
        assert args_type.model_config.get("extra") != "forbid", (
            "Expected model_config to NOT have extra='forbid'"
        )

        # Extra fields are silently ignored, not rejected
        m = args_type(query="HNSW", _otel={"traceparent": "00-abc"})
        assert m.query == "HNSW"
        assert not hasattr(m, "_otel"), "Extra field silently dropped"
        assert "_otel" not in m.model_dump(), "Extra field not in model_dump"
        print("Extras silently ignored at model construction")
        print("Carrier is LOST — silent strip, not hard reject")

    @pytest.mark.asyncio
    async def test_workbench_call_tool_silently_strips(self):
        """DISCOVERY: Workbench.call_tool() delegates to run_json(),
        which delegates to model_validate(). All three entry points
        silently strip extra fields. The tool executes successfully
        but the carrier is gone.

        All three entry points (run, run_json, call_tool) converge
        on the same silent strip behavior."""
        from autogen_core.tools import FunctionTool, StaticWorkbench
        from autogen_core import CancellationToken

        async def search(query: str) -> str:
            return f"Results for {query}"

        tool = FunctionTool(search, description="Search")
        workbench = StaticWorkbench(tools=[tool])
        await workbench.start()
        ct = CancellationToken()

        # Valid call
        result = await workbench.call_tool("search", {"query": "HNSW"}, ct)
        assert not result.is_error

        # Call with extra fields — silently stripped, no error
        result_with_extra = await workbench.call_tool(
            "search",
            {"query": "HNSW", "_otel": {"traceparent": "00-abc"}},
            ct,
        )
        assert not result_with_extra.is_error, (
            "Extras are silently stripped — tool should succeed"
        )
        print(f"\nWorkbench.call_tool with extras: is_error={result_with_extra.is_error}")
        print("All 3 entry points silently strip extras")
        print("Classification: SILENT STRIP across all entry points")

        await workbench.stop()
