"""LlamaIndex — verify actual tool call envelope shape.

Tests the real LlamaIndex ToolSelection, FunctionTool, and tool
invocation paths to confirm:
1. ToolSelection object type (dataclass vs Pydantic) and field structure
2. tool_kwargs coercion behavior for non-dict arguments
3. Whether extras survive in tool_kwargs dict
4. FunctionTool schema — does it allow or reject extra fields?
5. All entry points for tool execution

Research classification: "Compatible for extra dict keys in normal path,
but malformed/non-dict args can degrade to {}"

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

llama_index_core = pytest.importorskip("llama_index.core")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-llamaindex-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestLlamaIndexEnvelopeShape:
    """Verify LlamaIndex's actual tool call envelope structure."""

    def test_tool_selection_object_type(self):
        """Step 1 of per-framework workflow: verify whether ToolSelection
        is a dataclass, Pydantic model, or something else."""
        from llama_index.core.llms.llm import ToolSelection
        import dataclasses

        ts = ToolSelection(
            tool_id="call_123",
            tool_name="web_search",
            tool_kwargs={"query": "HNSW"},
        )

        is_dataclass = dataclasses.is_dataclass(ts)
        has_model_dump = hasattr(ts, "model_dump")
        has_model_config = hasattr(ts, "model_config")

        print(f"\nToolSelection type: {type(ts)}")
        print(f"Is dataclass: {is_dataclass}")
        print(f"Has model_dump: {has_model_dump}")
        print(f"Has model_config: {has_model_config}")

        if is_dataclass:
            fields = {f.name: f.type for f in dataclasses.fields(ts)}
            print(f"Dataclass fields: {fields}")
        if has_model_dump:
            print(f"Pydantic model_config: {getattr(ts, 'model_config', 'N/A')}")
            print(f"model_dump: {ts.model_dump()}")

    def test_tool_kwargs_extras_survive(self, tracing):
        """Test whether extra fields in tool_kwargs dict survive.
        Research says extras inside a valid dict are not stripped
        at the envelope level."""
        from llama_index.core.llms.llm import ToolSelection

        tracer, exporter = tracing
        carrier = {}

        with tracer.start_as_current_span("chat") as span:
            inject(carrier)
            parent_ctx = span.get_span_context()

        ts = ToolSelection(
            tool_id="call_456",
            tool_name="web_search",
            tool_kwargs={"query": "HNSW", "_otel": carrier},
        )

        # Check if extras survive in tool_kwargs
        assert isinstance(ts.tool_kwargs, dict)
        if "_otel" in ts.tool_kwargs:
            print(f"\nExtras SURVIVE in tool_kwargs: {list(ts.tool_kwargs.keys())}")
            recovered = ts.tool_kwargs["_otel"]

            # Verify causality can be established
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
        else:
            print(f"\nExtras STRIPPED from tool_kwargs: {list(ts.tool_kwargs.keys())}")
            pytest.fail("Extras were stripped — unexpected based on research")

    def test_tool_kwargs_non_dict_coercion(self):
        """Research says ToolSelection coerces non-dict tool_kwargs to {}.
        This is the silent strip variant we modeled in our simulated test.
        Verify this is actually the case."""
        from llama_index.core.llms.llm import ToolSelection

        # Pass a string instead of a dict
        try:
            ts = ToolSelection(
                tool_id="call_789",
                tool_name="search",
                tool_kwargs='{"query": "HNSW", "_otel": {"traceparent": "00-abc"}}',
            )
            print(f"\nNon-dict tool_kwargs result: {ts.tool_kwargs}")
            print(f"Type: {type(ts.tool_kwargs)}")

            if ts.tool_kwargs == {}:
                print("CONFIRMED: non-dict coerced to {} — silent strip")
            elif isinstance(ts.tool_kwargs, str):
                print("String preserved as-is — no coercion")
            else:
                print(f"Unexpected result: {ts.tool_kwargs}")
        except Exception as e:
            print(f"\nNon-dict tool_kwargs REJECTED: {type(e).__name__}: {e}")

    def test_function_tool_schema_and_type(self):
        """Inspect FunctionTool's structure — is it Pydantic?
        What does the generated schema look like?
        Does it set additionalProperties?"""
        from llama_index.core.tools import FunctionTool

        def web_search(query: str, top_k: int = 10) -> str:
            """Search the web for information."""
            return f"Results for {query}"

        tool = FunctionTool.from_defaults(fn=web_search)

        print(f"\nFunctionTool type: {type(tool)}")
        print(f"Has model_dump: {hasattr(tool, 'model_dump')}")

        # Get the tool's metadata/schema
        metadata = tool.metadata
        print(f"Metadata type: {type(metadata)}")
        print(f"Tool name: {metadata.name}")
        print(f"Tool description: {metadata.description}")

        if hasattr(metadata, 'fn_schema'):
            print(f"fn_schema: {metadata.fn_schema}")
            if metadata.fn_schema:
                schema = metadata.fn_schema.model_json_schema()
                print(f"JSON schema: {json.dumps(schema, indent=2)}")
                additional = schema.get("additionalProperties")
                print(f"additionalProperties: {additional}")

        # Check get_parameters_dict
        if hasattr(metadata, 'get_parameters_dict'):
            params = metadata.get_parameters_dict()
            print(f"Parameters dict: {json.dumps(params, indent=2)}")

    def test_function_tool_call_with_extras(self):
        """Test what happens when FunctionTool.call() receives extra
        fields in the kwargs. Does it pass them through to the function
        or strip/reject them?"""
        from llama_index.core.tools import FunctionTool, ToolOutput

        def web_search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        tool = FunctionTool.from_defaults(fn=web_search)

        # Clean call
        result = tool.call(query="HNSW")
        print(f"\nClean call result type: {type(result)}")
        print(f"Clean call result: {result}")

        # Call with extras
        try:
            result_extra = tool.call(
                query="HNSW",
                _otel={"traceparent": "00-abc"},
            )
            print(f"Extras ACCEPTED by call(): {result_extra}")
        except Exception as e:
            print(f"Extras REJECTED by call(): {type(e).__name__}: {e}")

    def test_function_tool_call_with_kwargs_function(self):
        """Test if a function with **kwargs allows extras to pass through."""
        from llama_index.core.tools import FunctionTool

        def flexible_search(query: str, **kwargs) -> str:
            """Search with flexible params."""
            return f"Results for {query}, extras: {list(kwargs.keys())}"

        tool = FunctionTool.from_defaults(fn=flexible_search)

        try:
            result = tool.call(
                query="HNSW",
                _otel={"traceparent": "00-abc"},
            )
            print(f"\nFlexible function result: {result}")
            if "_otel" in str(result):
                print("Carrier SURVIVES through **kwargs function")
            else:
                print("Carrier NOT in result — stripped before function")
        except Exception as e:
            print(f"Flexible function REJECTED extras: {type(e).__name__}: {e}")
