"""Google ADK — verify actual tool call envelope shape.

Tests the real Google Agent Development Kit tool objects to confirm:
1. Tool/FunctionTool object type (dataclass vs Pydantic vs other)
2. How tool parameters are validated — strict or permissive?
3. All entry points for tool execution
4. Whether a sidecar/metadata field exists for carrier injection

Research classification: "Unknown — assumed strict based on Protobuf heritage"

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

google_adk = pytest.importorskip("google.adk")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-google-adk-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestGoogleADKEnvelopeShape:
    """Verify Google ADK's actual tool call envelope structure."""

    def test_tool_object_type(self):
        """Step 1: verify whether ADK's tool objects are dataclass,
        Pydantic, Protobuf, or something else."""
        from google.adk.tools import FunctionTool
        import dataclasses

        def web_search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        tool = FunctionTool(web_search)

        is_dataclass = dataclasses.is_dataclass(tool)
        has_model_dump = hasattr(tool, "model_dump")
        has_model_config = hasattr(tool, "model_config")

        print(f"\nFunctionTool type: {type(tool)}")
        print(f"Is dataclass: {is_dataclass}")
        print(f"Has model_dump: {has_model_dump}")
        print(f"Has model_config: {has_model_config}")
        print(f"MRO: {[c.__name__ for c in type(tool).__mro__]}")

        # Inspect attributes
        tool_attrs = [a for a in dir(tool) if not a.startswith('_')]
        print(f"Public attributes: {tool_attrs}")

    def test_tool_declaration_schema(self):
        """Inspect the schema/declaration that ADK generates for a tool.
        Check if additionalProperties is set."""
        from google.adk.tools import FunctionTool

        def web_search(query: str, top_k: int = 10) -> str:
            """Search the web for information."""
            return f"Results for {query}"

        tool = FunctionTool(web_search)

        # Check for schema/declaration methods
        if hasattr(tool, 'declaration'):
            decl = tool.declaration
            print(f"\nDeclaration type: {type(decl)}")
            print(f"Declaration: {decl}")
        if hasattr(tool, 'get_declaration'):
            decl = tool.get_declaration()
            print(f"\nget_declaration type: {type(decl)}")
            print(f"get_declaration: {decl}")
        if hasattr(tool, 'schema'):
            print(f"\nSchema: {tool.schema}")

        # Look for parameter-related attributes
        for attr in dir(tool):
            if any(kw in attr.lower() for kw in ['param', 'schema', 'decl', 'spec']):
                if not attr.startswith('_'):
                    val = getattr(tool, attr, None)
                    print(f"  {attr}: {type(val)}")

    def test_tool_run_with_extras(self):
        """Test what happens when a tool receives extra arguments
        beyond its declared parameters."""
        from google.adk.tools import FunctionTool

        def web_search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        tool = FunctionTool(web_search)

        # Find the execution method
        entry_points = []
        for attr in dir(tool):
            if any(kw in attr.lower() for kw in ['run', 'invoke', 'execute', 'call']):
                if not attr.startswith('__') and callable(getattr(tool, attr, None)):
                    entry_points.append(attr)
        print(f"\nEntry points found: {entry_points}")

        # Try running with extras through each entry point
        for ep_name in entry_points:
            ep = getattr(tool, ep_name)
            try:
                # Try with keyword args
                result = ep(query="HNSW", _otel={"traceparent": "00-abc"})
                print(f"{ep_name}(query, _otel): ACCEPTED — {result}")
            except TypeError as e:
                print(f"{ep_name}(query, _otel): TypeError — {e}")
            except Exception as e:
                print(f"{ep_name}(query, _otel): {type(e).__name__} — {e}")

    def test_tool_run_with_kwargs_function(self):
        """Test if a function with **kwargs allows extras to pass through."""
        from google.adk.tools import FunctionTool

        def flexible_search(query: str, **kwargs) -> str:
            """Search with flexible params."""
            return f"Results for {query}, extras: {list(kwargs.keys())}"

        tool = FunctionTool(flexible_search)

        entry_points = [a for a in dir(tool)
                        if any(kw in a.lower() for kw in ['run', 'call'])
                        and not a.startswith('__')
                        and callable(getattr(tool, a, None))]

        for ep_name in entry_points:
            ep = getattr(tool, ep_name)
            try:
                result = ep(query="HNSW", _otel={"traceparent": "00-abc"})
                print(f"\n{ep_name} with **kwargs fn: {result}")
                if "_otel" in str(result):
                    print(f"Carrier SURVIVES through {ep_name}")
            except Exception as e:
                print(f"\n{ep_name} with **kwargs fn: {type(e).__name__}: {e}")

    def test_tool_context_or_metadata_field(self):
        """Check if ADK tools have a context, metadata, or extensions
        field that could serve as a sidecar for carrier injection."""
        from google.adk.tools import FunctionTool

        def search(query: str) -> str:
            """Search."""
            return f"Results for {query}"

        tool = FunctionTool(search)

        # Look for sidecar-like attributes
        sidecar_keywords = ['context', 'metadata', 'extra', 'extensions',
                           'attrs', 'properties', 'config', 'state']
        for attr in dir(tool):
            if any(kw in attr.lower() for kw in sidecar_keywords):
                if not attr.startswith('_'):
                    val = getattr(tool, attr, None)
                    print(f"  {attr}: {type(val)} = {repr(val)[:100]}")

        # Check if ToolContext exists
        try:
            from google.adk.tools import ToolContext
            print(f"\nToolContext type: {type(ToolContext)}")
            print(f"ToolContext MRO: {[c.__name__ for c in ToolContext.__mro__]}")
            tc_attrs = [a for a in dir(ToolContext) if not a.startswith('_')]
            print(f"ToolContext attrs: {tc_attrs}")
        except ImportError:
            print("\nNo ToolContext class found")
