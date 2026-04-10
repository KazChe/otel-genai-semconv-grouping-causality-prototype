"""CrewAI — verify actual tool call envelope shape.

Tests the real CrewAI BaseTool, tool invocation, and args_schema
to confirm:
1. Tool object type (dataclass vs Pydantic)
2. Whether args_schema allows or rejects extra fields
3. All entry points for tool execution
4. Generated MCP tool behavior (research says rejects extras)

Research classification: "Unknown overall; Hard reject for generated
MCP tools; some integrations show arg-loss bugs"

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

crewai = pytest.importorskip("crewai")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-crewai-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestCrewAIEnvelopeShape:
    """Verify CrewAI's actual tool call envelope structure."""

    def test_base_tool_object_type(self):
        """Step 1: verify whether BaseTool is a dataclass, Pydantic model,
        or something else."""
        from crewai.tools import BaseTool
        import dataclasses
        from pydantic import BaseModel

        print(f"\nBaseTool type: {type(BaseTool)}")
        print(f"Is dataclass: {dataclasses.is_dataclass(BaseTool)}")
        print(f"Is Pydantic subclass: {issubclass(BaseTool, BaseModel)}")
        print(f"Has model_dump: {hasattr(BaseTool, 'model_dump')}")
        print(f"Has model_config: {hasattr(BaseTool, 'model_config')}")

        if hasattr(BaseTool, 'model_config'):
            print(f"model_config: {BaseTool.model_config}")

        # Check MRO
        print(f"MRO: {[c.__name__ for c in BaseTool.__mro__]}")

    def test_tool_with_args_schema_extras(self):
        """Test a custom tool with an explicit args_schema.
        Check if extras are allowed or rejected."""
        from crewai.tools import BaseTool
        from pydantic import BaseModel

        class SearchArgs(BaseModel):
            query: str

        class SearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search the web"
            args_schema: type = SearchArgs

            def _run(self, query: str) -> str:
                return f"Results for {query}"

        tool = SearchTool()

        print(f"\nTool type: {type(tool)}")
        print(f"args_schema: {tool.args_schema}")
        print(f"args_schema model_config: {tool.args_schema.model_config}")

        # Check if args_schema rejects extras
        from pydantic import ValidationError
        try:
            validated = tool.args_schema(query="HNSW", _otel={"traceparent": "00-abc"})
            print(f"Extras ACCEPTED by args_schema: {validated}")
            if hasattr(validated, '_otel'):
                print("_otel accessible on validated model")
            else:
                print("_otel NOT accessible — silently ignored")
                dumped = validated.model_dump()
                print(f"model_dump: {dumped}")
        except ValidationError as e:
            print(f"Extras REJECTED by args_schema: {e}")

    def test_tool_run_with_extras(self):
        """Test what happens when tool._run() receives extra kwargs."""
        from crewai.tools import BaseTool

        class SearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search the web"

            def _run(self, query: str) -> str:
                return f"Results for {query}"

        tool = SearchTool()

        # Clean run
        try:
            result = tool.run(query="HNSW")
            print(f"\nClean run result: {result}")
        except Exception as e:
            print(f"\nClean run failed: {type(e).__name__}: {e}")

        # Run with extras
        try:
            result_extra = tool.run(query="HNSW", _otel={"traceparent": "00-abc"})
            print(f"Extras ACCEPTED by run(): {result_extra}")
        except Exception as e:
            print(f"Extras REJECTED by run(): {type(e).__name__}: {e}")

    def test_tool_run_with_kwargs_function(self):
        """Test if a tool with **kwargs in _run allows extras."""
        from crewai.tools import BaseTool

        class FlexTool(BaseTool):
            name: str = "flex_search"
            description: str = "Flexible search"

            def _run(self, query: str, **kwargs) -> str:
                return f"Results for {query}, extras: {list(kwargs.keys())}"

        tool = FlexTool()

        try:
            result = tool.run(query="HNSW", _otel={"traceparent": "00-abc"})
            print(f"\nFlexible _run result: {result}")
            if "_otel" in str(result):
                print("Carrier SURVIVES through **kwargs")
            else:
                print("Carrier NOT in result")
        except Exception as e:
            print(f"Flexible _run REJECTED: {type(e).__name__}: {e}")

    def test_tool_schema_inspection(self):
        """Inspect the schema that CrewAI generates for tools.
        Check additionalProperties setting."""
        from crewai.tools import BaseTool
        from pydantic import BaseModel

        class SearchArgs(BaseModel):
            query: str
            top_k: int = 10

        class SearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search the web"
            args_schema: type = SearchArgs

            def _run(self, query: str, top_k: int = 10) -> str:
                return f"Results for {query}"

        tool = SearchTool()

        # Get schema
        schema = tool.args_schema.model_json_schema()
        print(f"\nargs_schema JSON schema: {json.dumps(schema, indent=2)}")
        print(f"additionalProperties: {schema.get('additionalProperties')}")

    def test_tool_invoke_entry_points(self):
        """Discover all entry points for tool execution in CrewAI.
        Check run(), _run(), invoke(), execute(), etc."""
        from crewai.tools import BaseTool
        import inspect

        class SearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search"

            def _run(self, query: str) -> str:
                return f"Results for {query}"

        tool = SearchTool()

        # Find all callable methods that look like execution entry points
        entry_points = []
        for attr in dir(tool):
            if any(kw in attr.lower() for kw in ['run', 'invoke', 'execute', 'call']):
                if not attr.startswith('__') and callable(getattr(tool, attr, None)):
                    entry_points.append(attr)

        print(f"\nPotential entry points: {entry_points}")

        # Check run() signature
        if hasattr(tool, 'run'):
            sig = inspect.signature(tool.run)
            print(f"run() signature: {sig}")

        # Check _run() signature
        sig_run = inspect.signature(tool._run)
        print(f"_run() signature: {sig_run}")
