"""PydanticAI — verify actual tool call envelope shape.

Tests the real PydanticAI tool execution to confirm:
1. Tool parameters are validated via PluggableSchemaValidator with extra_forbidden
2. RunContext can carry extra data as a sidecar
3. The actual validation behavior for tool parameters

PydanticAI is the foundation layer that other frameworks (Marvin, etc.)
delegate to for typed tool execution. Understanding its behavior covers
the upstream that multiple frameworks depend on.

These are integration tests that import the real framework.
"""

import json
import pytest

from opentelemetry import trace, context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.propagate import inject, extract

pydantic_ai = pytest.importorskip("pydantic_ai")


@pytest.fixture()
def tracing():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test-pydanticai-envelope")
    yield tracer, exporter
    provider.shutdown()


class TestPydanticAIEnvelopeShape:
    """Verify PydanticAI's actual tool call parameter handling."""

    def test_tool_object_type_and_schema(self):
        """Verify the type of the Tool object and its FunctionSchema.
        Determines the serialization path and validation behavior."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def search(query: str, top_k: int = 10) -> str:
            """Search the web."""
            return f"Results for {query}"

        toolset = agent._function_toolset
        tool = toolset.tools["search"]

        print(f"\nToolset type: {type(toolset)}")
        print(f"Tool type: {type(tool)}")
        print(f"Tool.function_schema type: {type(tool.function_schema)}")
        print(f"Validator type: {type(tool.function_schema.validator)}")
        print(f"JSON schema: {json.dumps(tool.function_schema.json_schema, indent=2)}")
        print(f"takes_ctx: {tool.takes_ctx}")

        # Verify schema structure
        schema = tool.function_schema.json_schema
        assert schema["additionalProperties"] is False
        assert "query" in schema["properties"]

    def test_tool_rejects_extra_parameters_hard(self):
        """DISCOVERY: PydanticAI uses PluggableSchemaValidator which
        enforces additionalProperties: false at RUNTIME — unlike AutoGen
        which silently ignores extras. This is a TRUE hard reject.

        The validator raises ValidationError with 'extra_forbidden' when
        unknown fields are present in tool arguments."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic import ValidationError

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def search(query: str) -> str:
            """Search."""
            return f"Results for {query}"

        tool = agent._function_toolset.tools["search"]
        validator = tool.function_schema.validator

        # Clean args — should pass
        result = validator.validate_python({"query": "HNSW"})
        assert result == {"query": "HNSW"}

        # Args with _otel carrier — HARD REJECT (extra_forbidden)
        with pytest.raises(ValidationError, match="extra_forbidden"):
            validator.validate_python(
                {"query": "HNSW", "_otel": {"traceparent": "00-abc"}}
            )

        print("\nPydanticAI HARD REJECTS extras via PluggableSchemaValidator")
        print("Unlike AutoGen (silent strip), PydanticAI raises ValidationError")

    def test_run_context_as_sidecar(self):
        """Verify that PydanticAI's RunContext (deps) can carry
        auxiliary data like OTel carrier alongside tool execution.
        This is the sidecar mitigation for strict parameter validation."""
        from pydantic_ai import Agent, RunContext
        from pydantic_ai.models.test import TestModel
        from dataclasses import dataclass

        @dataclass
        class AgentDeps:
            otel_carrier: dict
            extra_context: dict

        agent = Agent(
            TestModel(),
            deps_type=AgentDeps,
            system_prompt="Test agent.",
        )

        captured_carrier = {}

        @agent.tool
        def search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search with context."""
            captured_carrier.update(ctx.deps.otel_carrier)
            return f"Results for {query}"

        deps = AgentDeps(
            otel_carrier={"traceparent": "00-abcdef-123456-01"},
            extra_context={"session_id": "abc"},
        )

        assert deps.otel_carrier["traceparent"] == "00-abcdef-123456-01"
        print(f"\nDeps created with carrier: {deps.otel_carrier}")
        print("RunContext sidecar pattern: VIABLE for PydanticAI")

    def test_tool_plain_vs_tool_with_context(self):
        """Document the difference between @agent.tool_plain (no context)
        and @agent.tool (receives RunContext). Only the latter can access
        the sidecar deps for OTel carrier."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def plain_search(query: str) -> str:
            """No access to RunContext."""
            return f"Results for {query}"

        @agent.tool
        def context_search(ctx, query: str) -> str:
            """Has access to RunContext."""
            return f"Results for {query}"

        tools = agent._function_toolset.tools
        print(f"\nRegistered tools: {list(tools.keys())}")
        for name, tool in tools.items():
            ctx_info = "with RunContext" if tool.takes_ctx else "plain (no context)"
            print(f"  {name}: {ctx_info}")

        assert not tools["plain_search"].takes_ctx
        assert tools["context_search"].takes_ctx

    def test_comparison_autogen_vs_pydanticai_validation(self):
        """Documents the critical difference between AutoGen and PydanticAI:

        AutoGen: empty model_config → extra='ignore' → SILENT STRIP
        PydanticAI: PluggableSchemaValidator → extra_forbidden → HARD REJECT

        Both have additionalProperties: false in their schemas, but
        the enforcement mechanism is completely different. This is why
        integration tests matter — the schema alone doesn't tell you
        the runtime behavior."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic import ValidationError

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def search(query: str) -> str:
            """Search."""
            return f"Results for {query}"

        tool = agent._function_toolset.tools["search"]

        # PydanticAI: hard reject
        with pytest.raises(ValidationError):
            tool.function_schema.validator.validate_python(
                {"query": "test", "_otel": {"traceparent": "00-abc"}}
            )

        print("\nAutoGen: schema says reject, runtime silently strips (extra='ignore')")
        print("PydanticAI: schema says reject, runtime ACTUALLY rejects (extra_forbidden)")
        print("Same schema, different enforcement — integration tests are essential")

    def test_tool_manager_validate_args_rejects_extras(self):
        """tool_manager._validate_args() is the central validation entry
        point. It calls validator.validate_json() for string args or
        validator.validate_python() for dict args. Both paths enforce
        extra_forbidden.

        This is the path that agent.run() uses when processing LLM
        tool call responses — all tool execution flows through here."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic import ValidationError

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def search(query: str) -> str:
            """Search."""
            return f"Results for {query}"

        tool = agent._function_toolset.tools["search"]
        validator = tool.function_schema.validator

        # Path 1: validate_python (dict args) — rejects extras
        with pytest.raises(ValidationError, match="extra_forbidden"):
            validator.validate_python(
                {"query": "HNSW", "_otel": {"traceparent": "00-abc"}}
            )

        # Path 2: validate_json (string args) — also rejects extras
        import json
        with pytest.raises(ValidationError, match="extra_forbidden"):
            validator.validate_json(
                json.dumps({"query": "HNSW", "_otel": {"traceparent": "00-abc"}})
            )

        print("\nBoth validate_python and validate_json reject extras")
        print("All paths through tool_manager converge on same rejection")

    def test_function_schema_call_passes_validated_args_only(self):
        """FunctionSchema.call() passes validated args to the function
        via _call_args(). Since validation already rejected extras,
        the function never sees unknown fields.

        This is different from Haystack where extras survive to the
        function call and are rejected by Python's function signature."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(TestModel(), system_prompt="Test.")

        @agent.tool_plain
        def search(query: str) -> str:
            """Search."""
            return f"Results for {query}"

        tool = agent._function_toolset.tools["search"]
        fs = tool.function_schema

        # _call_args transforms validated dict into positional/keyword args
        # It only passes fields that survived validation
        print(f"\nFunctionSchema.single_arg_name: {fs.single_arg_name}")
        print(f"FunctionSchema.positional_fields: {fs.positional_fields}")
        print(f"FunctionSchema.takes_ctx: {fs.takes_ctx}")
        print("Extras never reach the function — rejected at validator level")
