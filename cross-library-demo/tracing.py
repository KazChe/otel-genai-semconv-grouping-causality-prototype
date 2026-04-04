"""OTEL tracing for cross-library demo: LangChain + LiteLLM.

Both OpenInference instrumentors registered on the SAME TracerProvider.
BaggageSpanProcessor copies grouping attributes to all spans regardless
of which library created them.

This is the setup that the reviewer said was impractical:
  "how would I add a link from an execute_tool span created by LangChain
   to an inference span created by LiteLLM?"
"""

import os

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor


def init_tracing(service_name="cross-library-demo"):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
    )

    provider = trace_sdk.TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # ── GROUPING ── BaggageSpanProcessor copies Baggage to span attributes
    # on ALL spans — whether created by LangChain or LiteLLM.
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))

    trace.set_tracer_provider(provider)

    # Both instrumentors share the same TracerProvider.
    # LangChain creates orchestration spans (execute_tool, chain, etc.)
    # LiteLLM creates inference spans (chat, completion, etc.)
    # They don't know about each other — but they share context.
    LangChainInstrumentor().instrument(tracer_provider=provider)
    LiteLLMInstrumentor().instrument(tracer_provider=provider)

    print(f"[tracing] Cross-library demo: LangChain + LiteLLM instrumented on shared TracerProvider")
    print(f"[tracing] Exporting to {endpoint}")
    return provider
