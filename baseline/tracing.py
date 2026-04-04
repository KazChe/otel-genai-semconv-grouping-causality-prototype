"""Standard OTEL tracing setup — NO BaggageSpanProcessor, NO grouping conventions.
This is the 'before' state that shows what backends see without the proposed primitives.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor


def init_tracing():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource.create(
        {
            "service.name": "baseline-demo",
            "service.version": "1.0.0",
        }
    )

    provider = trace_sdk.TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument LangGraph/LangChain
    LangChainInstrumentor().instrument(tracer_provider=provider)

    print(f"[tracing] Baseline OTEL initialized, exporting to {endpoint}")
    return provider
