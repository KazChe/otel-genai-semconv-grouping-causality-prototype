"""OTEL tracing setup WITH BaggageSpanProcessor — the 'after' state.

The BaggageSpanProcessor reads W3C Baggage from the active context and copies
baggage entries as span attributes on every span. This means any span created
while gen_ai.group.id is set in Baggage will automatically carry that attribute
— without the span creator needing to know about grouping.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS
from openinference.instrumentation.langchain import LangChainInstrumentor


def init_tracing(service_name="langgraph-demo"):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
    )

    provider = trace_sdk.TracerProvider(resource=resource)

    # Export spans to collector
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # ── GROUPING ── BaggageSpanProcessor copies Baggage entries to span attributes.
    # Any span created while gen_ai.group.id is in Baggage will automatically
    # carry it as an attribute — no manual span.set_attribute() needed.
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))

    trace.set_tracer_provider(provider)

    # Auto-instrument LangGraph/LangChain
    LangChainInstrumentor().instrument(tracer_provider=provider)

    print(f"[tracing] LangGraph demo OTEL initialized with BaggageSpanProcessor, exporting to {endpoint}")
    return provider
