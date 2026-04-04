"""OTEL tracing for AutoGen demo — with BaggageSpanProcessor.

AutoGen v0.4 uses the global TracerProvider set via trace.set_tracer_provider().
The runtime internally creates spans for invoke_agent, execute_tool, etc.
BaggageSpanProcessor copies gen_ai.group.id from Baggage to ALL spans —
including those created by AutoGen's runtime, not just our code.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS


def init_tracing(service_name="autogen-demo"):
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
    # on ALL spans — including AutoGen's internally created spans.
    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))

    trace.set_tracer_provider(provider)

    print(f"[tracing] AutoGen demo OTEL initialized with BaggageSpanProcessor")
    print(f"[tracing] Exporting to {endpoint}")
    return provider
