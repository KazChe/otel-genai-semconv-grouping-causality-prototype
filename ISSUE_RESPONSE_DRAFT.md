# Draft response for open-telemetry/semantic-conventions#3575

---

Following up on the great feedbacks from @Cirilla-zmh and @lmolkova the original proposal was desirable but not practical.I went back and prototyped a revised approach that uses existing OTel mechanisms rather than proposing new span types or custom attributes. Here are what I found.

## Revised approach: two layers, one contract

Instead of custom `gen_ai.group.id` attributes on span definitions and typed span links, the revised approach uses two existing OTel mechanisms:

**1. Grouping via W3C Baggage + `BaggageSpanProcessor`**

`gen_ai.group.id` and `gen_ai.group.type` are carried in [W3C Baggage](https://www.w3.org/TR/baggage/) and the official [`opentelemetry-processor-baggage`](https://pypi.org/project/opentelemetry-processor-baggage/) package copies them to span attributes automatically. No changes to span creation code needed as the processor attaches grouping data to every span in the active context, regardless of which library created the span.

**2. Causality via payload-level `traceparent` injection**

After an LLM returns tool calls, the current span's `traceparent` is injected into the tool call payload (e.g., `tool_call["_otel"] = carrier`). The tool executor extracts it and uses it as the parent context. This makes `execute_tool` a child of the `chat` span that triggered it — establishing a causal link in the trace tree. Same pattern as HTTP `traceparent` headers, applied to LLM message envelopes.

## What the prototype demonstrates

I built four demos ([prototype repo](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype)):

| Demo | Framework | Finding |
|------|-----------|---------|
| **baseline** | LangGraph | Before we see flat spans, no grouping or causal signal |
| **langgraph-demo** | LangGraph | Baggage grouping works. Payload traceparent causality works |
| **cross-library-demo** | LangChain + LiteLLM | Grouping works across library boundaries. Causality requires payload traceparent as in-process context alone does not survive across independent library span lifecycles. |
| **autogen-demo** | AutoGen v0.4 | Baggage survives AutoGen's async event-driven message dispatch. Runtime-created spans carry `gen_ai.group.id` without any framework-specific telemetry configuration. |

