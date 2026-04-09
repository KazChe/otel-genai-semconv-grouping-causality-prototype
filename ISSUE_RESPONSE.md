# Draft response for [open-telemetry/semantic-conventions#3575](https://github.com/open-telemetry/semantic-conventions/issues/3575)

> **Note:** This document is kept for historical reference so that the [link from the original GitHub issue comment](https://github.com/open-telemetry/semantic-conventions/issues/3575) continues to work. As [proposed by @Cirilla-zmh](https://github.com/open-telemetry/semantic-conventions/issues/3575), this proposal is being split into two separate issues — one for generic grouping attributes and one for causal span linking — and significantly expanded based on reviewer feedback and cross-framework integration testing (6 frameworks verified so far). The updated proposals will be posted as new issues on open-telemetry/semantic-conventions.

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
| **langgraph-demo (overlapping)** | LangGraph | Namespaced baggage keys solve overlapping group membership (skill + iteration + agent simultaneously) |
| **cross-library-demo** | LangChain + LiteLLM | Grouping works across library boundaries. Causality requires payload traceparent as in-process context alone does not survive across independent library span lifecycles. |
| **cross-library-demo (fragility tests)** | pytest + InMemorySpanExporter | Payload traceparent survives JSON, deepcopy, Pydantic (extra=allow), msgpack. Fails with strict schema stripping — mitigations tested. |
| **autogen-demo** | AutoGen v0.4 | Baggage survives AutoGen's async event-driven message dispatch. Runtime-created spans carry `gen_ai.group.id` without any framework-specific telemetry configuration. |

### Addressing @Cirilla-zmh's concerns

**Nested/multi-dimensional group membership:**

> "A group may be nested, meaning that an operation can belong to groups across multiple dimensions."

W3C Baggage supports multiple keys simultaneously, so a span can carry `gen_ai.group.id=round-1`, `gen_ai.group.type=react_iteration`, and `gen_ai.agent.id=main-agent` at the same time, each set via `baggage.set_baggage()` and copied to attributes by `BaggageSpanProcessor`. No arrays and no wrapper spans.

See: [`langgraph-demo/agent_multidim.py#L80-L82`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/langgraph-demo/agent_multidim.py#L80-L82) sets 3 baggage dimensions per round, and [`langgraph-demo/tracing.py#L36-L39`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/langgraph-demo/tracing.py#L36-L39) — `BaggageSpanProcessor` setup.

**Overlapping group membership (gen_ai.group.type mutual exclusivity):**

> "If gen_ai.group.type is of type StringAttributeKey, shouldn't its value be mutually exclusive? ...when a skill is read in the ReAct iteration, shouldn't it belong to both skill and react_iteration?"

This is a valid concern. The solution is to use **namespaced baggage keys** instead of overloading a single `gen_ai.group.type` key. Each group dimension gets its own key:

```
gen_ai.group.id              = round-1          (iteration identity)
gen_ai.group.iteration.type  = react            (iteration kind)
gen_ai.group.skill.id        = rag-retrieval    (skill identity)
gen_ai.group.skill.type      = rag              (skill kind)
gen_ai.agent.id              = main-agent       (agent identity)
```

> **Note:** These attribute names are **proposed, not existing** in the current OTel GenAI semantic conventions. The prototype validates that the underlying mechanism (W3C Baggage + `BaggageSpanProcessor`) supports this pattern. The actual attribute names and namespace structure would need to be agreed upon as part of the spec discussion.

A span simultaneously belongs to an iteration AND a skill — no arrays, no wrapper spans, no mutual exclusivity. When a skill is not active, the skill keys are simply absent from baggage and the span only carries the iteration dimension.

This is validated by automated tests:
- [`langgraph-demo/test_overlapping_groups.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/langgraph-demo/test_overlapping_groups.py) — 5 tests covering overlapping membership, independent dimension lifecycle, nested agent delegation, and queryability by any dimension.
- [`langgraph-demo/agent_overlapping_groups.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/langgraph-demo/agent_overlapping_groups.py) — runnable demo where rounds 1-2 have active skills (RAG, code-gen) and round 3 is pure reasoning with no skill dimension.

Jaeger query examples showing overlapping membership:
```
gen_ai.group.skill.id=rag-retrieval                            → all RAG skill spans
gen_ai.group.iteration.type=react                              → all ReAct iteration spans
gen_ai.group.skill.id=rag-retrieval gen_ai.group.id=round-1    → RAG in round 1
gen_ai.agent.id=main-agent gen_ai.group.skill.type=rag         → RAG by main agent
```

**Cross-library span linking (LangChain + LiteLLM):**

> "How would I add a link from an execute_tool span created by LangChain to an inference span created by LiteLLM?"

Two findings here:

1. **Grouping works without coordination.** Both instrumentors are registered on the same `TracerProvider`. `BaggageSpanProcessor` copies baggage to all spans regardless of which library created them — LiteLLM's `completion` span carries `gen_ai.group.id=round-1` even though LiteLLM knows nothing about the convention.
   See: [`cross-library-demo/tracing.py#L45-L50`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/tracing.py#L45-L50) — both `LangChainInstrumentor` and `LiteLLMInstrumentor` on the same provider.
   See: [`cross-library-demo/agent.py#L87-L88`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/agent.py#L87-L88) — baggage set before `litellm.completion()` call.

2. **Causality requires payload-level propagation.** When LiteLLM's `completion` span ends (after `litellm.completion()` returns), its context is no longer active, so in-process context propagation can't link the subsequent `execute_tool` span to it. The fix: inject `traceparent` into the tool call payload while the parent span is still active, and extract it at tool execution time. This is the same pattern HTTP uses for cross-service propagation, applied to LLM tool call envelopes.
   See: [`cross-library-demo/agent_with_causality.py#L93-L96`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/agent_with_causality.py#L93-L96) — inject traceparent while chat span is active.
   See: [`cross-library-demo/agent_with_causality.py#L111-L113`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/agent_with_causality.py#L111-L113) — extract traceparent at tool execution.

**Payload traceparent fragility:**

> "You can't always assume that you can access the tool-call payload and inject additional information into it. Parsing and copying may occur, so the context you inject into payload could be lost."

This is a legitimate concern. I built automated tests to map exactly which payload transformations preserve or destroy the carrier:

| Transformation | Carrier survives? | Test |
|---------------|-------------------|------|
| JSON round-trip (1 hop) | Yes | `test_json_round_trip` |
| JSON round-trip (3 hops) | Yes | `test_multiple_json_round_trips` |
| `copy.deepcopy` | Yes | `test_deep_copy` |
| Nested in message list + JSON | Yes | `test_nested_json_in_message_list` |
| Dataclass with metadata dict | Yes | `test_dataclass_payload` |
| Pydantic `extra="allow"` | Yes | `test_pydantic_model_with_extra_allowed` |
| MessagePack binary round-trip | Yes | `test_msgpack_round_trip` |
| Pydantic `extra="forbid"` | **No** | `test_pydantic_model_extra_forbid_loses_carrier` |
| Schema stripping unknown fields | **No** | `test_schema_strip_unknown_fields` |

See: [`cross-library-demo/test_payload_traceparent.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/test_payload_traceparent.py) — 12 tests total.

The carrier is just a `dict[str, str]` (key: `traceparent`, value: W3C trace context string), so it survives any serialization that preserves string key-value pairs. It fails when schemas actively reject unknown fields.

**Mitigations for strict-schema scenarios** (also tested):

1. **Metadata sidecar field**: Place the carrier in a `metadata` or `extensions` dict that schemas typically allow for extension data (`test_strict_schema_mitigation_via_metadata_field`).
2. **Out-of-band context store**: Store the carrier keyed by the tool call's correlation ID. The payload only carries the ID; the executor retrieves the carrier from the store (`test_out_of_band_context_store`).

The convention should recommend approach (1) as the default and document approach (2) for fully opaque payloads.

### Addressing @lmolkova's question

> "How does this problem manifest for users?"

The baseline demo shows it directly: a multi round ReAct agent produces a flat span list as `chat`, `execute_tool`, `chat`, `execute_tool`, `chat` with no signal for which spans belong to which round or which llm call triggered which tool. Debugging "why did my agent fail on attempt 3?" requires manually correlating spans by timestamp

### Addressing @Krishnachaitanyakc's question

> "Where do you see the line between 'this is just a grouping concern' and 'this needs its own operation'?"

The prototype confirms your intuition, they complement each other. Grouping (Baggage) handles loose structural correlation: "these spans belong to the same round." Dedicated operations (`execute_tool`, `chat`) handle phases with their own meaningful duration and hierarchy. Payload traceparent bridges the two by establishing causal parent-child links between operations that grouping alone can not express.

## Generalizability

The approach was tested across two framework categories (graph-based and async event-driven). 
Analysis of the broader landscape:

| Framework category | Baggage (grouping) | Payload traceparent (causality) | Status |
|-----------|-------------------|-------------------------------|--------|
| Graph-based (LangGraph, LlamaIndex) | Works | Works | Tested |
| Async event-driven (AutoGen v0.4) | Works | Message envelope is natural carrier | Tested |
| CrewAI | Works (built-in OTel) | Task delegation payload is carrier | Analysis |
| Cross-language (Python//.NET) | Won't cross boundary | Only viable mechanism | Analysis |
| MCP-based tools | Won't cross process | Already HTTP, traceparent native | Analysis |

These are not two independent ideas -they are two layers of the same contract that within a single runtime, use `Baggage`. At every boundary that crosses async,language, or process lines, use payload `traceparent`.

## Caveats

- **Baggage propagation depends on the runtime.** AutoGen preserves it because its runtime owns context dispatch. Frameworks without runtime-managed dispatch need explicit `context.copy()` at async boundaries, the convention should specify this as an instrumentation author responsibility.

- **AutoGen validation used the global `TracerProvider`**, not AutoGen's Core API telemetry setup. This means the pattern works with any framework that follows the standard OTel global provider pattern, but does not yet prove it works with AutoGen's `GrpcWorkerAgentRuntime` (cross-process).
- **Payload traceparent injection requires a convention** for where to put the carrier in the tool call envelope. The prototype uses `tool_call["_otel"]` — a real spec would need to standardize this field name.

## Prototype repo

https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype

Self contained with `docker compose up -d` starts the infrastructure (OTEL Collector + Aspire dashboard), then run any demo script. Screenshots in the repo show before/after comparisons.
