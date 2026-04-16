# [gen-ai] Add causal span linking for LLM-triggered tool execution

Split from [open-telemetry/semantic-conventions#3575](https://github.com/open-telemetry/semantic-conventions/issues/3575).
Companion issue: Generic Grouping Attributes for GenAI Spans (ISSUE_GROUPING.md)

### Area(s)

area:gen-ai

### What's missing?

GenAI semantic conventions have no standard mechanism for expressing causal relationships between LLM inference and tool execution spans. When an LLM returns tool calls, the subsequent `execute_tool` span appears as a sibling of the `chat` span — not as a child. There is no signal in the trace tree that says "this tool execution was triggered by that LLM call."

This matters because debugging agentic workflows requires understanding causality: which LLM decision led to which tool invocation, and in what order. Without causal links, users must reconstruct the chain manually by matching timestamps and tool names across a flat span list.

The problem is compounded when independent instrumentation libraries create inference and tool execution spans separately. Their span lifecycles do not overlap — the inference span ends when the LLM call returns; the tool execution span starts afterward. In-process OTel context propagation cannot link them because the parent span's context is no longer active when the child span is created. This is not specific to any particular framework combination; it affects any setup where inference and tool execution are instrumented by different libraries.

**Open concerns from #3575:**

**Cross-library span linking** (@Cirilla-zmh):
> "In an agent built with LiteLLM and LangChain, how would I add a link from an `execute_tool` span created by LangChain to an `inference` span created by LiteLLM?"

When instrumentation library A creates an inference span and library B creates a tool execution span independently, their span lifecycles do not overlap. The inference span ends when the LLM call returns; the tool execution span starts afterward. In-process context propagation cannot link them because the parent span's context is no longer active. The question is what mechanism can establish this causal link across independent library span lifecycles.

**Payload fragility** (@Cirilla-zmh):
> "You can't always assume that you can access the tool-call payload and inject additional information into it. ...parsing and copying may occur, so the context you inject into payload could be lost."

If the solution involves injecting trace context into tool call payloads (analogous to HTTP `traceparent` headers), the injected data must survive the processing pipeline between LLM response and tool execution. In practice, tool call arguments flow through a multi-layer pipeline — Model API to Orchestrator to Tool Executor — and may undergo:

- JSON serialization/deserialization (single-hop and multi-hop)
- State persistence round-trips (e.g., Checkpointer serialization via Pydantic models)
- Schema validation that strips unknown fields
- Strict Pydantic models with `extra="forbid"`
- Binary serialization (MessagePack, Protobuf)
- Typed object encapsulation (dataclasses, Pydantic models)

Any of these steps could silently discard the injected context. The most dangerous failure mode is not a crash but a **silent loss of observability** — the application continues to run but traces are disconnected. The concern is that the implementation cost of handling these edge cases may outweigh the value the causal linking provides.

**Grouping vs. dedicated operations** (@Krishnachaitanyakc):
> "Where do you see the line between 'this is just a grouping concern' and 'this needs its own operation'?"

Causal linking is distinct from grouping. Grouping says "these spans belong to the same round." Causality says "this span was triggered by that span." A `chat` span that triggers an `execute_tool` span has a causal relationship that grouping alone cannot express — the tool execution has a specific parent, not just a shared tag.

### Describe the solution you'd like

Payload-level `traceparent` propagation via framework-native sidecar mechanisms — the same W3C Trace Context standard used for HTTP, applied to LLM tool call envelopes through each framework's existing extension points rather than through tool call arguments.

#### 1. Key finding: carrier MUST NOT go in tool call arguments

Integration testing across 6 major frameworks revealed that **injecting the carrier into tool call arguments fails in 5 of 6 frameworks**. The carrier is either silently stripped or hard rejected before reaching the tool function:

| Framework | What happens to extras in args | Mechanism | Classification |
|-----------|-------------------------------|-----------|----------------|
| AutoGen | Silently discarded | `model_validate()` with `extra="ignore"` | Silent strip |
| CrewAI | Silently discarded | Internal filtering between `run()` and `_run()` | Silent strip |
| Google ADK | Silently discarded | `{k:v for k,v in args if k in valid_params}` | Silent strip |
| Haystack | Crashes | Python function signature rejects unknown kwargs | Hard reject |
| PydanticAI | Crashes | `PluggableSchemaValidator` raises `extra_forbidden` | Hard reject |
| LlamaIndex | Crashes | Python function signature rejects unknown kwargs | Hard reject |

The silent strip is the most dangerous failure mode — the application continues to run but traces are silently disconnected. Three of six frameworks exhibit this behavior. AutoGen and PydanticAI both declare `additionalProperties: false` in their tool schemas, but the runtime enforcement is opposite — AutoGen silently strips while PydanticAI raises `ValidationError`. The schema alone does not predict runtime behavior.

This means the original `tool_call["_otel"] = carrier` approach from the prototype is not viable as a general-purpose convention. The carrier must travel through a different channel.

#### 2. Mechanism: sidecar propagation (normative)

The carrier is a `dict[str, str]` containing `traceparent` (and optionally `tracestate`), injected via standard OTel `propagate.inject()` and extracted via `propagate.extract()`. This is not a new propagation mechanism — it is the existing W3C Trace Context standard applied to each framework's native extension point.

The convention should define:
- **The carrier format** — `dict[str, str]` with `traceparent` key (already standardized by W3C)
- **The propagation contract** — carrier MUST be placed in a framework's native sidecar, NOT in tool call arguments
- **The extraction contract** — instrumentation authors extract the carrier from the sidecar and use it as parent context for the tool execution span

#### 3. Framework sidecar mapping (strong guidance)

Integration testing found that 4 of 6 frameworks already have a native sidecar mechanism suitable for carrier injection:

| Framework | Native sidecar | How carrier rides |
|-----------|---------------|-------------------|
| Haystack | `ToolCall.extra` field | `tc.extra = {"_otel": carrier}` — directly on the tool call object, strongest native fit |
| Google ADK | `ToolContext.state` | Injected automatically when function declares `tool_context` parameter — rich context with state, session, artifacts |
| PydanticAI | `RunContext.deps` | Requires `@agent.tool` (not `@tool_plain`) — deps carry carrier alongside tool execution |
| Semantic Kernel | `KernelContent.Metadata` | Built-in metadata dict on function result content |
| AutoGen | None found | Out-of-band correlation required |
| LlamaIndex | None found | Out-of-band correlation required |
| CrewAI | None found | Out-of-band correlation required |

**Sidecar vs Out-of-Band — two different integration models:**

**Sidecar propagation** means the carrier rides inside a field that the framework already provides and manages as part of its normal data flow. The framework carries it for you — the instrumentor just needs to know which field to use. This is the preferred approach because it requires no custom infrastructure and survives whatever serialization the framework applies to its own objects.

**Out-of-Band Correlation** is the fallback for frameworks that don't provide a native sidecar (AutoGen, LlamaIndex, CrewAI). It means the carrier doesn't travel inside any framework object at all. Instead, the instrumentor stores the carrier in a separate data structure — typically a thread-local or async-local dict — keyed by the tool call's correlation ID. At tool execution time, the executor retrieves the carrier from this external store using the ID. The framework never sees or touches the carrier.

This works, but it's hand-rolled plumbing: every instrumentor for every framework without a native sidecar must independently implement the storage, keying, lifecycle management, and cleanup. Without a convention, each implementation reinvents this mapping differently — which is exactly the kind of fragmentation that a semantic convention should prevent.

The convention should:
- **Recommend sidecar propagation** as the primary approach for frameworks that provide an extension point
- **Define a standard out-of-band contract** (storage interface, key format, lifecycle) so instrumentors for frameworks without a native sidecar don't each invent their own
- **Encourage framework authors** to add a metadata/context/extra field if they don't have one — pointing to Haystack's `ToolCall.extra` and Google ADK's `ToolContext` as successful examples

#### 4. Serialization resilience (informational)

While the carrier must not go in tool call arguments, the carrier format itself (`dict[str, str]`) is resilient across common serialization paths. This matters for the sidecar mechanisms that do serialize:

| Transformation | Carrier survives? |
|----------------|-------------------|
| JSON round-trip (single and multi-hop) | Yes |
| State persistence (Pydantic `extra="allow"` round-trips) | Yes |
| `copy.deepcopy` | Yes |
| MessagePack binary serialization | Yes |
| Pydantic `extra="forbid"` | No — hard reject |
| Schema sanitization (known-fields filter) | No — silent strip |

#### 5. Relationship to grouping

Causal linking and grouping are complementary layers of the same contract:
- **Grouping** (baggage) handles loose structural correlation: "these spans belong to the same round"
- **Causality** (sidecar traceparent) handles directed relationships: "this tool execution was triggered by that LLM call"
- Together they provide both structural and temporal relationships in the trace tree

#### Prototype evidence

Prototype repo: https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype

- **Simulated tests:** 20 automated tests mapping the compatibility matrix — serialization resilience, failure modes, mitigations, and framework-specific envelope patterns ([`cross-library-demo/test_payload_traceparent.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/cross-library-demo/test_payload_traceparent.py))
- **Integration tests:** Real framework imports verifying actual envelope shapes and context propagation across 6 frameworks — AutoGen, Haystack, PydanticAI, LlamaIndex, CrewAI, Google ADK ([`frameworks/`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/tree/main/frameworks))
- **Runnable demos:** Same-process causality (LangGraph), cross-library causality, baseline comparison (flat siblings vs. causal tree)
- **Key discovery:** simulated tests alone would have given ~25% accuracy on envelope behavior — integration tests were essential for correct classification
