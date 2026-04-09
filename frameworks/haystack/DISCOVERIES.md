# Haystack — Integration Test Discoveries

Findings from running real Haystack imports vs our simulated tests.

## Envelope Shape

### ToolCall is a dataclass, not a Pydantic model
- **Our assumption:** Not explicitly stated, but simulated test modeled it as a plain dict
- **Reality:** `ToolCall` is a Python dataclass with fields: `tool_name: str`, `arguments: dict[str, Any]`, `id: str | None`, `extra: dict[str, Any] | None`
- **Impact:** No Pydantic-level `extra="allow"/"forbid"` on the envelope. Dataclass accepts whatever you pass.

### ToolCall has a built-in `extra` field — NATIVE SIDECAR
- **Our assumption:** No metadata/extensions field exists on ToolCall
- **Reality:** `ToolCall.extra` is a `dict[str, Any] | None` field, defaults to `None`
- **Impact:** This is a **native sidecar slot** for carrier injection. No need to inject into `arguments` (which may be schema-validated) or use out-of-band correlation. The carrier rides in `tc.extra = {"_otel": carrier}` and survives JSON round-trips.
- **This is the best native fit we've found** — even better than Semantic Kernel's `KernelContent.Metadata` because it's directly on the tool call object itself.

### ToolCall has an `id` field
- **Our assumption:** Unknown
- **Reality:** `ToolCall.id` is `str | None`, maps to the LLM provider's tool call ID
- **Impact:** Can be used as correlation key for out-of-band pattern, but the `extra` field makes this unnecessary in most cases.

### arguments dict preserves extra keys
- **Confirmed:** Extra fields in `arguments` dict survive — no schema validation at the ToolCall level
- **Caveat:** `tools_strict=True` sets `additionalProperties: false` on the JSON Schema sent to the LLM provider, which rejects extras at the **provider boundary** before Haystack even receives the response

## Tool Execution Entry Points

Three paths for tool execution were tested:

### Path 1: `Tool.invoke(**kwargs)` — direct invocation
- `Tool.invoke()` passes arguments as `**kwargs` to the underlying Python function
- If the function doesn't accept `**kwargs`, Python raises `TypeError` for unknown keyword arguments
- Haystack wraps this in `ToolInvocationError`
- **Result:** Extras in arguments are **hard rejected** at the Python function signature boundary
- This is different from Pydantic/schema-level validation — it's the Python function itself that rejects unknown kwargs

### Path 2: `Tool.invoke()` with `**kwargs` function — extras survive
- If the tool function is defined with `**kwargs` (e.g., `def search(query: str, **kwargs)`), extras pass through to the function
- The carrier IS accessible inside the function via `kwargs["_otel"]`
- **Result:** Carrier **survives** — but requires the tool function to be designed for it

### Path 3: `ToolInvoker.run()` — pipeline component path
- `ToolInvoker._prepare_tool_call_params()` does `tool_call.arguments.copy()` — extras in arguments survive this step
- Then calls `tool.invoke(**final_args)` — which hits the same function signature boundary as Path 1
- **Result:** Same as Path 1 — extras survive through the preparation stage but are rejected at function call

### Path 4: `ToolCall.extra` — out-of-band sidecar
- `ToolCall.extra` is NOT passed to `tool.invoke()` — it rides alongside the tool call
- Instrumentation or middleware must read `tc.extra` separately from the function call path
- **Result:** Carrier is **preserved** but must be accessed out-of-band, not via function parameters

### Summary of all entry points

| Entry point | Extras in arguments | ToolCall.extra sidecar |
|-------------|--------------------|-----------------------|
| `Tool.invoke(**kwargs)` — strict function | Hard reject (TypeError) | N/A (not on Tool) |
| `Tool.invoke(**kwargs)` — flexible `**kwargs` fn | Survives | N/A |
| `ToolInvoker.run()` — pipeline path | Hard reject at invoke | Extra preserved out-of-band |
| Direct `ToolCall.extra` access | N/A | Preserved — read separately |

## Context Propagation

### Baggage PROPAGATES across sync pipeline components
- **Our research classification:** "Requires manual propagation"
- **Reality:** Baggage set before `pipeline.run()` is visible in **both** sequential components in a sync `Pipeline`
- **Impact:** Our research classification was too conservative for the sync pipeline case. Baggage propagates automatically for sync pipelines.
- **Update:** Also tested with `AsyncPipeline.run_async()` — baggage propagates across both async components too. Haystack's `copy_context()` in the executor path preserves baggage.

### BaggageSpanProcessor works for caller-side spans
- **Confirmed:** Spans created in the caller context carry baggage attributes via BaggageSpanProcessor.

## Reclassifications

| Aspect | Research classification | Actual behavior |
|--------|----------------------|-----------------|
| Envelope extra fields | No metadata slot found | `ToolCall.extra` is a native sidecar — best fit found |
| Sync pipeline baggage | Requires manual propagation | Propagates automatically |
| Async pipeline baggage | Requires manual propagation | Propagates automatically (copy_context preserves it) |
| tools_strict=True | Hard reject | Confirmed — at provider boundary |

## Action items
- [x] Update `cross-library-demo/test_payload_traceparent.py::test_json_schema_additional_properties_false_haystack` — updated with ToolCall.extra sidecar mitigation
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_haystack_pipeline_component_dispatch` docstring — sync propagates confirmed, async TBD
- [x] Update `ISSUE_CAUSALITY.md` — Haystack native `extra` field documented as best sidecar fit
- [x] Update `ISSUE_GROUPING.md` — Haystack sync pipeline reclassified to "propagates"
- [x] Add `AsyncPipeline` test — baggage PROPAGATES in async pipeline too (copy_context preserves it)
