# AutoGen v0.4 — Integration Test Discoveries

Findings from running real AutoGen imports vs our simulated tests.

## Envelope Shape

### FunctionCall is a dataclass, not a Pydantic model
- **Our assumption:** `FunctionCall` has `model_dump()` (Pydantic)
- **Reality:** It's a plain Python dataclass. Use `dataclasses.asdict()` for serialization.
- **Impact:** Our simulated test in `cross-library-demo/test_payload_traceparent.py::test_string_wrapped_arguments_autogen` assumed Pydantic semantics. The actual serialization path is simpler — no `extra="allow"/"forbid"` on the envelope itself.

### arguments field accepts both strings AND dicts
- **Our assumption:** `arguments` is always a JSON string (based on research noting AutoGen stringifies dict args)
- **Reality:** AutoGen stores whatever you pass — string stays string, dict stays dict. No automatic coercion.
- **Impact:** The double-encoding scenario (carrier nested inside a JSON string) is one valid path, but direct dict arguments are also possible. Our simulated test only covers the string path.

### FunctionTool schema has additionalProperties: false — BUT NOT ENFORCED AT RUNTIME
- **Schema says:** `"additionalProperties": false`
- **Runtime does:** The generated Pydantic args model has **empty `model_config`** — Pydantic defaults to `extra="ignore"`, silently discarding unknown fields
- **Our assumption was wrong:** We classified this as "hard reject" based on the schema. The actual behavior is **silent strip** — the most dangerous failure mode
- **The distinction:** `additionalProperties: false` is a directive for the **LLM provider** (telling the LLM not to generate extras). It is NOT enforced at **runtime validation** by `model_validate()`
- **All three entry points** (`tool.run()`, `tool.run_json()`, `workbench.call_tool()`) converge on the same silent strip behavior
- **Impact:** Our simulated test and ISSUE_CAUSALITY.md need to reclassify AutoGen from "compatible at envelope level" to "silent strip at tool execution level"
- **Mitigation needed:** Sidecar pattern or out-of-band correlation — carrier cannot ride inside tool arguments

## Context Propagation

### Baggage LOST at message dispatch boundary
- **Confirmed:** Baggage set before `runtime.publish_message()` is NOT visible inside the agent's message handler.
- **Classification:** Requires manual propagation
- **Root cause:** `SingleThreadedAgentRuntime` processes each message in a separate `asyncio.create_task()`. While Python copies contextvars at task creation, the runtime's internal message queue and dispatch mechanism does not preserve the caller's context into the handler.

### BaggageSpanProcessor works for caller-side spans
- **Confirmed:** Spans created in the caller context (before dispatch) correctly carry baggage attributes via BaggageSpanProcessor.

## Action items
- [x] Update `cross-library-demo/test_payload_traceparent.py::test_string_wrapped_arguments_autogen` — rewritten to document two-layer behavior (envelope preserves, tool execution silently strips)
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_autogen_message_dispatch_in_process` docstring — confirmed baggage LOST at dispatch
- [x] Reclassify AutoGen in `ISSUE_CAUSALITY.md` — updated from "compatible at envelope level" to "silent strip at tool execution level"
- [ ] Update `research-notes.md` AutoGen classification — schema says reject, runtime says silent strip
