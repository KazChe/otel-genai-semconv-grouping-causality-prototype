# PydanticAI — Integration Test Discoveries

Findings from running real PydanticAI imports vs our simulated tests.

## Envelope Shape

### Tool parameters use PluggableSchemaValidator — TRUE HARD REJECT
- **Our assumption:** Strict Pydantic model with `extra="forbid"` — hard reject
- **Reality:** Uses `PluggableSchemaValidator` with `extra_forbidden` enforcement. `ValidationError` raised with `extra_forbidden` error type.
- **Critical comparison with AutoGen:** Both frameworks have `additionalProperties: false` in their JSON schemas. But AutoGen silently strips extras (empty `model_config`, defaults to `extra="ignore"`), while PydanticAI actually rejects them. Same schema, completely different runtime behavior. **This is why integration tests matter.**
- **Impact:** Our simulated test classification of "hard reject" was correct for PydanticAI.

### API differences from assumed structure
- **Our assumption:** `agent._function_tools` dict
- **Reality:** `agent._function_toolset` → `_AgentFunctionToolset` with `.tools` dict
- **Our assumption:** `TestModel(custom_result_text=...)`
- **Reality:** `TestModel(custom_output_text=...)`
- **Impact:** API names differ but the patterns are the same. Tests needed updating for correct attribute names.

### RunContext as sidecar — CONFIRMED VIABLE
- **Confirmed:** `RunContext[Deps]` carries arbitrary deps alongside tool execution
- **`@agent.tool` (with context)** receives `RunContext` — can access carrier via deps
- **`@agent.tool_plain` (no context)** does NOT receive RunContext — no sidecar access
- **Impact:** The sidecar mitigation works, but requires using `@agent.tool` instead of `@agent.tool_plain`

### Tool object structure
- Tool type: `pydantic_ai.tools.Tool`
- FunctionSchema type: `pydantic_ai._function_schema.FunctionSchema`
- Validator type: `pydantic.plugin._schema_validator.PluggableSchemaValidator`
- Schema has `additionalProperties: false` — enforced at runtime

## Context Propagation

### Baggage PROPAGATES into tool functions
- **Our research classification:** Expected "Propagates" for in-process tool execution
- **Confirmed:** Baggage set before `agent.run()` is visible in BOTH `@tool_plain` and `@tool` functions
- **Both baggage AND deps accessible** in the same `@tool` function — no conflict
- **Impact:** PydanticAI is in the "Propagates" category for baggage — no manual propagation needed

### BaggageSpanProcessor works for caller-side spans
- **Confirmed:** Spans created in the caller context carry baggage attributes via BaggageSpanProcessor.

## Reclassifications

| Aspect | Research classification | Actual behavior |
|--------|----------------------|-----------------|
| Tool parameter extras | Hard reject (assumed) | Hard reject (CONFIRMED — PluggableSchemaValidator) |
| RunContext sidecar | Viable (assumed) | CONFIRMED viable |
| Baggage propagation | Expected propagates | CONFIRMED propagates |

## Key insight: AutoGen vs PydanticAI

This is the most important finding across all framework tests so far:

| Framework | Schema | Runtime enforcement | Classification |
|-----------|--------|-------------------|----------------|
| AutoGen | `additionalProperties: false` | `extra="ignore"` (empty model_config) | **Silent strip** |
| PydanticAI | `additionalProperties: false` | `extra_forbidden` (PluggableSchemaValidator) | **Hard reject** |

Same schema declaration, opposite runtime behavior. You cannot determine a framework's behavior from its schema alone.

## Tool Execution Entry Points

All paths converge on the same `PluggableSchemaValidator` — extras are rejected before reaching the function.

### Path 1: `tool_manager._validate_args()` — the central validation entry point
- Called by `agent.run()` when processing LLM tool call responses
- Calls `validator.validate_json()` for string args or `validator.validate_python()` for dict args
- Both paths enforce `extra_forbidden` — carrier is rejected here
- **Result:** HARD REJECT — `ValidationError` with `extra_forbidden`

### Path 2: `FunctionSchema.call()` → `_call_args()` — function invocation
- `call()` passes validated args to the function via `_call_args()`
- Since validation already rejected extras, the function never sees unknown fields
- **This is different from Haystack** where extras survive to the function call and are rejected by Python's function signature
- **Result:** Extras never reach this layer — rejected upstream at validator

### Path 3: `FunctionToolset.call_tool()` → `tool.call_func()` — toolset orchestration
- `call_tool()` delegates to `tool.call_func()` which calls `FunctionSchema.call()`
- Same validation path — extras rejected before function invocation
- **Result:** Same HARD REJECT

### Comparison: where extras are rejected

| Framework | Rejection point | Mechanism |
|-----------|----------------|-----------|
| AutoGen | Pydantic `model_validate()` | Silent strip (`extra="ignore"`) |
| Haystack | Python function signature | `TypeError` for unknown kwargs |
| PydanticAI | `PluggableSchemaValidator` | `ValidationError` (`extra_forbidden`) |

All three reject extras, but at different layers with different behaviors:
- AutoGen: silently continues (most dangerous)
- Haystack: crashes at function call (late, but visible)
- PydanticAI: crashes at validation (early, fail-fast)

## Action items
- [x] Update `cross-library-demo/test_payload_traceparent.py::test_pydantic_ai_tool_call_envelope` — confirmed hard reject, updated to use PluggableSchemaValidator terminology
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_pydantic_ai_tool_dispatch` docstring — baggage propagates confirmed
- [x] Update `ISSUE_CAUSALITY.md` — PydanticAI confirmed as hard reject with RunContext sidecar mitigation
- [x] Update `ISSUE_GROUPING.md` — PydanticAI baggage propagates (added to propagates category)
- [x] Add AutoGen vs PydanticAI comparison note to ISSUE_CAUSALITY.md — same schema, different enforcement
