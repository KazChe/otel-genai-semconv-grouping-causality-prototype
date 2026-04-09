# LlamaIndex — Integration Test Discoveries

Findings from running real LlamaIndex imports vs our simulated tests.

## Envelope Shape

### ToolSelection is a Pydantic BaseModel (not a dataclass)
- **Our assumption:** Research mentioned Pydantic-backed `fn_schema` but we didn't confirm the ToolSelection object type
- **Reality:** `ToolSelection` is a Pydantic `BaseModel` with `model_config = {}` (empty — defaults to `extra="ignore"`)
- **Fields:** `tool_id: str`, `tool_name: str`, `tool_kwargs: Dict[str, Any]`
- **Location:** `llama_index.core.llms.llm.ToolSelection` (not re-exported from `llama_index.core.llms`)
- **Impact:** Unlike AutoGen (dataclass) and Haystack (dataclass), LlamaIndex uses Pydantic for the tool call envelope

### tool_kwargs extras SURVIVE in the dict
- **Confirmed:** Extra fields in `tool_kwargs` dict are preserved at the ToolSelection level
- **Why:** `tool_kwargs` is typed as `Dict[str, Any]` — it's a generic dict, not validated against a schema
- **Impact:** Carrier can ride inside `tool_kwargs["_otel"]` at the envelope level

### Non-dict tool_kwargs coercion to {} — CONFIRMED silent strip
- **Our simulated test assumption:** Non-dict args get coerced to `{}`
- **Confirmed:** `ToolSelection` has a `field_validator("tool_kwargs", mode="wrap")` called `ignore_non_dict_arguments` that catches `ValidationError` and returns `{}`
- **Impact:** Our simulated test in `test_non_dict_coercion_to_empty_llamaindex` was correct

### FunctionTool schema has NO additionalProperties
- **Reality:** The JSON schema generated for `FunctionTool` does not include `additionalProperties` at all (it's `None`)
- **Impact:** Unlike AutoGen (`additionalProperties: false`) and PydanticAI (`additionalProperties: false`), LlamaIndex doesn't declare a position on extras at the schema level

### FunctionTool is NOT a Pydantic model
- **Reality:** `FunctionTool` does not have `model_dump` — it's a custom class, not a Pydantic BaseModel
- **Impact:** Different serialization behavior from ToolSelection (which is Pydantic)

## Tool Execution Entry Points

### Path 1: `FunctionTool.call(**kwargs)` — sync invocation
- Passes kwargs directly to the underlying Python function
- If function doesn't accept `**kwargs`, Python raises `TypeError` for unknown arguments
- **Result:** Hard reject at function signature (same pattern as Haystack)

### Path 2: `FunctionTool.call()` with `**kwargs` function
- If tool function accepts `**kwargs`, extras pass through to the function
- Carrier IS accessible inside the function via `kwargs["_otel"]`
- **Result:** Carrier survives — requires tool function designed for it

### Path 3: `ToolSelection.tool_kwargs` — envelope level
- Extra fields in `tool_kwargs` dict survive at the ToolSelection level
- But when agent calls `tool.call(**tool_kwargs)`, extras hit the function signature boundary
- **Result:** Extras survive in envelope, rejected at function call (two-layer like AutoGen, but rejection mechanism differs — TypeError vs silent strip)

### Comparison: where extras are rejected

| Framework | Envelope | Tool execution | Rejection mechanism |
|-----------|----------|---------------|-------------------|
| LlamaIndex | Survives in `tool_kwargs` dict | Rejected at function signature | `TypeError` (hard reject) |
| AutoGen | Survives in `arguments` string/dict | Silently stripped by `model_validate()` | Silent strip |
| Haystack | Survives in `arguments` dict | Rejected at function signature | `TypeError` (hard reject) |
| PydanticAI | N/A (no separate envelope) | Rejected by `PluggableSchemaValidator` | `ValidationError` (hard reject) |

## Context Propagation

### Baggage PROPAGATES in sync FunctionTool.call()
- **Confirmed:** Baggage set before `tool.call()` is visible inside the tool function
- **Classification:** Propagates for direct sync calls

### Baggage LOST in async FunctionTool.acall()
- **Confirmed:** `baggage.get_baggage()` returns `None` inside the tool function when called via `acall()`
- **Root cause:** Likely `acall()` dispatches via `asyncio.to_thread()` or `run_in_executor()` without `copy_context()`
- **Classification:** Requires manual propagation for async calls
- **Impact:** Reclassify LlamaIndex as split — sync propagates, async requires manual propagation

### BaggageSpanProcessor works for caller-side spans
- **Confirmed:** Spans created in the caller context carry baggage attributes.

## Reclassifications

| Aspect | Research classification | Actual behavior |
|--------|----------------------|-----------------|
| ToolSelection type | Pydantic-backed (assumed) | Pydantic BaseModel (CONFIRMED) |
| tool_kwargs extras | Survive in dict (assumed) | CONFIRMED — survive at envelope level |
| Non-dict coercion | Coerced to {} (assumed) | CONFIRMED — field_validator catches and returns {} |
| FunctionTool schema | Unknown | No additionalProperties declared |
| Baggage (sync) | Requires manual propagation | PROPAGATES in sync call() |
| Baggage (async) | Requires manual propagation | CONFIRMED — lost in acall() |

## Action items
- [x] Update `cross-library-demo/test_payload_traceparent.py::test_non_dict_coercion_to_empty_llamaindex` — confirmed behavior, added ToolSelection Pydantic + field_validator details
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_llamaindex_workflow_step_dispatch` docstring — sync propagates, async lost
- [x] Update `ISSUE_GROUPING.md` — LlamaIndex split: sync propagates, async requires manual propagation
- [x] Update `ISSUE_CAUSALITY.md` — LlamaIndex extras survive in tool_kwargs but rejected at function signature
