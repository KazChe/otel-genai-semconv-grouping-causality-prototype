# Google ADK — Integration Test Discoveries

Findings from running real Google ADK imports vs our simulated tests.

## Envelope Shape

### FunctionTool is a plain Python class (not Pydantic, not dataclass)
- **Our assumption:** Strict, Protobuf-heritage schemas
- **Reality:** `FunctionTool` extends `BaseTool → ABC → object`. No Pydantic, no dataclass.
- **Impact:** No Pydantic-level `extra="allow"/"forbid"` on the tool itself. Validation is custom.

### run_async() takes structured args, not **kwargs
- **Key discovery:** `run_async(*, args: dict[str, Any], tool_context: ToolContext)` — args are passed as a dict, not as keyword arguments
- **The dict goes through explicit filtering:**
  ```python
  args_to_call = {k: v for k, v in args_to_call.items() if k in valid_params}
  ```
  This filters args to only declared function parameters.
- **Classification: SILENT STRIP** — extras in the args dict are silently removed before function invocation
- **Same family as AutoGen and CrewAI** but at a different layer — ADK's `run_async` does the filtering, not Pydantic

### ToolContext is a rich sidecar
- **Discovery:** `ToolContext` (aliased from `google.adk.tools.ToolContext`) is a substantial object with:
  - `state` — session state dict
  - `session` — session object
  - `actions` — action queue
  - `save_artifact` / `load_artifact` — artifact storage
  - `search_memory` / `add_memory` — memory management
  - `function_call_id` — correlation ID for the tool call
  - `invocation_id` — unique invocation identifier
- **Impact:** ToolContext is the natural sidecar for carrier injection. Tools that accept `tool_context` parameter can access carrier from `tool_context.state` or similar. No need to inject into args dict.

### custom_metadata field exists but is None by default
- **Discovery:** `FunctionTool.custom_metadata` is an attribute (defaults to `None`)
- **Impact:** Could potentially be used for carrier, but ToolContext.state is a richer option

### No additionalProperties in schema
- **Reality:** ADK generates its own tool declarations (not JSON Schema). Schema structure is Google-specific (proto-like).

## Tool Execution Entry Points

### Path 1: `run_async(args=dict, tool_context=ToolContext)` — primary entry point
- Takes args as a dict + ToolContext as separate parameter
- `_preprocess_args()` converts Pydantic models if needed
- Then filters: `{k: v for k, v in args_to_call.items() if k in valid_params}`
- **Result:** Extras SILENTLY STRIPPED — filtered to declared params only

### Path 2: `_invoke_callable(target, args_to_call)` — internal invocation
- Called by run_async after filtering
- Passes filtered args to the actual function
- **Result:** Extras already gone at this point

### Path 3: ToolContext parameter injection
- If function declares a `tool_context` parameter, ADK injects ToolContext automatically
- ToolContext carries state, session, artifacts, memory — rich sidecar
- **Result:** Carrier can ride in ToolContext.state — not in args

### Comparison: where extras are rejected

| Framework | Filtering mechanism | Where | Classification |
|-----------|-------------------|-------|----------------|
| Google ADK | `{k: v for k, v in args.items() if k in valid_params}` | `run_async()` | Silent strip |
| AutoGen | `model_validate()` with `extra="ignore"` | `run_json()` | Silent strip |
| CrewAI | Internal filtering between `run()` and `_run()` | `run()` | Silent strip |
| Haystack | Python function signature | `tool.invoke()` | Hard reject (TypeError) |
| PydanticAI | `PluggableSchemaValidator` | `validate_python()` | Hard reject (ValidationError) |
| LlamaIndex | Python function signature | `tool.call()` | Hard reject (TypeError) |

## Context Propagation

### Baggage PROPAGATES in direct function calls
- **Confirmed:** Baggage set before direct function call is visible inside the function
- **Note:** `run_async()` dispatch not tested for baggage — it's async and may use different context

## Reclassifications

| Aspect | Research assumption | Actual behavior |
|--------|-------------------|-----------------|
| Tool type | Strict Protobuf-heritage | Plain Python class (BaseTool → ABC) |
| Extra field handling | Assumed strict reject | Silent strip (filtered to valid_params) |
| Sidecar | No metadata slot found | ToolContext is a rich sidecar (state, session, artifacts) |
| Schema format | JSON Schema | Google-specific declaration (proto-like) |

## Action items
- [x] Update `cross-library-demo/test_payload_traceparent.py::test_google_adk_tool_call_envelope` — reclassify from assumed strict to silent strip, note ToolContext sidecar
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_google_adk_agent_dispatch` docstring
- [x] Update `ISSUE_CAUSALITY.md` — ADK silently strips via valid_params filter, ToolContext as sidecar
- [x] Update `ISSUE_GROUPING.md` — ADK baggage propagation status
