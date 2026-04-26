# CrewAI — Integration Test Discoveries

Findings from running real CrewAI imports vs our simulated tests.

## Envelope Shape

### BaseTool is a Pydantic BaseModel (+ ABC)
- **Our assumption:** Research mentioned Pydantic `args_schema` on `BaseTool` subclasses
- **Reality:** `BaseTool` extends `BaseModel` and `ABC`, with `model_config = {'arbitrary_types_allowed': True}`
- **MRO:** `BaseTool → BaseModel → ABC → object`
- **Impact:** Same Pydantic foundation as PydanticAI's tool handling, but different validation behavior

### args_schema extras: SILENT STRIP
- **Our research classification:** "Varies: user-authored models decide"
- **Reality:** Default `args_schema` has empty `model_config` — Pydantic defaults to `extra="ignore"`, silently dropping extras
- **Same as AutoGen** — extras are accepted at model construction but silently discarded
- **Impact:** Carrier injected into args_schema fields is silently lost

### run() accepts extras but strips them before _run()
- **Key discovery:** `run(*args, **kwargs)` has a flexible signature that accepts anything
- **But:** extras do NOT reach `_run()` — CrewAI's `run()` method filters/validates arguments before calling `_run()`
- **Even with `**kwargs` in `_run()`:** the carrier is stripped between `run()` and `_run()`
- **Impact:** There is NO path for carrier injection through the tool execution pipeline

### No additionalProperties in schema
- **Reality:** The JSON schema generated from `args_schema` does not include `additionalProperties`
- **Same as LlamaIndex** — no explicit position on extras at the schema level

## Tool Execution Entry Points

Four entry points found: `run`, `_run`, `arun`, `_arun`

### Path 1: `tool.run(**kwargs)` — primary entry point
- Accepts `*args, **kwargs` — flexible signature
- But internally validates/filters args before calling `_run()`
- **Result:** Extras accepted by `run()` but silently stripped before reaching `_run()`

### Path 2: `tool._run(**kwargs)` — internal implementation
- Tool author defines this method with their parameter signature
- If called directly (bypassing `run()`), follows Python function signature rules
- **Result:** Same as Haystack/LlamaIndex — hard reject (TypeError) unless function uses `**kwargs`

### Path 3: `tool.arun()` — async entry point
- Not yet tested with extras
- Likely same filtering behavior as `run()`

### Summary: CrewAI has a hidden filtering layer

| Layer | What happens to extras |
|-------|----------------------|
| `run(**kwargs)` | Accepted (flexible signature) |
| Internal filtering | **Silently stripped** |
| `_run()` | Never sees extras |
| `args_schema` validation | Silent strip (`extra="ignore"`) |

This is the most deceptive framework tested — `run()` accepts extras without error, but they vanish before reaching the actual tool function.

## Context Propagation

### Baggage PROPAGATES in direct tool.run()
- **Confirmed:** Baggage set before `tool.run()` is visible inside `_run()`
- **Classification:** Propagates for direct sync calls

### Baggage PROPAGATES through kickoff_async / akickoff dispatch boundaries
- **Confirmed:** Baggage set in caller context is visible inside the worker
  callable when dispatched via:
  - `asyncio.to_thread(...)` (the mechanism `Crew.kickoff_async()` uses
    internally) — Python copies the current contextvars snapshot into the
    worker thread (PEP 3156)
  - native async coroutine (the path `Crew.akickoff()` uses) — same task
    lineage as the caller, no context handoff needed
- **Classification:** Propagates for both async entrypoints at the
  dispatch-boundary level
- **What was tested:** the dispatch boundary mechanism, not a full Crew
  run with an LLM. Both `Crew.kickoff_async()` and `Crew.akickoff()`
  ultimately drive tool execution through the boundary mechanisms above,
  so this is the determining factor for whether caller baggage flows
  into tool execution
- **Residual gap:** full LLM-driven Crew kickoff still not exercised end
  to end. Tool-side propagation works; whether CrewAI's own
  agent/task/crew layer mutates baggage between rounds is a separate
  question

## Reclassifications

| Aspect | Research classification | Actual behavior |
|--------|----------------------|-----------------|
| BaseTool type | Pydantic (assumed) | Pydantic BaseModel + ABC (CONFIRMED) |
| args_schema extras | Varies / Unknown | Silent strip (empty model_config, extra="ignore") |
| run() with extras | Unknown | Accepts but silently strips before _run() |
| Schema additionalProperties | Unknown | Not declared (None) |
| Baggage (sync) | Requires manual propagation | PROPAGATES in direct run() |
| Baggage (kickoff_async dispatch boundary) | Requires manual propagation | PROPAGATES through asyncio.to_thread() |
| Baggage (akickoff dispatch boundary) | Requires manual propagation | PROPAGATES through native async |

## Action items
- [x] Update `langgraph-demo/test_overlapping_groups.py::test_crewai_to_thread_dispatch` docstring — sync propagates, kickoff TBD
- [x] Update `ISSUE_GROUPING.md` — CrewAI sync propagates
- [x] Update `ISSUE_CAUSALITY.md` — CrewAI run() silently strips extras before _run()
- [x] Test kickoff_async / akickoff dispatch boundaries (both PROPAGATE)
- [ ] Full LLM-driven Crew kickoff end-to-end test (residual gap)
