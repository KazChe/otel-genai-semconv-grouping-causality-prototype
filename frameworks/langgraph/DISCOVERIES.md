# LangGraph — Integration Test Discoveries

Findings from running real LangGraph imports against the test conventions
established in `frameworks/`.

## Role in This Repository

LangGraph serves as the **same-process demonstrator** for both proposals.
Unlike the `frameworks/` tests (which verify behavior against opaque
third-party dispatch models), these tests verify behavior against a dispatch
model we control. LangGraph proves the mechanism works in a controlled
environment; the frameworks/ tests prove it works (or fails) against real
frameworks.

## Envelope Shape

### config["configurable"] as sidecar

- **Our assumption:** `config["configurable"]` is a `dict[str, Any]` that
  can carry the `_otel` carrier into graph nodes as a sidecar.
- **Reality:** Confirmed. The carrier survives and can establish causal
  parent-child links. However, the node function's `config` parameter
  **must** be typed as `RunnableConfig` (from `langchain_core.runnables`),
  not as `dict`. Using `dict` causes a `TypeError` because LangGraph's
  internal `_RunnableCallable` only injects config when it detects the
  correct type annotation.
- **Impact:** Minor, but important for instrumentation authors. The type
  annotation is part of the contract.

### State is TypedDict-based (not Pydantic)

- **Our assumption:** LangGraph state uses TypedDict, so there is no
  Pydantic validation layer that would strip extra fields from messages.
- **Reality:** Confirmed. `_otel` carrier injected into message dicts
  within the state survives graph execution. No silent stripping, no hard
  rejection. This is in contrast to frameworks like AutoGen (silent strip),
  PydanticAI (hard reject), and CrewAI (silent strip) where extra fields
  in tool call arguments are lost.

### Checkpoint serialization

- **Our assumption:** Carrier in state should survive MemorySaver
  checkpoint round-trips since MemorySaver stores in-memory.
- **Reality:** Confirmed. The `_otel` carrier in state messages survives
  across graph steps with checkpointing enabled. Note: this test uses
  `MemorySaver` (in-memory). `SqliteSaver` and other persistent
  checkpointers use `JsonPlusSerializer` (MessagePack-based) which has
  been tested separately in `cross-library-demo/test_payload_traceparent.py`
  and confirmed to preserve `dict[str, str]` carriers.

## Context Propagation

### Baggage through graph nodes

- **Our assumption:** Baggage set before `graph.invoke()` should be
  visible inside node functions since LangGraph executes nodes in the
  caller's context.
- **Reality:** Confirmed. Baggage propagates into all nodes including
  sequential nodes and conditional edge targets. BaggageSpanProcessor
  copies baggage entries to span attributes on manually created spans
  within nodes.

### Sequential node propagation

- **Our assumption:** Baggage should persist across sequential node
  transitions without context reset.
- **Reality:** Confirmed. Both the first and second nodes in a sequential
  graph see the same baggage values.

### Conditional edge propagation

- **Our assumption:** Baggage should survive conditional edge routing
  (the `should_continue` pattern).
- **Reality:** Confirmed. Both the router function and the target node
  see baggage. This validates the pattern used in
  `agent_overlapping_groups.py` where the ReAct loop routes between
  `llm_call` and `tool_call` nodes.

### Classification

| Propagation path | Baggage status | Notes |
|------------------|---------------|-------|
| `graph.invoke()` into nodes | Propagates | Caller context inherited |
| Sequential node transitions | Propagates | Context not reset between nodes |
| Conditional edge routing | Propagates | Router and target both see baggage |
| Manual spans within nodes | Propagates | BaggageSpanProcessor copies to attributes |

**Overall classification: Propagates (sync)**

LangGraph's sync dispatch preserves context across all tested paths.
Async dispatch (`graph.ainvoke()`) is not yet tested.

## Reclassifications

| Aspect | Previous assumption | Actual behavior | Change |
|--------|-------------------|-----------------|--------|
| config param type | `dict` works | Must be `RunnableConfig` | Documentation fix |

No reclassifications on envelope or propagation behavior. All assumptions
confirmed.

## Action items

- [x] Document `RunnableConfig` type requirement in test docstring
- [ ] Test async dispatch (`graph.ainvoke()`) for baggage propagation
- [ ] Test `SqliteSaver` checkpoint round-trip directly (currently covered
  indirectly by MessagePack tests in cross-library-demo)
