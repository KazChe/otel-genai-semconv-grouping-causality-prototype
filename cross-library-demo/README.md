# Cross-Library Demo — LangChain + LiteLLM

> **Note:** This README reflects the original demo state. The demos and screenshots remain valid, they prove the core concepts. However, integration testing across 6 frameworks has since shown that injecting the carrier into tool call arguments (`tool_call["_otel"]`) fails in 5/6 frameworks (silent strip or hard reject). The recommended approach is now sidecar propagation via framework-native extension points. This directory also now contains `test_payload_traceparent.py` — 20 automated tests mapping the carrier compatibility matrix across serialization paths and framework envelope patterns. See `ISSUE_CAUSALITY.md` in the repo root for the updated proposal.

Demonstrates grouping and causality across two independent instrumentation libraries — LangChain and LiteLLM — neither of which knows about the other's spans. Both instrumented on the same `TracerProvider`.

## What this demo proves about Grouping

- Both LangChain's orchestration spans and LiteLLM's inference spans carry `gen_ai.group.id` from Baggage
- `BaggageSpanProcessor` copies baggage to ALL spans regardless of which library created them
- Grouping works across library boundaries without either library needing to know about the convention

## What this demo proves about Causality

**Finding:** In the cross-library case, `execute_tool` and `completion` spans are **flat siblings** — not parent-child. This is because LiteLLM's `completion` span ends when `litellm.completion()` returns, before tool execution begins. In-process execution context alone cannot link them.

**This is exactly why payload `traceparent` injection is needed.** When the LLM call and the tool execution are handled by different libraries with different span lifecycles, the only way to establish a causal link is to carry the span context in the tool call payload — just like HTTP `traceparent` carries context across service boundaries.

The LangGraph causality demo (`langgraph-demo/agent_causality.py`) proves that payload injection works when both sides are in the same library. This cross-library demo shows where it's needed most — and why the convention must be specified at the protocol level, not left to individual libraries.

## Run

```bash
# From repo root — start infra first
docker compose up -d

# Reuse baseline venv
cd cross-library-demo
source ../baseline/.venv/bin/activate
pip install langchain litellm openinference-instrumentation-litellm

# Run each scenario separately
python agent.py                  # grouping only (flat siblings — shows WHY causality is needed)
python agent_with_causality.py   # grouping + causality (nested spans)
```

Check Aspire at http://localhost:18888 — each script creates a trace with a distinct service name.

## Scripts

| Script                    | What it demonstrates                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `agent.py`                | Grouping works across LangChain + LiteLLM, but spans are flat siblings — shows why payload traceparent is needed |
| `agent_with_causality.py` | Grouping + causality — LiteLLM's `completion` and `execute_tool` both nested under `chat`                        |

## Screenshots

### Grouping works across libraries — completion span (LiteLLM)

![Cross-library completion span with grouping](../screenshots/cross-library-demo-completion-grouping-needs-payload-traceparent.png)

LiteLLM's `completion` span carries `gen_ai.group.id=round-1` and `gen_ai.group.type=react_iteration` — even though LiteLLM knows nothing about our grouping convention. `BaggageSpanProcessor` copied the baggage automatically. Also visible: LiteLLM's own attributes (`llm.model_name`, `llm.input_messages`, `llm.invocation_parameters`) alongside the grouping attributes. **Grouping across library boundaries: proven.**

### Grouping works across libraries — execute_tool span

![Cross-library execute_tool span with grouping](../screenshots/cross-library-demo-executetool-grouping-needs-payload-traceparent.png)

`execute_tool` span also carries `gen_ai.group.id=round-1` — same round as the `completion` span above. Both spans are tagged with the same group ID despite being created by different libraries. Note the flat trace tree (Depth 2) — `execute_tool` is a sibling of `completion`, not a child. **This is the evidence that payload `traceparent` injection is needed for causality in the cross-library case.**

### With causality — `agent_with_causality.py`

After adding payload traceparent injection, the trace tree changes from Depth 2 (flat) to Depth 3 (nested). Both `completion` (LiteLLM) and `execute_tool` are now **children** of the `chat` span.

#### chat span — parent of both completion and execute_tool

![Cross-library causality chat span](../screenshots/cross-library-causality-demo-chat.png)

The `chat` span (113.49ms) wraps both the LiteLLM inference call and the tool execution. It carries `gen_ai.group.id=round-1` and `gen_ai.response.finish_reasons=tool_calls`. The traceparent is injected from inside this span while it's still active — so both children inherit it.

#### completion span (LiteLLM) — child of chat, with grouping

![Cross-library causality completion span](../screenshots/cross-library-causality-demo-grouping-with-causality-completion.png)

LiteLLM's `completion` span (11.18ms) is nested under `chat` — it carries `gen_ai.group.id=round-1` from Baggage AND is a child span via execution context. LiteLLM created this span; our code didn't touch it. **Grouping + causality across library boundaries: proven.**

#### execute_tool span — child of chat, with grouping

![Cross-library causality execute_tool span](../screenshots/cross-library-causality-demo-grouping-with-causality-executetool.png)

`execute_tool` (102.03ms) is also a child of `chat`, carrying `gen_ai.group.id=round-1`. The causal link was established via payload traceparent injection — the tool executor extracted the chat span's context from the tool call payload. **Cross-library grouping + causality: working.**
