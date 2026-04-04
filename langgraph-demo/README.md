# LangGraph Demo — After Conventions

Same agent as baseline, but with **Grouping** via W3C Baggage and `BaggageSpanProcessor`.

## What this demo proves about Grouping

- `gen_ai.group.id` set in Baggage at each ReAct round boundary
- `BaggageSpanProcessor` (official package) automatically copies baggage to span attributes
- All spans in a round carry `gen_ai.group.id=round-N` — no wrapper spans needed
- Multi-dimensional membership works: a span carries both `gen_ai.group.id` and `gen_ai.group.type` simultaneously
- `agent_multidim.py` demonstrates multi-dimensional nesting — a single span belongs to a round, an agent, AND a phase (4 dimensions) simultaneously

## What this demo proves about Causality

`agent_causality.py` injects `traceparent` into tool call payloads so `execute_tool` spans become **children** of the `chat` spans that triggered them — not siblings.

- After LLM returns `tool_calls`, the current span's context is injected into the tool call payload via `inject(carrier)` → `tool_call["_otel"] = carrier`
- The tool executor extracts it via `extract(tool_call["_otel"])` and uses it as the parent context
- Same pattern as HTTP `traceparent` headers, applied to LLM message envelopes
- Works regardless of which library created each span

## Run

```bash
# From repo root — start infra first
docker compose up -d

# Reuse baseline venv (same deps + baggage processor)
cd langgraph-demo
source ../baseline/.venv/bin/activate
pip install opentelemetry-processor-baggage

# Run
python agent.py
```

Check Aspire at http://localhost:18888 — compare with the baseline trace. Spans now carry `gen_ai.group.id` and `gen_ai.group.type` attributes.

## Screenshots

### Single-dimension grouping (before / after)

![Grouping before and after](../screenshots/grouping-before-after-aspire.png)

Left: `gen_ai.group.id=round-1` and `gen_ai.group.type=react_iteration` present. Right (baseline): no grouping attributes.

<!-- ### Multi-dimensional grouping — chat span (before / after)

![Multi-dim chat span](../screenshots/multidim-chat-span-before-after.png)

Left: `chat` span carries 4 dimensions — `gen_ai.agent.id=main-agent`, `gen_ai.group.id=round-1`, `gen_ai.group.type=react_iteration`, `gen_ai.phase=reasoning`. Right (baseline): only standard GenAI attributes.

### Multi-dimensional grouping — execute_tool span (before / after)

![Multi-dim execute_tool span](../screenshots/multidim-execute-tool-before-after.png)

Left: `execute_tool` carries same round and agent, but `gen_ai.phase=execution` (not `reasoning`). Same round, same agent, different phase — multi-dimensional membership proven. Right (baseline): no grouping. -->

### Causality — execute_tool parented to chat

![Causality demo](../screenshots/langgraph-causality-demo.png)

**Left:** Payload `traceparent` injection worked — `execute_tool` spans are **nested under** the `chat` spans that triggered them. The LLM call injects its span context into the tool call payload via `inject(carrier)` → `tool_call["_otel"] = carrier`. The tool executor extracts it via `extract(tool_call["_otel"])` and uses it as the parent context. This establishes a causal parent-child link in the trace tree. **Right (baseline):** `chat` and `execute_tool` are flat siblings with no causal signal.

> Note: If you see a `gen_ai.causality` attribute on spans in earlier screenshots, this is a **debug label only** — not a proposed semantic convention. The causality is proven by the parent-child relationship in the trace tree itself.
