# OTel GenAI Semantic Conventions — Grouping & Causality Prototype

Prototype for [open-telemetry/semantic-conventions#3575](https://github.com/open-telemetry/semantic-conventions/issues/3575).

This repo demonstrates two proposed primitives for OpenTelemetry GenAI semantic conventions:

- **Grouping** — `gen_ai.group.id` carried in W3C Baggage, copied to span attributes via `BaggageSpanProcessor`. Allows spans to be grouped into logical units (ReAct rounds, tasks, steps) without creating wrapper spans.
- **Causality** — Payload-level `traceparent` injection into LLM tool call metadata. Establishes causal links ("this tool was triggered by that LLM call") across library and process boundaries.

## Before / After

|               | Before (baseline)                            | After (with conventions)                                |
| ------------- | -------------------------------------------- | ------------------------------------------------------- |
| **Grouping**  | Flat spans — backend guesses from timestamps | Each span tagged with `gen_ai.group.id=round-N`         |
| **Causality** | Tool spans are siblings of LLM spans         | Tool spans parented to the LLM call that triggered them |

### Before (Aspire) — Baseline

![Baseline — flat spans](screenshots/baseline-aspire-before.png)

No `gen_ai.group.id`, no causal links. Backend guesses from timestamps.

### After (Aspire) — Single-dimension Grouping

![Grouping before/after](screenshots/grouping-before-after-aspire.png)

Spans now carry `gen_ai.group.id=round-1` and `gen_ai.group.type=react_iteration`.

<!-- ### After (Aspire) — Multi-dimensional Grouping

![Multi-dim chat span](screenshots/multidim-chat-span-before-after.png)

A single `chat` span belongs to 4 dimensions simultaneously: round, round type, agent, and phase.

![Multi-dim execute_tool span](screenshots/multidim-execute-tool-before-after.png)

`execute_tool` span: same round and agent as `chat`, but `gen_ai.phase=execution` instead of `reasoning`. -->

### After (Aspire) — Causality via Payload Traceparent

![Causality demo](screenshots/langgraph-causality-demo.png)

**Left (causality demo):** `execute_tool` spans are **nested under** the `chat` spans that triggered them — a parent-child relationship established via payload `traceparent` injection. The LLM call's span context was injected into the tool call payload (`tool_call["_otel"] = carrier`), and the tool executor extracted it to set the parent (`extract(tool_call["_otel"])`). **Right (baseline):** `chat` and `execute_tool` are flat siblings — no causal link, backend cannot tell which LLM call triggered which tool.

### Cross-Library (LangChain + LiteLLM) — Grouping works, Causality needs payload traceparent

![Cross-library completion span](screenshots/cross-library-demo-completion-grouping-needs-payload-traceparent.png)

LiteLLM's `completion` span carries `gen_ai.group.id=round-1` — **grouping works across library boundaries**. LiteLLM knows nothing about our convention; `BaggageSpanProcessor` copied it automatically.

![Cross-library execute_tool span](screenshots/cross-library-demo-executetool-grouping-needs-payload-traceparent.png)

`execute_tool` is a flat sibling of `completion`, not a child — **causality requires payload traceparent injection** when spans are created by different libraries with different lifecycles. This is the evidence that the convention must specify traceparent propagation at the message/payload level.

### Cross-Library WITH Causality — `agent_with_causality.py`

![Cross-library causality chat span](screenshots/cross-library-causality-demo-chat.png)

After adding payload traceparent injection: trace goes from Depth 2 (flat) to **Depth 3 (nested)**. The `chat` span wraps both LiteLLM's `completion` and `execute_tool` as children.

![Cross-library causality completion](screenshots/cross-library-causality-demo-grouping-with-causality-completion.png)

LiteLLM's `completion` span — created by LiteLLM's instrumentor, not our code — carries `gen_ai.group.id=round-1` (grouping) AND is nested under `chat` (causality). **Both primitives working across library boundaries.**

![Cross-library causality execute_tool](screenshots/cross-library-causality-demo-grouping-with-causality-executetool.png)

`execute_tool` is also a child of `chat`, with `gen_ai.group.id=round-1`. Cross-library grouping + causality — **working.**

> Note: If you see a `gen_ai.causality` attribute on spans in earlier screenshots, this is a **debug label only** — not a proposed semantic convention. The causality is proven by the parent-child relationship in the trace tree itself.

## Demos

| Demo                                       | Framework           | Purpose                                                                        |
| ------------------------------------------ | ------------------- | ------------------------------------------------------------------------------ |
| [baseline/](baseline/)                     | LangGraph           | **Before** — flat spans, no conventions                                        |
| [langgraph-demo/](langgraph-demo/)         | LangGraph           | **After** — Baggage grouping + payload traceparent                             |
| [cross-library-demo/](cross-library-demo/) | LangChain + LiteLLM | Cross-library span linking — grouping works, causality via payload traceparent |
| [autogen-demo/](autogen-demo/)             | AutoGen v0.4        | Adversarial validation — Baggage survives async event-driven runtime           |

### AutoGen — Baggage Survives Async Dispatch

![Baggage survives AutoGen async dispatch](screenshots/does-baggage-survive-autogen-async-dispatch.png)

AutoGen's internally created `autogen process worker` span carries `gen_ai.group.id=autogen-session-1`, `gen_ai.group.type=agent_collaboration`, and `gen_ai.agent.id=planner-worker-team`. We did not create this span, AutoGen's runtime did. The `BaggageSpanProcessor` copied our Baggage to it automatically, across async message passing between the planner and worker agents. AutoGen knows nothing about these attributes, it just created its autogen process worker span as usual, and our processor attached the grouping data automatically. We proved that baggage-based grouping can survive an async, event-driven agent runtime and be applied to spans created internally by the framework itselfnot only spans created by application code - Note: this does not prove yet that every async framework preserves baggage the same way that causality is fully modeled, only that AutoGen’s built-in hierarchy appears intact in this case

## Backends

- **Aspire** (open source) anyone can reproduce, shows raw span data with attributes
- **Aegis** (open source) for another UX - I felt it has better querying capabilities
- **Galileo** still work in progress (production AI observability) — will show what an observability backend can build with these conventions

## Quick Start

```bash
# Start infrastructure (Collector + Aspire + Jaeger)
docker compose up -d

# Run baseline (before — flat spans, no conventions)
cd baseline && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent.py

# Run langgraph demos (after — grouping + causality)
cd ../langgraph-demo && source ../baseline/.venv/bin/activate
pip install opentelemetry-processor-baggage
python agent.py               # grouping only
python agent_multidim.py       # multi-dimensional grouping
python agent_causality.py      # grouping + causality

# Run cross-library demos (LangChain + LiteLLM)
cd ../cross-library-demo && source ../baseline/.venv/bin/activate
pip install langchain litellm openinference-instrumentation-litellm
python agent.py                  # grouping only (shows WHY causality is needed)
python agent_with_causality.py   # grouping + causality

# Run AutoGen demo (async event-driven validation — requires OPENAI_API_KEY in .env)
cd ../autogen-demo && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent.py
```

Aspire dashboard: http://localhost:18888

## Generalizability Matrix

| Framework                     | Baggage (grouping)    | Payload traceparent (causality)     | Status   |
| ----------------------------- | --------------------- | ----------------------------------- | -------- |
| LangGraph / LlamaIndex        | Works                 | Works                               | Tested   |
| AutoGen v0.4 (async)          | Works                 | Message envelope is natural carrier | Tested   |
| CrewAI                        | Works (built-in OTel) | Task delegation payload is carrier  | Analysis |
| Cross-language (Python//.NET) | Won't cross boundary  | Only viable mechanism               | Analysis |
| MCP-based tools               | Won't cross process   | Already HTTP, traceparent native    | Analysis |

> These are not two independent ideas. They are two layers of the same contract. Within a single runtime, use Baggage. At every boundary that crosses async, language, or process lines, use payload traceparent.
