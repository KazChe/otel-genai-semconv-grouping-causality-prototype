# Baseline — Before Conventions

This is the **before** state. A multi-round ReAct agent with standard OpenTelemetry instrumentation — no grouping, no causality conventions.

## What this shows about Grouping

**Nothing.** Spans are flat. The backend sees `chat`, `execute_tool`, `chat`, `execute_tool`, `chat` — with no signal for which spans belong to round 1 vs round 2. Grouping must be inferred from timestamps, which is brittle and breaks with retries or concurrent execution.

## What this shows about Causality

**Nothing.** Tool execution spans are siblings of LLM call spans, not children. There is no signal indicating which LLM call triggered which tool execution. The backend cannot trace the decision chain.

## Run

```bash
# From repo root — start infra first
docker compose up -d

# Install deps
cd baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
python agent.py
```

Check Aspire at http://localhost:18888 — you'll see a flat span list.

## Screenshot (Aspire — Before)

![Baseline trace in Aspire — flat spans, no grouping, no causality](../screenshots/baseline-aspire-before.png)

Notice: `chat` and `execute_tool` spans are flat at the bottom with no round grouping and no causal links to show which LLM call triggered which tool execution.
