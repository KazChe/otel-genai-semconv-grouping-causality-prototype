# LangGraph Demo — Same-Process Demonstrator

LangGraph serves as the **same-process demonstrator** for both the grouping and
causality proposals. Unlike the other frameworks in this directory (which verify
behavior against opaque third-party dispatch models), LangGraph proves the
mechanisms work in a controlled environment where the dispatch model is under
our control.

## What this directory contains

### Demo scripts

| Script | What it demonstrates |
|--------|---------------------|
| `agent.py` | Original single-key grouping (historical, kept for reference) |
| `agent_overlapping_groups.py` | Full demo: namespaced baggage keys, payload traceparent causality, agent delegation, mid-round skill transitions |

### Tests

| File | What it tests |
|------|--------------|
| `test_envelope_shape.py` | config["configurable"] as sidecar, carrier in TypedDict state, checkpoint round-trip |
| `test_context_propagation.py` | Baggage through graph nodes, sequential nodes, conditional edges, manual spans |
| `DISCOVERIES.md` | Findings from integration tests, reclassifications |

Pure OTel mechanism tests (overlapping group membership, propagation boundaries,
mitigations) live in `frameworks/otel-only/test_grouping_mechanism.py`.

## What this demo proves

### Grouping

- Namespaced baggage keys (`gen_ai.group.id`, `gen_ai.group.iteration.type`,
  `gen_ai.group.skill.id`, `gen_ai.group.skill.type`) enable overlapping
  group membership
- `BaggageSpanProcessor` automatically copies baggage to span attributes
- Baggage propagates through LangGraph graph nodes, conditional edges, and
  checkpoint round-trips

### Causality

- `config["configurable"]` works as a native sidecar for carrier injection
  (analogous to Haystack's `ToolCall.extra` or PydanticAI's `RunContext.deps`)
- Carrier in TypedDict state survives graph execution (no Pydantic stripping)
- Payload traceparent injection makes `execute_tool` spans children of the
  `chat` spans that triggered them

## Run tests

```bash
cd frameworks/langgraph
# Create venv and install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run all tests
.venv/bin/python -m pytest test_envelope_shape.py test_context_propagation.py -v
```

## Run demos

```bash
# Start infra (Aspire dashboard)
docker compose up -d

cd frameworks/langgraph
source .venv/bin/activate

python agent_overlapping_groups.py    # full demo with self-documenting output
```

Check Aspire at http://localhost:18888.
