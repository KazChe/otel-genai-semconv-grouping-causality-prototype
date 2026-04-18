**This repo is the research and prototype evidence base for two related GenAI semconv proposals.**

That framing matches where you are now much better. Your current issue drafts are clearly split into:

* grouping representation plus interoperability guidance 
* causal span linking plus sidecar/out-of-band guidance 

## How I would go about redoing it

I would change the README from a demo-first artifact into a **proposal-support artifact** with this structure:

1. **What this repo is**
2. **What changed from the original prototype**
3. **The two proposal tracks**
4. **What is actually tested**
5. **What is demonstrated vs what is only analyzed**
6. **Repo layout**
7. **How to run specific evidence paths**
8. **Key findings**
9. **How this repo supports the two issue drafts**

That way, someone like Cirilla-zmh can skim it and quickly understand:

* what question you were trying to answer
* what was disproven
* what survived
* where the real evidence is

## The main tone shift

Old README tone:

* "here are two primitives"
* "before/after"
* "this is the convention"

New README tone:

* "this repo investigates whether these patterns hold across frameworks"
* "some original ideas were refuted"
* "the outcome is two narrower proposals"
* "the repo should be read as evidence, not as the spec itself"

That will make the repo feel much more credible.

---

## Draft README

You can use this as a starting point.

---

# OTel GenAI Semantic Conventions — Grouping & Causality Research Prototype

This repository contains the research, prototype code, and framework tests that support two related OpenTelemetry GenAI semantic convention proposals:

* **Generic grouping attributes for agentic workflow spans**
* **Causal span linking for LLM-triggered tool execution**

These proposals were originally explored together, but are now split into two narrower issue drafts because the problems are related but distinct:

* **Grouping** answers: "which spans belong to the same logical unit?"
* **Causality** answers: "which LLM decision triggered which tool execution?"

This repo is best read as an **evidence base and prototype harness**, not as the spec itself.

## Current status

The original prototype started from a simpler model:

* grouping via a single `gen_ai.group.id`
* causality via payload-level `traceparent` injection into tool call data

Cross-framework testing showed that the reality is more nuanced:

* the grouping problem and the causality problem should be treated separately
* baggage-based grouping is promising, but continuity depends on execution boundaries
* injecting `traceparent` directly into tool-call arguments is **not** a reliable general-purpose convention across frameworks
* framework-native sidecars and out-of-band correlation are more realistic causal-linking patterns than modifying tool-call arguments directly

As a result, the proposal evolved into two separate drafts.

## What this repo is for

This repo exists to answer four practical questions:

1. Can grouping information propagate across spans created by different libraries?
2. What execution boundaries preserve baggage automatically, and which do not?
3. Can causal parent context safely ride inside tool-call payloads?
4. If not, what alternative carrier patterns are viable across real frameworks?

## The two proposal tracks

### 1. Grouping

The grouping proposal explores how spans can be grouped into logical units such as rounds, skills, tasks, or phases without inventing wrapper spans for every agentic pattern.

Current direction:

* represent grouping as attributes under a `gen_ai.group.*` namespace
* use W3C Baggage as the recommended transport
* treat execution-boundary behavior as interoperability guidance, not semantic definition 

### 2. Causality

The causality proposal explores how to represent that a specific `execute_tool` span was triggered by a specific LLM inference span.

Current direction:

* do **not** rely on injecting context into tool-call arguments as a general-purpose convention
* prefer framework-native sidecar carriers where available
* use out-of-band correlation as a fallback when no native sidecar exists 

## Key findings from this repo

### Grouping

* Baggage-based grouping can work across library boundaries and across some framework-managed spans.
* Propagation behavior is framework- and boundary-dependent.
* Sync, async, thread, and process boundaries should not be treated as equivalent.

### Causality

* Injecting context into tool-call arguments failed in most tested frameworks through either:

  * **silent strip**, or
  * **hard reject**
* Schema shape alone did not reliably predict runtime behavior.
* The carrier format itself is generally durable; the fragile part is the framework's validation/sanitization path. 

## What is tested here

### Simulated tests

The simulated tests focus on failure modes and durability questions such as:

* JSON round-trips
* multi-hop serialization
* Pydantic `extra="allow"` vs `extra="forbid"`
* schema sanitization
* MessagePack round-trips
* sidecar and out-of-band mitigations

These tests are useful for reasoning about the space, but they were **not sufficient on their own** to predict real framework behavior.

### Integration tests

The framework directories contain real imports and targeted tests for:

* envelope/tool-call shape
* unknown-field behavior
* context/baggage propagation behavior

Frameworks currently covered include:

* AutoGen
* Haystack
* PydanticAI
* LlamaIndex
* CrewAI
* Google ADK

These integration tests were essential because several assumptions that looked reasonable in simulated tests did **not** hold in real frameworks. 

## What is demonstrated vs analyzed

This repo contains both runnable demos and broader research notes.

### Runnable demos

These are primarily useful for showing visual trace effects and proving same-process or cross-library patterns:

* `langgraph-demo/` — grouping and causality demonstrations in a controlled environment
* `cross-library-demo/` — cross-library behavior and why grouping and causality diverge

### Framework evidence

The `frameworks/` directory is the most important part of the repo for proposal review. It contains:

* targeted tests
* per-framework findings
* evidence backing the proposal claims

## Repo layout

```text
langgraph-demo/        # Same-process demonstration environment + grouping tests
cross-library-demo/    # Cross-library demonstration environment + causality tests
frameworks/            # Integration tests and findings by framework
research-notes.md      # Consolidated research notes and interpretation
ISSUE_GROUPING.md      # Current grouping proposal draft
ISSUE_CAUSALITY.md     # Current causality proposal draft
```

## How to read this repo

If you are reviewing the proposal direction, start here:

1. `ISSUE_GROUPING.md`
2. `ISSUE_CAUSALITY.md`
3. `research-notes.md`
4. `frameworks/*/DISCOVERIES.md`
5. the corresponding `test_envelope_shape.py` and `test_context_propagation.py` files

If you want to run the demos, then use the demo folders afterward.

## Running the tests

Tests use `InMemorySpanExporter` — no Docker needed.

### Simulated tests

```bash
# Causality — payload traceparent resilience (20 tests)
cd cross-library-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest test_payload_traceparent.py -v

# Grouping — overlapping membership + baggage propagation
cd langgraph-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest test_overlapping_groups.py -v
```

### Framework integration tests

Run these selectively by framework. Each has its own venv to avoid dependency conflicts:

```bash
cd frameworks/haystack
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -v -s

# Same pattern for: autogen, pydantic-ai, llamaindex, crewai, google-adk
```

## Running the demos

Demos send traces to an OTLP collector for visualization. Start infrastructure first:

```bash
docker compose up -d

# Run the current demo (grouping + causality + delegation + skill transitions)
cd langgraph-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent_overlapping_groups.py
```

Aspire dashboard: http://localhost:18888
Jaeger UI: http://localhost:16686

The demo prints a self-documenting span summary to stdout — you can see the attributes without needing a trace viewer.

## Important caveat

The code and examples in this repo should not be read as the final proposed semantic convention shape.

Some names and patterns in the prototype reflect intermediate exploration rather than settled proposal language. The issue drafts are the current source of truth for proposal framing.

## Related drafts

* `ISSUE_GROUPING.md`- generic grouping attributes for agentic workflow spans
* `ISSUE_CAUSALITY.md` — causal span linking for LLM-triggered tool execution
