# [gen-ai] Add generic grouping attributes for agentic workflow spans

Split from [open-telemetry/semantic-conventions#3575](https://github.com/open-telemetry/semantic-conventions/issues/3575).
Companion issue: Causal Span Linking for GenAI Tool Calls (ISSUE_CAUSALITY.md)

### Area(s)

area:gen-ai

### What's missing?

> **Note:** This proposal introduces new attributes under the `gen_ai.group.*` namespace. This namespace does not currently exist in the GenAI semantic conventions. Existing `gen_ai.*` attributes (`gen_ai.operation.name`, `gen_ai.agent.id`, etc.) are not modified.

GenAI semantic conventions lack a standard way to group related spans into logical units (iterations, tasks, skills, phases). When a user opens a trace from an agentic workflow, they see a flat or inconsistently structured list of spans, `chat`, `execute_tool`, `chat`, `execute_tool`, `chat`, with no reliable signal for which spans belong to the same logical unit.

Debugging "why did my agent fail on attempt 3?" requires manually correlating spans by timestamp. Observability platforms cannot build generic iteration-level views or aggregations because the grouping signal is not in the data in any standardized form.

Every time a new agentic pattern emerges (ReAct iterations, planning phases, tool-selection rounds), the only recourse today is to propose a new span type — leading to an N+1 span type problem. This is not specific to any one framework; it affects graph-based orchestrators, async event-driven runtimes, and multi-agent delegation patterns alike.

**Why not solve this with existing operations?**

The existing GenAI semconv already defines `execute_tool`, `invoke_agent`, and `invoke_workflow` as dedicated operations. This proposal does not replace them. Grouping is a separate concern — it adds correlation structure *around* existing operations rather than replacing them:

- **Why not another wrapper span?** Wrapper spans represent real work with meaningful duration. Grouping is a tag, not a phase — "these spans belong to round 2" does not imply a span that starts and ends with round 2.
- **Why isn't `invoke_workflow` enough?** Workflow spans capture workflow-level duration and hierarchy. They don't express overlapping membership — a span can be part of an iteration AND a skill AND an agent simultaneously.
- **Why baggage rather than direct attributes?** Direct attributes require every instrumentation library to explicitly set grouping on every span. Baggage with `BaggageSpanProcessor` makes this transparent — grouping is set once and propagates automatically, even to spans created by libraries that know nothing about the convention.

**Open concerns from #3575:**

**Overlapping group membership** (@Cirilla-zmh):

> "A group may be nested, meaning that an operation can belong to groups across multiple dimensions. For example, it may belong to the 'main agent', while also being part of the second ReAct iteration."

If `gen_ai.group.type` is modeled as a single `StringAttributeKey`, its value is mutually exclusive. But in practice, groups are not always mutually exclusive — a span executing a skill during a ReAct iteration should belong to both `skill` and `react_iteration` simultaneously. A single-valued `gen_ai.group.type` does not achieve the goal of avoiding the need to define multiple span types.

**Instrumentation complexity** (@Cirilla-zmh):

> "If we do not introduce such a span and instead add a `group.id` to each of its intended child spans, the instrumentation implementation would become more difficult and complex."

Adding group attributes directly to each child span shifts the burden to instrumentation library authors, who must thread grouping context through their span creation code. The question is whether the grouping mechanism can be made transparent to instrumentors.

**Grouping vs. dedicated operations** (@Krishnachaitanyakc):

> "Where do you see the line between 'this is just a grouping concern' and 'this needs its own operation'?"

An agent planning phase illustrates the tension: the `chat` call that generates a plan is causally a child of the planning decision, not a sibling that happens to share a group tag. `execute_tool` exists as its own operation for the same reason. The question is how grouping and dedicated operations complement each other rather than replace each other.

**How does this manifest for users?** (@lmolkova):

> "How does this problem manifest for users? What is the real problem, why not having grouping is a problem?"

A multi-round ReAct agent produces a flat span list with no signal for which spans belong to which round or which LLM call triggered which tool execution. Observability platforms cannot build round-level or phase-level views because there is no standardized grouping signal in the trace data.

### Describe the solution you'd like

This proposal standardizes the grouping representation (namespaced attributes under `gen_ai.group.*`) and recommends W3C Baggage as the transport. It also provides interoperability guidance for maintaining grouping continuity across common execution boundaries found in LLM orchestration frameworks.

#### 1. Attribute model: namespaced grouping keys (normative)

**How we arrived at this model**

The original proposal in #3575 used a single `gen_ai.group.type` + `gen_ai.group.id` pair. During review, @Cirilla-zmh identified a fundamental flaw: a single-valued `gen_ai.group.type` is mutually exclusive, but real agent systems require overlapping membership — a span executing a skill during a ReAct iteration belongs to _both_ groups simultaneously.

We explored several alternatives:

| Alternative                                                    | Why it was rejected                                                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Single `gen_ai.group.type` string                              | Mutually exclusive — the original problem. Can't express "this span is both a skill and an iteration" |
| JSON blob attribute (ex. `{"session": "abc", "skill": "rag"}`) | Harder for telemetry backends to query and index. Not idiomatic for OTel span attributes              |
| Array of mixed IDs (ex. `["abc123", "wf-42"]`)                 | No way to know which ID means what without external schema                                            |
| Wrapper spans per group                                        | Adds artificial spans that inflate trace size without representing real work                          |

The solution is **namespaced grouping keys**: instead of overloading a single key, give each grouping concept its own distinct attribute under a shared `gen_ai.group.*` namespace.

**Why namespacing matters**

The namespace does two things:

1. **Prevents collisions** : each grouping dimension is independent. Setting `gen_ai.group.skill.id` does not overwrite `gen_ai.group.iteration.type`.
2. **Makes the category self-describing**: the key name tells you what kind of group it represents, without needing external documentation to interpret a generic ID.

Think of it like HTTP headers: `X-Session-Id`, `X-Workflow-Id`, and `X-Toolchain-Id` are cleaner than packing everything into one `X-Context: session=abc;workflow=wf-42` header.

**Before (single key — mutually exclusive):**

```
gen_ai.group.type = "react_iteration"
gen_ai.group.id   = "round-2"
# Can't also express: this span is part of skill "rag-retrieval"
```

**After (namespaced keys — overlapping membership):**

```
# NEW — proposed gen_ai.group.* attributes (used in prototype)
gen_ai.group.id              = round-2
gen_ai.group.iteration.type  = react
gen_ai.group.skill.id        = rag-retrieval
gen_ai.group.skill.type      = rag

# EXISTING — already defined in GenAI semconv, shown for context
gen_ai.agent.id              = main-agent
```

A single span now carries all grouping dimensions simultaneously alongside existing attributes. The new `gen_ai.group.*` keys are additive and do not modify or conflict with existing `gen_ai.*` attributes. When a dimension does not apply, its keys are simply not set rather than being set to a placeholder like `"none"` or `"N/A"`.

**Why this works with baggage specifically**

Namespaced keys are not just a naming convention, they are what makes W3C Baggage viable as the transport for grouping. W3C Baggage is a flat key-value store: each `baggage.set_baggage(key, value)` call adds an independent entry. A `BaggageSpanProcessor` copies _all_ baggage entries to span attributes automatically. This means:

- **No instrumentation burden**: span creators do not need to know about grouping at all. If baggage is set in the active context, the processor handles it.
- **Overlapping membership is free**: adding a new group dimension is just another `set_baggage()` call. No schema changes, no new span types.
- **Cross-library transparency**: if two instrumentation libraries share the same `TracerProvider` with a `BaggageSpanProcessor`, both libraries' spans carry grouping attributes without either library knowing about the convention.

Without namespaced keys, baggage would require encoding multiple groups into a single value (ex. `gen_ai.group.type=react_iteration,skill`) and parsing them back out — defeating the simplicity of the flat key-value model.

**Queryability:** Backends can ask simple questions like `gen_ai.group.id = round-2` or `gen_ai.group.skill.type = rag` without parsing structured values. This is what lets observability UIs build generic round-level, phase-level, or skill-level views without vendor-specific parsing logic — the grouping signal is in the attributes themselves, not encoded in a blob.

**What this looks like in a trace:**

```text
invoke_agent react_agent
│
├─ chat gpt-4o                                  round-2, react, rag-retrieval
│    gen_ai.group.id              = round-2
│    gen_ai.group.iteration.type  = react
│    gen_ai.group.skill.id        = rag-retrieval
│    gen_ai.group.skill.type      = rag
│    gen_ai.agent.id              = main-agent        ← existing attribute
│
├─ execute_tool search_docs                      round-2, react, rag-retrieval
│    gen_ai.group.id              = round-2
│    gen_ai.group.iteration.type  = react
│    gen_ai.group.skill.id        = rag-retrieval
│    gen_ai.group.skill.type      = rag
│
├─ chat gpt-4o                                  round-3, react (no skill)
│    gen_ai.group.id              = round-3
│    gen_ai.group.iteration.type  = react
│    ← no gen_ai.group.skill.* keys — skill dimension absent
```

Both `chat` and `execute_tool` in round-2 carry the same iteration AND skill dimensions set once via baggage, copied to both spans automatically by `BaggageSpanProcessor`. Round-3's `chat` carries the iteration but no skill — the skill keys are simply absent. A backend can filter `gen_ai.group.skill.id = rag-retrieval` to see all RAG-related spans across rounds, or `gen_ai.group.id = round-2` to see everything in round 2.

These attribute names are **proposed, not existing** in the current GenAI semantic conventions. The attribute names used here (`gen_ai.group.id`, `gen_ai.group.iteration.type`, `gen_ai.group.skill.id`, `gen_ai.group.skill.type`) match the prototype's test suite and demo code. The actual names and namespace structure would be agreed upon as part of the spec discussion.

#### 2. Recommended transport: W3C Baggage + BaggageSpanProcessor

Grouping attributes are carried in [W3C Baggage](https://www.w3.org/TR/baggage/) and the official [`opentelemetry-processor-baggage`](https://pypi.org/project/opentelemetry-processor-baggage/) package copies them to span attributes automatically. This reduces instrumentation burden: span creators do not need to call `span.set_attribute()` for grouping attributes manually on every span if a `BaggageSpanProcessor` or equivalent copier is configured.

Some environments may still choose direct attribute setting as a fallback or compatibility path.

#### 3. Interoperability guidance: grouping continuity across execution boundaries

Baggage transport alone does not guarantee grouping continuity. Whether baggage survives depends on the execution boundaries that LLM orchestration frameworks introduce. Integration testing across 6 frameworks produced the following observed behavior:

**Observed propagation behavior in tested implementations:**

| Propagation state | Behavior | Frameworks (verified by integration tests) |
|-------------------|----------|---------------------------------------------|
| **Propagates (sync and async)** | Baggage survives automatically across all execution paths | Haystack (`copy_context()` preserves baggage in executor), PydanticAI (both `@tool_plain` and `@tool`) |
| **Propagates (sync only)** | Baggage survives in sync calls but lost in async dispatch | LlamaIndex (sync `call()` propagates, async `acall()` lost), CrewAI (direct `tool.run()` propagates), Google ADK (direct function calls) |
| **Requires manual propagation** | Context not preserved across dispatch boundary | AutoGen (in-process — baggage lost at `SingleThreadedAgentRuntime` message dispatch) |
| **Breaks** | Baggage completely lost | AutoGen `GrpcWorkerAgentRuntime` (cross-process) |
| **Not yet tested** | Classification from research only | Semantic Kernel, DSPy, Instructor, ControlFlow |

This is implementation evidence that informs guidance, not a semantic requirement. The key insight is that propagation behavior depends on boundary type and sometimes runtime path — it is not simply "works" or "doesn't work."

Based on this evidence, the convention should include the following interoperability guidance:

**In-process, same execution context:**
Baggage is the recommended carrier for grouping attributes. Baggage propagation generally works automatically for many sync tool execution paths tested here — no additional action needed in those cases.

**Async or thread dispatch boundaries:**
Instrumentations and frameworks that dispatch execution across tasks or threads **should preserve the active context** into the dispatched work using `contextvars.copy_context()` (as Haystack does). If the runtime does not do this automatically, instrumentation authors **should manually capture and re-attach context** (e.g., `context.get_current()` before dispatch, `context.attach()` on receiver).

**Process / network / agent-runtime boundaries:**
Grouping context does not survive process boundaries via Python `contextvars`. If continuity across process or network boundaries is desired, grouping attributes should be serialized into message metadata or protocol headers. This is an operational consequence of the transport choice, not a semantic requirement.

#### 4. Relationship to dedicated operations

Grouping and dedicated operations complement each other:

- **Grouping** (baggage) handles loose structural correlation: "these spans belong to the same round"
- **Dedicated operations** (`execute_tool`, `chat`) handle phases with their own meaningful duration and hierarchy
- **Causal linking** (see companion issue) bridges the two by establishing parent-child links between operations

#### Prototype evidence

Prototype repo: https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype

- **Overlapping membership:** 5 automated tests covering overlapping skill+iteration+agent membership, independent dimension lifecycle, nested agent delegation, and queryability by any dimension ([`langgraph-demo/test_overlapping_groups.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/langgraph-demo/test_overlapping_groups.py))
- **Baggage propagation:** Simulated boundary tests + framework-specific patterns, with 1 implemented (Haystack `copy_context()` pattern) and stubs for remaining frameworks
- **Integration tests:** Real framework imports verifying baggage propagation across 6 frameworks — AutoGen (lost at dispatch), Haystack (propagates sync+async), PydanticAI (propagates), LlamaIndex (sync only), CrewAI (sync only), Google ADK (direct calls) ([`frameworks/`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/tree/main/frameworks))
- **Runnable demos:** Baseline (flat spans) vs. grouping-enabled across LangGraph, cross-library (LangChain + LiteLLM), and AutoGen v0.4
