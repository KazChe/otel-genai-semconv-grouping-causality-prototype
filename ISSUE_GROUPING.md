# [gen-ai] Add generic grouping attributes for agentic workflow spans

Split from [open-telemetry/semantic-conventions#3575](https://github.com/open-telemetry/semantic-conventions/issues/3575).
Companion issue: [Causal Span Linking for GenAI Tool Calls](https://github.com/open-telemetry/semantic-conventions/issues/3662) 

area:gen-ai

### What's missing?

GenAI semantic conventions lack a standard way to group related spans into logical units (iterations, tasks, skills, phases). When a user opens a trace from an agentic workflow, they see a flat or inconsistently structured list of spans, `chat`, `execute_tool`, `chat`, `execute_tool`, `chat`, with no reliable signal for which spans belong to the same logical unit.

Debugging "why did my agent fail on attempt 3?" requires manually correlating spans by timestamp. Observability platforms cannot build generic iteration-level views or aggregations because the grouping signal is not in the data in any standardized form.

Every time a new agentic pattern emerges (ReAct iterations, planning phases, tool-selection rounds), the only recourse today is to propose a new span type, leading to an N+1 span type problem. This is not specific to any one framework; it affects graph-based orchestrators, async event-driven runtimes, and multi-agent delegation patterns alike.

### Describe the solution you'd like

> **Note:** This proposal introduces a new `gen_ai.group.*` attribute namespace with the following keys:
>
> - `gen_ai.group.id`: identifier for a specific logical unit of work, such as a single ReAct round (e.g. `round-2`)
> - `gen_ai.group.iteration.type`: the iteration pattern being executed (e.g. `react`, `plan_execute`)
> - `gen_ai.group.skill.id`: identifier for the active skill invoked within the group (e.g. `rag-retrieval`, `code-generation`)
> - `gen_ai.group.skill.type`: the category of the active skill (e.g. `rag`, `code_gen`, `compute`)

This proposal addresses feedback from @Cirilla-zmh, @Krishnachaitanyakc, and @lmolkova on #3575 around overlapping membership, instrumentation complexity, relationship to dedicated operations, and user-facing value.

This proposal standardizes the grouping representation (namespaced attributes under `gen_ai.group.*`) and recommends W3C Baggage as the transport. It also provides interoperability guidance for maintaining grouping continuity across common execution boundaries found in LLM orchestration frameworks.

#### 1. Attribute model: namespaced grouping keys (normative)

The recommendation is **namespaced grouping keys**: instead of overloading a single key, give each grouping concept its own distinct attribute under a shared `gen_ai.group.*` namespace.

**Why namespacing matters**

The namespace does two things:

1. **Prevents collisions** : each grouping dimension is independent. Setting `gen_ai.group.skill.id` does not overwrite `gen_ai.group.iteration.type`.
2. **Makes the category self-describing**: the key name tells you what kind of group it represents, without needing external documentation to interpret a generic ID.

**Using namespaced keys and overlapping membership):**

```
# Proposed gen_ai.group.* attributes (used in prototype)
gen_ai.group.id              = round-2
gen_ai.group.iteration.type  = react
gen_ai.group.skill.id        = rag-retrieval
gen_ai.group.skill.type      = rag
# EXISTING — already defined in GenAI semconv, shown for context
gen_ai.agent.id              = main-agent
```
A single span carries all grouping dimensions simultaneously. Dimensions that do not apply are simply absent, no placeholders.

**Queryability:** Backends can filter on any single dimension (e.g., `gen_ai.group.id = round-2` or `gen_ai.group.skill.type = rag`) without parsing structured values. The grouping signal is in the attributes themselves, not encoded in a blob.

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

Baggage set once in the active context is copied to every span by `BaggageSpanProcessor`. Round-3's skill keys are absent because that dimension no longer applies.

#### 2. Recommended transport: W3C Baggage + BaggageSpanProcessor

Grouping attributes are carried in [W3C Baggage](https://www.w3.org/TR/baggage/) and the official [`opentelemetry-processor-baggage`](https://pypi.org/project/opentelemetry-processor-baggage/) package copies them to span attributes automatically. This reduces instrumentation burden: span creators do not need to call `span.set_attribute()` for grouping attributes manually on every span if a `BaggageSpanProcessor` or equivalent copier is configured.

Namespaced keys are what make this viable. W3C Baggage is a flat key-value store, so each grouping dimension maps directly to one baggage entry. Overlapping membership is free — adding a new dimension is just another `set_baggage()` call, with no schema changes or new span types.

**Why baggage rather than direct attributes?** Direct attributes require every instrumentation library to explicitly set grouping on every span. Baggage with `BaggageSpanProcessor` makes this transparent, grouping is set once and propagates automatically, even to spans created by libraries that know nothing about the convention.

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

Semantic Kernel, DSPy, Instructor, and ControlFlow were reviewed from documentation only and have not yet been integration-tested.

This is implementation evidence that informs guidance, not a semantic requirement. The key insight is that propagation behavior depends on boundary type and sometimes runtime path — it is not simply "works" or "doesn't work."

Based on this evidence, the convention should include the following interoperability guidance:

**In-process, same execution context:**
Baggage is the recommended carrier for grouping attributes. Baggage propagation generally works automatically for many sync tool execution paths tested here and no additional action needed in those cases.

**Async or thread dispatch boundaries:**
Instrumentations and frameworks that dispatch execution across tasks or threads **should preserve the active OTel context** into the dispatched work. Haystack demonstrates this pattern in Python using `contextvars.copy_context()`. If the runtime does not do this automatically, instrumentation authors **should manually capture the active context before dispatch and re-attach it on the receiver side**, using whatever context API the language's OTel SDK provides.

**Process / network / agent-runtime boundaries:**
Grouping context is held in W3C Baggage, which lives inside OTel's Context. Per the OTel Context specification, cross-cutting concerns "access their data **in-process** using the same shared `Context` object"; transmission across process boundaries is handled by `Propagator`s, which "send their state to the next process" via explicit message-format read/write operations ([OTel Context spec][1], [OTel Propagators spec][2]).

If continuity across process or network boundaries is desired, grouping attributes **should be propagated using the W3C Baggage Propagator** (the standard OTel mechanism for serializing baggage into message metadata or HTTP headers). This reuses existing propagation machinery, no new format or protocol is required.

[1]: https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/context/README.md
[2]: https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/context/api-propagators.md

#### 4. Relationship to dedicated operations

Grouping and dedicated operations complement each other:

- **Grouping** (baggage) handles loose structural correlation: "these spans belong to the same round"
- **Dedicated operations** (`execute_tool`, `chat`) handle phases with their own meaningful duration and hierarchy
- **Causal linking** (see companion issue) bridges the two by establishing parent-child links between operations

#### Precedent in OTel semantic conventions

This proposal's interoperability guidance (Section 3) follows established patterns in OTel semconv:

- **[AWS Lambda semconv][lambda]** — recommends a specific propagator (`xray-lambda`), says SDKs should provide it, gives configuration pseudocode, and defines parent/link behavior for SQS spans. This is precedent for semconv material telling instrumentation authors to adhere to particular propagation flows.
- **[AWS compatibility guidance][aws-compat]** — says an AWS-supported propagation format should be used on outgoing requests, discusses baggage and tracestate, and covers operational caveats. Same scope as our cross-process baggage propagation recommendation.

[lambda]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/faas/aws-lambda.md
[aws-compat]: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/non-normative/compatibility/aws.md

#### Prototype evidence

Prototype repo: https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype

- **Overlapping membership:** 5 automated tests covering overlapping skill+iteration+agent membership, independent dimension lifecycle, nested agent delegation, and queryability by any dimension ([`frameworks/otel-only/test_grouping_mechanism.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/frameworks/otel-only/test_grouping_mechanism.py))
- **Baggage propagation boundaries:** 8 mechanism tests covering sync, `asyncio.create_task()`, `ThreadPoolExecutor`, `asyncio.to_thread()`, and cross-process boundaries, plus mitigation patterns (capture/reattach, `copy_context()`, serialize) ([`frameworks/otel-only/test_grouping_mechanism.py`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/blob/main/frameworks/otel-only/test_grouping_mechanism.py))
- **Integration tests:** Real framework imports verifying baggage propagation across 6 frameworks — AutoGen (lost at dispatch), Haystack (propagates sync+async), PydanticAI (propagates), LlamaIndex (sync only), CrewAI (sync only), Google ADK (direct calls) ([`frameworks/`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/tree/main/frameworks))
- **Runnable demo:** Same-process grouping demonstration via LangGraph, with overlapping dimensions, agent delegation, and mid-round skill transitions ([`frameworks/langgraph/`](https://github.com/KazChe/otel-genai-semconv-grouping-causality-prototype/tree/main/frameworks/langgraph))
