"""Overlapping group membership + causality demo — proves both proposals
work together as complementary layers of the same contract.

Directly addresses concerns from issue #3575:

  1. Overlapping membership: "If gen_ai.group.type is of type
     StringAttributeKey, shouldn't its value be mutually exclusive?"
     Namespaced baggage keys solve this. Each grouping concept gets its
     own key under gen_ai.group.* — overlapping membership is free
     because W3C Baggage is a flat key-value store and BaggageSpanProcessor
     copies all entries to span attributes automatically.
     See ISSUE_GROUPING.md "Attribute model" section for the full rationale
     on how we evolved from single-key to namespaced-key model.

  2. Causality: "How would I add a link from an execute_tool span
     to an inference span?"
     Payload traceparent injection solves this

This demo combines:
  - Namespaced baggage keys for overlapping group membership
  - Payload traceparent for causal parent-child linking
  - Agent delegation (main-agent -> research-sub-agent)
  - Mid-round skill transitions (RAG -> code-gen within same round)
  - Self-documenting span output (prints attributes without needing Jaeger)

Jaeger query examples:
  gen_ai.group.skill.id=rag-retrieval -> all RAG skill spans
  gen_ai.group.iteration.type=react -> all ReAct iteration spans
  gen_ai.group.skill.id=rag-retrieval gen_ai.group.id=round-1 -> RAG in round 1
  gen_ai.agent.id=research-sub-agent -> delegated agent spans
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tracing import init_tracing
from tools import TOOLS

from opentelemetry import trace, baggage, context
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

tracer = trace.get_tracer("langgraph-overlapping-groups-demo")

# In-memory exporter for self-documenting output
_memory_exporter = InMemorySpanExporter()


class AgentState(TypedDict):
    question: str
    messages: list
    round: int
    final_answer: str


# Simulated rounds demonstrating:
# - Round 1: RAG skill active, then mid-round transition to code-gen
# - Round 2: Delegation to research-sub-agent (nested agent)
# - Round 3: Pure reasoning, no skill, no delegation
SIMULATED_ROUNDS = [
    {
        "thought": "I need to retrieve knowledge about HNSW using RAG.",
        "tool": "web_search",
        "tool_input": "HNSW algorithm vector databases",
        "skill": {"id": "rag-retrieval", "type": "rag"},
        "follow_up_skill": {"id": "code-generation", "type": "code_gen"},
        "delegate_to": None,
    },
    {
        "thought": "Let me delegate deeper research to a sub-agent.",
        "tool": "calculator",
        "tool_input": "2 ** 10",
        "skill": {"id": "analysis", "type": "compute"},
        "follow_up_skill": None,
        "delegate_to": "research-sub-agent",
    },
    {
        "thought": "I have enough information to synthesize the final answer.",
        "tool": None,
        "skill": None,
        "follow_up_skill": None,
        "delegate_to": None,
        "final_answer": (
            "HNSW is a graph-based ANN algorithm (O(log n) search). "
            "Example: a 10-dimensional space has 2^10 = 1024 possible binary vectors. "
            "This answer combined RAG retrieval, code generation, and delegated analysis."
        ),
    },
]


async def llm_call(state: AgentState) -> dict:
    """LLM call with overlapping groups + causality injection + delegation."""
    round_idx = state["round"]
    sim = SIMULATED_ROUNDS[round_idx]
    round_num = round_idx + 1

    # ── OVERLAPPING GROUPS ──
    # Always set iteration dimension
    ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_num}")
    ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
    ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)

    # Conditionally add skill dimension — when present, the span belongs
    # to BOTH the iteration AND the skill simultaneously
    if sim.get("skill"):
        ctx = baggage.set_baggage("gen_ai.group.skill.id", sim["skill"]["id"], ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.type", sim["skill"]["type"], ctx)

    token = context.attach(ctx)

    try:
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", "gpt-4o-mini")

            await asyncio.sleep(0.2)

            messages = list(state["messages"])
            messages.append({"role": "assistant", "content": sim.get("thought", "")})

            if sim["tool"] is None:
                span.set_attribute("gen_ai.response.finish_reasons", "stop")
                return {
                    "messages": messages,
                    "final_answer": sim["final_answer"],
                    "round": round_num,
                }
            else:
                span.set_attribute("gen_ai.response.finish_reasons", "tool_calls")

                # ── CAUSALITY ── Inject traceparent into tool call payload
                carrier = {}
                inject(carrier)

                tool_msg = {
                    "role": "tool_call",
                    "tool": sim["tool"],
                    "input": sim["tool_input"],
                    "_otel": carrier,  # traceparent rides in the payload
                }

                # Pass delegation and follow-up skill info through state
                if sim.get("delegate_to"):
                    tool_msg["_delegate_to"] = sim["delegate_to"]
                if sim.get("follow_up_skill"):
                    tool_msg["_follow_up_skill"] = sim["follow_up_skill"]

                messages.append(tool_msg)
                return {"messages": messages, "round": round_num}
    finally:
        context.detach(token)


async def tool_call(state: AgentState) -> dict:
    """Tool execution with overlapping groups, causality extraction,
    agent delegation, and mid-round skill transition."""
    messages = list(state["messages"])
    last_msg = messages[-1]

    if last_msg.get("role") != "tool_call":
        return {"messages": messages}

    tool_name = last_msg["tool"]
    tool_input = last_msg["input"]
    round_idx = state["round"]

    sim = SIMULATED_ROUNDS[round_idx - 1] if round_idx > 0 else SIMULATED_ROUNDS[0]

    # Determine agent identity (main or delegated)
    agent_id = last_msg.get("_delegate_to", "main-agent")

    # ── OVERLAPPING GROUPS ──
    ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_idx}")
    ctx = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx)
    ctx = baggage.set_baggage("gen_ai.agent.id", agent_id, ctx)

    if sim.get("skill"):
        ctx = baggage.set_baggage("gen_ai.group.skill.id", sim["skill"]["id"], ctx)
        ctx = baggage.set_baggage("gen_ai.group.skill.type", sim["skill"]["type"], ctx)

    # If delegated, add delegation provenance
    if agent_id != "main-agent":
        ctx = baggage.set_baggage("gen_ai.group.delegated_from", "main-agent", ctx)

    # ── CAUSALITY ── Extract traceparent from tool call payload
    otel_carrier = last_msg.get("_otel", {})
    if otel_carrier:
        ctx = extract(otel_carrier, context=ctx)

    token = context.attach(ctx)

    try:
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", tool_name)

            tool_fn = TOOLS.get(tool_name)
            if tool_fn:
                result = tool_fn(tool_input)
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append({"role": "tool_result", "tool": tool_name, "result": result})

        # ── MID-ROUND SKILL TRANSITION ──
        # After the first tool execution, if there's a follow-up skill,
        # create a second span under a different skill dimension within
        # the same round — proves dimensions are truly independent
        follow_up = last_msg.get("_follow_up_skill")
        if follow_up:
            ctx2 = baggage.set_baggage("gen_ai.group.id", f"round-{round_idx}")
            ctx2 = baggage.set_baggage("gen_ai.group.iteration.type", "react", ctx2)
            ctx2 = baggage.set_baggage("gen_ai.agent.id", agent_id, ctx2)
            ctx2 = baggage.set_baggage("gen_ai.group.skill.id", follow_up["id"], ctx2)
            ctx2 = baggage.set_baggage("gen_ai.group.skill.type", follow_up["type"], ctx2)

            if otel_carrier:
                ctx2 = extract(otel_carrier, context=ctx2)

            token2 = context.attach(ctx2)
            try:
                with tracer.start_as_current_span("execute_tool") as span2:
                    span2.set_attribute("gen_ai.operation.name", "execute_tool")
                    span2.set_attribute("gen_ai.tool.name", "code_snippet")
                    span2.set_attribute("gen_ai.tool.description",
                                        "follow-up skill transition within same round")

                    await asyncio.sleep(0.05)
                    messages.append({
                        "role": "tool_result",
                        "tool": "code_snippet",
                        "result": "def hnsw_search(query, k=10): ...",
                    })
            finally:
                context.detach(token2)

        return {"messages": messages}
    finally:
        context.detach(token)


def should_continue(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "tool_call":
        return "tool"
    return "end"


builder = StateGraph(AgentState)
builder.add_node("llm", llm_call)
builder.add_node("tool", tool_call)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"tool": "tool", "end": END})
builder.add_edge("tool", "llm")
graph = builder.compile()


def _print_span_summary(exporter):
    """Print a self-documenting summary of span attributes — no Jaeger needed."""
    spans = exporter.get_finished_spans()
    if not spans:
        print("[spans] No spans captured in memory exporter.")
        return

    print(f"\n{'='*80}")
    print(f"SPAN SUMMARY ({len(spans)} spans)")
    print(f"{'='*80}")

    for span in spans:
        attrs = dict(span.attributes) if span.attributes else {}
        parent_info = ""
        if span.parent:
            parent_info = f" (parent: {span.parent.span_id:#018x})"

        print(f"\n  {span.name}{parent_info}")
        print(f"    span_id: {span.context.span_id:#018x}")

        # Group the attributes for readability
        group_attrs = {k: v for k, v in attrs.items() if k.startswith("gen_ai.group.")}
        agent_attrs = {k: v for k, v in attrs.items() if k.startswith("gen_ai.agent.")}
        other_attrs = {k: v for k, v in attrs.items()
                       if not k.startswith("gen_ai.group.") and not k.startswith("gen_ai.agent.")}

        if group_attrs:
            print(f"    groups: {group_attrs}")
        if agent_attrs:
            print(f"    agent:  {agent_attrs}")
        if other_attrs:
            dims = {k: v for k, v in other_attrs.items() if k.startswith("gen_ai.")}
            if dims:
                print(f"    genai:  {dims}")

    print(f"\n{'='*80}")


async def main():
    provider = init_tracing(service_name="langgraph-overlapping-groups-demo")

    # Add in-memory exporter for self-documenting output
    provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))

    with tracer.start_as_current_span("invoke_agent") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "react_agent_with_skills")

        result = await graph.ainvoke(
            {
                "question": "Tell me about HNSW and demonstrate with a calculation",
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about HNSW and demonstrate with a calculation",
                    }
                ],
                "round": 0,
                "final_answer": "",
            }
        )

        print(f"\n--- Agent Response ---")
        print(result["final_answer"])
        print(f"--- End ({result['round']} rounds) ---\n")

    # Print span summary before flushing
    _print_span_summary(_memory_exporter)

    await asyncio.sleep(2)
    provider.shutdown()
    print("\n[tracing] Spans flushed.")
    print("[tracing] Aspire: http://localhost:18888")
    print("[tracing] Jaeger: http://localhost:16686")
    print()
    print("[tracing] What to look for:")
    print("  1. CAUSALITY: execute_tool spans are CHILDREN of chat spans (not siblings)")
    print("  2. OVERLAPPING GROUPS: spans carry both skill and iteration dimensions")
    print("  3. SKILL TRANSITION: round-1 has two execute_tool spans with different skills")
    print("  4. DELEGATION: round-2 execute_tool has gen_ai.agent.id=research-sub-agent")
    print()
    print("[tracing] Jaeger queries:")
    print("  gen_ai.group.skill.id=rag-retrieval -> RAG skill spans")
    print("  gen_ai.group.skill.id=code-generation -> code-gen skill spans")
    print("  gen_ai.group.iteration.type=react -> all ReAct spans")
    print("  gen_ai.agent.id=research-sub-agent -> delegated agent spans")
    print("  gen_ai.group.delegated_from=main-agent -> all delegated work")


if __name__ == "__main__":
    asyncio.run(main())
