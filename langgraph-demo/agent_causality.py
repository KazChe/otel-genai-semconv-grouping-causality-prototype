"""NOTE: This is an early prototype kept for historical reference. It uses
tool_call["_otel"] = carrier (argument injection) for causality, which
integration testing showed fails in 5/6 frameworks. The recommended
approach is now sidecar propagation via framework-native extension points.
See agent_overlapping_groups.py for the current version and ISSUE_CAUSALITY.md
for the updated proposal.

LangGraph ReAct agent with BOTH Grouping AND Causality.

Grouping: gen_ai.group.id in Baggage (same as agent.py)
Causality: After LLM returns tool_calls, we inject the current span's
traceparent into the tool call payload. The tool executor extracts it
and uses it as the parent context — so execute_tool becomes a CHILD of
the chat span that triggered it, not just a sibling.

This addresses the second half of the reviewer's critique from #3575:
  "how would I add a link from an execute_tool span to an inference span?"

Answer: inject traceparent into the tool call payload. Same pattern as
HTTP traceparent headers, applied to LLM message envelopes.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tracing import init_tracing
from tools import TOOLS

from opentelemetry import trace, baggage, context
from opentelemetry.propagate import inject, extract
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

tracer = trace.get_tracer("langgraph-causality-demo")


class AgentState(TypedDict):
    question: str
    messages: list
    round: int
    final_answer: str


SIMULATED_ROUNDS = [
    {
        "thought": "I need to search for information about HNSW.",
        "tool": "web_search",
        "tool_input": "HNSW algorithm vector databases",
    },
    {
        "thought": "Now let me check the weather to give a complete answer.",
        "tool": "get_weather",
        "tool_input": "San Francisco",
    },
    {
        "thought": "I have enough information to answer the user's question.",
        "tool": None,
        "final_answer": (
            "HNSW is a graph-based ANN algorithm used in vector databases. "
            "It was created by Malkov & Yashunin in 2016 and offers O(log n) search. "
            "Also, the weather in San Francisco is 72°F and partly cloudy."
        ),
    },
]


async def llm_call(state: AgentState) -> dict:
    """Simulate an LLM inference call with grouping + causality injection."""
    round_idx = state["round"]
    sim = SIMULATED_ROUNDS[round_idx]

    # ── GROUPING ── Set gen_ai.group.id for this ReAct round
    group_id = f"round-{round_idx + 1}"
    ctx = baggage.set_baggage("gen_ai.group.id", group_id)
    ctx = baggage.set_baggage("gen_ai.group.type", "react_iteration", ctx)
    token = context.attach(ctx)

    try:
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", "gpt-4o-mini")

            await asyncio.sleep(0.2)  # simulate LLM latency

            messages = list(state["messages"])
            messages.append({"role": "assistant", "content": sim.get("thought", "")})

            if sim["tool"] is None:
                span.set_attribute("gen_ai.response.finish_reasons", "stop")
                return {
                    "messages": messages,
                    "final_answer": sim["final_answer"],
                    "round": round_idx + 1,
                }
            else:
                span.set_attribute("gen_ai.response.finish_reasons", "tool_calls")

                # ── CAUSALITY ── Inject traceparent into tool call payload.
                # This captures the current span's context (the chat span)
                # so the tool executor can parent its span to THIS chat span.
                # Same pattern as HTTP traceparent headers, applied to LLM
                # tool call envelopes.
                carrier = {}
                inject(carrier)

                messages.append(
                    {
                        "role": "tool_call",
                        "tool": sim["tool"],
                        "input": sim["tool_input"],
                        "_otel": carrier,  # traceparent rides in the payload
                    }
                )
                return {"messages": messages, "round": round_idx + 1}
    finally:
        context.detach(token)


async def tool_call(state: AgentState) -> dict:
    """Execute tool with causal link back to the LLM call that triggered it."""
    messages = list(state["messages"])
    last_msg = messages[-1]

    if last_msg.get("role") != "tool_call":
        return {"messages": messages}

    tool_name = last_msg["tool"]
    tool_input = last_msg["input"]

    # ── GROUPING ── Same round as the LLM call
    round_idx = state["round"]
    group_id = f"round-{round_idx}"
    ctx = baggage.set_baggage("gen_ai.group.id", group_id)
    ctx = baggage.set_baggage("gen_ai.group.type", "react_iteration", ctx)

    # ── CAUSALITY ── Extract traceparent from tool call payload.
    # This recovers the chat span's context and uses it as the parent,
    # so execute_tool becomes a CHILD of the chat span — not a sibling.
    otel_carrier = last_msg.get("_otel", {})
    if otel_carrier:
        ctx = extract(otel_carrier, context=ctx)

    token = context.attach(ctx)

    try:
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", tool_name)
            # Causality is proven by the parent-child relationship
            # in the trace tree — no extra attribute needed

            tool_fn = TOOLS.get(tool_name)
            if tool_fn:
                result = tool_fn(tool_input)
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append({"role": "tool_result", "tool": tool_name, "result": result})
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


async def main():
    provider = init_tracing(service_name="langgraph-causality-demo")

    with tracer.start_as_current_span("invoke_agent") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "react_agent")

        result = await graph.ainvoke(
            {
                "question": "Tell me about HNSW and the weather in SF",
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about HNSW and the weather in SF",
                    }
                ],
                "round": 0,
                "final_answer": "",
            }
        )

        print(f"\n--- Agent Response ---")
        print(result["final_answer"])
        print(f"--- End ({result['round']} rounds) ---\n")

    await asyncio.sleep(2)
    provider.shutdown()
    print("[tracing] Spans flushed. Check Aspire at http://localhost:18888 - langgraph-causality-demo")
    print("[tracing] Look for execute_tool spans parented to chat spans (causality)")


if __name__ == "__main__":
    asyncio.run(main())
