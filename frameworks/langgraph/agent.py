"""NOTE: This is an early prototype kept for historical reference and demo
screenshots. It uses the original single-key model (gen_ai.group.id).
See agent_overlapping_groups.py for the current version with namespaced
baggage keys, causality, agent delegation, and mid-round skill transitions.

LangGraph ReAct agent WITH Baggage grouping — the 'after' state.

Same agent logic as the baseline, but with gen_ai.group.id set in W3C Baggage
at each ReAct round boundary. The BaggageSpanProcessor automatically copies
this to every span's attributes — so all spans in a round are tagged with
the same group ID without any wrapper spans.

Compare the trace from this demo with the baseline trace in Aspire to see
the difference: spans now carry gen_ai.group.id and gen_ai.group.type attributes.
"""

import asyncio
import sys
import os
from uuid import uuid4

sys.path.insert(0, os.path.dirname(__file__))

from tracing import init_tracing
from tools import TOOLS

from opentelemetry import trace, baggage, context
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

tracer = trace.get_tracer("langgraph-demo")


class AgentState(TypedDict):
    question: str
    messages: list
    round: int
    final_answer: str


# --- Simulated LLM responses for a multi-round ReAct ---

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
    """Simulate an LLM inference call."""
    round_idx = state["round"]
    sim = SIMULATED_ROUNDS[round_idx]

    # ── GROUPING ── Set gen_ai.group.id for this ReAct round.
    # BaggageSpanProcessor will copy this to all spans created in this context.
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
                messages.append(
                    {
                        "role": "tool_call",
                        "tool": sim["tool"],
                        "input": sim["tool_input"],
                    }
                )
                return {"messages": messages, "round": round_idx + 1}
    finally:
        context.detach(token)


async def tool_call(state: AgentState) -> dict:
    """Execute the tool from the last message."""
    messages = list(state["messages"])
    last_msg = messages[-1]

    if last_msg.get("role") != "tool_call":
        return {"messages": messages}

    tool_name = last_msg["tool"]
    tool_input = last_msg["input"]

    # ── GROUPING ── Same round group as the LLM call that triggered it.
    round_idx = state["round"]
    group_id = f"round-{round_idx}"
    ctx = baggage.set_baggage("gen_ai.group.id", group_id)
    ctx = baggage.set_baggage("gen_ai.group.type", "react_iteration", ctx)
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
            return {"messages": messages}
    finally:
        context.detach(token)


def should_continue(state: AgentState) -> str:
    """Route: if we have a final answer, end. Otherwise, call a tool."""
    if state.get("final_answer"):
        return "end"
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "tool_call":
        return "tool"
    return "end"


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("llm", llm_call)
builder.add_node("tool", tool_call)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"tool": "tool", "end": END})
builder.add_edge("tool", "llm")
graph = builder.compile()


async def main():
    provider = init_tracing()

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

    # Flush spans
    await asyncio.sleep(2)
    provider.shutdown()
    print("[tracing] Spans flushed. Check Aspire at http://localhost:18888 - langgraph-demo")

if __name__ == "__main__":
    asyncio.run(main())
