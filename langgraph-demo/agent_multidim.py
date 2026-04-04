"""Multi-dimensional grouping demo — proves a span can belong to multiple groups.

Directly addresses the reviewer's nesting concern from issue #3575:
  "A group may be nested, meaning that an operation can belong to groups
   across multiple dimensions. For example, it may belong to the 'main agent',
   while also being part of the second ReAct iteration."

This demo sets THREE baggage dimensions simultaneously:
  - gen_ai.group.id    = round-N                   (which ReAct iteration)
  - gen_ai.group.type  = react_reasoning|react_execution  (what kind of work)
  - gen_ai.agent.id    = main-agent                (which agent owns this span)

Jaeger query examples:
  gen_ai.group.id=round-1                                → all spans in round 1
  gen_ai.group.type=react_reasoning                      → all LLM reasoning spans
  gen_ai.group.id=round-1 gen_ai.group.type=react_execution → tool execution in round 1
  gen_ai.agent.id=main-agent gen_ai.group.type=react_reasoning → reasoning by main-agent
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tracing import init_tracing
from tools import TOOLS

from opentelemetry import trace, baggage, context
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

tracer = trace.get_tracer("langgraph-demo-multidim")


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
        "thought": "Let me do a calculation to show multi-round behavior.",
        "tool": "calculator",
        "tool_input": "2 ** 10",
    },
    {
        "thought": "I have enough information to answer.",
        "tool": None,
        "final_answer": (
            "HNSW is a graph-based ANN algorithm. 2^10 = 1024. "
            "This response was generated across 3 ReAct rounds."
        ),
    },
]


async def llm_call(state: AgentState) -> dict:
    """Simulate an LLM inference call with multi-dimensional grouping."""
    round_idx = state["round"]
    sim = SIMULATED_ROUNDS[round_idx]

    # ── GROUPING (MULTI-DIMENSIONAL) ──
    # Set multiple baggage dimensions. BaggageSpanProcessor copies ALL of them
    # to span attributes. A span belongs to a round, a group type, and an agent
    # simultaneously — without wrapper spans or arrays.
    #
    # Jaeger query examples:
    #   gen_ai.group.id=round-1                              → all spans in round 1
    #   gen_ai.group.type=react_reasoning                    → all LLM reasoning spans
    #   gen_ai.group.id=round-1 gen_ai.group.type=react_reasoning → reasoning in round 1
    #   gen_ai.agent.id=main-agent                           → all spans from this agent
    ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_idx + 1}")
    ctx = baggage.set_baggage("gen_ai.group.type", "react_reasoning", ctx)
    ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)
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
    """Execute tool with multi-dimensional grouping — same round, different phase."""
    messages = list(state["messages"])
    last_msg = messages[-1]

    if last_msg.get("role") != "tool_call":
        return {"messages": messages}

    tool_name = last_msg["tool"]
    tool_input = last_msg["input"]

    round_idx = state["round"]

    # ── GROUPING (MULTI-DIMENSIONAL) ──
    # Same round and agent as the LLM call, but group.type = "react_execution"
    # instead of "react_reasoning". A backend can now filter:
    #   gen_ai.group.id=round-1 gen_ai.group.type=react_execution
    ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_idx}")
    ctx = baggage.set_baggage("gen_ai.group.type", "react_execution", ctx)
    ctx = baggage.set_baggage("gen_ai.agent.id", "main-agent", ctx)
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
    provider = init_tracing(service_name="langgraph-multidim-demo")

    with tracer.start_as_current_span("invoke_agent") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "react_agent")

        result = await graph.ainvoke(
            {
                "question": "Tell me about HNSW and compute 2^10",
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about HNSW and compute 2^10",
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
    print("[tracing] Spans flushed.")
    print("[tracing] Aspire: http://localhost:18888")
    print("[tracing] Jaeger: http://localhost:16686")
    print("[tracing] Try Jaeger tag queries:")
    print("  gen_ai.group.id=round-1")
    print("  gen_ai.group.type=react_reasoning")
    print("  gen_ai.group.id=round-1 gen_ai.group.type=react_execution")


if __name__ == "__main__":
    asyncio.run(main())
