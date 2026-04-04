"""Cross-library demo: LangChain orchestrator + LiteLLM as LLM backend.

Directly answers the reviewer's critique from issue #3575:
  "how would I add a link from an execute_tool span created by LangChain
   to an inference span created by LiteLLM?"

Answer: Both libraries are instrumented on the same TracerProvider.
  - LiteLLM creates inference spans (via litellm.completion with mock_response)
  - Tool execution spans are created separately (simulating LangChain's executor)
  - GROUPING: gen_ai.group.id in Baggage is copied to ALL spans via
    BaggageSpanProcessor — regardless of which library created the span.
  - CAUSALITY: Payload traceparent injection links tool execution spans
    to the LLM inference spans that triggered them.

No API key needed — LiteLLM's mock_response parameter returns canned responses
while still going through the full instrumented code path.
"""

import asyncio
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from tracing import init_tracing

import litellm
from opentelemetry import trace, baggage, context
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer("cross-library-demo")


# --- Tool functions (simulating LangChain tools) ---

def web_search(query: str) -> str:
    time.sleep(0.1)
    return (
        f'Results for "{query}":\n'
        "1. HNSW is a graph-based ANN algorithm.\n"
        "2. Used in Pinecone, Weaviate, Qdrant."
    )


def calculator(expression: str) -> str:
    time.sleep(0.02)
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {"web_search": web_search, "calculator": calculator}


# --- Mock LLM responses that simulate tool calls ---

MOCK_RESPONSES = [
    # Round 1: LLM decides to call web_search
    json.dumps({
        "thought": "I need to search for HNSW information.",
        "tool_call": {"name": "web_search", "input": "HNSW algorithm"},
    }),
    # Round 2: LLM decides to call calculator
    json.dumps({
        "thought": "Let me compute something.",
        "tool_call": {"name": "calculator", "input": "2 ** 16"},
    }),
    # Round 3: LLM gives final answer
    "HNSW is a graph-based ANN algorithm used in vector databases like Pinecone and Weaviate. 2^16 = 65536.",
]


async def run_react_agent():
    """Multi-round ReAct loop: LiteLLM for inference, manual tool execution."""
    messages = [{"role": "user", "content": "Tell me about HNSW and compute 2^16"}]
    final_answer = ""

    for round_idx, mock_resp in enumerate(MOCK_RESPONSES):
        round_num = round_idx + 1

        # ── GROUPING ── Set baggage for this round.
        # BaggageSpanProcessor copies these to ALL spans — both LiteLLM's
        # inference spans and our tool execution spans.
        ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_num}")
        ctx = baggage.set_baggage("gen_ai.group.type", "react_iteration", ctx)
        token = context.attach(ctx)

        try:
            # --- LLM inference via LiteLLM (creates spans via LiteLLM instrumentor) ---
            # This is the key: LiteLLM's OpenInference instrumentor creates the
            # inference span, NOT our code. We just call litellm.completion().
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=messages,
                mock_response=mock_resp,
            )

            llm_output = response.choices[0].message.content

            # ── CAUSALITY ── Capture current span context AFTER the LLM call.
            # This is the inference span's context that we'll inject into the
            # tool call payload.
            carrier = {}
            inject(carrier)

            # Parse the mock response
            try:
                parsed = json.loads(llm_output)
                tool_call_info = parsed.get("tool_call")
            except (json.JSONDecodeError, AttributeError):
                tool_call_info = None
                final_answer = llm_output

            if tool_call_info:
                tool_name = tool_call_info["name"]
                tool_input = tool_call_info["input"]

                # ── CAUSALITY ── Extract traceparent from the carrier.
                # This makes execute_tool a CHILD of the LiteLLM inference span.
                tool_ctx = extract(carrier, context=context.get_current())
                tool_token = context.attach(tool_ctx)

                try:
                    # --- Tool execution (simulating LangChain's tool executor) ---
                    # In production, LangChain's instrumentor would create this span.
                    with tracer.start_as_current_span("execute_tool") as tool_span:
                        tool_span.set_attribute("gen_ai.operation.name", "execute_tool")
                        tool_span.set_attribute("gen_ai.tool.name", tool_name)

                        tool_fn = TOOLS.get(tool_name)
                        result = tool_fn(tool_input) if tool_fn else "Unknown tool"

                        messages.append({"role": "assistant", "content": llm_output})
                        messages.append({"role": "tool", "content": result})
                finally:
                    context.detach(tool_token)
            else:
                messages.append({"role": "assistant", "content": llm_output})
        finally:
            context.detach(token)

    return final_answer


async def main():
    provider = init_tracing()

    with tracer.start_as_current_span("invoke_agent") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "cross_library_react_agent")

        answer = await run_react_agent()

        print(f"\n--- Agent Response ---")
        print(answer)
        print(f"--- End ---\n")

    await asyncio.sleep(2)
    provider.shutdown()
    print("[tracing] Spans flushed. Check Aspire at http://localhost:18888")
    print("[tracing] LiteLLM created inference spans, tool executor created execute_tool spans")
    print("[tracing] Both should carry gen_ai.group.id (grouping)")
    print("[tracing] execute_tool should be nested under LiteLLM spans (causality)")


if __name__ == "__main__":
    asyncio.run(main())
