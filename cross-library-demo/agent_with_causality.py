"""NOTE: This is an early prototype kept for historical reference. It uses
payload argument injection for causality, which integration testing showed
fails in 5/6 frameworks. The recommended approach is now sidecar propagation.
See ISSUE_CAUSALITY.md in the repo root for the updated proposal.

Cross-library demo WITH causality — LangChain + LiteLLM + payload traceparent.

This script builds on agent.py by wrapping the LiteLLM call in a parent span
that stays active during tool execution. The traceparent from this wrapper span
is injected into the tool call payload, so execute_tool becomes a child.

This proves that payload traceparent injection solves the cross-library causality
problem — even when the two libraries (LiteLLM for inference, tool executor for
execution) have independent span lifecycles.

Compare with agent.py (no causality — flat siblings) to see the difference.
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

tracer = trace.get_tracer("cross-library-causality-demo")


# --- Tool functions ---

def web_search(query: str) -> str:
    time.sleep(0.1)
    return f'Results for "{query}": HNSW is a graph-based ANN algorithm.'


def calculator(expression: str) -> str:
    time.sleep(0.02)
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {"web_search": web_search, "calculator": calculator}


MOCK_RESPONSES = [
    json.dumps({
        "thought": "I need to search for HNSW information.",
        "tool_call": {"name": "web_search", "input": "HNSW algorithm"},
    }),
    json.dumps({
        "thought": "Let me compute something.",
        "tool_call": {"name": "calculator", "input": "2 ** 16"},
    }),
    "HNSW is a graph-based ANN algorithm. 2^16 = 65536.",
]


async def run_react_agent():
    messages = [{"role": "user", "content": "Tell me about HNSW and compute 2^16"}]
    final_answer = ""

    for round_idx, mock_resp in enumerate(MOCK_RESPONSES):
        round_num = round_idx + 1

        # ── GROUPING ── Set baggage for this round
        ctx = baggage.set_baggage("gen_ai.group.id", f"round-{round_num}")
        ctx = baggage.set_baggage("gen_ai.group.type", "react_iteration", ctx)
        token = context.attach(ctx)

        try:
            # ── CAUSALITY ── Wrap the LLM call + tool execution in a "chat" span.
            # This span stays active while both the LiteLLM call AND the tool
            # execution happen — so the tool span becomes a child of this span,
            # which is itself the parent of LiteLLM's completion span.
            with tracer.start_as_current_span("chat") as chat_span:
                chat_span.set_attribute("gen_ai.operation.name", "chat")
                chat_span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")

                # LiteLLM creates its own "completion" span inside this context.
                # The completion span becomes a CHILD of our chat span.
                response = litellm.completion(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    mock_response=mock_resp,
                )

                llm_output = response.choices[0].message.content

                # ── CAUSALITY ── Inject traceparent WHILE chat span is still active.
                # This captures the chat span's context for the tool executor.
                carrier = {}
                inject(carrier)

                try:
                    parsed = json.loads(llm_output)
                    tool_call_info = parsed.get("tool_call")
                except (json.JSONDecodeError, AttributeError):
                    tool_call_info = None
                    final_answer = llm_output
                    chat_span.set_attribute("gen_ai.response.finish_reasons", "stop")

                if tool_call_info:
                    chat_span.set_attribute("gen_ai.response.finish_reasons", "tool_calls")
                    tool_name = tool_call_info["name"]
                    tool_input = tool_call_info["input"]

                    # ── CAUSALITY ── Extract traceparent from the carrier.
                    # execute_tool becomes a CHILD of the chat span.
                    tool_ctx = extract(carrier, context=context.get_current())
                    tool_token = context.attach(tool_ctx)

                    try:
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
    provider = init_tracing(service_name="cross-library-causality-demo")

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
    print("[tracing] completion (LiteLLM) and execute_tool should both be CHILDREN of chat")
    print("[tracing] Compare with agent.py where they are flat siblings")


if __name__ == "__main__":
    asyncio.run(main())
