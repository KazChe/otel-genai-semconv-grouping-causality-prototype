"""AutoGen v0.4 demo — async event-driven runtime with Baggage grouping.

This is the adversarial validationwith AutoGen it uses asynchronous message passing
between agents via a runtime. The question is whether gen_ai.group.id set
in Baggage BEFORE the agent invocation survives AutoGen's async dispatch
and appears on the spans created by AutoGen's runtime

Uses real OpenAI API calls (requires OPENAI_API_KEY in .env)
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Load .env from repo root
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(env_path)

from tracing import init_tracing

from opentelemetry import trace, baggage, context

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

tracer = trace.get_tracer("autogen-demo")


# --- Tools for the executor agent ---

async def web_search(query: str) -> str:
    """Search the web for information."""
    await asyncio.sleep(0.1)
    return (
        f'Results for "{query}":\n'
        "1. HNSW is a graph-based ANN algorithm (Malkov & Yashunin, 2016).\n"
        "2. Used in vector databases like Pinecone, Weaviate, Qdrant."
    )


async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    await asyncio.sleep(0.02)
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


async def main():
    provider = init_tracing()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required. Set it in ../.env")
        return

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    # ── GROUPING ── set baggage BEFORE creating agents and running the team
    # The question: does this survive AutoGen's async message dispatch?
    ctx = baggage.set_baggage("gen_ai.group.id", "autogen-session-1")
    ctx = baggage.set_baggage("gen_ai.group.type", "agent_collaboration", ctx)
    ctx = baggage.set_baggage("gen_ai.agent.id", "planner-worker-team", ctx)
    token = context.attach(ctx)

    try:
        # Planner agent — decides what to do
        planner = AssistantAgent(
            name="planner",
            model_client=model_client,
            description="Plans tasks and delegates to the worker",
            system_message=(
                "You are a task planner. When given a question, decide what tools "
                "the worker should use and ask the worker to use them. Be concise. "
                "After the worker reports results, summarize the answer."
            ),
        )

        # Worker agent — executes tools
        worker = AssistantAgent(
            name="worker",
            model_client=model_client,
            description="Executes tasks using available tools",
            tools=[web_search, calculator],
            system_message=(
                "You are a task executor. Use the available tools to complete "
                "tasks assigned by the planner. Report results clearly."
            ),
        )

        # Create team — round-robinhood: planner  worker -> planner -> ...
        team = RoundRobinGroupChat(
            participants=[planner, worker],
            termination_condition=MaxMessageTermination(max_messages=6),
        )

        print("\n--- Running AutoGen multi-agent team ---\n")

        result = await team.run(
            task="Search for 'HNSW algorithm' and then calculate 2^16. Give me both results."
        )

        print("\n--- Conversation ---")
        for msg in result.messages:
            print(f"  [{msg.source}]: {str(msg.content)[:150]}")
        print(f"\n--- End (stop reason: {result.stop_reason}) ---\n")

    finally:
        context.detach(token)

    # Flush spans
    await asyncio.sleep(3)
    provider.shutdown()
    print("[tracing] Spans flushed.")
    print("[tracing] Aspire: http://localhost:18888")
    print("[tracing] Jaeger: http://localhost:16686")
    print("[tracing] Check if AutoGen's spans carry gen_ai.group.id=autogen-session-1")


if __name__ == "__main__":
    asyncio.run(main())
