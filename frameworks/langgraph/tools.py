"""Simulated tools for the ReAct agent. Same across all demos."""

import time


def web_search(query: str) -> str:
    """Simulate a web search tool."""
    time.sleep(0.1)  # simulate latency
    return (
        f'Results for "{query}":\n'
        "1. HNSW is a graph-based approximate nearest neighbor algorithm (Malkov & Yashunin, 2016).\n"
        "2. Used in vector databases like Pinecone, Weaviate, Qdrant, pgvector.\n"
        "3. Time complexity: O(log n) for search."
    )


def get_weather(location: str) -> str:
    """Simulate a weather lookup tool."""
    time.sleep(0.05)
    return f"Weather in {location}: 72°F, partly cloudy, humidity 45%."


def calculator(expression: str) -> str:
    """Simulate a calculator tool."""
    time.sleep(0.02)
    try:
        result = eval(expression)  # safe enough for a demo
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "web_search": web_search,
    "get_weather": get_weather,
    "calculator": calculator,
}
