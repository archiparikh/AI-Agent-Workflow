"""
examples/custom_tool.py – Shows how to register a custom tool and use it.

Run with:
    python examples/custom_tool.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import Agent, ToolRegistry
from agent.tools import ToolSpec


def reverse_string(text: str) -> str:
    """Return the reversed version of *text*."""
    return text[::-1]


def main() -> None:
    # Build a fresh registry with just our custom tool
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="reverse_string",
            description="Reverse the characters in a string.",
            func=reverse_string,
            parameters={"text": "str – the string to reverse"},
        )
    )

    # Extend the agent's _think to handle our custom task pattern
    agent = Agent(name="ReverseAgent", registry=registry, verbose=True)

    import re

    def custom_think(task: str, context: list):
        task_lower = task.lower()
        prior = " ".join(context[1:])
        match = re.search(r"reverse [\"'](.+)[\"']", task_lower)
        if match and "reverse_string" not in prior:
            return (
                "I'll reverse the provided string.",
                "reverse_string",
                {"text": match.group(1)},
            )
        if match and "reverse_string" in prior:
            obs = re.search(r"Observation:\s*(.+)", prior)
            ans = obs.group(1) if obs else "unknown"
            return (f"The reversed string is: {ans}", None, None)
        return ("I cannot handle this task.", None, None)

    agent._think = custom_think  # type: ignore[method-assign]

    result = agent.run("Please reverse 'Hello, World!'")
    result.pretty_print()


if __name__ == "__main__":
    main()
