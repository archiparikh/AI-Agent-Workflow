"""
tools.py – Built-in tools that an Agent can call.

Each tool is a plain Python function decorated with @tool.  The decorator
registers the function in a global ToolRegistry so the Agent can discover
and invoke it by name.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool registry & decorator
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """Metadata for a single tool."""
    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Collection of callable tools available to an Agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def call(self, name: str, **kwargs: Any) -> Any:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool '{name}'. Available: {list(self._tools)}")
        return spec.func(**kwargs)

    def describe(self) -> str:
        """Return a human-readable summary of all registered tools."""
        lines: List[str] = []
        for spec in self._tools.values():
            params = ", ".join(
                f"{k}: {v}" for k, v in spec.parameters.items()
            )
            lines.append(f"  • {spec.name}({params}) – {spec.description}")
        return "\n".join(lines)


# Module-level default registry
_default_registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Callable:
    """Decorator that registers a function as an agent tool."""
    def decorator(func: Callable) -> Callable:
        _name = name or func.__name__
        _registry = registry or _default_registry
        _registry.register(
            ToolSpec(
                name=_name,
                description=description or (func.__doc__ or "").strip(),
                func=func,
                parameters=parameters or {},
            )
        )
        return func
    return decorator


def get_default_registry() -> ToolRegistry:
    return _default_registry


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@tool(
    description="Evaluate a mathematical expression and return the numeric result.",
    parameters={"expression": "str – a valid Python math expression (e.g. '2 + 3 * 4')"},
)
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Allow only safe characters
    allowed = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%\^]+$")
    expr = expression.replace("^", "**")
    if not allowed.match(expr):
        return f"Error: unsafe expression '{expression}'"
    try:
        result = eval(expr, {"__builtins__": {}}, {"math": math})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool(
    description="Search a list of facts for entries that match a query string.",
    parameters={
        "query": "str – keyword(s) to look for",
        "knowledge_base": "str – JSON-encoded list of strings to search",
    },
)
def text_search(query: str, knowledge_base: str) -> str:
    """Full-text keyword search over a JSON list of strings."""
    try:
        facts: List[str] = json.loads(knowledge_base)
    except json.JSONDecodeError:
        return "Error: knowledge_base must be a JSON-encoded list of strings"
    q = query.lower()
    hits = [f for f in facts if q in f.lower()]
    if not hits:
        return f"No results found for '{query}'."
    return "\n".join(f"  [{i+1}] {h}" for i, h in enumerate(hits))


@tool(
    description="Return a mock weather report for a given city.",
    parameters={"city": "str – city name"},
)
def get_weather(city: str) -> str:
    """Return a stubbed weather report (replace with a real API call in production)."""
    reports = {
        "london": "Overcast, 12 °C, humidity 80 %, light rain expected.",
        "new york": "Partly cloudy, 18 °C, humidity 55 %, mild breeze.",
        "tokyo": "Sunny, 22 °C, humidity 60 %, clear skies.",
        "sydney": "Sunny, 26 °C, humidity 45 %, strong UV.",
        "paris": "Cloudy, 14 °C, humidity 70 %, chance of drizzle.",
    }
    return reports.get(city.lower(), f"Weather data not available for '{city}'.")


@tool(
    description="Summarize a text by returning its first N sentences.",
    parameters={
        "text": "str – the input text",
        "num_sentences": "int – how many sentences to keep (default 2)",
    },
)
def summarize_text(text: str, num_sentences: int = 2) -> str:
    """Return the first *num_sentences* sentences of the supplied text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:num_sentences]) or text
