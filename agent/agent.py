"""
agent.py – Core ReAct-style agent.

The Agent follows the **Thought → Action → Observation** loop:

  1. Receives a task.
  2. Thinks about what to do (Thought).
  3. Calls a tool (Action).
  4. Records the result (Observation).
  5. Repeats until it decides it has an answer.

In this demo the "LLM" reasoning is implemented by a rule-based
``_think`` method.  Swap it out for a real LLM call (OpenAI, Anthropic,
Gemini, etc.) to make the agent fully autonomous.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tools import ToolRegistry, get_default_registry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single step in the agent's reasoning trace."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    """Final result returned by the agent."""
    task: str
    answer: str
    steps: List[Step] = field(default_factory=list)
    success: bool = True

    def pretty_print(self) -> None:
        print(f"\n{'='*60}")
        print(f"Task   : {self.task}")
        print(f"{'='*60}")
        for i, step in enumerate(self.steps, start=1):
            print(f"\n[Step {i}]")
            print(f"  Thought     : {step.thought}")
            if step.action:
                print(f"  Action      : {step.action}")
                print(f"  Action Input: {json.dumps(step.action_input, indent=4)}")
            if step.observation:
                print(f"  Observation : {step.observation}")
        print(f"\n{'─'*60}")
        print(f"Answer : {self.answer}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    A ReAct-style agent that iterates Thought/Action/Observation steps.

    Parameters
    ----------
    name:
        Human-readable identifier for this agent.
    system_prompt:
        High-level instructions that shape the agent's behaviour.
    registry:
        Tool registry to use.  Defaults to the module-level registry
        populated by the ``@tool`` decorator.
    max_steps:
        Maximum number of Thought/Action/Observation iterations before
        giving up.
    verbose:
        Print each step to stdout while running.
    """

    def __init__(
        self,
        name: str = "Agent",
        system_prompt: str = "You are a helpful AI assistant.",
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.registry = registry or get_default_registry()
        self.max_steps = max_steps
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Execute *task* and return an :class:`AgentResult`."""
        if self.verbose:
            print(f"\n[{self.name}] Starting task: {task}")

        steps: List[Step] = []
        context: List[str] = [f"Task: {task}"]

        for _ in range(self.max_steps):
            thought, action, action_input = self._think(task, context)
            step = Step(thought=thought, action=action, action_input=action_input)

            if self.verbose:
                print(f"  Thought: {thought}")

            if action is None:
                # Agent has decided it's done — thought IS the answer.
                answer = thought
                result = AgentResult(task=task, answer=answer, steps=steps, success=True)
                if self.verbose:
                    print(f"  [Done] {answer}")
                return result

            # Execute the tool
            try:
                observation = self.registry.call(action, **(action_input or {}))
            except Exception as exc:
                observation = f"Tool error: {exc}"

            step.observation = str(observation)
            steps.append(step)
            context.append(
                f"Thought: {thought}\nAction: {action}\n"
                f"Input: {json.dumps(action_input)}\nObservation: {observation}"
            )

            if self.verbose:
                print(f"  Action: {action}({action_input})")
                print(f"  Observation: {observation}")

        # Exceeded max_steps
        answer = "I was unable to complete the task within the step limit."
        return AgentResult(task=task, answer=answer, steps=steps, success=False)

    # ------------------------------------------------------------------
    # "LLM" stub – replace with a real model call
    # ------------------------------------------------------------------

    def _think(
        self,
        task: str,
        context: List[str],
    ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Decide the next Thought/Action/ActionInput triple.

        This is a rule-based stub that recognises a small set of task
        patterns.  In a real system you would call an LLM here and parse
        its structured output (e.g. JSON or function-calling schema).

        Returns
        -------
        (thought, action_name, action_kwargs)
            If *action_name* is None the agent considers itself done and
            *thought* is treated as the final answer.
        """
        task_lower = task.lower()
        prior_actions = [s for s in context[1:]]  # everything after the initial task line

        # ── Maths ──────────────────────────────────────────────────────
        math_match = re.search(
            r"(?:calculate|compute|what is|evaluate|solve)\s+([\d\s\+\-\*\/\.\(\)\^%]+)",
            task_lower,
        )
        if math_match and not any("calculator" in a for a in prior_actions):
            expr = math_match.group(1).strip()
            return (
                f"I need to calculate '{expr}'. I'll use the calculator tool.",
                "calculator",
                {"expression": expr},
            )
        if math_match and any("calculator" in a for a in prior_actions):
            # Extract last observation
            last_obs = self._last_observation(context)
            return (f"The result of the calculation is {last_obs}.", None, None)

        # ── Weather ────────────────────────────────────────────────────
        weather_match = re.search(r"weather (?:in|for) ([a-z\s]+)", task_lower)
        if weather_match and not any("get_weather" in a for a in prior_actions):
            city = weather_match.group(1).strip()
            return (
                f"I need the current weather for {city.title()}.",
                "get_weather",
                {"city": city},
            )
        if weather_match and any("get_weather" in a for a in prior_actions):
            last_obs = self._last_observation(context)
            city = weather_match.group(1).strip().title()
            return (f"The weather in {city} is: {last_obs}", None, None)

        # ── Search ─────────────────────────────────────────────────────
        # Try quoted form first: search for 'query' / find "query"
        search_match = re.search(
            r"(?:search|find|look up|lookup)(?:\s+for)?\s+[\"']([^\"']+)[\"']",
            task_lower,
        )
        # Fall back to unquoted form: search <query>
        if not search_match:
            search_match = re.search(
                r"(?:search|find|look up|lookup)\s+(?!for\b)(\w[\w\s]*?)(?:\s+in\s+|\s*$)",
                task_lower,
            )
        if search_match and not any("text_search" in a for a in prior_actions):
            query = search_match.group(1).strip()
            # Use a small built-in knowledge base for the demo
            kb = json.dumps(_DEMO_KNOWLEDGE_BASE)
            return (
                f"I'll search for '{query}' in the knowledge base.",
                "text_search",
                {"query": query, "knowledge_base": kb},
            )
        if search_match and any("text_search" in a for a in prior_actions):
            last_obs = self._last_observation(context)
            return (f"Search results: {last_obs}", None, None)

        # ── Summarize ──────────────────────────────────────────────────
        summary_match = re.search(r"summarize\s+[\"'](.+)[\"']", task, re.IGNORECASE)
        if summary_match and not any("summarize_text" in a for a in prior_actions):
            text = summary_match.group(1).strip()
            return (
                "I'll summarize the provided text.",
                "summarize_text",
                {"text": text, "num_sentences": 2},
            )
        if summary_match and any("summarize_text" in a for a in prior_actions):
            last_obs = self._last_observation(context)
            return (f"Summary: {last_obs}", None, None)

        # ── Fallback ───────────────────────────────────────────────────
        return (
            f"I have processed the task. Based on the available information: {task}",
            None,
            None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_observation(context: List[str]) -> str:
        """Extract the observation text from the most recent context entry."""
        for entry in reversed(context):
            match = re.search(r"Observation:\s*(.+)", entry, re.DOTALL)
            if match:
                return match.group(1).strip()
        return "(no observation)"


# ---------------------------------------------------------------------------
# Small demo knowledge base used by the text_search tool
# ---------------------------------------------------------------------------

_DEMO_KNOWLEDGE_BASE: List[str] = [
    "Python is a high-level, interpreted programming language known for its readability.",
    "Machine learning is a subset of artificial intelligence focused on learning from data.",
    "Neural networks are computing systems inspired by biological neural networks in brains.",
    "Reinforcement learning trains agents by rewarding desired behaviours.",
    "Natural language processing (NLP) enables computers to understand human language.",
    "Transformers are a deep-learning architecture introduced in the paper 'Attention Is All You Need'.",
    "LangChain is a framework for building applications powered by language models.",
    "An AI agent is a system that perceives its environment and takes actions to achieve goals.",
    "The ReAct pattern combines reasoning and acting in language model agents.",
    "Vector databases store high-dimensional embeddings for semantic similarity search.",
]
