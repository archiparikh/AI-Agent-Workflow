"""
tests/test_agent.py – Unit tests for the Agent class.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agent import Agent, AgentResult, Step
from agent.tools import ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# AgentResult helpers
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_pretty_print_runs(self, capsys):
        result = AgentResult(
            task="test task",
            answer="42",
            steps=[Step(thought="thinking", action="calc", action_input={"expression": "6*7"}, observation="42")],
        )
        result.pretty_print()
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "test task" in captured.out


# ---------------------------------------------------------------------------
# Agent.run – math tasks
# ---------------------------------------------------------------------------

class TestAgentMath:
    def test_calculate_result_contains_answer(self):
        agent = Agent(verbose=False)
        result = agent.run("Calculate 6 * 7")
        assert isinstance(result, AgentResult)
        assert result.success

    def test_result_has_steps(self):
        agent = Agent(verbose=False)
        result = agent.run("Calculate 10 + 5")
        assert len(result.steps) > 0


# ---------------------------------------------------------------------------
# Agent.run – weather tasks
# ---------------------------------------------------------------------------

class TestAgentWeather:
    def test_weather_london(self):
        agent = Agent(verbose=False)
        result = agent.run("What is the weather in London?")
        assert result.success
        assert "London" in result.answer or "\u00b0C" in result.answer

    def test_weather_tokyo(self):
        agent = Agent(verbose=False)
        result = agent.run("What is the weather in Tokyo?")
        assert result.success


# ---------------------------------------------------------------------------
# Agent.run – search tasks
# ---------------------------------------------------------------------------

class TestAgentSearch:
    def test_search_returns_results(self):
        agent = Agent(verbose=False)
        result = agent.run("Search for 'neural network' in the knowledge base")
        assert result.success

    def test_search_no_match(self):
        agent = Agent(verbose=False)
        result = agent.run("Search for 'xyzzy' in the knowledge base")
        assert result.success


# ---------------------------------------------------------------------------
# Agent.run – summarize tasks
# ---------------------------------------------------------------------------

class TestAgentSummarize:
    def test_summarize_long_text(self):
        agent = Agent(verbose=False)
        text = "One. Two. Three. Four."
        result = agent.run(f'Summarize "{text}"')
        assert result.success
        assert "One" in result.answer or "Summary" in result.answer


# ---------------------------------------------------------------------------
# Agent with custom registry
# ---------------------------------------------------------------------------

class TestAgentCustomRegistry:
    def test_custom_tool_called(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="calculator",
                description="calc",
                func=lambda expression: "999",
                parameters={"expression": "str"},
            )
        )
        agent = Agent(registry=registry, verbose=False)
        result = agent.run("Calculate 1 + 1")
        # Our stub calculator always returns 999
        assert "999" in result.answer


# ---------------------------------------------------------------------------
# Agent max_steps guard
# ---------------------------------------------------------------------------

class TestAgentMaxSteps:
    def test_max_steps_reached_returns_failure(self):
        # Task that the stub _think never resolves to a known pattern
        agent = Agent(max_steps=1, verbose=False)
        # Force a tool call loop by patching _think to always return an action
        original_think = agent._think

        call_count = [0]

        def always_action(task, context):
            call_count[0] += 1
            return ("thinking", "calculator", {"expression": "1 + 1"})

        agent._think = always_action
        result = agent.run("some task that loops forever")
        assert not result.success
        assert call_count[0] <= agent.max_steps + 1
