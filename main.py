"""
main.py – Runnable demos for the AI Agent Workflow.

Run with:
    python main.py

Each demo illustrates a different capability of the framework.
"""

from __future__ import annotations

from agent import Agent, Workflow, WorkflowStep
from agent.tools import get_default_registry


# ---------------------------------------------------------------------------
# Demo 1 – Single agent, single math task
# ---------------------------------------------------------------------------

def demo_calculator() -> None:
    print("\n" + "=" * 60)
    print("DEMO 1: Calculator Agent")
    print("=" * 60)
    agent = Agent(name="MathAgent", verbose=True)
    result = agent.run("Calculate 15 * 4 + 7")
    result.pretty_print()


# ---------------------------------------------------------------------------
# Demo 2 – Single agent, weather look-up
# ---------------------------------------------------------------------------

def demo_weather() -> None:
    print("\n" + "=" * 60)
    print("DEMO 2: Weather Agent")
    print("=" * 60)
    agent = Agent(name="WeatherAgent", verbose=True)
    result = agent.run("What is the weather in Tokyo?")
    result.pretty_print()


# ---------------------------------------------------------------------------
# Demo 3 – Single agent, knowledge-base search
# ---------------------------------------------------------------------------

def demo_search() -> None:
    print("\n" + "=" * 60)
    print("DEMO 3: Search Agent")
    print("=" * 60)
    agent = Agent(name="SearchAgent", verbose=True)
    result = agent.run("Search for 'transformer' in the knowledge base")
    result.pretty_print()


# ---------------------------------------------------------------------------
# Demo 4 – Single agent, text summarization
# ---------------------------------------------------------------------------

def demo_summarize() -> None:
    print("\n" + "=" * 60)
    print("DEMO 4: Summarization Agent")
    print("=" * 60)
    agent = Agent(name="SummarizeAgent", verbose=True)
    long_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines. "
        "It involves learning, reasoning, and self-correction. "
        "Modern AI applications include speech recognition, computer vision, "
        "natural language processing, and autonomous vehicles. "
        "The field was founded in 1956 at Dartmouth College."
    )
    result = agent.run(f'Summarize "{long_text}"')
    result.pretty_print()


# ---------------------------------------------------------------------------
# Demo 5 – Multi-step workflow (chained tasks)
# ---------------------------------------------------------------------------

def demo_workflow() -> None:
    print("\n" + "=" * 60)
    print("DEMO 5: Multi-step Chained Workflow")
    print("=" * 60)

    agent = Agent(name="WorkflowAgent", verbose=True)

    workflow = Workflow(name="Research + Summarize Pipeline", default_agent=agent)

    workflow.add_step(
        WorkflowStep(
            name="Search knowledge base",
            task_template="Search for 'neural network' in the knowledge base",
        )
    )
    workflow.add_step(
        WorkflowStep(
            name="Summarize search result",
            task_template='Summarize "{prev_answer}"',
        )
    )

    wf_result = workflow.run()
    wf_result.pretty_print()


# ---------------------------------------------------------------------------
# Demo 6 – List available tools
# ---------------------------------------------------------------------------

def demo_list_tools() -> None:
    print("\n" + "=" * 60)
    print("DEMO 6: Available Tools")
    print("=" * 60)
    registry = get_default_registry()
    print("Registered tools:\n")
    print(registry.describe())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_list_tools()
    demo_calculator()
    demo_weather()
    demo_search()
    demo_summarize()
    demo_workflow()
