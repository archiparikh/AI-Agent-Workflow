"""
examples/multi_agent_workflow.py – Demonstrates a workflow with two
specialised agents working on different steps.

Run with:
    python examples/multi_agent_workflow.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import Agent, Workflow, WorkflowStep


def main() -> None:
    # Agent specialised in data retrieval (search / weather)
    retrieval_agent = Agent(
        name="RetrievalAgent",
        system_prompt="You retrieve information from tools and knowledge bases.",
        verbose=True,
    )

    # Agent specialised in summarization
    summary_agent = Agent(
        name="SummaryAgent",
        system_prompt="You distill observations into concise answers.",
        verbose=True,
    )

    workflow = Workflow(name="Retrieve-then-Summarize", verbose=True)

    workflow.add_step(
        WorkflowStep(
            name="Retrieve weather",
            task_template="What is the weather in Paris?",
            agent=retrieval_agent,
        )
    )
    workflow.add_step(
        WorkflowStep(
            name="Summarize result",
            task_template='Summarize "{prev_answer}"',
            agent=summary_agent,
        )
    )

    wf_result = workflow.run()
    wf_result.pretty_print()


if __name__ == "__main__":
    main()
