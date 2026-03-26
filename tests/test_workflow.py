"""
tests/test_workflow.py – Unit tests for the Workflow orchestrator.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agent import Agent
from agent.workflow import Workflow, WorkflowStep, WorkflowResult


class TestWorkflowStep:
    def test_build_task_no_prev(self):
        step = WorkflowStep(name="s1", task_template="Do something")
        assert step.build_task() == "Do something"

    def test_build_task_with_prev_answer(self):
        from agent.agent import AgentResult
        prev = AgentResult(task="t", answer="prev_answer_here")
        step = WorkflowStep(
            name="s2",
            task_template="Based on: {prev_answer}, do more",
        )
        task = step.build_task(prev)
        assert "prev_answer_here" in task

    def test_build_task_with_transform(self):
        from agent.agent import AgentResult
        prev = AgentResult(task="t", answer="42")
        step = WorkflowStep(
            name="s3",
            task_template="unused",
            transform=lambda r: f"The value is {r.answer}",
        )
        assert step.build_task(prev) == "The value is 42"


class TestWorkflow:
    def _agent(self) -> Agent:
        return Agent(verbose=False)

    def test_sequential_factory(self):
        wf = Workflow.sequential(
            name="test",
            tasks=["Calculate 1 + 1", "Calculate 2 + 2"],
            agent=self._agent(),
            verbose=False,
        )
        assert len(wf.steps) == 2

    def test_run_returns_workflow_result(self):
        wf = Workflow(
            name="simple",
            default_agent=self._agent(),
            verbose=False,
        )
        wf.add_step(WorkflowStep(name="step1", task_template="Calculate 3 + 3"))
        result = wf.run()
        assert isinstance(result, WorkflowResult)
        assert len(result.step_results) == 1

    def test_chained_steps(self):
        wf = Workflow(
            name="chained",
            default_agent=self._agent(),
            verbose=False,
        )
        wf.add_step(WorkflowStep(name="s1", task_template="What is the weather in London?"))
        wf.add_step(
            WorkflowStep(
                name="s2",
                task_template='Summarize "{prev_answer}"',
            )
        )
        result = wf.run()
        assert result.success
        assert len(result.step_results) == 2

    def test_workflow_result_final_answer(self):
        from agent.agent import AgentResult
        wr = WorkflowResult(
            workflow_name="test",
            step_results=[
                AgentResult(task="t1", answer="first"),
                AgentResult(task="t2", answer="last"),
            ],
        )
        assert wr.final_answer == "last"

    def test_empty_workflow_result(self):
        wr = WorkflowResult(workflow_name="empty")
        assert wr.final_answer == ""
        assert wr.success  # vacuously true

    def test_add_step_chaining(self):
        wf = Workflow(name="chain", verbose=False)
        result = wf.add_step(WorkflowStep(name="s", task_template="t"))
        assert result is wf  # returns self

    def test_elapsed_seconds_recorded(self):
        wf = Workflow(
            name="timed",
            default_agent=self._agent(),
            verbose=False,
        )
        wf.add_step(WorkflowStep(name="s", task_template="Calculate 1 + 1"))
        result = wf.run()
        assert result.elapsed_seconds >= 0
