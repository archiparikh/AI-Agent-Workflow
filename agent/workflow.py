"""
workflow.py – Multi-step, multi-agent workflow orchestration.

A :class:`Workflow` is a named sequence of :class:`WorkflowStep` objects.
Each step can run the same agent on a new task, pass the previous step's
answer into the next task (chaining), or fan-out to multiple agents in
parallel (simulated here sequentially for simplicity).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, AgentResult


# ---------------------------------------------------------------------------
# WorkflowStep
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """
    A single step in a :class:`Workflow`.

    Parameters
    ----------
    name:
        Human-readable label for this step.
    task_template:
        Task string given to the agent.  Use ``{prev_answer}`` as a
        placeholder that will be filled with the previous step's answer.
    agent:
        The :class:`Agent` to run this step.  If *None* the workflow's
        default agent is used.
    transform:
        Optional callable ``(prev_result: AgentResult) -> str`` that
        produces the task string from the previous result.  Takes
        precedence over *task_template*.
    """
    name: str
    task_template: str
    agent: Optional[Agent] = None
    transform: Optional[Callable[[AgentResult], str]] = None

    def build_task(self, prev_result: Optional[AgentResult] = None) -> str:
        if self.transform and prev_result is not None:
            return self.transform(prev_result)
        if prev_result is not None:
            return self.task_template.format(prev_answer=prev_result.answer)
        return self.task_template


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Aggregated result of an entire workflow run."""
    workflow_name: str
    step_results: List[AgentResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def final_answer(self) -> str:
        return self.step_results[-1].answer if self.step_results else ""

    @property
    def success(self) -> bool:
        return all(r.success for r in self.step_results)

    def pretty_print(self) -> None:
        print(f"\n{'#'*60}")
        print(f"Workflow : {self.workflow_name}")
        print(f"Steps    : {len(self.step_results)}")
        print(f"Success  : {self.success}")
        print(f"Elapsed  : {self.elapsed_seconds:.2f}s")
        print(f"{'#'*60}")
        for result in self.step_results:
            result.pretty_print()
        print(f"\n>>> Final answer: {self.final_answer}")
        print(f"{'#'*60}\n")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class Workflow:
    """
    Orchestrates a sequence of agent steps.

    Parameters
    ----------
    name:
        Descriptive name for the workflow.
    default_agent:
        Fallback agent used for steps that don't specify their own.
    steps:
        Ordered list of :class:`WorkflowStep` objects.
    verbose:
        Print progress to stdout.
    """

    def __init__(
        self,
        name: str,
        default_agent: Optional[Agent] = None,
        steps: Optional[List[WorkflowStep]] = None,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.default_agent = default_agent or Agent(name="DefaultAgent")
        self.steps: List[WorkflowStep] = steps or []
        self.verbose = verbose

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """Append a step and return *self* for chaining."""
        self.steps.append(step)
        return self

    def run(self) -> WorkflowResult:
        """Execute all steps in order and return a :class:`WorkflowResult`."""
        if self.verbose:
            print(f"\n{'*'*60}")
            print(f"Starting workflow: {self.name}")
            print(f"Total steps: {len(self.steps)}")
            print(f"{'*'*60}")

        start = time.monotonic()
        step_results: List[AgentResult] = []
        prev_result: Optional[AgentResult] = None

        for i, step in enumerate(self.steps, start=1):
            if self.verbose:
                print(f"\n--- Workflow step {i}/{len(self.steps)}: {step.name} ---")

            agent = step.agent or self.default_agent
            task = step.build_task(prev_result)
            result = agent.run(task)
            step_results.append(result)
            prev_result = result

        elapsed = time.monotonic() - start
        return WorkflowResult(
            workflow_name=self.name,
            step_results=step_results,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def sequential(
        cls,
        name: str,
        tasks: List[str],
        agent: Optional[Agent] = None,
        verbose: bool = True,
    ) -> "Workflow":
        """Build a simple sequential workflow from a flat list of task strings."""
        wf = cls(name=name, default_agent=agent, verbose=verbose)
        for task in tasks:
            wf.add_step(WorkflowStep(name=task[:50], task_template=task))
        return wf
