# AI Agent Workflow

A lightweight, dependency-free Python framework demonstrating the core
concepts behind modern AI agent architectures.  It implements the
**ReAct** (Reasoning + Acting) loop, a pluggable **tool registry**, and a
**workflow orchestrator** that chains multiple agents together — all
runnable with the Python standard library alone (no API keys required).

---

## Architecture

```
┌────────────────────────────────────────────────┐
│                   Workflow                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │  Step 1  │──▶│  Step 2  │──▶│  Step N  │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│       │              │              │           │
│  ┌────▼─────┐   ┌────▼─────┐   ┌───▼──────┐   │
│  │  Agent   │   │  Agent   │   │  Agent   │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│       │              │              │           │
│  ┌────▼─────────────────────────────▼─────┐    │
│  │              Tool Registry              │    │
│  │  calculator │ text_search │ get_weather │    │
│  │             │ summarize_text            │    │
│  └────────────────────────────────────────┘    │
└────────────────────────────────────────────────┘
```

### ReAct Loop (per Agent step)

```
Task
 │
 ▼
Thought ──▶ Action (tool call) ──▶ Observation
   ▲                                    │
   └────────────── (loop) ──────────────┘
                        │
                    Final Answer
```

---

## Project Layout

```
AI-Agent-Workflow/
├── agent/
│   ├── __init__.py      # Package exports
│   ├── agent.py         # ReAct agent (Thought/Action/Observation loop)
│   ├── tools.py         # Tool registry + built-in tools
│   └── workflow.py      # Multi-step workflow orchestrator
├── examples/
│   ├── custom_tool.py          # Register & use a custom tool
│   └── multi_agent_workflow.py # Two specialised agents in one workflow
├── tests/
│   ├── test_agent.py    # Agent unit tests
│   ├── test_tools.py    # Tool unit tests
│   └── test_workflow.py # Workflow unit tests
├── main.py              # Runnable demos (6 scenarios)
└── requirements.txt
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/archiparikh/AI-Agent-Workflow.git
cd AI-Agent-Workflow

# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install test dependencies
pip install -r requirements.txt

# Run all demos
python main.py

# Run the test suite
python -m pytest tests/ -v
```

---

## Demos

| # | Demo | What it shows |
|---|------|---------------|
| 1 | Calculator Agent | Math expression evaluation via the `calculator` tool |
| 2 | Weather Agent | Mock weather look-up via the `get_weather` tool |
| 3 | Search Agent | Keyword search over an in-memory knowledge base |
| 4 | Summarisation Agent | First-N-sentences extraction via `summarize_text` |
| 5 | Chained Workflow | Search → Summarise pipeline using `Workflow` |
| 6 | List Tools | Print all registered tools with their signatures |

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `calculator(expression)` | Evaluates safe Python math expressions |
| `text_search(query, knowledge_base)` | Keyword search over a JSON list of strings |
| `get_weather(city)` | Mock weather report (replace with a real API) |
| `summarize_text(text, num_sentences)` | Returns the first N sentences of a text |

---

## Adding a Custom Tool

```python
from agent.tools import ToolRegistry, ToolSpec

registry = ToolRegistry()
registry.register(
    ToolSpec(
        name="shout",
        description="Convert text to upper-case.",
        func=lambda text: text.upper(),
        parameters={"text": "str"},
    )
)
```

Pass the registry to any `Agent`:

```python
from agent import Agent
agent = Agent(name="ShoutAgent", registry=registry)
```

See [`examples/custom_tool.py`](examples/custom_tool.py) for a complete example.

---

## Plugging in a Real LLM

Replace the `_think` method on any `Agent` instance (or subclass) with a
call to your preferred LLM:

```python
import openai

def openai_think(task, context):
    messages = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user",   "content": "\n".join(context)},
    ]
    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages,
        # use function-calling / tool_choice to get structured output
    )
    # parse thought / action / action_input from response …

agent._think = openai_think
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

41 tests covering tools, agent logic, and workflow orchestration.
