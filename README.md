![](https://img.shields.io/github/license/wh1isper/langfuse-pydantic-ai)
![](https://img.shields.io/github/v/release/wh1isper/langfuse-pydantic-ai)
![](https://img.shields.io/github/last-commit/wh1isper/langfuse-pydantic-ai)
![](https://img.shields.io/pypi/pyversions/langfuse-pydantic-ai)

# langfuse_pydantic_ai

> This is a third-party package, not officially maintained by Langfuse. If Langfuse requires for this package, feel free to contact me.

A simple wrapper, send trace to langfuse when using pydantic-ai

## Install

`pip install langfuse-pydantic-ai`

## Usage

TL;DR

```python
from langfuse_pydantic_ai import observed_agent

agent = observed_agent(agent)
```

Full example:

```python
import asyncio

from pydantic_ai.agent import Agent
from langfuse.decorators import observe
from langfuse_pydantic_ai import observed_agent

@observe # Add this decorator to span a trace
async def main():
    agent = Agent(
        "google-gla:gemini-1.5-flash",
        # Register a static system prompt using a keyword argument to the agent.
        # For more complex dynamically-generated system prompts, see the example below.
        system_prompt="Be concise, reply with one sentence.",
    )
    agent = observed_agent(agent)
    result = await agent.run('Where does "hello world" come from?')
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

If using custom model, use `observed_model` instead

```python
from pydantic_ai.agent import Agent
from langfuse_pydantic_ai import observed_model

model = observed_model(model)
agent = Agent(model=model)
```

If using agent factory function, use `@use_observed_agent` directly

```python
from pydantic_ai.agent import Agent
from langfuse_pydantic_ai import use_observed_agent

@use_observed_agent
def init_agent() -> Agent:
    return Agent(
        "google-gla:gemini-1.5-flash",
        # Register a static system prompt using a keyword argument to the agent.
        # For more complex dynamically-generated system prompts, see the example below.
        system_prompt="Be concise, reply with one sentence.",
    )
```

Configuration via environment variables:

```bash
LANGFUSE_HOST=<langfuse_host>
LANGFUSE_PUBLIC_KEY=<langfuse_public_key>
LANGFUSE_SECRET_KEY=<langfuse_secret_key>
```

## Develop

Install pre-commit before commit

```
pip install pre-commit
pre-commit install
```

Install package locally

```
pip install -e .[test]
```

Run unit-test before PR

```
pytest -v
```
