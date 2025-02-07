from contextlib import AbstractAsyncContextManager
from functools import wraps
from typing import Any, Coroutine

from pydantic_ai.agent import Agent
from pydantic_ai.result import RunResult, StreamedRunResult


def _warp_run(agent: Agent) -> Agent:
    original_run = agent.run

    @wraps(original_run)
    async def _warpped(*args: Any, **kwargs: Any) -> Coroutine[Any, Any, RunResult[str]]:
        result = await original_run(*args, **kwargs)
        return result

    agent.run = _warpped
    return agent


def _warp_stream(agent: Agent) -> Agent:
    original_run_stream = agent.run_stream

    @wraps(original_run_stream)
    async def _warpped(
        *args: Any, **kwargs: Any
    ) -> AbstractAsyncContextManager[StreamedRunResult[None, str], bool | None]:
        result = await original_run_stream(*args, **kwargs)
        return result

    agent.run_stream = _warpped
    return agent


def observed_agent(agent: Agent) -> Agent:
    agent = _warp_run(agent)
    agent = _warp_stream(agent)
    return agent
