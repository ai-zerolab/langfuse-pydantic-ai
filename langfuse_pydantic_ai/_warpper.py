from functools import wraps
from typing import Any, AsyncIterator

from langfuse.decorators import langfuse_context, observe
from pydantic_ai.agent import Agent
from pydantic_ai.models import Model, StreamedResponse
from pydantic_ai.result import RunResult


def _warp_model_request(model: Model) -> Model:
    origin_request = model.request

    @observe(name="model-request", as_type="generation")
    @wraps(origin_request)
    async def _warpped(*args: Any, **kwargs: Any) -> RunResult[str]:

        model_settings = model.settings

        result = origin_request(*args, **kwargs)

        return result

    model.request = _warpped

    return model


def _warp_model_request_stream(model: Model) -> Model:
    origin_request_stream = model.request_stream

    @observe(name="model-request-stream", as_type="generation")
    @wraps(origin_request_stream)
    async def _warpped(*args: Any, **kwargs: Any) -> AsyncIterator[StreamedResponse]:
        return origin_request_stream(*args, **kwargs)

    model.request_stream = _warpped

    return model


def observe_model(model: Model) -> Model:
    model = _warp_model_request(model)
    model = _warp_model_request_stream(model)
    return model


def observed_agent(agent: Agent) -> Agent:
    agent.model = observe_model(agent.model)
    return agent
