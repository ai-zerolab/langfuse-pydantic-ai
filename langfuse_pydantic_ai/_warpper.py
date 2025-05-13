from contextlib import asynccontextmanager
from functools import wraps
from inspect import signature
from typing import Any, AsyncIterator, Coroutine

from langfuse.decorators import langfuse_context, observe
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, StreamedResponse
from pydantic_ai.result import Usage


def _warp_model_request(model: Model) -> Model:
    origin_request = model.request
    sig = signature(origin_request)

    @wraps(origin_request)
    @observe(name="model-request", as_type="generation")
    async def _warpped(*args: Any, **kwargs: Any) -> Coroutine[Any, Any, ModelResponse]:
        bound_args = sig.bind(*args, **kwargs)
        bound_kwargs = dict(bound_args.arguments)

        langfuse_context.update_current_observation(
            input=bound_kwargs["messages"],
            model=model.model_name,
            model_parameters=bound_kwargs.get("model_parameters", None),
            metadata=bound_kwargs,
        )

        response = await origin_request(*args, **kwargs)
        usage = response.usage

        langfuse_context.update_current_observation(
            output=response,
            usage={
                "input": usage.request_tokens,
                "output": usage.response_tokens,
                "total": usage.total_tokens,
            },
        )

        return response

    model.request = _warpped

    return model


def _warp_model_request_stream(model: Model) -> Model:
    origin_request_stream = model.request_stream
    sig = signature(origin_request_stream)

    @wraps(origin_request_stream)
    @asynccontextmanager
    @observe(name="model-request-stream", as_type="generation")
    async def _warpped(*args: Any, **kwargs: Any) -> AsyncIterator[StreamedResponse]:
        bound_args = sig.bind(*args, **kwargs)
        bound_kwargs = dict(bound_args.arguments)

        langfuse_context.update_current_observation(
            input=bound_kwargs["messages"],
            model=model.model_name,
            model_parameters=bound_kwargs.get("model_parameters", None),
            metadata=bound_kwargs,
        )

        response: StreamedResponse
        async with origin_request_stream(*args, **kwargs) as response:
            yield response
            usage = response.usage()
            langfuse_context.update_current_observation(
                output=response.get(),
                usage={
                    "input": usage.request_tokens,
                    "output": usage.response_tokens,
                    "total": usage.total_tokens,
                },
            )

    model.request_stream = _warpped

    return model


def observed_model(model: Model) -> Model:
    model = _warp_model_request(model)
    model = _warp_model_request_stream(model)
    return model


def observed_agent(agent: Agent) -> Agent:
    if not agent.model:
        # FIXME: Maybe warming for no model to observe
        return agent
    agent.model = observed_model(agent.model)
    return agent


def use_observed_agent(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        agent = func(*args, **kwargs)
        return observed_agent(agent)

    return wrapper
