"""
This file is copied from pydantic-ai, warped to use langfuse-pydantic-ai observed
"""

import os
from datetime import timezone
from typing import Any, Iterator

import pytest
from dirty_equals import IsNow as _IsNow
from dirty_equals import IsStr
from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Usage

from langfuse_pydantic_ai import observed_agent


def Agent(*args, **kwargs) -> PydanticAgent:
    return observed_agent(PydanticAgent(*args, **kwargs))


def IsNow(*args: Any, **kwargs: Any):
    # Increase the default value of `delta` to 10 to reduce test flakiness on overburdened machines
    if "delta" not in kwargs:
        kwargs["delta"] = 10
    return _IsNow(*args, **kwargs)


pytestmark = pytest.mark.anyio


class TestEnv:
    __test__ = False

    def __init__(self):
        self.envars: dict[str, str | None] = {}

    def set(self, name: str, value: str) -> None:
        self.envars[name] = os.getenv(name)
        os.environ[name] = value

    def remove(self, name: str) -> None:
        self.envars[name] = os.environ.pop(name, None)

    def reset(self) -> None:  # pragma: no cover
        for name, value in self.envars.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def env() -> Iterator[TestEnv]:
    test_env = TestEnv()

    yield test_env

    test_env.reset()


def test_result_tuple():
    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), result_type=tuple[str, str])

    result = agent.run_sync("Hello")
    assert result.output == ("foo", "bar")


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    result = agent.run_sync("Hello")
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {"a": 1, "b": "foo"}


async def test_streamed_text_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f"{x}-apple"

    async with test_agent.run_stream("Hello") as result:
        assert test_agent.name == "test_agent"
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="Hello", timestamp=IsNow(tz=timezone.utc))]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name="ret_a", args={"x": "a"}, tool_call_id=IsStr())],
                    usage=Usage(request_tokens=51, response_tokens=0, total_tokens=51),
                    model_name="test",
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="ret_a",
                            tool_call_id=IsStr(),
                            content="a-apple",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ]
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=5,
                total_tokens=108,
            )
        )
        response = await result.get_output()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="Hello", timestamp=IsNow(tz=timezone.utc))]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name="ret_a", args={"x": "a"}, tool_call_id=IsStr())],
                    usage=Usage(request_tokens=51, response_tokens=0, total_tokens=51),
                    model_name="test",
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="ret_a",
                            tool_call_id=IsStr(),
                            content="a-apple",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='{"ret_a":"a-apple"}')],
                    usage=Usage(request_tokens=52, response_tokens=11, total_tokens=63),
                    model_name="test",
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=11,
                total_tokens=114,
            )
        )
