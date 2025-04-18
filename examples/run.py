import asyncio

from pydantic_ai.agent import Agent

from langfuse_pydantic_ai import observed_agent


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
