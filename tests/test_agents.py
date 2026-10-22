"""Unit tests for Graphflow Agent and Tool loop execution."""

from graphflow import Agent, Application, MockModel, tool
from graphflow.models.base import ModelResponse, ToolCall


def test_single_agent_direct_response():
    mock_model = MockModel(responses=["Hello, I am a test agent!"])
    agent = Agent(name="greeter", model=mock_model)

    res = agent.execute({"input": "Hello!"})
    assert res["output"] == "Hello, I am a test agent!"


def test_agent_tool_calling_loop():
    @tool
    def get_weather(city: str) -> str:
        """Returns the weather in a city."""
        return f"Sunny in {city}, 22C"

    # Simulated ReAct sequence:
    # 1. Model requests tool call to get_weather(city="Paris")
    # 2. Model receives tool output and gives final response
    mock_model = MockModel(
        responses=[
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})],
            ),
            ModelResponse(content="The weather in Paris is sunny, 22C."),
        ]
    )

    agent = Agent(
        name="weather_agent",
        model=mock_model,
        tools=[get_weather],
    )

    app = Application("agent_app")
    app.register(agent)

    result = app.run({"query": "What is the weather in Paris?"})
    assert result["output"] == "The weather in Paris is sunny, 22C."
