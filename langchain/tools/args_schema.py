from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

from llm_config import llm
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5

class WeatherInput(BaseModel):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="Temperature unit")
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


# # 调用数学工具
# print(tool1.invoke({"x": 467}))
#
# # 调用查询天气工具
# print(get_weather.invoke({"location": "杭州", "include_forecast": True}))

# 创建智能体，并添加工具
agent = create_agent(
    llm,
    tools=[tool1, get_weather],
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。"
)

# # 调用智能体
# for token, metadata in agent.stream(
#         {"messages": [HumanMessage(content="467的平方根是多少?")]},
#         stream_mode="messages"
# ):
#     print(token.content, end="", flush=True)
for chunk in agent.stream(
    {"messages": [HumanMessage(content="467、529的平方根是多少?")]},
    stream_mode="updates"
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
        print()

for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="北京和杭州接下来几天天气如何?")]},
        stream_mode="messages"
):
    print(token.content, end="", flush=True)