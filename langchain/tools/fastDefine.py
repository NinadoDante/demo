from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from llm_config import llm
# 1. 用 @tool 装饰器定义工具
@tool
def get_weather(location: str) -> str:
    """Get the weather in a given location."""
    return f"Current weather in {location} is sunny"

# 2. 创建 Agent 并绑定工具
agent = create_agent(llm, tools=[get_weather])

# 3. 调用——Agent 会自动判断需要调用 get_weather
response = agent.invoke({
    "messages": [HumanMessage(content="杭州今天天气如何?")]
})
for msg in response['messages']:
    msg.pretty_print()