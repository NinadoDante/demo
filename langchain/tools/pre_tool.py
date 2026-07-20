from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from llm_config import llm

load_dotenv()

from langchain_tavily import TavilySearch

# tool = TavilySearch(max_results=5, topic="general")

tavily = TavilySearch(max_results=5, topic="general")

@tool
def web_search(query: str):
    """Search the web for information"""
    return tavily.invoke(query)
from pydantic import BaseModel, Field

class Reference(BaseModel):
    title: str = Field(description="The title of the cited web page")
    url: str = Field(description="The url of the cited web page")

class AnswerInfo(BaseModel):
    answer: str = Field(description="The final answer for user")
    reference: list[Reference] = Field(description="The web pages cited")

agent = create_agent(
    llm,
    tools=[web_search],
    system_prompt="你是一个智能助手，使用工具解决问题。回答时请引用信息来源。",
    response_format=AnswerInfo
)

for event in agent.stream(
    {"messages": [HumanMessage(content="蒸蚌是什么梗？")]},
    stream_mode=["messages", "updates"]
):
    mode, data = event
    if mode == "messages":
        chunk, metadata = data
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    elif mode == "updates":
        for node_name, node_output in data.items():
            if "structured_response" in node_output:
                result = node_output["structured_response"]

print()
print(f"\n回答：{result.answer}")
print(f"引用：{result.reference}")


