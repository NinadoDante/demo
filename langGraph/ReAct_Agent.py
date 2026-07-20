from langchain.tools import tool
from langchain.chat_models import init_chat_model

from llm_config import llm

from typing import Annotated, TypedDict, Literal

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, CachePolicy, interrupt
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.errors import NodeError

from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, ToolMessage
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from IPython.display import Image, display
from operator import add
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
load_dotenv()

# 定义两个工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    weather_data = {"北京": "晴天 25度", "上海": "多云 28度"}
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def square_root(x: float) -> str:
    """计算平方根"""
    return str(x ** 0.5)

tools = [get_weather, square_root]
tools_by_name = {t.name: t for t in tools}

# 初始化模型并绑定工具
model = llm
model_with_tools = model.bind_tools(tools)

from langgraph.graph.message import MessagesState
from langchain_core.messages import ToolMessage

def llm_node(state: MessagesState):
    """LLM 节点：调用模型，可能产生工具调用请求"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: MessagesState):
    """工具节点：执行 LLM 请求的工具，并封装结果"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": results}

from langgraph.graph import StateGraph, START, END
from typing import Literal

# 路由函数：LLM 返回的消息中有工具调用，就去 tools 节点，否则结束
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# 构建图
agent_graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", should_continue) # LLM 节点后根据条件跳转
    .add_edge("tools", "llm")                      # 工具节点执行后，回到 LLM 节点
    .compile()
)

display(Image(agent_graph.get_graph().draw_mermaid_png()))

# 调用
print("测试1: 不需要工具调用\n")
result = agent_graph.invoke({
    "messages": [HumanMessage(content="你好")]
})
for m in result['messages']:
    m.pretty_print()

print("测试2: 需要工具调用\n")
result = agent_graph.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？")]
})
for m in result['messages']:
    m.pretty_print()

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

agent_with_memory = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", should_continue) # LLM 节点后根据条件跳转
    .add_edge("tools", "llm")                      # 工具节点执行后，回到 LLM 节点
    .compile(checkpointer=InMemorySaver())         # 关键：注入检查点
)

# 调用时，通过 thread_id 区分会话
config = {"configurable": {"thread_id": "baize-001"}}
config2 = {"configurable": {"thread_id": "baize-002"}}
print("第 1 轮会话")
result = agent_with_memory.invoke({"messages": [HumanMessage("你好，我的名字是白泽")]}, config)
for m in result['messages']:
    m.pretty_print()

print("\n\n第 2 轮会话")
result = agent_with_memory.invoke({"messages": [HumanMessage("你知道我是谁吗？")]}, config2)
for m in result['messages']:
    m.pretty_print()