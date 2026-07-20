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

def llm_node(state: MessagesState):
    """LLM 节点：调用模型，可能产生工具调用请求"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: MessagesState):
    """工具节点：执行前中断，等待人工确认（HITL）"""
    last_message = state["messages"][-1]

    tool_calls_info = [
        {"name": tc["name"], "args": tc["args"]}
        for tc in last_message.tool_calls
    ]
    human_decision = interrupt({
        "question": "是否执行以下工具调用？",
        "tool_calls": tool_calls_info
    })

    if human_decision != "approved":
        return {"messages": [ToolMessage(
            content="用户拒绝了工具执行",
            tool_call_id=last_message.tool_calls[0]["id"]
        )]}

    results = []
    for tool_call in last_message.tool_calls:
        t = tools_by_name[tool_call["name"]]
        result = t.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": results}

# 路由函数：LLM 返回的消息中有工具调用，就去 tools 节点，否则结束
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# 构建图（必须带 checkpointer 才能支持 interrupt）
agent_graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", should_continue)
    .add_edge("tools", "llm")
    .compile(checkpointer=InMemorySaver())
)

display(Image(agent_graph.get_graph().draw_mermaid_png()))

# ===== HITL 调用演示 =====
config = {"configurable": {"thread_id": "hitl-demo-001"}}

print("测试: 需要工具调用（触发人工确认）\n")
result = agent_graph.invoke(
    {"messages": [HumanMessage(content="北京今天天气怎么样？")]},
    config
)

# 第一次 invoke 会在 interrupt 处暂停，检查图状态
snapshot = agent_graph.get_state(config)
if snapshot.next:
    print("⏸️  图已中断，等待人工确认...")
    print(f"中断信息: {snapshot.tasks[0].interrupts[0].value}")

    # 模拟人工确认：approved 表示同意执行
    human_input = input("是否执行工具？(approved/rejected): ").strip()

    result = agent_graph.invoke(
        Command(resume=human_input),
        config
    )

print("\n最终对话记录：")
for m in result['messages']:
    m.pretty_print()

# ===== 不需要工具的测试（不会触发中断）=====
print("\n\n测试: 不需要工具调用（无中断）\n")
config2 = {"configurable": {"thread_id": "hitl-demo-002"}}
result = agent_graph.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config2
)
for m in result['messages']:
    m.pretty_print()
