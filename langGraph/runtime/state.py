from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from llm_config import llm


from langchain.agents import AgentState
from typing import NotRequired

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from datetime import datetime


class CustomState(AgentState):
    """Agent的自定义任务状态"""
    model_call_count: NotRequired[int]  # 模型调用次数
    session_start: NotRequired[str]     # 会话开始时间


@tool
def update_state(runtime: ToolRuntime):
    """更新Agent状态：计数并记录开始时间"""
    # 获取当前状态
    count = runtime.state.get("model_call_count", 0)

    # 构造更新指令
    command = {
        "model_call_count": count + 1,
        "messages": [ToolMessage("状态已更新", tool_call_id=runtime.tool_call_id)]
    }
    # 在首次调用时记录会话开始时间
    if count == 0:
        command['session_start'] = datetime.now().isoformat()

    # 通过 Command 返回对 State 的更新
    return Command(update=command)

agent = create_agent(
    llm,
    tools=[update_state],
    state_schema=CustomState, # 关键：注册自定义State
    checkpointer=InMemorySaver(),
    system_prompt="你是一个助手，每次收到用户消息都必须调用 update_state 工具。"
)

config = {"configurable": {"thread_id": "baize-003"}}
print("第 1 轮对话")
result = agent.invoke({"messages": [HumanMessage("你好，我是白泽")]}, config)
for m in result['messages']:
    m.pretty_print()
print("\n\n第 2 轮对话")
result = agent.invoke({"messages": [HumanMessage("你知道我是谁吗？")]}, config)
for m in result['messages']:
    m.pretty_print()
print("\n\n第 3 轮对话")
result = agent.invoke({"messages": [HumanMessage("讲个小红帽的故事")]}, config)
for m in result['messages']:
    m.pretty_print()