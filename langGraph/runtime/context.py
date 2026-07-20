from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore


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

# 1. 创建 Store
store = InMemoryStore()
# 1. 定义 Context 结构
@dataclass
class UserContext:
    user_id: str = ""

# 2. 在工具中访问 Context
@tool
def get_my_profile(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的偏好设置（从偏好存储中读取）"""
    if runtime.store is None:
        return "Store not available"
    user_id = runtime.context.user_id
    # 从 preferences 命名空间读取
    pref = runtime.store.get(("preferences",), user_id)
    if pref is None:
        return "未找到您的偏好信息，请先设置偏好。"
    return f"您的偏好：{pref.value}"

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    保存用户的偏好设置（键值对形式），例如：
    key: "style", value: "简洁易懂"
    key: "language", value: "中文"
    """
    if runtime.store is None:
        return "Store not available"
    user_id = runtime.context.user_id
    # 读取已有的偏好（如果有）
    existing = runtime.store.get(("preferences",), user_id)
    prefs = existing.value.copy() if existing else {}
    # 更新键值
    prefs[key] = value
    # 写入 store
    runtime.store.put(("preferences",), user_id, prefs)
    return f"已保存偏好：{key} -> {value}"

# 3. 创建 Agent 并注册 Context
agent = create_agent(
    llm,
    tools=[get_my_profile, save_user_preference],
    store=store,
    context_schema=UserContext  # 注册 Context 类型
)

# 4. 调用时传入 Context 数据
def chat_with_agent(user_id: str, message: str):
    """辅助函数，方便多次调用"""
    response = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        context=UserContext(user_id=user_id)
    )
    for msg in response['messages']:
        msg.pretty_print()
    print("\n" + "="*50 + "\n")

# 第一次：查询偏好（此时还没有）
chat_with_agent("user_001", "我的偏好是什么？")

# 第二次：让 Agent 记住偏好
chat_with_agent("user_001", "你帮我记一下我的偏好呗，我比较喜欢简洁易懂的中文回复")

# 第三次：再次查询偏好（应该能查到了）
chat_with_agent("user_001", "我的偏好是什么？")