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


# 1. State: 定义共享数据结构
class SimpleState(TypedDict):
    name: str
    greeting: str

# 2. Node: 定义处理逻辑
def greet_node(state: SimpleState):
    print(f"接收名字: {state['name']}")
    return {"greeting": f"Hello, {state['name']}!"}

def uppercase_node(state: SimpleState):
    print(f"接收问候语: {state['greeting']}")
    return {"greeting": state["greeting"].upper()}

def end_node(state: SimpleState):
    print(f"接收问候语: {state['greeting']}")
    return {"greeting": f"{state['greeting']}!    end"}

# 3. Edge: 编排执行顺序
# 3.1. 用State创建图
graph_builder = StateGraph(SimpleState)
# 3.2. 注册节点node
graph_builder.add_node("greet", greet_node)
graph_builder.add_node("uppercase", uppercase_node)
graph_builder.add_node("end", end_node)

# 3.3. 创建边edge，连接各个节点
graph_builder.add_edge(START, "greet")       # START -> greet
graph_builder.add_edge("greet", "uppercase") # greet -> uppercase
# graph_builder.add_edge("uppercase", END)     # uppercase -> END
graph_builder.add_edge("uppercase", "end")
graph_builder.add_edge("end", END)


# 3.4. 编译成可执行图
graph = graph_builder.compile()

# 执行（必须传入初始化的 State）
result = graph.invoke({"name": "World"})
print(f"结果: {result['greeting']}") # 输出: HELLO, WORLD!

# # 可视化图结构
# display(Image(graph.get_graph().draw_mermaid_png()))

