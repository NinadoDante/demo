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
# 默认 Reducer，实现数据覆盖
# 1. State: 定义共享数据结构，这里采用默认Reducer
class DefaultReducerState(TypedDict):
    val: str

# 2. Node: 定义处理逻辑
def node_1(state: DefaultReducerState):
    print(f"节点1接收值: {state['val']}")
    return {"val": "node_1"}

def node_2(state: DefaultReducerState):
    print(f"节点2接收值: {state['val']}")
    return {"val": "node_2"}

# 3. Edge: 编排执行顺序
default_reducer_graph = (
    StateGraph(DefaultReducerState)
    .add_node("node_1", node_1)
    .add_node("node_2", node_2)
    .add_edge(START, "node_1")
    .add_edge("node_1", "node_2")
    .add_edge("node_2", END)
    .compile()
)

result = default_reducer_graph.invoke(DefaultReducerState(val="default"))
print(result)


# 自定义 Reducer，实现数据累加而非覆盖
from operator import add
from typing import Annotated, TypedDict

class CustomReducerState(TypedDict):
    count: int                       # 这个字段没有定义Reducer，也就是默认会覆盖旧值
    nodes: Annotated[list[str], add] # 这里使用add作为Reducer，也就是会拼接结果


# 2. Node: 定义处理逻辑
def node_1(state: CustomReducerState):
    print(f"节点1接收count: {state['count']}")
    return {"count": 1, "nodes": ["node_1"]} # 直接修改nodes值，不用自己拼接

def node_2(state: CustomReducerState):
    print(f"节点2接收count: {state['count']}")
    return {"count": 2, "nodes": ["node_2"]} # 直接修改nodes值，不用自己拼接

# 3. Edge: 编排执行顺序
custom_reducer_graph = (
    StateGraph(CustomReducerState)
    .add_node("node_1", node_1)
    .add_node("node_2", node_2)
    .add_edge(START, "node_1")
    .add_edge("node_1", "node_2")
    .add_edge("node_2", END)
    .compile()
)

result = custom_reducer_graph.invoke(CustomReducerState(count=0, nodes=[]))
print(result)