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



class ParallelState(TypedDict):
    nodes: Annotated[list[str], add]

def node_a(state: ParallelState):
    print(f"执行a节点: {state['nodes']}")
    return {"nodes": ["a"]}

def node_b(state: ParallelState):
    print(f"执行b节点: {state['nodes']}")
    return {"nodes": ["b"]}

def node_c(state: ParallelState):
    print(f"执行c节点: {state['nodes']}")
    return {"nodes": ["c"]}

def node_d(state: ParallelState):
    print(f"执行d节点: {state['nodes']}")
    return {"nodes": ["d"]}

parallel_graph = (
    StateGraph(ParallelState)
        .add_node("a", node_a)
        .add_node("b", node_b)
        .add_node("c", node_c)
        .add_node("d", node_d)
        .add_edge(START, "a") # start -> a
        .add_edge("a", "b")   # a -> b
        .add_edge("a", "c")   # a -> c
        .add_edge("b", "d")   # b -> d
        .add_edge("c", "d")   # c -> d
        .add_edge("d", END)   # d -> end
        .compile()
)

display(Image(parallel_graph.get_graph().draw_mermaid_png()))

result = parallel_graph.invoke(ParallelState(nodes=[]))
print(result)