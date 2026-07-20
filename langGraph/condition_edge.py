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


# 条件边示例：根据state决定走哪条路
class RouteState(TypedDict):
    score: int
    result: str

def scorer(state: RouteState):
    score = int(input("请输入分数:"))
    print(score)
    return {"score": score}  # 模拟评分

def pass_node(state: RouteState):
    return {"result": "通过！"}

def fail_node(state: RouteState):
    return {"result": "不通过"}

def router(state: RouteState) -> Literal["pass", "fail"]:
    """根据score决定下一节点"""
    if state["score"] >= 60:
        return "pass"
    return "fail"

condition_graph = (
    StateGraph(RouteState)
    .add_node("scorer", scorer)         # 打分节点
    .add_node("pass", pass_node)        # 成功节点
    .add_node("fail", fail_node)        # 失败节点
    .add_edge(START, "scorer")               # start -> scorer
    .add_conditional_edges("scorer", router) # scorer -> conditional_edge(router)
    .add_edge("pass", END)                   # pass -> end
    .add_edge("fail", END)                   # fail -> end
    .compile()
)

display(Image(condition_graph.get_graph().draw_mermaid_png()))

result = condition_graph.invoke({"score": 0, "result": ""})
print(f"score={result['score']} -> {result['result']}")