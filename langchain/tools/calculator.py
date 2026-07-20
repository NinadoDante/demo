
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

from llm_config import llm


class CalculatorInput(BaseModel):
    num1: float = Field(description="第一个数字")
    num2: float = Field(description="第二个数字")
    operator: Literal["+", "-", "*", "/"] = Field(description="运算符：+加、-减、*乘、/除")


@tool(args_schema=CalculatorInput)
def calculator(num1: float, num2: float, operator: str) -> str:
    """Perform basic arithmetic operations: addition, subtraction, multiplication, division."""
    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        if num2 == 0:
            return "错误：除数不能为零"
        result = num1 / num2
    else:
        return f"不支持的运算符: {operator}"
    return f"{num1} {operator} {num2} = {result}"


agent = create_agent(
    llm,
    tools=[calculator],
    system_prompt="你是一个智能计算助手，使用工具来完成数学运算。"
)

for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="123 加 456 等于多少？")]},
        stream_mode="messages"
):
    print(token.content, end="", flush=True)
