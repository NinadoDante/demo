from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

# 1. 定义工具：用 @tool 装饰器和函数文档字符串来描述工具
@tool
def get_weather(location: str) -> str:
    """
    Get the weather in a given location.
    Args:
        location: city name or coordinates
    """
    return f"Current weather in {location} is sunny"

@tool
def get_math_executor(symbol: str, a:int, b:int) -> int | str:
    """
    Perform basic arithmetic calculation with two integers.
    Args:
        symbol: arithmetic operator, one of '+', '-', '*', '/'
        a: first integer operand
        b: second integer operand
    """
    if symbol == "+":
        return a + b
    elif symbol == "-":
        return a - b
    elif symbol == "*":
        return a * b
    elif symbol == "/":
        return a // b
    else:
        return "Invalid symbol"

# 2. 创建 LLM：使用 SiliconFlow 平台（兼容 OpenAI 接口）
# llm = ChatOpenAI(
#     model="Qwen/Qwen3.6-27B",
#     api_key=os.getenv("API_KEY", "sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc"),
#     base_url="https://api.siliconflow.cn/v1"
# )
llm = init_chat_model(
    model="Qwen/Qwen3.6-27B",
    model_provider="openai",
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
# 3. 创建 Agent：传入 LLM 实例和工具集
agent = create_agent(
    llm
)
multimodal_message = HumanMessage(
    content=[
        {"type": "image", "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
        {"type": "text", "text": "这张图片描绘了什么内容？"}
    ]
)

for token, metadata in agent.stream(
    {"messages": [multimodal_message]},
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)

# # 4. 调用 Agent（流式输出，逐 token）
# print("🚀 正在调用大模型...")
#
# for token, metadata in agent.stream(
#     {"messages": [{"role": "user", "content": "帮我计算 2341+123等于多少?"}]},
#     stream_mode="messages"
# ):
#     if token.content:
#         print(token.content, end="", flush=True)
#
# print()

# # 4. 调用 Agent（流式输出）
# print("🚀 正在调用大模型...")
# input_messages = {
#     "messages": [
#         {"role": "user", "content": "帮我计算 2341+123等于多少?"}
#     ]
# }
#
# for event in agent.stream(input_messages, stream_mode="updates"):
#     for node_name, node_output in event.items():
#         if "messages" in node_output:
#             for msg in node_output["messages"]:
#                 if msg.type == "ai" and msg.content:
#                     print(f"🤖 助手回复: {msg.content}")
#                 elif msg.type == "tool":
#                     print(f"🔧 工具调用 [{msg.name}]: {msg.content}")
#

# # 5. 输出结果
# messages = response["messages"]
# for msg in messages:
#     if msg.type == "ai" and msg.content:
#         print(f"🤖 助手回复: {msg.content}")
#     elif msg.type == "tool":
#         print(f"🔧 工具调用 [{msg.name}]: {msg.content}")
