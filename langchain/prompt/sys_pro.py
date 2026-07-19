from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model="Qwen/Qwen3.6-27B",
    model_provider="openai",
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
# agent = create_agent(
#     llm,
#     system_prompt="像海盗一样说话."
# )
#
# for token, metadata in agent.stream(
#     {"messages": [HumanMessage(content="你是谁？")]},
#     stream_mode="messages"
# ):
#    print(token.content, end="", flush=True)

# from pydantic import BaseModel
#
#
# class CapitalInfo(BaseModel):
#    name: str
#    location: str
#    vibe: str
#    economy: str

system_prompt = """
# 身份
你是一个城市美食鉴赏家
输出包含城市名、三道特色菜和一句总结
# 示例
用户：成都
助手：成都的特色菜有火锅、串串香和兔头，非常值得尝试。
"""
agent = create_agent(
   llm,
   system_prompt = system_prompt
)

for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="北京")]},
    stream_mode="messages"
):
    print(token.content, end="", flush=True)
