import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from langchain_core.tools import tool
import requests
from datetime import datetime
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv

from llm_config import llm

load_dotenv()


# ========== 工具定义 ==========

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    url = f"http://wttr.in/{city}?format=j1"
    try:
        data = requests.get(url).json()
        current = data.get("current_condition", [{}])[0]
        if current:
            return f"{city}当前温度：{current['temp_C']}°C，天气：{current['weatherDesc'][0]['value']}"
        return f"未找到{city}的天气信息"
    except Exception as e:
        return f"天气查询失败: {e}"

@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"


tavily = TavilySearch(max_results=5, topic="general")

@tool
def web_search(query: str) -> str:
    """使用 Tavily 搜索引擎进行网络检索"""
    return tavily.invoke(query)

# ========== 初始化模型 ==========
# llm = ChatOpenAI(
#     base_url=os.getenv("SILICONFLOW_BASE_URL"),
#     api_key=os.getenv("SILICONFLOW_API_KEY"),
#     model="Qwen/Qwen3-8B"
# )

# ========== 创建 Agent ==========
agent = create_agent(model=llm, tools=[get_weather, get_current_time, web_search])

async def stream_llm_response(query: str):
    """异步生成器，流式返回 Agent 的回复"""
    # 使用 agent.stream 发送消息，stream_mode="messages" 可逐 token 输出
    async for chunk, metadata in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages"
    ):
        # 只处理 AI 消息的内容
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content