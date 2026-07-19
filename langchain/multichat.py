from langchain.messages import SystemMessage, HumanMessage, AIMessage
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

messages = [
    SystemMessage(content="你是一个记忆力很好的AI助手，用户会告诉你一些信息，请在最后准确复述所有信息。"),

    # 第一轮对话
    HumanMessage(content="你好，我叫张三，今年25岁，在北京工作。"),
    AIMessage(content="你好张三！记住了，你25岁，在北京工作。"),

    # 第二轮对话
    HumanMessage(content="我喜欢的编程语言是Python，最近在学LangChain框架。"),
    AIMessage(content="好的，你喜欢Python，正在学习LangChain框架。"),

    # 第三轮对话：验证记忆
    HumanMessage(content="请你复述一下关于我的所有信息。"),
]

response = llm.invoke(messages)

print(response.content)
