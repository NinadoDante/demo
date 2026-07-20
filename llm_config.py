import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

llm = init_chat_model(
    model="Qwen/Qwen3.6-35B-A3B",
    model_provider="openai",
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
llm0 = ChatOpenAI(
    model="Qwen/Qwen3.6-35B-A3B",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL")
)
# deepseek-ai/DeepSeek-V4-Flash
# Qwen/Qwen3.6-27B
# Qwen/Qwen3.6-35B-A3B
