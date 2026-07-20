import os
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import dotenv
dotenv.load_dotenv()


# 1. 初始化组件
tavily = TavilySearch(max_results=5, topic="general")


def get_current_time() -> str:
    return datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')


llm = ChatOpenAI(
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="Qwen/Qwen3-8B",
    temperature=0.7
)

# 2. Prompt 模板
prompt = ChatPromptTemplate.from_template(
    "当前时间是：{current_time}\n"
    "网络搜索结果：\n{search_result}\n\n"
    "用户原始问题：{query}\n"
    "请结合当前时间点和最新搜索结果，准确回答用户的问题。"
)

# 3. 串行的 LCEL 链
chain = (
    # 第一步：先获取时间，此时数据流变为 {"query": "...", "current_time": "..."}
        RunnablePassthrough.assign(current_time=lambda _: get_current_time())

        # 第二步：带着时间和query去搜索（依赖上一步生成的 current_time）
        | RunnablePassthrough.assign(
    search_result=lambda x: tavily.invoke(f"{x['current_time']} {x['query']}")
)

        # 第三步：格式化并交给 LLM
        | prompt
        | llm
        | StrOutputParser()
)

# 4. 执行测试
result = chain.invoke({"query": "世界杯赛况"})
print(result)



