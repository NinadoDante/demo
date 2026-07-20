from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph_cli.schemas import IndexConfig
from langchain_community.embeddings import DashScopeEmbeddings
import os
from langchain.tools import tool
from llm_config import llm

# 1. 创建语义搜索 Store
store = InMemoryStore(index=IndexConfig(
    embed=DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")),
    dims=1024  # 向量维度
))

# 2. 存储数据（与基础 Store 完全一样）
store.put(("users",), "user_001", {"name": "白泽", "department": "总裁办"})
store.put(("users",), "user_002", {"name": "张三", "department": "技术部"})
store.put(("users",), "user_003", {"name": "李四", "department": "市场部"})

# 3. 语义搜索（用自然语言查询，而不是死板的字段匹配）
results = store.search(("users",), query="技术人员", limit=5)
# 张三会因“技术部”与查询“技术人员”语义相近而排在首位
print(results)

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """获取用户信息"""
    # 从 Store 中获取用户信息
    if runtime.store is None:
        return "Store not available"

    # 通过runtime获取store，读取其中的数据
    user_info = runtime.store.get(("users",), user_id)

    if user_info is None:
        return "没有找到用户"

    return f"用户信息: {user_info.value}"

# 创建 Agent 时，只需传入 store 参数
agent = create_agent(
    llm,
    tools=[get_user_info],
    store=store  # 注册长期记忆
)

response = agent.invoke({
    "messages": [HumanMessage("帮我查询user_001的信息")]
})
for message in response['messages']:
    message.pretty_print()