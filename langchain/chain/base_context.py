from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm_config import llm


def get_context(topic: str) -> str:
    """模拟从数据库或知识库或联网搜索获取上下文"""
    context_map = {
        "人工智能": "人工智能是研究如何使计算机模拟人类智能的领域...",
        "大语言模型": "大语言模型是指具有数十亿参数的深度学习模型..."
    }
    return context_map.get(topic, f"关于{topic}的通用信息")

prompt = ChatPromptTemplate.from_template(
    "参考上下文：{context}\n回答关于{topic}的问题：{question}"
)

# # 方法1：使用 RunnableParallel 显式构建字典
# multi_input_chain = (
#     RunnableParallel({
#         "topic": lambda x: x["topic"],
#         "question": lambda x: x["question"],
#         "context": lambda x: get_context(x["topic"])
#     })
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# response = multi_input_chain.invoke({
#     "topic": "人工智能",
#     "question": "什么是大语言模型？"
# })
# print(response)

# 方法2：使用 RunnablePassthrough.assign 更简洁（推荐）
multi_input_chain = (
    RunnablePassthrough.assign(
        context=lambda x: get_context(x["topic"])
    )
    | prompt
    | llm
    | StrOutputParser()
)

response = multi_input_chain.invoke({
    "topic": "人工智能",
    "question": "什么是大语言模型？"
})
print(response)