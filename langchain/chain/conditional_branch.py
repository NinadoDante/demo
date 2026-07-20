from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_config import llm

# 定义各专业链
python_prompt = ChatPromptTemplate.from_template(
    "你是Python专家，请用技术术语回答，包含代码示例：\n问题：{question}"
)
python_chain = python_prompt | llm | StrOutputParser()

general_prompt = ChatPromptTemplate.from_template(
    "请简洁回答（不超过100字）：\n问题：{question}"
)
general_chain = general_prompt | llm | StrOutputParser()

# 条件分支
branch_chain = RunnableBranch(
    (lambda x: "python" in x["question"], python_chain),
    (lambda x: "天气" in x["question"], general_chain),  # 可加更多条件
    general_chain  # 默认分支
)

# 构建完整链：先通过 RunnablePassthrough 透传，再路由
conditional_chain = (
    RunnablePassthrough.assign(question=lambda x: x)  # 将原始字符串转为字典
    | branch_chain
)

# # 测试
# print(conditional_chain.invoke({"question":"Python中如何反转列表？"}))
# print(conditional_chain.invoke({"question":"今天天气怎么样？"}))
# print(conditional_chain.invoke({"question":"你好"}))

from langchain_core.runnables import RunnableLambda

def route_by_keyword(question: str):
    if "python" in question.lower():
        return python_chain
    elif "天气" in question:
        return general_chain
    else:
        return general_chain

conditional_chain = RunnableLambda(route_by_keyword)

# 注意：RunnableLambda 直接返回链，调用方式不变
print(conditional_chain.invoke("Python如何排序？"))