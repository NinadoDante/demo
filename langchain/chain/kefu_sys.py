import os
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm_config import llm


# 1. 定义各子链
consult_chain = ChatPromptTemplate.from_template("你是客服咨询顾问，请回答：{input}") | llm | StrOutputParser()
complaint_chain = ChatPromptTemplate.from_template("你是投诉处理专员，请妥善处理：{input}") | llm | StrOutputParser()
general_chain = ChatPromptTemplate.from_template("你是一个友好的聊天助手，请回复：{input}") | llm | StrOutputParser()

# 2. ✅ 意图识别链（使用结构化输出替代字符串匹配，彻底解决不稳定性）
class IntentResult(BaseModel):
    intent: str = Field(description="用户意图，仅限'咨询'、'投诉'、'闲聊'三者之一")

intent_chain = (
    ChatPromptTemplate.from_template("识别用户意图：\n用户消息：{input}")
    | llm.with_structured_output(IntentResult)
)

# 3. 使用 RunnableBranch 路由（基于结构化字段精确判断）
branch_chain = RunnableBranch(
    (lambda x: x["intent_result"].intent == "投诉", complaint_chain),
    (lambda x: x["intent_result"].intent == "咨询", consult_chain),
    general_chain  # 默认
)

# 4. 组合完整工作流
customer_service_chain = (
    RunnablePassthrough.assign(intent_result = lambda x:intent_chain.invoke({"input":x}))
    | branch_chain
)

# 5. 测试
print(customer_service_chain.invoke({"input":"你们的产品怎么退货？"}))
print(customer_service_chain.invoke({"input":"今天心情真好！"}))

# RunnablePassthrough.assign(
#     intent_result=lambda x: intent_chain.invoke({"input": x})
# )









