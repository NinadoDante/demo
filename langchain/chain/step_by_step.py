from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from llm_config import llm

# 链1：生成产品名
name_prompt = ChatPromptTemplate.from_template("为{product}起一个简洁的产品名称，只需输出名称")
name_chain = name_prompt | llm | StrOutputParser()

# 链2：生成描述（需要原始输入 + 链1的输出）
desc_prompt = ChatPromptTemplate.from_template(
    "根据产品类型{product}和名称{name}，写50字描述"
)
# 使用 RunnablePassthrough.assign 串联：先保留原始输入，再追加 name 字段
desc_chain = (
    RunnablePassthrough.assign(name=name_chain)  # name_chain 接收原始 product
    | desc_prompt
    | llm
    | StrOutputParser()
)

# 最终组合：并行获取两个结果
full_chain = RunnableParallel(
    product_name=name_chain,
    product_description=desc_chain
)

result = full_chain.invoke({"product": "智能手表"})
print(f"产品名称: {result['product_name']}")
print(f"产品描述: {result['product_description']}")