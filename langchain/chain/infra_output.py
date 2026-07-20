import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from llm_config import llm


# 1. 定义结构化输出的数据模型
class ProductInfo(BaseModel):
    name: str = Field(description="创新的产品名称，要求吸引人，不超过10个字")
    price: float = Field(description="预估的市场零售价（单位：元）")
    tags: list[str] = Field(description="产品的3个核心卖点标签")

# # 2. 初始化大模型
# llm = ChatOpenAI(
#     base_url=os.getenv("SILICONFLOW_BASE_URL"),
#     api_key=os.getenv("SILICONFLOW_API_KEY"),
#     model="Qwen/Qwen3-8B"
# )

# 3. 定义 Prompt 模板
prompt = ChatPromptTemplate.from_template(
    "你是一位资深的电商产品经理。请为以下概念产品生成名称、预估价格和核心标签。\n"
    "产品概念：{product}"
)

# 4. ✅ 构建 LCEL 链：绑定结构化输出
# with_structured_output 会自动将 ProductInfo 的 Field 描述转换为模型的 tool/function 定义
structured_chain = (
    prompt | llm.with_structured_output(ProductInfo)
)

# 5. 执行测试
print("正在调用大模型生成结构化数据...\n")

# invoke 传入的字典 key 必须与 prompt 模板中的变量名一致
result = structured_chain.invoke({
    "product": "一款能自动监测睡眠脑电波，并在浅睡期释放助眠香薰的智能眼罩"
})

# 6. 直接使用 Pydantic 对象的属性，享受代码提示和类型安全
print(f"📦 产品名称: {result.name}")
print(f"💰 预估价格: ￥{result.price}")
print(f"🏷️ 核心标签: {', '.join(result.tags)}")