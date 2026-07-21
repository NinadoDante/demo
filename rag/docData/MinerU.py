import os
import dotenv
dotenv.load_dotenv()

from dashscope import api_key
from langchain_mineru import MinerULoader

# loader = MinerULoader(
#     source="./docs/老年健康指南.pdf",
#     mode="flash",   # flash（免登录，<20页）或 precision（需Token，<200页）
# )
# docs = loader.load()  # 直接返回 LangChain Document 对象
# print(docs[0].metadata)
# print(docs[0].page_content[:500])


loader = MinerULoader(
    source="./docs/test.jpg",
    mode="precision",
    token = os.getenv("MINERU_TOKEN"),
    ocr=True   # 开启 OCR 识别扫描件中的文字
)
docs = loader.load()
print(docs[0].metadata)
print(docs[0].page_content[:500])



