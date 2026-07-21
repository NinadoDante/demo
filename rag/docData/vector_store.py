from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mineru import MinerULoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)
vectorstore = Chroma(
    collection_name="my_knowledge_base",
    embedding_function=embeddings,        # 复用之前的向量模型
    persist_directory="./db/chroma_db"    # 持久化目录
)
# 确认实际的键名

# results = vectorstore.similarity_search(
#     "高血压饮食注意事项",  # 查询文本
#     k=3  # 返回 Top-3 最相关的片段
# )
# for i, doc in enumerate(results):
#     print(f"结果 {i+1}: {doc.page_content[:200]}...")

results = vectorstore.search(
    query="高血压饮食注意事项",
    search_type="similarity",
    k=3,
    filter={"filename": "./docs/老年健康指南.pdf"}  # 仅搜索特定文件
)
for i, doc in enumerate(results):
    print(f"结果 {i+1}: {doc.page_content[:200]}...")
