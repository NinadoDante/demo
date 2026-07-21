from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

# # 单条向量化
# vec = embeddings.embed_query("我爱上班")
# print(f"维度: {len(vec)}")
#
# 批量向量化
vectors = embeddings.embed_documents(["我要躺平", "我爱工作", "拒绝加班"])
print(f"批量: {len(vectors)} 条, 维度: {len(vectors[0])}")

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

base_vec = embeddings.embed_query("我爱上班")
for text, vec in zip(["我要躺平", "我爱工作", "拒绝加班"], vectors):
    sim = cosine_similarity(base_vec, vec)
    print(f"「{text}」与「我爱上班」相似度: {sim:.4f}")
# 预期：「我爱工作」相似度最高，> 0.8


from langchain_chroma import Chroma
from langchain_core.documents import Document

vectorstore = Chroma(
    collection_name="my_knowledge_base",
    embedding_function=embeddings,        # 复用之前的向量模型
    persist_directory="./db/chroma_db"    # 持久化目录
)

# 添加文档（自动完成向量化+存储）
docs = [Document(page_content=text) for text in ["我要躺平", "我爱工作", "拒绝加班"]]
vec = vectorstore.add_documents(docs)
print(f"知识库共 {len(vec)} 个片段")


