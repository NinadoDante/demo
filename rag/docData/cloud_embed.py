import os

import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
import dotenv
dotenv.load_dotenv()

dashscope_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 向量化单条文本
text = "我爱上班"
vector = dashscope_embeddings.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"向量前5维: {vector[:5]}")

# 批量向量化
texts = ["我要躺平", "我爱工作", "拒绝加班"]
vectors = dashscope_embeddings.embed_documents(texts)
print(f"\n批量向量化: {len(vectors)} 条, 维度: {len(vectors[0])}")

for v in vectors:
    similarity = cosine_similarity(vector, v)
    print("Cosine Similarity:", similarity)

