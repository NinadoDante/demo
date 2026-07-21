from typing import Dict, List

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

from llm_config import llm
from rag.Agent.index_opt import bm25_search

load_dotenv()

# 读取 Markdown 文档（《白鹿原》全文）
with open("./docs/白鹿原.md", "r", encoding="utf-8") as f:
    docs = "".join(line for line in f.readlines())

# 按标题层级切分
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = markdown_splitter.split_text(docs)

idx = 1
# 给每个文档片段补充所属章节标题和唯一 ID
def handle_doc_header(doc: Document):
    global idx
    m = doc.metadata
    # 安全获取 Header 3
    header = m.get("Header 3", "")
    doc.page_content = f"### {header}\n{doc.page_content}"
    doc.id = f"doc_{idx}"
    idx += 1
    return doc

ds = [handle_doc_header(doc) for doc in chunks]

# 向量化并存入向量库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(ds)
print(f"知识库已就绪，{len(chunks)} 个文档")

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k=60):
    """
    ranked_lists: List[List[Dict]]
                  每个检索器返回的文档dict列表，包含id和content，按排名从高到低。
                  注意：这里不需要score，在 RRF 中只使用排名位置。
    """
    rrf_scores = {}  # 字典，key=doc_id, value=累加的 RRF 贡献
    results = {}  # 字典，key=doc_id, value=doc
    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            # 每个文档在每个列表中独立贡献
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            results[doc_id] = doc

    # 按 RRF 总分降序排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [results[doc_id] for doc_id, score in sorted_docs]

# 测试《白鹿原》混合检索
query = "白孝文的结局如何"

# 稠密检索（向量）
vector_results = vectorstore.similarity_search_with_score(query, k=3)
# 稀疏检索（bm25）
bm25_results = bm25_search(query, k=3)

# 处理成List[dict], dict包含id和content
vector_rs = [{"id": doc.id, "content": doc.page_content} for doc, _ in vector_results]
bm25_rs = [doc for doc, _ in bm25_results]

# rrf
ranked_results = reciprocal_rank_fusion([vector_rs, bm25_rs])

for i, doc in enumerate(ranked_results):
    print(f"=============rank: {i+1}=====id: {doc['id']}===========")
    print(doc["content"])