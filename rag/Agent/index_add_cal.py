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


from typing import Dict, List, Tuple


def min_max_normalize(scores_dict: Dict[str, float]):
    """对字典值进行 Min-Max 归一化"""
    if not scores_dict:
        return {}
    scores = list(scores_dict.values())
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores_dict.items()}


# 测试
scores = {
    "doc_001": 10,
    "doc_002": 12,
    "doc_003": 9,
    "doc_004": 18,
    "doc_005": 15,
    "doc_006": 30,
}

print(min_max_normalize(scores))


def weighted_sum_fusion(
    results_list: List[dict[str, float]],
    weights: List[float],
    normalized: List[bool]
) -> List[Tuple[str, float]]:
    """
    加权求和融合多个检索结果
    results_list: [检索器1的{doc_id: score}, 检索器2的{doc_id: score}, ...]
    weights: 对应权重，应总和为1
    normalized: 对应结果集是否需要归一化
    返回: [(doc_id, final_score), ...] 按分数降序
    """
    assert len(results_list) == len(weights)
    assert len(normalized) == len(weights)

    # 1. 收集所有文档 ID
    all_doc_ids = set()
    for results in results_list:
        all_doc_ids.update(results.keys())

    # 2. 可选：归一化每个检索器的分数
    normalized_results = []
    for i, results in enumerate(results_list):
        normalized_results.append(
            min_max_normalize(results) if normalized[i] else results
        )

    # 3. 加权求和
    final_scores = {}
    for doc_id in all_doc_ids:
        total = 0.0
        for i, norm_results in enumerate(normalized_results):
            score = norm_results.get(doc_id, 0.0)  # 未出现得0分
            total += weights[i] * score
        final_scores[doc_id] = total

    # 4. 重新排序并返回
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
# 测试《白鹿原》加权融合
query = "黑娃的结局"
vector_results = vectorstore.similarity_search_with_score(query, k=3)
bm25_results = bm25_search(query, k=3)

# docs
raw_docs = {}
vector_rs = {}
bm25_rs = {}
for doc, score in vector_results:
    raw_docs[doc.id] = doc
    vector_rs[doc.id] = score
for doc, score in bm25_results:
    raw_docs[doc["id"]] = doc
    bm25_rs[doc["id"]] = score

ranked_results = weighted_sum_fusion([vector_rs, bm25_rs], [0.5, 0.5], [False, True])

for id, score in ranked_results:
    print(f"=========id: {id} , score: {score}===========")
    print(raw_docs[id])