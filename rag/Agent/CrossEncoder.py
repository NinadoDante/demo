# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_core.documents import Document
# from langchain_community.embeddings import DashScopeEmbeddings
# import os
from dotenv import load_dotenv

from llm_config import llm

load_dotenv()
#
# # 读取 Markdown 文档（《白鹿原》全文）
# with open("./docs/白鹿原.md", "r", encoding="utf-8") as f:
#     docs = "".join(line for line in f.readlines())
#
# # 按标题层级切分
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
# chunks = markdown_splitter.split_text(docs)
#
# idx = 1
# # 给每个文档片段补充所属章节标题和唯一 ID
# def handle_doc_header(doc: Document):
#     global idx
#     m = doc.metadata
#     # 安全获取 Header 3
#     header = m.get("Header 3", "")
#     doc.page_content = f"### {header}\n{doc.page_content}"
#     doc.id = f"doc_{idx}"
#     idx += 1
#     return doc
#
# ds = [handle_doc_header(doc) for doc in chunks]
#
# # 向量化并存入向量库
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )
# vectorstore = InMemoryVectorStore(embeddings)
# vectorstore.add_documents(ds)
# print(f"知识库已就绪，{len(chunks)} 个文档")
#
# # 创建检索器
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3}
# )

import os
import requests
from typing import Sequence, List
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field


# ========== 自定义 Reranker 类 ==========
class SiliconFlowReranker(BaseDocumentCompressor):
    """基于 SiliconFlow API 的文档重排序器"""

    model: str = Field(default="Qwen/Qwen3-Reranker-0.6B")
    api_key: str = Field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))
    api_base: str = Field(default="https://api.siliconflow.cn/v1")
    top_n: int = Field(default=3)
    score_threshold: float = Field(default=0.0)

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks=None,
    ) -> List[Document]:
        # 提取文档文本
        texts = [doc.page_content for doc in documents]

        # 构造请求
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": self.top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 调用 API
        response = requests.post(
            f"{self.api_base}/rerank",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()["results"]

        # 按分数排序并过滤阈值
        reranked_docs = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            if score >= self.score_threshold:
                doc = documents[idx]
                doc.metadata["relevance_score"] = score
                reranked_docs.append(doc)

        return reranked_docs


reranker = SiliconFlowReranker(
    model="Qwen/Qwen3-Reranker-0.6B",
    top_n=3,
    score_threshold=0.1,
)

# ========== 准备《白鹿原》数据 ==========
query = "朱先生是什么人物？"
passages = [
    "朱先生是白鹿原上的圣人，德高望重，曾编纂县志，劝退清军，名震关中。",
    "白嘉轩是白鹿村的族长，一生娶过七房女人，腰杆挺直，性格刚毅。",
    "鹿子霖是白鹿村的乡约，善于钻营，好色成性，与白嘉轩明争暗斗。",
    "黑娃原名鹿兆谦，曾当土匪，后归顺保安团，最终被镇压。",
]
documents = [Document(page_content=p) for p in passages]

# ========== 执行精排 ==========
scored_docs = reranker.compress_documents(documents=documents, query=query)

# ========== 输出结果 ==========
for doc in scored_docs:
    print(f"[{doc.metadata['relevance_score']:.4f}] {doc.page_content}")