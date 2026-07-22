from langchain_core.documents import Document
from typing import List, Dict

from llm_config import llm
from rag.Agent.CrossEncoder import reranker
from rag.Agent.index_opt import bm25_search
from rag.Agent.index_rrf import reciprocal_rank_fusion
from rag.Agent.ready_rag import vectorstore


def cross_encoder_rerank(query: str, docs: List[Dict], top_k: int = 3):
    # 将 docs 转换为 Document 对象列表（保留 id）
    documents = [Document(page_content=doc["content"], metadata={"id": doc["id"]}) for doc in docs]
    # 调用 reranker 的 compress_documents，它会返回排序后的 Document 列表，并附带 relevance_score
    scored_docs = reranker.compress_documents(documents=documents, query=query)
    # 只取 top_k 个（compress_documents 已按 top_n 截断，但为了保险）
    scored_docs = scored_docs[:top_k]
    # 构建返回列表
    result = []
    for doc in scored_docs:
        result.append({
            "id": doc.metadata.get("id", ""),
            "content": doc.page_content,
            "score": doc.metadata.get("relevance_score", 0.0)
        })
    return result

from langchain.tools import tool
from langchain.agents import create_agent
import time

# 将检索器包装为Tool，Agent自主决定调用
@tool
def search_knowledge_base(query: str) -> str:
    """搜索《白鹿原》知识库，获取小说情节、人物命运、历史背景等知识。需要查找资料时调用。"""
    top_k = 3
    start = time.perf_counter()
    # 1.稠密检索（向量）
    vector_results = vectorstore.similarity_search(query, k=top_k)
    end = time.perf_counter()
    print(f"稠密检索完成,耗时:{(end - start) * 1000}ms~")
    start = end
    # 2.稀疏检索（bm25）
    bm25_results = bm25_search(query, k=top_k)
    end = time.perf_counter()
    print(f"稀疏检索完成,耗时:{(end - start) * 1000}ms~")
    start = end

    if not vector_results or not bm25_results:
        return "未找到相关文档"

    # 3.RRF
    # 3.1.处理成List[dict], dict包含id和content
    vector_rs = [
        {"id": doc.id, "content": doc.page_content} for doc in vector_results
    ]
    bm25_rs = [doc for doc, _ in bm25_results]
    end = time.perf_counter()
    print(f"rrf前置文档处理完成,耗时:{(end - start) * 1000}ms~")
    start = end

    # 3.2.rrf
    rrf_results = reciprocal_rank_fusion([vector_rs, bm25_rs])
    end = time.perf_counter()
    print(f"rrf完成,耗时:{(end - start) * 1000}ms~")
    start = end

    # 4.cross-encoder精排
    final_docs = cross_encoder_rerank(query, rrf_results, top_k)
    end = time.perf_counter()
    print(f"cross-encoder完成,耗时:{(end - start) * 1000}ms~")

    # 5.拼接文档
    docs_content = "\n\n".join(doc["content"] for doc in final_docs)

    # 6.输出日志
    print(f"\n\n{'=' * 30}Tool Message{'=' * 30}")
    print(f"检索到与问题'{query}'相关文档：")
    print(f"\n\n".join([f"=====score: {doc['score']}=======\n\n{doc['content']}" for doc in final_docs]))
    print("=" * 30 + "AI Message" + "=" * 30)

    return docs_content

rag_agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt="""
    你是一个专业的《白鹿原》文学知识专家。您的职责是帮助用户解决有关小说《白鹿原》的相关问题。
    产品说明:
    1. 如果用户问了一个你不确定的问题，或者涉及小说具体情节、人物关系、历史背景等专业知识，你必须使用`search_knowledge_base`工具来查阅相关文档。
    2. 在引用文档时，要清楚地总结包括内容中的相关上下文。
    3. 如果获取文档失败，请告诉用户，并以您最好的专家理解继续进行。
    在回答用户关于《白鹿原》的问题之前，您必须查阅工具以获取最新信息。你的回答应该清晰、简洁、准确。不要有过多解释除非用户询问。
    """,
)

print("Agentic RAG Agent 创建完成")


from langchain.messages import AIMessage
# 测试知识检索
response = rag_agent.stream(
    {"messages": [{"role": "user", "content": "白孝文的结局如何？"}]},
    stream_mode="messages"
)

for chunk, metadata in response:
    if isinstance(chunk, AIMessage) and chunk.content:
        print(chunk.content, end="")