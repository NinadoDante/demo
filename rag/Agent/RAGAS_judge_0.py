# 加载知识库（《白鹿原》小说全文）
import sys
import types

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import dotenv
import logging

# ===== 时序日志配置 =====
logging.basicConfig(level=logging.INFO, format="[RAGAS_EVAL] %(message)s", force=True)
logger = logging.getLogger("ragas_eval")

dotenv.load_dotenv()
from llm_config import llm



# 1. 加载并切分《白鹿原》小说
with open("./docs/白鹿原.md", "r", encoding="utf-8") as f:
    docs = f.read()
logger.info("[步骤01] 读取 ./docs/白鹿原.md 完成，字符数: %d", len(docs))

# 先用md结构切分
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = markdown_splitter.split_text(docs)
logger.info("[步骤02] Markdown标题切分完成，得到 %d 个块", len(chunks))

# 创建递归切分器
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。"]  # 优先级从高到低
)

ds = []
idx = 1
# 定义方法，作用是给文档内容拼上三级标题和id
def handle_doc_header(doc: Document):
    global idx
    m = doc.metadata
    # 安全获取 Header 3
    header = m.get("Header 3", "")
    doc.page_content = f"### {header}\n{doc.page_content}"
    doc.id = f"doc_{idx}"
    idx = idx + 1
    return doc

# 遍历文档，判断大小，如果超过了200，再用递归大小切分
for doc in chunks:
    m = doc.metadata
    content = doc.page_content
    if len(content) > 500:
        sub_docs = recursive_splitter.split_documents([doc])
        logger.info("[步骤03] 块(len=%d)>500，递归切分为 %d 个子块", len(content), len(sub_docs))
        ds.extend(handle_doc_header(d) for d in sub_docs)
    else:
        ds.append(handle_doc_header(doc))

# for chunk in ds:
#     print(len(chunk.page_content))
#     print(chunk.id)
#     print(chunk)
#     print("=" * 60)

# 2. 向量化（DashScope Embedding）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(ds)
logger.info("[步骤04] DashScope向量化完成，共索引 %d 个文档切片", len(ds))
retriever = vectorstore.as_retriever(kw_args={"k": 8})

print(f"知识库已就绪: {len(ds)} 个文档切片")

# 3. BM25 稀疏检索
import os
import bm25s
import jieba


# 0. 准备一个初始化BM25库的工具
def create_bm25_index(metadata_corpus, index_path="./db/my_index2.bm25", k1=1.5, b=0.75):
    if os.path.exists(index_path):
        print(f"索引文件已存在，正在加载: {index_path}")
        retriever = bm25s.BM25.load(index_path, load_corpus=True)
    else:
        print(f"未找到索引文件，正在创建新索引...")
        # 对语料进行分词（注意：jieba.cut 返回生成器，需用 list() 转为列表）
        corpus_tokens = [list(jieba.cut(doc["content"])) for doc in metadata_corpus]

        # 基于原始文档创建索引库，将来检索出来的也是原始文档
        retriever = bm25s.BM25(k1=k1, b=b, corpus=metadata_corpus)
        # 创建索引
        retriever.index(corpus_tokens)
        # 保存到本地
        retriever.save(index_path)
        logger.info("[步骤05] jieba分词 + BM25索引创建并保存至 %s", index_path)
        print(f"索引已保存至: {index_path}")

    return retriever


# 1. 把之前切分的文档处理成dict，与doc_id一一映射
metadata_corpus = [
    {"id": doc.id, "content": doc.page_content} for doc in ds
]
# 2. 创建BM25索引库
bm25_retriever = create_bm25_index(metadata_corpus)

from typing import List, Tuple, Dict


# 3. 封装查询方法
def bm25_search(query: str, k: int = 3) -> List[Tuple[Dict, float]]:
    # 查询分词（同样需要 list() 转换）
    query_tokens = [list(jieba.cut(query))]

    # 检索，返回top-k结果，形式为(docs, scores)
    # docs和scores都是二维数组 shape (n_queries, k)
    results, scores = bm25_retriever.retrieve(query_tokens, k=k)
    logger.info("[步骤10] BM25稀疏检索完成，返回 %d 条结果", results.shape[1])

    # 封装结果
    return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]


print("BM25 检索器已就绪")


# 4. RRF 融合
def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    results = {}
    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            results[doc_id] = doc
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info("[步骤11] RRF融合完成，共融合 %d 个文档", len(sorted_docs))
    return [results[doc_id] for doc_id, score in sorted_docs]


print("RRF 融合已就绪")

# 5. Cross-Encoder 重排序
import os
import requests
from typing import Sequence, List, Dict
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field
from langchain_core.documents import Document


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
        logger.info("[步骤12] SiliconFlow Rerank API返回 %d 条重排结果", len(results))

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
    top_n=20,           # 初筛阶段多返回，给后续精排留空间
    score_threshold=0.0,
)


def cross_encoder_rerank(query: str, docs: List[Dict], top_k: int = 8):
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


print("Cross-Encoder 已就绪")
print("\n所有检索组件准备完成！")
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



# 构建 Agent
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState


def retrieve_docs(query: str, top_k: int):
    """混合检索：稠密 + 稀疏 + RRF + CrossEncoder"""
    # 1. 稠密检索
    vector_results = vectorstore.similarity_search(query, k=top_k)
    logger.info("[步骤09] 稠密检索完成，返回 %d 条结果", len(vector_results))
    # 2. 稀疏检索 BM25
    bm25_results_list = bm25_search(query, k=top_k)
    if not vector_results or not bm25_results_list:
        return []
    # 3. RRF 融合
    vector_rs = [{"id": doc.id, "content": doc.page_content} for doc in vector_results]
    bm25_rs = [doc for doc, _ in bm25_results_list]
    rrf_results = reciprocal_rank_fusion([vector_rs, bm25_rs])
    logger.info("[步骤11] RRF融合结果: %d 条，准备CrossEncoder重排", len(rrf_results))
    # 4. Cross-Encoder 重排序
    return cross_encoder_rerank(query, rrf_results, top_k)


# tool 内部顺手把检索结果存下来，供评估时提取 context，避免检索两次
contexts_store = {}


@tool
def search_knowledge_base(query: str, runtime: ToolRuntime[AgentState]) -> str:
    """搜索《白鹿原》知识库，获取小说情节、人物命运、历史背景等知识。需要查找资料时使用。"""
    # 查询扩展：生成多个检索角度
    rewrite_prompt = f"""将以下问题改写为3个不同角度的检索查询，用于在小说全文中搜索相关段落。
    改写要求：围绕问题的核心实体和关键事件，覆盖"起因、经过、结局/命运"等不同维度，每行一个查询：
    问题：{query}"""
    rewritten = llm.invoke(rewrite_prompt).content.strip().split("\n")
    queries = [query] + [q.strip() for q in rewritten if q.strip()]
    logger.info("[步骤08] 查询改写完成，原始查询: '%s'，改写后共 %d 个查询", query[:30], len(queries))

    # 对每个查询分别检索，合并去重（初筛多取，给重排序更大选择空间）
    all_docs = {}
    for q in queries:
        for doc in retrieve_docs(q, 12):
            all_docs[doc["id"]] = doc
    # 重排序后保留最相关的 8 条（平衡精度与召回）
    final_docs = cross_encoder_rerank(query, list(all_docs.values()), 8)
    logger.info("[步骤12] 最终重排完成，保留 %d 条文档", len(final_docs))
    if not final_docs:
        return "未找到相关文档"

    # 顺手存 context，评估取用
    msg = runtime.state['messages'][0].content
    cs = contexts_store.get(msg, [])
    cs.extend([doc["content"] for doc in final_docs])
    contexts_store[msg] = list(set(cs))
    logger.info("[步骤12] 检索上下文已存入contexts_store，共 %d 条", len(contexts_store[msg]))

    return "\n\n".join(doc["content"] for doc in final_docs)


rag_agent = create_agent(
    model= llm,
    tools=[search_knowledge_base],
    system_prompt="""
    你是一个《白鹿原》文学知识专家。你必须严格遵守以下规则：

    1. 有关《白鹿原》的问题**只能**调用`search_knowledge_base`工具来查阅相关文档，根据文档信息回答问题
    2. **每个问题最多调用一次工具**，禁止重复调用
    3. **禁止**添加任何你自己的知识、理解或推断
    4. **禁止**扩展、举例或给出建议
    5. 尽最大努力从参考文档中提取与问题相关的信息作答；只有当文档中**完全没有任何相关信息**时，才回复"根据提供的资料，无法回答此问题"
    6. 回答要**直接、简洁地回应问题本身**，先概括要点，再引用原文关键句作为佐证
    7. 不要大段照搬原文，而是提炼出与问题直接相关的信息作答
    """,
)

print("RAG Agent 创建完成！（混合检索：稠密+BM25+RRF+CrossEncoder）")



# 收集 Agent 回答和检索上下文，构建 ragas.Dataset
import json
import time
# 兼容旧版 ragas 对已移除模块的导入
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except (ImportError, ModuleNotFoundError):
    _fake_module = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        ChatVertexAI = None
    _fake_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_module

from ragas import Dataset
import os


def create_dataset():
    """创建dataset，先尝试本地加载，如果没有则重新生成"""

    # 1. 创建 ragas Dataset
    dataset = Dataset(name="rag_eval", backend="local/csv", root_dir="./experiments")
    if os.path.exists("./experiments/datasets/rag_eval.csv"):
        # 尝试读取本地数据
        dataset.reload()
        logger.info("[步骤06] 本地评估数据集已加载，共 %d 条", len(dataset))
    # 如果有数据，直接返回，没有则重新生成数据集
    if len(dataset) > 0:
        print(f"加载了 {len(dataset)} 个测试数据集，不再重新生成")
        return dataset

    # 2. 加载评估问题（《白鹿原》相关）
    with open("./docs/ragas_eval_dataset.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    logger.info("[步骤06] 加载评估问题集，共 %d 个问题", len(eval_data))
    print(f"加载了 {len(eval_data)} 个评估问题")

    # 3.调用Agent，生成response和context
    # 每次 Agent 调用后的等待时间（秒），避免触发模型提供商限流
    # DeepSeek 免费版建议 1~2s，付费版可适当降低，根据实际报错调整
    REQUEST_DELAY = 3

    print(f"开始收集回答和上下文（共 {len(eval_data)} 题，间隔 {REQUEST_DELAY}s）...")
    print("-" * 50)

    for i, item in enumerate(eval_data):
        query = item["question"]
        logger.info("[步骤07] 开始处理第 %d/%d 题: '%s'", i + 1, len(eval_data), query[:40])

        try:
            # 调用 Agent —— 内部 search_knowledge_base tool 会将检索结果存入 contexts_store
            response = rag_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": f"eval-{i}"}, "recursion_limit": 25}
            )
            answer = response["messages"][-1].content
            logger.info("[步骤13] Agent生成答案完成，答案长度: %d 字符", len(answer))
        except Exception as e:
            print(f"[{i + 1}/{len(eval_data)}] {query[:40]}... ✗ Agent异常: {e}")
            answer = "根据提供的资料，无法回答此问题"

        # 从 tool 的 store 中取 context（一次检索，不重复）
        contexts = contexts_store.get(query, [])
        logger.info("[步骤14] 收集答案与上下文完成，contexts: %d 条，写入Dataset", len(contexts))

        # 写入数据集
        dataset.append({
            "user_input": query,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        })

        print(f"[{i + 1}/{len(eval_data)}] {query[:40]}... ✓ (contexts: {len(contexts)}条)")

        # 等待一会儿再发下一个请求，避免触发限流
        if i < len(eval_data) - 1:  # 最后一条不用等
            time.sleep(REQUEST_DELAY)

    dataset.save()
    logger.info("[步骤14] 评估数据集已保存，共 %d 个样本", len(eval_data))
    print(f"\n评估数据集已保存: {len(eval_data)} 个样本 -> ./experiments/rag_eval/")
    return dataset


# 调用函数，创建dataset
dataset = create_dataset()


# 新 API：@experiment 装饰器 + 标准指标
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI
import httpx
import os

# ----- 统一使用 SiliconFlow（OpenAI 兼容接口）-----
timeout = httpx.Timeout(180.0, connect=60.0)
http_client = httpx.AsyncClient(timeout=timeout)

siliconflow_client = AsyncOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    http_client=http_client,
    max_retries=3,
)
logger.info("[步骤15] SiliconFlow AsyncOpenAI评估客户端创建完成")

# 推理模型（SiliconFlow 支持的模型）
evaluator_llm = llm_factory(
    model="Qwen/Qwen3-8B",   # 可更换为 "deepseek-ai/DeepSeek-V3" 等
    client=siliconflow_client,
    max_tokens=4096,         # 增大 token 上限，避免长 context 评估时输出截断
)

# Embedding 模型（SiliconFlow 也提供）
evaluator_embeddings = embedding_factory(
    model="Qwen/Qwen3-Embedding-0.6B",  # 或其他 SiliconFlow 上的 embedding 模型
    client=siliconflow_client,
)
logger.info("[步骤16] 评估器初始化完成: LLM(Qwen3-8B) + Embedding(Qwen3-Embedding-0.6B)")

print("评估器已配置完成（使用 SiliconFlow）")


print(dataset.__getitem__(3))

row = dataset.__getitem__(3)
user_input = row.get("user_input")
response = row.get("response")
retrieved_contexts = row.get("retrieved_contexts")
reference = row.get("reference")

# CSV 加载后 retrieved_contexts 可能是字符串，需要还原为列表
if isinstance(retrieved_contexts, str):
    import ast
    try:
        retrieved_contexts = ast.literal_eval(retrieved_contexts)
    except (ValueError, SyntaxError):
        retrieved_contexts = [retrieved_contexts]
print(f"retrieved_contexts 类型: {type(retrieved_contexts)}, 长度: {len(retrieved_contexts)}")
logger.info("[步骤16] 评估样本准备完成: user_input='%s', contexts=%d条", user_input[:30], len(retrieved_contexts))

# 6个核心指标（从 collections 导入）
from ragas.metrics.collections import (
    Faithfulness,  # 忠实度
    AnswerRelevancy,  # 回答相关性
    ContextPrecision,  # 上下文精度
    ContextEntityRecall,  # 上下文实体召回
    NoiseSensitivity,  # 噪声敏感度
    ContextRecall  # 上下文召回
)

print(f"========开始评估：input:{user_input}===========")

import asyncio

async def run_evaluation():
    results = {}
    logger.info("[步骤17] 开始评估，user_input: '%s'", user_input[:40])

    # ---- 检索阶段指标 ----
    print("[1/6] 正在评估 ContextPrecision（等待 SiliconFlow 响应）...")
    try:
        cp = ContextPrecision(llm=evaluator_llm)
        cp_result = await cp.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        results["ContextPrecision"] = cp_result
        logger.info("[步骤17] ContextPrecision评估完成，score=%.4f，进度1/6", float(cp_result))
        print(f"运行完成context precision评估,score: {cp_result},进度1/6\n")
    except Exception as e:
        print(f"[1/6] ContextPrecision 评估失败: {e}\n")

    print("[2/6] 正在评估 ContextRecall...")
    try:
        cr = ContextRecall(llm=evaluator_llm)
        cr_result = await cr.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        results["ContextRecall"] = cr_result
        logger.info("[步骤18] ContextRecall评估完成，score=%.4f，进度2/6", float(cr_result))
        print(f"运行完成context recall评估,score: {cr_result},进度2/6\n")
    except Exception as e:
        print(f"[2/6] ContextRecall 评估失败: {e}\n")

    print("[3/6] 正在评估 ContextEntityRecall...")
    try:
        cer = ContextEntityRecall(llm=evaluator_llm)
        cer_result = await cer.ascore(
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        results["ContextEntityRecall"] = cer_result
        logger.info("[步骤19] ContextEntityRecall评估完成，score=%.4f，进度3/6", float(cer_result))
        print(f"运行完成ContextEntityRecall评估,score: {cer_result},进度3/6\n")
    except Exception as e:
        print(f"[3/6] ContextEntityRecall 评估失败: {e}\n")

    # ---- 生成阶段指标 ----
    print("[4/6] 正在评估 Faithfulness...")
    try:
        faith = Faithfulness(llm=evaluator_llm)
        faith_result = await faith.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        results["Faithfulness"] = faith_result
        logger.info("[步骤20] Faithfulness评估完成，score=%.4f，进度4/6", float(faith_result))
        print(f"运行完成Faithfulness评估,score: {faith_result},进度4/6\n")
    except Exception as e:
        print(f"[4/6] Faithfulness 评估失败: {e}\n")

    print("[5/6] 正在评估 AnswerRelevancy...")
    try:
        ar = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        ar_result = await ar.ascore(
            user_input=user_input,
            response=response,
        )
        results["AnswerRelevancy"] = ar_result
        logger.info("[步骤21] AnswerRelevancy评估完成，score=%.4f，进度5/6", float(ar_result))
        print(f"运行完成AnswerRelevancy评估,score: {ar_result},进度5/6\n")
    except Exception as e:
        print(f"[5/6] AnswerRelevancy 评估失败: {e}\n")

    print("[6/6] 正在评估 NoiseSensitivity...")
    try:
        ns = NoiseSensitivity(llm=evaluator_llm)
        ns_result = await ns.ascore(
            user_input=user_input,
            response=response,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        results["NoiseSensitivity"] = ns_result
        logger.info("[步骤22] NoiseSensitivity评估完成，score=%.4f，进度6/6", float(ns_result))
        print(f"运行完成NoiseSensitivity评估,score: {ns_result},进度6/6\n")
    except Exception as e:
        print(f"[6/6] NoiseSensitivity 评估失败: {e}\n")

    # ---- 汇总 ----
    logger.info("[步骤22] 全部评估完成，成功 %d/6 项", len(results))
    print("=" * 50)
    print("评估结果汇总：")
    for name, score in results.items():
        print(f"  {name}: {score}")
    print(f"成功 {len(results)}/6 项")
    print("=" * 50)

asyncio.run(run_evaluation())
