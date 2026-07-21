from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.messages import AIMessage
import os
from dotenv import load_dotenv

from llm_config import llm

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



import os
import bm25s
import jieba


# 0. 准备一个初始化BM25库的工具
def create_bm25_index(metadata_corpus, index_path="./db/my_index.bm25", k1=1.5, b=0.75):
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

    # 封装结果
    return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]


# 4. 测试《白鹿原》相关查询
# query = "白灵是怎么死的"
query = "黑娃都干了什么大事"
ranked_docs = bm25_search(query, k=3)

for i, (doc, score) in enumerate(ranked_docs):
    print(f"======================Rank {i + 1} (score: {score:.2f})=================")
    print(f"doc: {doc}")