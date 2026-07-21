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




query = "白鹿原中谁组织了农协？"
rewrite_prompt = f"将以下问题改写为适合检索的关键词形式，提取核心概念，用空格分隔。只输出关键词不要解释。\n问题: {query}\n关键词:"
rewritten = llm.invoke(rewrite_prompt).content.strip()
print(f"重写: '{query}' → '{rewritten}'")

hyde_prompt = f"请根据你的知识，生成一个对以下问题的可能答案（简短但要关键，50字左右）：\n问题: {query}\n答案:"
fake_answer = llm.invoke(hyde_prompt).content.strip()
print(f"虚构答案: '{fake_answer}'")
# 用虚构答案去检索，准确度大幅提高
retrieved_docs = vectorstore.similarity_search(fake_answer, k=3)
print(f"检索到的文档: {[doc.metadata for doc in retrieved_docs]}")

import json
query = "白鹿原中白嘉轩和鹿子霖的关系如何？"
split_prompt = f"将用户问题拆分为多个子问题，不要任何解释，直接返回JSON数组。\n问题: {query}\n子问题:"
sub_queries = json.loads(llm.invoke(split_prompt).content)
print(f"拆分后的子问题: {sub_queries}")