from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
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
