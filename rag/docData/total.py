import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from langchain_mineru import MinerULoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 加载文档
print("⏳ 开始加载文档...")
t0 = time.time()
# loader = MinerULoader(source="./docs/老年健康指南.pdf", mode="flash")
# docs = loader.load()
loader = PyPDFLoader("./docs/老年健康指南.pdf")
docs = loader.load()
print(f"✅ 文档加载完成，耗时 {time.time() - t0:.1f}s，共 {len(docs)} 个文档")

# 2. 文本切分
print("⏳ 开始切分文本...")
t0 = time.time()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"✅ 切分完成，耗时 {time.time() - t0:.1f}s，共 {len(chunks)} 个片段")

# 3. 向量化 + 存入向量库
print("⏳ 开始向量化（依赖 Ollama，请确认已启动）...")
t0 = time.time()
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vectorstore = Chroma(
    collection_name="my_rag_db",
    embedding_function=embeddings,
    persist_directory="./db/chroma_db"
)
vectorstore.add_documents(chunks)
print(f"✅ 知识库构建完成！耗时 {time.time() - t0:.1f}s，共 {len(chunks)} 个片段")

# 4. 创建检索器
print("⏳ 开始检索...")
t0 = time.time()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("高血压饮食注意事项")
print(f"✅ 检索完成，耗时 {time.time() - t0:.1f}s，检索到 {len(docs)} 个相关文档")
print(f"内容: {docs[0].page_content[:100]}...")
