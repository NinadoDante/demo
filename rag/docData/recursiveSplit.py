from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mineru import MinerULoader

loader = MinerULoader(
    source="./docs/老年健康指南.pdf",
    mode="flash",
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 目标块大小（字符数）
    chunk_overlap=100,     # 块间重叠（10%-20%）
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 优先级从高到低
)

chunks = splitter.split_documents(docs)
print(f"原始文档数: {len(docs)}，切分后块数: {len(chunks)}")
for chunk in chunks:
    print(chunk)