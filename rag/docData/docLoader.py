from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="docs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")
print(documents[0].page_content[:100])


