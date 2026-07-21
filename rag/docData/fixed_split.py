from langchain_text_splitters import CharacterTextSplitter

# ---------- 定义一段长文本 ----------
long_text = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的构建块，可以轻松组合成强大的工作流。

LCEL（LangChain 表达式语言）是 LangChain 中一种声明式编排方式，允许使用管道符（|）将不同的组件串联起来，形成可执行的链。这种语法简洁直观，并且天然支持流式输出。

Runnable 协议是 LCEL 的基石，所有主要组件（如模型、提示词模板、输出解析器）都实现了这个接口，使得它们可以无缝组合。

与传统的 Agent 不同，确定性链（Chain）可以预先定义执行步骤，从而避免模型自主决策带来的不稳定性。例如，在需要联网搜索的场景中，可以强制先获取当前时间，再基于时间构造搜索词，最后将结果传递给 LLM 生成回答。

输出解析器负责将模型的原始输出转换为程序可用的格式，例如字符串、JSON 或 Pydantic 对象。结构化输出（Structured Output）通过 with_structured_output 方法可以更方便地实现。

Agent 则更加灵活，它允许 LLM 自主决定调用哪些工具以及调用顺序，适用于需要复杂推理和动态规划的任务。然而，这种灵活性也带来了不确定性，需要通过精心设计的系统提示词和工具描述来约束。

在构建生产级应用时，通常需要结合链和 Agent 的优势：用链处理确定性流程，用 Agent 处理开放式决策。LangGraph 进一步扩展了这种能力，允许构建具有循环、条件分支和多角色协作的复杂状态机。

RAG（检索增强生成）是另一个重要应用场景，它通过向量数据库检索相关文档，然后将这些文档作为上下文注入到 LLM 的提示词中，从而生成更加准确和事实性的回答。LangChain 提供了完整的 RAG 工具链，包括文档加载器、文本分割器、向量存储和检索器。

文本分割是 RAG 流程中的关键步骤。LangChain 提供了多种分割器，例如按字符、按 token、按 Markdown 标题等。CharacterTextSplitter 是最基础的分割器，而 RecursiveCharacterTextSplitter 会递归尝试不同分隔符（如换行、句号、逗号）以保持语义完整性。

对于中文文本，建议使用 RecursiveCharacterTextSplitter 并指定分隔符为 ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]，这样能更好地保留句子和段落的完整性。

此外，LangChain 还支持异步调用、流式输出、回调处理等高级特性，可以轻松集成到 FastAPI、Streamlit 等 Web 框架中，构建实时响应的 AI 应用。

总之，LangChain 为构建复杂的 AI 应用提供了强大而灵活的工具集，无论你是初学者还是资深开发者，都能从中受益。
"""

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # 使用 cl100k_base 编码（对应 GPT-4 等模型）
    chunk_size=500,               # 每个块的最大 token 数
    chunk_overlap=200              # 块之间的重叠 token 数
)
chunks = splitter.split_text(long_text)

# ---------- 打印分割结果 ----------
print(f"总共分割为 {len(chunks)} 个块\n")
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i} (字符数: {len(chunk)}，预估 token 约 {len(chunk)//4})")
    print(chunk)
    print("-" * 50)