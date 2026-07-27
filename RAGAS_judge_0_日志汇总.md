# RAGAS_judge_0.py 日志汇总

## 一、日志格式

```
[RAGAS_EVAL] [步骤XX] 内容
```

- 前缀：`[RAGAS_EVAL]`
- 编号：`[步骤XX]`（对应时序图边编号）
- 内容：具体日志信息

---

## 二、时序边说明

| 编号 | 方向 | 说明 |
|------|------|------|
| 01 | Main → FileSystem | 读取《白鹿原》Markdown 全文 |
| 02 | Main → TextSplitter | MarkdownHeaderTextSplitter 按标题结构切分 |
| 03 | Main → TextSplitter | 超长块(>500字)用 RecursiveCharacterTextSplitter 二次切分 |
| 04 | Main → VectorStore | DashScope text-embedding-v3 向量化，写入 InMemoryVectorStore |
| 05 | Main → BM25Index | jieba 分词后创建/加载 BM25 稀疏索引 |
| 06 | Main → FileSystem | 读取 ragas_eval_dataset.json 评估问题集 |
| 07 | Main → RAG Agent | 调用 rag_agent.invoke() 处理每道评估题 |
| 08 | Agent → LLM | 查询改写：将原始问题改写为 3 个多角度检索查询 |
| 09 | Agent → VectorStore | 稠密检索 similarity_search(k=12) |
| 10 | Agent → BM25Index | BM25 稀疏检索(k=12) |
| 11 | Agent → Agent | RRF（Reciprocal Rank Fusion）融合两路结果 |
| 12 | Agent → Reranker | CrossEncoder 重排序（SiliconFlow Qwen3-Reranker-0.6B） |
| 13 | Agent → LLM | 基于精排文档，LLM 生成最终答案 |
| 14 | Main → Main | 将 answer + contexts + reference 写入 ragas Dataset |
| 15 | Main → SiliconFlow | 创建 AsyncOpenAI 评估客户端 |
| 16 | Main → SiliconFlow | 初始化评估器 LLM(Qwen3-8B) + Embedding(Qwen3-Embedding-0.6B) |
| 17 | Main → SiliconFlow | ContextPrecision 评估（检索阶段） |
| 18 | Main → SiliconFlow | ContextRecall 评估（检索阶段） |
| 19 | Main → SiliconFlow | ContextEntityRecall 评估（检索阶段） |
| 20 | Main → SiliconFlow | Faithfulness 评估（生成阶段） |
| 21 | Main → SiliconFlow | AnswerRelevancy 评估（生成阶段） |
| 22 | Main → SiliconFlow | NoiseSensitivity 评估（生成阶段） |

---

## 三、日志位点汇总

| 步骤编号 | 日志内容前缀 | 代码位置 |
|----------|-------------|----------|
| 01 | `读取 ./docs/白鹿原.md 完成，字符数: N` | 文件读取后 |
| 02 | `Markdown标题切分完成，得到 N 个块` | markdown_splitter.split_text() 后 |
| 03 | `块(len=N)>500，递归切分为 M 个子块` | recursive_splitter.split_documents() 后 |
| 04 | `DashScope向量化完成，共索引 N 个文档切片` | vectorstore.add_documents() 后 |
| 05 | `jieba分词 + BM25索引创建并保存至 path` | retriever.save() 后 |
| 06 | `本地评估数据集已加载，共 N 条` | dataset.reload() 后（有缓存时） |
| 06 | `加载评估问题集，共 N 个问题` | json.load() 后（无缓存时） |
| 07 | `开始处理第 i/N 题: 'X'` | 循环体开始（rag_agent.invoke 前） |
| 08 | `查询改写完成，原始查询: 'X'，改写后共 N 个查询` | llm.invoke(rewrite_prompt) 后 |
| 09 | `稠密检索完成，返回 N 条结果` | vectorstore.similarity_search() 后 |
| 10 | `BM25稀疏检索完成，返回 N 条结果` | bm25_retriever.retrieve() 后 |
| 11 | `RRF融合完成，共融合 N 个文档` | reciprocal_rank_fusion() 排序后 |
| 11 | `RRF融合结果: N 条，准备CrossEncoder重排` | retrieve_docs() 中 RRF 后 |
| 12 | `SiliconFlow Rerank API返回 N 条重排结果` | response.json() 解析后 |
| 12 | `最终重排完成，保留 N 条文档` | cross_encoder_rerank() 返回后 |
| 12 | `检索上下文已存入contexts_store，共 N 条` | contexts_store 写入后 |
| 13 | `Agent生成答案完成，答案长度: N 字符` | rag_agent.invoke() 返回后 |
| 14 | `收集答案与上下文完成，contexts: N 条，写入Dataset` | contexts_store.get() 后 |
| 14 | `评估数据集已保存，共 N 个样本` | dataset.save() 后 |
| 15 | `SiliconFlow AsyncOpenAI评估客户端创建完成` | AsyncOpenAI() 构造后 |
| 16 | `评估器初始化完成: LLM(Qwen3-8B) + Embedding(Qwen3-Embedding-0.6B)` | embedding_factory() 后 |
| 16 | `评估样本准备完成: user_input='X', contexts=N条` | 样本数据提取后 |
| 17 | `开始评估，user_input: 'X'` | run_evaluation() 入口 |
| 17 | `ContextPrecision评估完成，score=X.XXXX，进度1/6` | cp.ascore() 后 |
| 18 | `ContextRecall评估完成，score=X.XXXX，进度2/6` | cr.ascore() 后 |
| 19 | `ContextEntityRecall评估完成，score=X.XXXX，进度3/6` | cer.ascore() 后 |
| 20 | `Faithfulness评估完成，score=X.XXXX，进度4/6` | faith.ascore() 后 |
| 21 | `AnswerRelevancy评估完成，score=X.XXXX，进度5/6` | ar.ascore() 后 |
| 22 | `NoiseSensitivity评估完成，score=X.XXXX，进度6/6` | ns.ascore() 后 |
| 22 | `全部评估完成，成功 N/6 项` | 汇总输出前 |

---

## 四、日志输出示例

```
[RAGAS_EVAL] [步骤01] 读取 ./docs/白鹿原.md 完成，字符数: 523841
[RAGAS_EVAL] [步骤02] Markdown标题切分完成，得到 186 个块
[RAGAS_EVAL] [步骤03] 块(len=1204)>500，递归切分为 3 个子块
[RAGAS_EVAL] [步骤04] DashScope向量化完成，共索引 312 个文档切片
[RAGAS_EVAL] [步骤05] jieba分词 + BM25索引创建并保存至 ./db/my_index2.bm25
[RAGAS_EVAL] [步骤06] 加载评估问题集，共 12 个问题
[RAGAS_EVAL] [步骤07] 开始处理第 1/12 题: '白嘉轩一生娶过几房妻子？'
[RAGAS_EVAL] [步骤08] 查询改写完成，原始查询: '白嘉轩一生娶过几房妻子？'，改写后共 4 个查询
[RAGAS_EVAL] [步骤09] 稠密检索完成，返回 12 条结果
[RAGAS_EVAL] [步骤10] BM25稀疏检索完成，返回 12 条结果
[RAGAS_EVAL] [步骤11] RRF融合完成，共融合 18 个文档
[RAGAS_EVAL] [步骤12] SiliconFlow Rerank API返回 18 条重排结果
[RAGAS_EVAL] [步骤12] 最终重排完成，保留 8 条文档
[RAGAS_EVAL] [步骤12] 检索上下文已存入contexts_store，共 8 条
[RAGAS_EVAL] [步骤13] Agent生成答案完成，答案长度: 256 字符
[RAGAS_EVAL] [步骤14] 收集答案与上下文完成，contexts: 8 条，写入Dataset
[RAGAS_EVAL] [步骤15] SiliconFlow AsyncOpenAI评估客户端创建完成
[RAGAS_EVAL] [步骤16] 评估器初始化完成: LLM(Qwen3-8B) + Embedding(Qwen3-Embedding-0.6B)
[RAGAS_EVAL] [步骤16] 评估样本准备完成: user_input='白嘉轩一生娶过几房妻子？', contexts=8条
[RAGAS_EVAL] [步骤17] 开始评估，user_input: '白嘉轩一生娶过几房妻子？'
[RAGAS_EVAL] [步骤17] ContextPrecision评估完成，score=0.7500，进度1/6
[RAGAS_EVAL] [步骤18] ContextRecall评估完成，score=0.8333，进度2/6
[RAGAS_EVAL] [步骤19] ContextEntityRecall评估完成，score=0.7143，进度3/6
[RAGAS_EVAL] [步骤20] Faithfulness评估完成，score=0.9000，进度4/6
[RAGAS_EVAL] [步骤21] AnswerRelevancy评估完成，score=0.8571，进度5/6
[RAGAS_EVAL] [步骤22] NoiseSensitivity评估完成，score=0.1250，进度6/6
[RAGAS_EVAL] [步骤22] 全部评估完成，成功 6/6 项
```
