from langchain_text_splitters import MarkdownHeaderTextSplitter

# ---------- 定义一段示例 Markdown 文本 ----------
markdown_text = """
# 第一章：概述
这是第一章的引言部分，介绍整体框架。

## 第一节：背景
本节描述项目的背景和动机。

### 1.1 现状分析
当前存在的一些问题和挑战。

### 1.2 研究目标
明确本项目的核心目标。

## 第二节：相关工作
回顾已有研究和相关技术。

# 第二章：方法论
本章详细介绍所采用的方法和实验设计。

## 第一节：数据采集
描述数据来源和预处理步骤。

## 第二节：模型架构
阐述模型的具体结构和训练策略。
"""

# ---------- 原有代码保持不变 ----------
headers_to_split = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split)
chunks = splitter.split_text(markdown_text)

for doc in chunks:
    # 打印每个片段的元数据（标题层级）和前80个字符内容
    print(f"{doc.metadata} → {doc.page_content[:80]}...")