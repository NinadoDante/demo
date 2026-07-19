from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 定义角色和用户输入
system_msg = SystemMessagePromptTemplate.from_template(
    "你是一位资深{role}，用{style}风格回答问题。"
)
human_msg = HumanMessagePromptTemplate.from_template(
    "我的问题是：{question}"
)

# 组合成聊天模板
chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# 格式化
messages = chat_prompt.format_messages(
    role="机器学习工程师",
    style="专业严谨",
    question="如何理解 Transformer 的注意力机制？"
)

# for msg in messages:
#     print(f"[{msg.type}]: {msg.content}")


from langchain_core.prompts import AIMessagePromptTemplate

# 历史对话
history = [
    HumanMessagePromptTemplate.from_template("Python 怎么定义函数？"),
    AIMessagePromptTemplate.from_template("使用 def 关键字，例如：def my_func():")
]

# 新问题
new_question = HumanMessagePromptTemplate.from_template("那如何定义带参数的函数？")

# 组合完整对话
full_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一位专业的 Python 编程助手"),
    *history,
    new_question
])

messages = full_prompt.format_messages()
for msg in messages:
    print(f"{msg.type}: {msg.content}")
