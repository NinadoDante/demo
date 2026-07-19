from langchain_core.prompts import PromptTemplate

# context = {"date": "2026年7月15日", "event": "AI开发者大会", "speaker": "Yann LeCun"}
# template = "欢迎参加{event}！今天是{date}，主讲嘉宾是{speaker}。"
# prompt = PromptTemplate.from_template(template)
# print(prompt.format(**context))  # ** 解包字典

# header = PromptTemplate.from_template("主题：{title}\n")
# body = PromptTemplate.from_template("主要内容：{content}\n")
# footer = PromptTemplate.from_template("---\n联系人：{contact}")
#
# def build_report(title, content, contact):
#     return header.format(title=title) + body.format(content=content) + footer.format(contact=contact)
#
#
# print(f"报告内容:\n{build_report('AI技术前沿', '大模型最新进展', 'admin@example.com')}")
#


prompt = PromptTemplate(
    template="欢迎{name}，您的会员等级是{level}",
    input_variables=["name", "level"]  # 显式声明必需变量
)
# prompt.format(name="张三")  # 缺少 level 会抛出 KeyError
print(prompt.format(name="白泽", level=3))

# 大括号转义：{{ 和 }} 不会被解析为变量
# template = "这个{{符号}}不会被解析，但{name}会被替换"
# prompt = PromptTemplate.from_template(template)
# print(prompt.format(name="张三"))
# 输出：这个{符号}不会被解析，但张三会被替换

