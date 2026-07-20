from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from llm_config import llm

prompt = ChatPromptTemplate.from_template("回答关于{topic}的问题：{question}")
qa_chain = prompt | llm | StrOutputParser()

response = qa_chain.invoke({
    "topic": "人工智能",
    "question": "什么是大语言模型？"
})
print(response)