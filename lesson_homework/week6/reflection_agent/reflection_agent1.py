#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/19 14:52
@Author :lancelot.sheng
@File   :reflection_agent1.py
"""
import getpass
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from IPython.display import Markdown, display

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Reflection"

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a writing assistant tasked with creating well-crafted, coherent, and engaging articles based on the user's request."
            " Focus on clarity, structure, and quality to produce the best possible piece of writing."
            " If the user provides feedback or suggestions, revise and improve the writing to better align with their expectations.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# writer = writer_prompt | ChatOpenAI(
#     openai_api_base="https://ai-yyds.com/v1",
#     model="gpt-4o-mini",
#     max_tokens=8192,
#     temperature=1.2,
# )

writer = writer_prompt | ChatOllama(
    # base_url="http://eez-7io3hrtvr2tzdn7re-qz0zflpil-custom.service.onethingrobot.com:11434",
    model="llama3.1:8b-instruct-q8_0",
    max_tokens=8192,
    temperature=1.2,
)

article = ""

topic = HumanMessage(
    content="参考水浒传的风格，改写吴承恩的西游记中任意篇章"
)

for chunk in writer.stream({"messages": [topic]}):
    print(chunk.content, end="")
    article += chunk.content

# 使用Markdown显示优化后的格式
# display(Markdown(article))

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an article submission. writer critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",

        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# reflect = reflection_prompt | ChatOpenAI(
#     openai_api_base="https://ai-yyds.com/v1",
#     model="gpt-4o-mini",
#     max_tokens=8192,
#     temperature=0.2,
# )

reflect = reflection_prompt | ChatOllama(
    model="llama3.1:8b-instruct-q8_0",
    max_tokens=8192,
    temperature=0.2,
)

reflection = ""

# 将主题（topic）和生成的文章（article）作为输入发送给反思智能体
for chunk in reflect.stream({"messages": [topic, HumanMessage(content=article)]}):
    print(chunk.content, end="")
    reflection += chunk.content
