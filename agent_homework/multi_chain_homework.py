#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/5 17:17
@Author :lancelot.sheng
@File   :multi_chain_homework.py
"""
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

java_chain = (
        ChatPromptTemplate.from_template(
            "请用java 语言实现下面的功能需求，给出示例代码: \n 需求: {query}"
        )
        | ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini")
        | StrOutputParser()
)

python_chain = (
        ChatPromptTemplate.from_template(
            "请用python 语言实现下面的功能需求，给出示例代码: \n 需求: {query}"
        )
        | ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini")
        | StrOutputParser()
)


def response_process(x):
    # print(str(x))
    return x


process_chain = {
                    "java_code": java_chain,
                    "python_code": python_chain
                } | RunnableLambda(response_process)

rs = process_chain.invoke({"query": "请写一个快速排序的例子"})
# print(str(rs))
print(f"java:  \n {rs['java_code']}\n")
print(f"python:  \n {rs['python_code']}\n")
