#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/5 13:33
@Author :lancelot.sheng
@File   :rag_homework.py
"""
from operator import itemgetter

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


doc_url = "https://blog.csdn.net/qq_19600291/article/details/141858744"

bs4_strainer = bs4.SoupStrainer(class_="htmledit_views")

loader = WebBaseLoader(
    web_paths=(doc_url,),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

# 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(openai_api_base="https://ai-yyds.com/v1")
)

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
llm = ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini")

# 使用 hub 模块拉取 rag 提示词模板
prompt = hub.pull("rlm/rag-prompt")

# 使用 LCEL 构建 RAG Chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

#1. 使用其他的线上文档或离线文件，重新构建向量数据库，尝试提出3个相关问题，测试 LCEL 构建的 RAG Chain 是否能成功召回。
#流式生成回答
for chunk in rag_chain.stream("中国游戏出海最多的种类"):
    print(chunk, end="", flush=True)
print("\n\n")

for chunk in rag_chain.stream("中国游戏在全球移动游戏市场的份额"):
    print(chunk, end="", flush=True)
print("\n\n")

# 流式生成回答
for chunk in rag_chain.stream("全球游戏市场规模"):
    print(chunk, end="", flush=True)
print("\n\n")


# 2. 重新设计或在 LangChain Hub 上找一个可用的 RAG 提示词模板，测试对比两者的召回率和生成质量。
prompt2 = hub.pull("morningstar/base-rag-prompt")

compare_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一个对答案进行评分的专家，你会得到一个原始的问题以及与它相关的一些文档资料，并且会的得到几个针对该问题的回答。\n你会依据原始问题和相关的资料, 对每一个答案依次进行打分，并输出你的依据"),
        ("human", "原始问题:{question}\n 文档资料:\n{docs} \n\n回答1:\n{answer_1}\n\n回答2:\n{answer_2}"),
    ])


def print_middle(x):
    print(x + "\n\n")
    return x


rag_chain1 = prompt | llm | StrOutputParser() | print_middle
rag_chain2 = prompt2 | llm | StrOutputParser() | print_middle


compare_chain = (
        {
            "answer_1": {"context": itemgetter("context"), "question": itemgetter("question")} | rag_chain1,
            "answer_2": {"context": itemgetter("context"), "question": itemgetter("question")} | rag_chain2,
            "docs": itemgetter("context"),
            "question": itemgetter("question"),
        }
        | compare_prompt | llm | StrOutputParser()
)

for chunk in compare_chain.stream(
        {"question": "中国游戏出海最多的种类", "context": format_docs(retriever.invoke("中国游戏出海最多的种类"))}):
    print(chunk, end="", flush=True)
