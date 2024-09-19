#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/19 17:03
@Author :lancelot.sheng
@File   :reflection_agent_homework.py
"""
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from typing import Annotated  # 用于类型注解
from langgraph.graph import END, StateGraph, START  # 导入状态图的相关常量和类
from langgraph.graph.message import add_messages  # 用于在状态中处理消息
from langgraph.checkpoint.memory import MemorySaver  # 内存保存机制，用于保存检查点
from typing_extensions import TypedDict  # 用于定义带有键值对的字典类型

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Reflection"

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名写作助理，负责根据老板的要求编写指定主题的调研报告。"
            "专注于清晰度、结构和质量，以创作出最符合需求的报告。"
            "如果用户提供反馈或建议，请根据建议重写出一篇改进后的完整报告（不需要具体的改进的步骤），以更好地符合他们的期望。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

writer = writer_prompt | ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1:8b-instruct-q8_0",
    max_tokens=8192,
    temperature=1.2,
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名报告评论专家，对提交的报告进行评分。作者对用户提交的报告提出批评和建议。"
            "提供详细的建议，包括对长度、深度、风格、质量等的要求",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1:8b-instruct-q8_0",
    max_tokens=8192,
    temperature=0.2,
)


# 定义状态类，使用TypedDict以保存消息
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 使用注解确保消息列表使用add_messages方法处理


# 异步生成节点函数：生成内容（如作文）
# 输入状态，输出包含新生成消息的状态
def generation_node(state: State) -> State:
    # 调用生成器(writer)，并将消息存储到新的状态中返回
    res = writer.invoke(state['messages'])
    print(res.content)
    print("===============================")
    return {"messages": [res]}


# 异步反思节点函数：对生成的内容进行反思和反馈
# 输入状态，输出带有反思反馈的状态
def reflection_node(state: State) -> State:
    # 创建一个消息类型映射，ai消息映射为HumanMessage，human消息映射为AIMessage
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # 处理消息，保持用户的原始请求（第一个消息），转换其余消息的类型
    translated = [state['messages'][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state['messages'][1:]
    ]

    # 调用反思器(reflect)，将转换后的消息传入，获取反思结果
    res = reflect.invoke(translated)

    print(res.content)
    print("===============================")
    # 返回新的状态，其中包含反思后的消息
    return {"messages": [HumanMessage(content=res.content)]}


MAX_ROUND = 6

# 定义条件函数，决定是否继续反思过程
# 如果消息数量超过6条，则终止流程
def should_continue(state: State):
    if len(state["messages"]) > MAX_ROUND:
        return END  # 达到条件时，流程结束
    return "reflect"  # 否则继续进入反思节点


# 创建状态图，传入初始状态结构
builder = StateGraph(State)

# 在状态图中添加"writer"节点，节点负责生成内容
builder.add_node("writer", generation_node)

# 在状态图中添加"reflect"节点，节点负责生成反思反馈
builder.add_node("reflect", reflection_node)

# 定义起始状态到"writer"节点的边，从起点开始调用生成器
builder.add_edge(START, "writer")


# 在"writer"节点和"reflect"节点之间添加条件边
# 判断是否需要继续反思，或者结束
builder.add_conditional_edges("writer", should_continue)

# 添加从"reflect"节点回到"writer"节点的边，进行反复的生成-反思循环
builder.add_edge("reflect", "writer")

# 创建内存保存机制，允许在流程中保存中间状态和检查点
memory = MemorySaver()

# 编译状态图，使用检查点机制
graph = builder.compile(checkpointer=memory)

inputs = {
    "messages": [
        HumanMessage(content="请写一篇关于人工智能是否能取代程序员来写代码的可行性报告")
    ],
}

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(inputs, config):
    # print(chunk, end="")
    pass