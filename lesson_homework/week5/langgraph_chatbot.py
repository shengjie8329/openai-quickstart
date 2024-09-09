#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/9 11:31
@Author :lancelot.sheng
@File   :langgraph_chatbot.py
"""
import os

# 开启 LangSmith 跟踪，便于调试和查看详细执行信息
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph ChatBot"
# os.environ["LANGCHAIN_API_KEY"] = "LangGraph ChatBot"

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


# 定义状态类型，继承自 TypedDict，并使用 add_messages 函数将消息追加到现有列表
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建一个状态图对象，传入状态定义
graph_builder = StateGraph(State)

# 初始化一个 GPT-4o-mini 模型
chat_model = ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini")


# 定义聊天机器人的节点函数，接收当前状态并返回更新的消息列表
def chatbot(state: State):
    return {"messages": [chat_model.invoke(state["messages"])]}


# 第一个参数是唯一的节点名称，第二个参数是每次节点被调用时的函数或对象
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译状态图并生成可执行图对象
graph = graph_builder.compile()


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
#
# try:
#     # 使用 Mermaid 生成图表并保存为文件
#     mermaid_code = graph.get_graph().draw_mermaid_png()
#     with open("graph.jpg", "wb") as f:
#         f.write(mermaid_code)
#
#     # 使用 matplotlib 显示图像
#     img = mpimg.imread("graph.jpg")
#     plt.imshow(img)
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
# except Exception as e:
#     print(f"An error occurred: {e}")

# 开始一个简单的聊天循环
while True:
    # 获取用户输入
    user_input = input("User: ")

    # 可以随时通过输入 "quit"、"exit" 或 "q" 退出聊天循环
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")  # 打印告别信息
        break  # 结束循环，退出聊天

    # 将每次用户输入的内容传递给 graph.stream，用于聊天机器人状态处理
    # "messages": ("user", user_input) 表示传递的消息是用户输入的内容
    for event in graph.stream({"messages": ("user", user_input)}):

        # 遍历每个事件的值
        for value in event.values():
            # 打印输出 chatbot 生成的最新消息
            print("Assistant:", value["messages"][-1].content)


