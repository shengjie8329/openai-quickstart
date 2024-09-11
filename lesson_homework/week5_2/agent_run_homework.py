#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/11 14:37
@Author :lancelot.sheng
@File   :agent_run_homework.py
"""
import getpass
import os
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from agent_tools import tavily_tool, python_repl
import operator
from typing import Annotated, Sequence, TypedDict
import functools
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
# 导入预构建的工具节点
from langgraph.prebuilt import ToolNode
from typing import Literal
from agent_tools import agent_node
from agent_tools import create_agent
from agent_tools import AgentState

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


# 使用 OpenAI 模型
research_llm = ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini", temperature=0.5)
chart_llm = ChatOpenAI(openai_api_base="https://ai-yyds.com/v1", model="gpt-4o-mini", temperature=0.2)

# 研究智能体及其节点
research_agent = create_agent(
    research_llm,  # 使用 research_llm 作为研究智能体的语言模型
    [tavily_tool],  # 研究智能体使用 Tavily 搜索工具
    system_message="你是一个帮助分析用户提问的专家，如果你能回答该问题就回答。如果无法回答也可借助与工具。 如果要使用搜索工具，在使用搜索工具之前，仔细思考并弄清楚要查询的是什么。"
                   "然后，进行一次性的搜索，一次性找到所要查询问题的所有方面的答案。注意只需要找到对应的数据或者答案并将结果输出就可以，特别是不要自己生成代码,也不要输出FINAL ANSWER，这项工作会交由其他助手继续处理",
    # 系统消息，指导智能体如何使用搜索工具
)
# 使用 functools.partial 创建研究智能体的节点，指定该节点的名称为 "Researcher"
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# 图表生成器智能体及其节点
table_agent = create_agent(
    chart_llm,  # 使用 chart_llm 作为表格生成器智能体的语言模型
    [python_repl],  # 表格生成器智能体使用 Python REPL 工具
    system_message="根据用户需求以及其他助手得到的相应事实数据， 编写出相应的python代码用于在控制台上构造一个清晰直观的表格数据，并调用工具执行这些代码。",  # 系统消息，指导智能体如何生成表格 receive the code to generate clear and user-friendly table on the console based on the provided data.
)
# 使用 functools.partial 创建表格生成器智能体的节点，指定该节点的名称为 "table_generator"
table_node = functools.partial(agent_node, agent=table_agent, name="table_generator")

# 定义工具列表，包括 Tavily 搜索工具和 Python REPL 工具
tools = [tavily_tool, python_repl]

# 创建工具节点，负责工具的调用
tool_node = ToolNode(tools)

# 路由器函数，用于决定下一步是执行工具还是结束任务
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]  # 获取当前状态中的消息列表
    last_message = messages[-1]  # 获取最新的一条消息

    # 如果最新消息包含工具调用，则返回 "call_tool"，指示执行工具
    if last_message.tool_calls:
        return "call_tool"

    # 如果最新消息中包含 "FINAL ANSWER"，表示任务已完成，返回 "__end__" 结束工作流
    if "FINAL ANSWER" in last_message.content:
        return "__end__"

    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"

# 创建一个状态图 workflow，使用 AgentState 来管理状态
workflow = StateGraph(AgentState)

# 将研究智能体节点、图表生成器智能体节点和工具节点添加到状态图中
workflow.add_node("Researcher", research_node)
workflow.add_node("table_generator", table_node)
workflow.add_node("call_tool", tool_node)

# 为 "Researcher" 智能体节点添加条件边，根据 router 函数的返回值进行分支
workflow.add_conditional_edges(
    "Researcher",
    router,  # 路由器函数决定下一步
    {
        "continue": "table_generator",  # 如果 router 返回 "continue"，则传递到 chart_generator
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "chart_generator" 智能体节点添加条件边
workflow.add_conditional_edges(
    "table_generator",
    router,  # 同样使用 router 函数决定下一步
    {
        "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
# 工具调用节点不更新 sender 字段，这意味着边将返回给调用工具的智能体
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
    {
        "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
        "table_generator": "table_generator",  # 如果 sender 是 chart_generator，则返回给 chart_generator
    },
)

# 添加开始节点，将流程从 START 节点连接到 Researcher 节点
workflow.add_edge(START, "Researcher")

# 编译状态图以便后续使用
graph = workflow.compile()

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="获取2024年巴黎奥运会奖牌榜前五名国家的金，银，铜牌的情况, "
            "并且生成一段python代码在控制台上生成一张表格，列头分别是 '国家'、'金牌数'、'银牌数'、'铜牌数'. 并且在执行这段python代码实际在当前控制台上输出这个表格后，代表完成了整个任务。"
            )
        ],
    },
    # 设置最大递归限制
    {"recursion_limit": 20},
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()  # 打印消息内容


