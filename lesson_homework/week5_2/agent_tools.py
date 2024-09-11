#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/11 11:41
@Author :lancelot.sheng
@File   :agent_tools.py
"""
import operator
from typing import Annotated, TypedDict, Sequence

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# 定义 Tavily 搜索工具和 Python 代码执行工具
# Tavily 搜索工具，用于搜索最多 5 条结果
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()


# Python REPL 工具，用于执行 Python 代码
@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code(not json string). If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        # 如果代码执行失败，返回错误信息
        return f"Failed to execute. Error: {repr(e)}"

    # 成功执行的返回信息，包含执行的代码和标准输出
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    # 返回结果，并提示如果任务已完成，请回复 FINAL ANSWER
    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# 创建智能体的函数，绑定 LLM（大型语言模型） 和工具
def create_agent(llm, tools, system_message: str):
    """创建一个智能体。"""
    # 定义智能体的提示模板，包含系统消息和工具信息
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",  # 系统消息部分，描述智能体的行为逻辑
                "你是一个有帮助的 AI 助手，正在与其他助手协作。"
                " 使用提供的工具逐步解决问题。"
                " 如果你无法完全回答，没关系，另一个助手会接手，使用不同的工具继续。"
                " 尽你所能执行操作以取得进展。"
                " 如果你或其他助手有最终答案或可交付物，请在回复中加上 FINAL ANSWER 作为前缀，"
                " 让团队知道可以停止操作。"
                " 你可以使用以下工具：{tool_names}。\n{system_message}",  # 提供的工具名称和系统消息
            ),
            MessagesPlaceholder(variable_name="messages"),  # 用于替换的消息占位符
        ]
    )

    # 将系统消息部分和工具名称插入到提示模板中
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    # 将提示模板与语言模型和工具绑定
    return prompt | llm.bind_tools(tools)


# 定义图中传递的对象，包含消息和发送者信息
class AgentState(TypedDict):
    # messages 是传递的消息，使用 Annotated 和 Sequence 来标记类型
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # sender 是发送消息的智能体
    sender: str


# 辅助函数：为智能体创建一个节点
def agent_node(state, agent, name):
    # 调用智能体，获取结果
    result = agent.invoke(state)
    print("result---> " + str(result))
    # 将智能体的输出转换为适合追加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass  # 如果是工具消息，跳过处理
    else:
        # 将结果转换为 AIMessage，并排除部分字段
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    # 返回更新后的状态，包括消息和发送者
    return {
        "messages": [result],  # 包含新生成的消息
        # 我们使用严格的工作流程，通过记录发送者来知道接下来传递给谁
        "sender": name,
    }