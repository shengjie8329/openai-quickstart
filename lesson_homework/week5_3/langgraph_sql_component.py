#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/12 15:04
@Author :lancelot.sheng
@File   :langgraph_sql_component.py
"""
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode


# 创建具有回退机制的工具节点
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    创建一个 ToolNode（工具节点），并为它添加回退机制。回退机制用于
    在工具调用出现错误时处理这些错误，并将错误信息传递给 Agent。

    tools: 传入的工具列表，每个工具可以执行某种操作
    返回值: 包含回退逻辑的 ToolNode
    """
    return ToolNode(tools).with_fallbacks(
        # 添加回退逻辑，使用 RunnableLambda 运行 handle_tool_error 方法来处理错误
        [RunnableLambda(handle_tool_error)],
        # 指定当出现 "error" 时触发回退机制
        exception_key="error"
    )


# 处理工具调用时发生的错误
def handle_tool_error(state) -> dict:
    """
    处理工具调用过程中发生的错误，并返回包含错误信息的消息列表。

    state: 当前的状态，包含工具调用的信息和发生的错误
    返回值: 包含错误信息的消息字典
    """
    # 获取错误信息
    error = state.get("error")

    # 获取最后一个消息中的工具调用列表
    tool_calls = state["messages"][-1].tool_calls

    # 返回带有错误信息的 ToolMessage 列表
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # 生成错误内容
                tool_call_id=tc["id"],  # 记录工具调用的唯一ID
            )
            for tc in tool_calls  # 为每个工具调用创建对应的错误消息
        ]
    }

