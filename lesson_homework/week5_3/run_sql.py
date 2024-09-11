#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/11 18:13
@Author :lancelot.sheng
@File   :run_sql.py
"""
# 导入 SQLDatabase 模块
from langchain_community.utilities import SQLDatabase

# 连接 SQLite 数据库
db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())

# 执行 SQL 查询
rs = db.run("SELECT * FROM Artist LIMIT 10;")
print(str(rs))