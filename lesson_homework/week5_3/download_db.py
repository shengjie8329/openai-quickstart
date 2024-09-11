#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   :2024/9/11 18:03
@Author :lancelot.sheng
@File   :download_db.py
"""
import requests

# 下载 Chinook 数据库
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'  # https -> http
        }

response = requests.get(url, proxies=proxies)
if response.status_code == 200:
    # 将下载的内容保存为 Chinook.db
    with open("data/Chinook.db", "wb") as file:
        file.write(response.content)
    print("文件已下载并保存为 Chinook.db")
else:
    print(f"文件下载失败，状态码: {response.status_code}")