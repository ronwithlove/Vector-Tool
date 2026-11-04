#!/bin/bash
# 双击运行（macOS）。第一次会自动建虚拟环境并安装依赖，之后秒开。
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "首次运行：正在安装依赖（约 1 分钟）…"
  python3 -m venv .venv || { echo "需要先安装 Python 3：https://www.python.org/downloads/"; read -r; exit 1; }
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt || { echo "依赖安装失败"; read -r; exit 1; }
fi
.venv/bin/python server.py
