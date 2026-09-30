"""
Workspace Hub 模块命令行直接执行入口
====================================
运行命令：python -m tools.workspace_hub
"""

import os
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.workspace_hub.app import main

if __name__ == "__main__":
    main()
