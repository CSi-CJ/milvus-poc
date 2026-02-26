#!/usr/bin/env python3
"""
多模态文件索引器 - 主入口文件
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_indexer.cli import main as cli_main

if __name__ == "__main__":
    cli_main()