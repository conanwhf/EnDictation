import os
import sys

# 让 tests/ 下的测试可以直接导入仓库根目录的 app、tasks 等模块
sys.path.insert(0, os.path.dirname(__file__))
