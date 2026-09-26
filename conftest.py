import os
import sys
import tempfile

# 让 tests/ 下的测试可以直接导入仓库根目录的 app、tasks 等模块
sys.path.insert(0, os.path.dirname(__file__))

# 不加载开发者的真实配置；生产不读取环境变量。
import config
config.DATA_DIR = tempfile.mkdtemp(prefix="endictation-test-data-")
