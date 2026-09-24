import os
import sys
import tempfile

# 让 tests/ 下的测试可以直接导入仓库根目录的 app、tasks 等模块
sys.path.insert(0, os.path.dirname(__file__))

# app.py 在导入时执行启动检查：测试环境使用固定假密钥与临时数据目录
# （测试中使用固定假密钥不受限制，生产禁止回退默认值）
os.environ.setdefault("SECRET_KEY", "test-secret-key-do-not-use-in-production")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="endictation-test-data-"))
