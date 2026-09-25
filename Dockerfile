# NAS 迁移版容器镜像（docs/NAS_MIGRATION.md §5）
# Azure Speech SDK 已按 §3.5 决策移除，改用 REST 接口，无需安装其系统库。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py tasks.py ./
COPY templates ./templates

# 非 root 运行；预创建 /data/tasks 并归属应用用户。
# 空 named volume 首次挂载时会继承该目录的属主与权限；已有卷若不可写，
# 应用启动探测会直接失败并输出路径与所需权限（app.bootstrap_runtime）。
RUN useradd --create-home --uid 1000 endictation \
    && mkdir -p /data/tasks \
    && chown -R endictation:endictation /data /app

USER endictation

EXPOSE 5001

# 单 worker + 4 线程；不使用 --preload：任务执行器必须在 worker 进程内初始化
CMD ["gunicorn", "--workers", "1", "--worker-class", "gthread", "--threads", "4", "--bind", "0.0.0.0:5001", "app:app"]
