# NAS 迁移版容器镜像（docs/NAS_MIGRATION.md §5）
# Azure Speech SDK 已按 §3.5 决策移除，改用 REST 接口，无需安装其系统库。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py tasks.py config.py config.default.json ./
COPY templates ./templates
COPY static ./static
COPY Dockerfile docker_start.py ./
COPY docker_start.py /opt/endictation/docker_start.py

# 非 root 运行；预创建运行目录并归属应用用户。
# 空 named volume 首次挂载时会继承该目录的属主与权限；已有卷若不可写，
# 应用启动探测会直接失败并输出路径与所需权限（app.bootstrap_runtime）。
RUN useradd --create-home --uid 1000 endictation \
    && mkdir -p /app/.local-data/tasks \
    && chown -R endictation:endictation /app

USER endictation

EXPOSE 5001

# 每次容器启动先更新代码，再 exec 单 worker + 4 线程的 Gunicorn。
CMD ["python", "/opt/endictation/docker_start.py"]
