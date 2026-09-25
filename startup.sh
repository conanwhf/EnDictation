#!/bin/bash
# 启动脚本：配置与目录权限检查通过后，以 exec 方式启动 Gunicorn。
# Gunicorn worker 与 python app.py 开发入口共用 app.bootstrap_runtime 的同一套启动检查。
set -euo pipefail

: "${SECRET_KEY:?启动失败：必须设置 SECRET_KEY 环境变量（会话签名密钥），不得使用回退默认值}"

DATA_DIR="${DATA_DIR:-.local-data}"
TASKS_DIR="$DATA_DIR/tasks"
PORT="${PORT:-5001}"

if ! mkdir -p "$TASKS_DIR" 2>/dev/null; then
    echo "启动失败：任务目录 $TASKS_DIR 无法创建（需要写入权限）" >&2
    exit 1
fi
if ! touch "$TASKS_DIR/.startup-probe" 2>/dev/null; then
    echo "启动失败：任务目录 $TASKS_DIR 不可写（需要写入权限）" >&2
    exit 1
fi
rm -f "$TASKS_DIR/.startup-probe"

cd "$(dirname "$0")"
exec gunicorn --workers 1 --worker-class gthread --threads 4 \
    --bind "0.0.0.0:$PORT" app:app
