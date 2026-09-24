"""进程内任务模型：串行 FIFO 执行、容量限制、owner 校验与过期清理。

对应 docs/NAS_MIGRATION.md §3.1/§3.3/§3.4 的任务生命周期部分。本模块不依赖
Flask 与任何外部服务，可独立测试；OCR/TTS 等具体执行逻辑由 app.py 以执行函数
形式传入。

设计要点：
- 全局最多一个生成任务在执行；其余有效任务进入 FIFO 队列，提交立即返回 queued。
- 使用同一把锁原子地完成容量检查、任务登记、输入保存与执行器提交；
  提交失败时撤销登记并清理任务目录（含上传文件）。
- 任务记录与输出目录都按 task_id 组织；owner 不匹配时与「任务不存在」同样处理，
  不向请求方泄露任务是否曾经存在。
- 排队中与执行中的任务不参与过期清理；只有 succeeded/failed 任务按完成时间过期，
  清理时同时移除记录和任务目录。
- OCR 原图在任务执行结束（成功或失败）后的 finally 中删除。
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

DEFAULT_MAX_TASKS = 100
DEFAULT_RETENTION_SECONDS = 24 * 3600
DEFAULT_CLEANUP_INTERVAL_SECONDS = 60


class CapacityError(RuntimeError):
    """任务记录数量达到上限且无过期项可清理。对应 HTTP 429 与错误类型 capacity。"""


class TaskFailure(Exception):
    """任务执行中的稳定失败；code 是对前端的稳定错误类型。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class TaskContext:
    """传给执行函数的上下文：输出目录、进度回调与执行起始时间。"""

    def __init__(self, manager, record):
        self._manager = manager
        self._task_id = record["task_id"]
        self.task_id = record["task_id"]
        self.task_dir = Path(record["task_dir"])
        self.owner = record["owner"]
        self.started_monotonic = time.monotonic()

    def set_progress(self, current, total, message=None, warnings=None):
        """更新任务进度快照；任务已不存在时静默忽略（如被并发清理）。"""
        self._manager._update(self._task_id, current=current, total=total,
                              message=message, warnings=warnings)

    def elapsed_seconds(self):
        """从实际开始执行起的秒数；任务级执行预算以此计算，不计排队时间。"""
        return time.monotonic() - self.started_monotonic


class TaskManager:
    """集中管理任务字典、锁、串行 FIFO 执行、过期清理和生命周期。

    不抽象通用任务框架：执行函数签名固定为 execute(context) -> (result, warnings)。
    """

    def __init__(self, tasks_root, *, max_tasks=DEFAULT_MAX_TASKS,
                 retention_seconds=DEFAULT_RETENTION_SECONDS,
                 cleanup_interval_seconds=DEFAULT_CLEANUP_INTERVAL_SECONDS):
        self.tasks_root = Path(tasks_root)
        self.max_tasks = max_tasks
        self.retention_seconds = retention_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._lock = threading.RLock()
        # task_id -> record；OrderedDict 的插入序即受理序
        self._tasks = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="endictation-task")
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = None

    # ---- 提交与执行 ----

    def submit(self, *, stage, owner, execute, input_saver=None):
        """受理一个任务，返回 queued 快照。

        input_saver(task_dir) 在锁内把上传文件写入任务目录并返回其路径；
        容量不足抛 CapacityError；登记或提交失败时撤销登记并清理任务目录。
        """
        with self._lock:
            self._cleanup_expired_locked()
            if len(self._tasks) >= self.max_tasks:
                raise CapacityError(
                    f"任务数量已达上限 {self.max_tasks}，且当前没有过期任务可清理，本次请求未被受理")
            task_id = str(uuid.uuid4())
            task_dir = self.tasks_root / task_id
            record = {
                "task_id": task_id,
                "stage": stage,
                "status": STATUS_QUEUED,
                "owner": owner,
                "current": 0,
                "total": 0,
                "message": "排队中",
                "warnings": None,
                "result": None,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "task_dir": str(task_dir),
                "input_path": None,
                "_execute": execute,
            }
            self._tasks[task_id] = record
            try:
                task_dir.mkdir(parents=True, exist_ok=True)
                if input_saver is not None:
                    record["input_path"] = str(input_saver(task_dir))
                self._executor.submit(self._run, task_id)
            except Exception:
                self._tasks.pop(task_id, None)
                shutil.rmtree(task_dir, ignore_errors=True)
                raise
            return self._snapshot(record)

    def _run(self, task_id):
        """在单 worker 执行器线程中运行；这里是任务执行边界的兜底捕获点。"""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record["status"] != STATUS_QUEUED:
                return
            record["status"] = STATUS_RUNNING
            record["started_at"] = time.time()
            record["message"] = "正在处理"

        context = TaskContext(self, record)
        try:
            result, warnings = record["_execute"](context)
            with self._lock:
                record["status"] = STATUS_SUCCEEDED
                record["result"] = result
                record["warnings"] = list(warnings) if warnings else None
                record["message"] = "完成"
                record["finished_at"] = time.time()
        except TaskFailure as exc:
            with self._lock:
                record["status"] = STATUS_FAILED
                record["error"] = {"code": exc.code, "message": exc.message}
                record["finished_at"] = time.time()
        except Exception:
            logger.exception("任务 %s 执行中出现未预期异常", task_id)
            fallback_code = "ocr_failed" if record["stage"] == "ocr" else "tts_failed"
            with self._lock:
                record["status"] = STATUS_FAILED
                record["error"] = {
                    "code": fallback_code,
                    "message": "任务执行出现未预期异常，已记录服务端日志",
                }
                record["finished_at"] = time.time()
        finally:
            # OCR 原图在成功或失败后都删除，不保留到过期清理
            input_path = record.get("input_path")
            if input_path:
                try:
                    os.remove(input_path)
                except OSError:
                    logger.warning("任务 %s 的输入文件删除失败: %s", task_id, input_path)

    # ---- 查询 ----

    def get(self, task_id, owner):
        """按 owner 校验并返回任务快照；不存在或无权限统一返回 None。"""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record["owner"] != owner:
                return None
            return self._snapshot(record)

    def task_dir(self, task_id, owner):
        """按 owner 校验并返回任务输出目录（Path）；不存在或无权限返回 None。"""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record["owner"] != owner:
                return None
            return Path(record["task_dir"])

    def task_count(self):
        with self._lock:
            return len(self._tasks)

    def _update(self, task_id, *, current=None, total=None, message=None, warnings=None):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            if current is not None:
                record["current"] = current
            if total is not None:
                record["total"] = total
            if message is not None:
                record["message"] = message
            if warnings is not None:
                record["warnings"] = list(warnings)

    def _snapshot(self, record):
        total = record["total"]
        current = record["current"]
        progress = int(current / total * 100) if total else 0
        return {
            "task_id": record["task_id"],
            "stage": record["stage"],
            "status": record["status"],
            "current": current,
            "total": total,
            "progress": progress,
            "message": record["message"],
            "warnings": list(record["warnings"]) if record["warnings"] else None,
            "result": copy.deepcopy(record["result"]),
            "error": dict(record["error"]) if record["error"] else None,
        }

    # ---- 过期清理 ----

    def _cleanup_expired_locked(self):
        now = time.time()
        expired = [
            task_id for task_id, record in self._tasks.items()
            if record["status"] in (STATUS_SUCCEEDED, STATUS_FAILED)
            and record["finished_at"] is not None
            and now - record["finished_at"] > self.retention_seconds
        ]
        for task_id in expired:
            record = self._tasks.pop(task_id)
            shutil.rmtree(record["task_dir"], ignore_errors=True)
            logger.info("已清理过期任务 %s", task_id)

    def cleanup_expired(self):
        with self._lock:
            self._cleanup_expired_locked()

    def start_cleanup_loop(self):
        """启动周期清理线程；只在应用 worker 进程内调用，测试默认不启动。"""
        if self._cleanup_thread is not None:
            return

        def loop():
            while not self._cleanup_stop.wait(self.cleanup_interval_seconds):
                try:
                    self.cleanup_expired()
                except Exception:  # noqa: BLE001 - 清理线程不能因单次失败退出
                    logger.exception("周期清理任务时出现异常")

        self._cleanup_thread = threading.Thread(
            target=loop, daemon=True, name="endictation-cleanup")
        self._cleanup_thread.start()

    def stop_cleanup_loop(self):
        self._cleanup_stop.set()

    def purge_all_task_dirs(self):
        """worker 启动时删除遗留任务目录：进程重启后旧任务一律失效。

        只清理本应用专属的 tasks_root 下的子目录，不遍历或删除其他挂载内容。
        """
        if not self.tasks_root.exists():
            return
        for child in self.tasks_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        logger.info("已清理任务根目录 %s 下的遗留任务目录", self.tasks_root)

    def close(self):
        """停止清理线程并关闭执行器（用于测试与应用退出）。"""
        self.stop_cleanup_loop()
        self._executor.shutdown(wait=True)
