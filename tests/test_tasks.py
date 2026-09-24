"""任务模型针对性测试（NAS_MIGRATION 步骤 2）。

覆盖：任务创建、FIFO 排队、容量竞争、owner 校验、过期清理、失败后继续执行、
输入文件生命周期、进度更新与快照隔离。外部操作全部使用可控制的假实现，不触网。
"""

import threading
import time
from pathlib import Path

import pytest

from tasks import (
    CapacityError,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    TaskFailure,
    TaskManager,
)


def wait_until(cond, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class BlockableJob:
    """可阻塞的假执行函数：开始时置 started，等待 release 后返回结果或抛出失败。"""

    def __init__(self, result=None, error=None, on_start=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.result = result if result is not None else {"ok": True}
        self.error = error
        self.on_start = on_start
        self.call_count = 0

    def __call__(self, context):
        self.call_count += 1
        if self.on_start:
            self.on_start(context)
        self.started.set()
        assert self.release.wait(10), "测试执行函数被意外阻塞超过 10 秒"
        if self.error is not None:
            raise self.error
        return self.result, None


def make_manager(tmp_path, **kwargs):
    return TaskManager(tmp_path / "tasks", **kwargs)


def test_submit_returns_queued_then_runs_to_success(tmp_path):
    manager = make_manager(tmp_path)
    job = BlockableJob(result={"sentences": []})
    snap = manager.submit(stage="ocr", owner="session-a", execute=job)

    assert snap["task_id"]
    assert snap["status"] == STATUS_QUEUED
    assert snap["stage"] == "ocr"
    assert snap["message"] == "排队中"
    assert snap["result"] is None

    assert job.started.wait(2)
    running = manager.get(snap["task_id"], "session-a")
    assert running["status"] == STATUS_RUNNING

    job.release.set()
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)
    done = manager.get(task_id, "session-a")
    assert done["result"] == {"sentences": []}
    assert done["warnings"] is None
    manager.close()


def test_fifo_serial_execution_order(tmp_path):
    manager = make_manager(tmp_path)
    order = []
    jobs = []
    for name in ("a", "b", "c"):
        job = BlockableJob(on_start=lambda ctx, n=name: order.append(n))
        jobs.append(job)
        manager.submit(stage="tts", owner="session-a", execute=job)

    # 串行：同一时刻最多一个在执行
    assert jobs[0].started.wait(2)
    time.sleep(0.2)
    assert jobs[1].call_count == 0 and jobs[2].call_count == 0

    for job in jobs:
        job.release.set()

    assert wait_until(lambda: len(order) == 3)
    assert order == ["a", "b", "c"], "执行顺序必须等于受理顺序"
    manager.close()


def test_second_submission_accepted_while_first_running(tmp_path):
    manager = make_manager(tmp_path)
    first = BlockableJob()
    second = BlockableJob()
    snap_a = manager.submit(stage="ocr", owner="session-a", execute=first)
    assert first.started.wait(2)

    snap_b = manager.submit(stage="ocr", owner="session-b", execute=second)
    assert snap_b["status"] == STATUS_QUEUED
    assert manager.get(snap_b["task_id"], "session-b")["status"] == STATUS_QUEUED
    assert manager.task_count() == 2

    first.release.set()
    second.release.set()
    assert wait_until(lambda: manager.get(snap_b["task_id"], "session-b")["status"] == STATUS_SUCCEEDED)
    manager.close()


def test_failed_task_then_next_starts_automatically(tmp_path):
    manager = make_manager(tmp_path)
    first = BlockableJob(error=TaskFailure("ocr_failed", "OCR 服务返回错误"))
    second = BlockableJob()
    snap_a = manager.submit(stage="ocr", owner="session-a", execute=first)
    snap_b = manager.submit(stage="ocr", owner="session-a", execute=second)

    first.release.set()
    task_a, task_b = snap_a["task_id"], snap_b["task_id"]
    assert wait_until(lambda: manager.get(task_a, "session-a")["status"] == STATUS_FAILED)
    failed = manager.get(task_a, "session-a")
    assert failed["error"]["code"] == "ocr_failed"
    assert "OCR" in failed["error"]["message"]

    # 首项失败后，下一项自动开始，无需重新提交
    assert second.started.wait(2)
    second.release.set()
    assert wait_until(lambda: manager.get(task_b, "session-a")["status"] == STATUS_SUCCEEDED)
    manager.close()


def test_unexpected_exception_marks_failed_with_stage_code(tmp_path):
    manager = make_manager(tmp_path)

    def boom(context):
        raise ValueError("未预期异常")

    snap = manager.submit(stage="tts", owner="session-a", execute=boom)
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_FAILED)
    failed = manager.get(task_id, "session-a")
    assert failed["error"]["code"] == "tts_failed"
    assert "未预期" in failed["error"]["message"]
    manager.close()


def test_owner_isolation(tmp_path):
    manager = make_manager(tmp_path)
    job = BlockableJob()
    snap = manager.submit(stage="tts", owner="session-a", execute=job)
    task_id = snap["task_id"]
    job.release.set()
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)

    # 其他会话查询不到任务的存在，也不区分「不存在」与「无权限」
    assert manager.get(task_id, "session-b") is None
    assert manager.get(task_id, "session-a") is not None
    assert manager.task_dir(task_id, "session-b") is None
    assert manager.task_dir(task_id, "session-a") is not None
    assert manager.get("不存在", "session-a") is None
    manager.close()


def test_capacity_rejects_then_recovers_after_expiry(tmp_path):
    manager = make_manager(tmp_path, max_tasks=2, retention_seconds=0.2)
    first = BlockableJob()
    second = BlockableJob()
    manager.submit(stage="tts", owner="session-a", execute=first)
    manager.submit(stage="tts", owner="session-a", execute=second)
    first.release.set()
    second.release.set()
    task_ids = list(manager._tasks.keys())
    assert wait_until(lambda: all(
        manager._tasks[t]["status"] == STATUS_SUCCEEDED for t in task_ids))

    # 数量上限且无过期项可清理 → 明确拒绝
    with pytest.raises(CapacityError):
        manager.submit(stage="tts", owner="session-a", execute=BlockableJob())

    # 超过保留期后，提交时自动清理过期项并恢复受理
    time.sleep(0.3)
    third = BlockableJob()
    snap = manager.submit(stage="tts", owner="session-a", execute=third)
    third.release.set()
    assert wait_until(lambda: manager.get(snap["task_id"], "session-a")["status"] == STATUS_SUCCEEDED)
    assert manager.get(task_ids[0], "session-a") is None, "过期任务记录应被移除"
    manager.close()


def test_expiry_removes_record_and_task_dir(tmp_path):
    manager = make_manager(tmp_path, retention_seconds=24 * 3600)
    job = BlockableJob()
    snap = manager.submit(stage="tts", owner="session-a", execute=job)
    job.release.set()
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)
    task_dir = manager.task_dir(task_id, "session-a")
    (task_dir / "sentence_0.mp3").write_bytes(b"fake")

    # 白盒：把完成时间拨回过期之前
    with manager._lock:
        manager._tasks[task_id]["finished_at"] -= manager.retention_seconds + 1
    manager.cleanup_expired()

    assert manager.get(task_id, "session-a") is None
    assert not task_dir.exists(), "清理时应同时移除任务记录和音频目录"
    manager.close()


def test_queued_and_running_tasks_never_expire(tmp_path):
    manager = make_manager(tmp_path, retention_seconds=0.1)
    first = BlockableJob()
    queued = BlockableJob()
    snap_a = manager.submit(stage="ocr", owner="session-a", execute=first)
    snap_b = manager.submit(stage="ocr", owner="session-a", execute=queued)
    assert first.started.wait(2)
    time.sleep(0.3)

    with manager._lock:
        manager._tasks[snap_b["task_id"]]["created_at"] -= 10 * 365 * 24 * 3600

    manager.cleanup_expired()
    assert manager.get(snap_a["task_id"], "session-a") is not None, "执行中任务不参与过期清理"
    assert manager.get(snap_b["task_id"], "session-a") is not None, "排队中任务不参与过期清理"

    first.release.set()
    queued.release.set()
    assert wait_until(lambda: manager.get(snap_b["task_id"], "session-a")["status"] == STATUS_SUCCEEDED)
    manager.close()


def test_input_file_kept_while_queued_deleted_after_run(tmp_path):
    manager = make_manager(tmp_path)

    def save_input(task_dir):
        path = task_dir / "input.jpg"
        path.write_bytes(b"fake-jpeg")
        return str(path)

    first = BlockableJob()
    manager.submit(stage="ocr", owner="session-a", execute=first)
    assert first.started.wait(2)

    queued_job = BlockableJob()
    queued_snap = manager.submit(stage="ocr", owner="session-b",
                                 execute=queued_job, input_saver=save_input)
    input_path = manager._tasks[queued_snap["task_id"]]["input_path"]
    assert input_path and Path(input_path).exists(), "排队 OCR 的输入文件必须保留到任务执行结束"

    first.release.set()
    queued_job.release.set()
    assert wait_until(lambda: manager.get(queued_snap["task_id"], "session-b")["status"] == STATUS_SUCCEEDED)
    assert not Path(input_path).exists(), "OCR 原图应在任务结束后删除"
    assert manager.task_dir(queued_snap["task_id"], "session-b").exists(), "输出目录保留至过期清理"
    manager.close()


def test_progress_update_and_zero_total(tmp_path):
    manager = make_manager(tmp_path)

    def job_with_progress(context):
        context.set_progress(3, 10, "正在处理第 3/10 项")
        return {"done": True}, None

    snap = manager.submit(stage="tts", owner="session-a", execute=job_with_progress)
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)
    done = manager.get(task_id, "session-a")
    assert done["current"] == 3 and done["total"] == 10
    assert done["progress"] == 30

    def job_zero_total(context):
        context.set_progress(0, 0, "识别中")
        return {"done": True}, None

    snap2 = manager.submit(stage="ocr", owner="session-a", execute=job_zero_total)
    task_id2 = snap2["task_id"]
    assert wait_until(lambda: manager.get(task_id2, "session-a")["status"] == STATUS_SUCCEEDED)
    assert manager.get(task_id2, "session-a")["progress"] == 0
    manager.close()


def test_warnings_surfaced_on_partial_success(tmp_path):
    manager = make_manager(tmp_path)

    def partial(context):
        return {"sentences": [1, 2]}, ["句子 3 音频生成失败：Azure TTS取消", "重点词 quickly 音频生成失败"]

    snap = manager.submit(stage="tts", owner="session-a", execute=partial)
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)
    done = manager.get(task_id, "session-a")
    assert done["warnings"] == ["句子 3 音频生成失败：Azure TTS取消", "重点词 quickly 音频生成失败"]
    manager.close()


def test_snapshot_is_isolated_copy(tmp_path):
    manager = make_manager(tmp_path)
    job = BlockableJob(result={"nested": ["a"]})
    snap = manager.submit(stage="tts", owner="session-a", execute=job)
    task_id = snap["task_id"]
    job.release.set()
    assert wait_until(lambda: manager.get(task_id, "session-a")["status"] == STATUS_SUCCEEDED)

    current = manager.get(task_id, "session-a")
    current["message"] = "被篡改"
    current["result"]["nested"].append("被篡改")
    again = manager.get(task_id, "session-a")
    assert again["message"] != "被篡改"
    assert again["result"]["nested"] == ["a"]
    manager.close()


def test_submit_failure_undoes_registration_and_files(tmp_path):
    manager = make_manager(tmp_path)
    manager.close()  # 关闭执行器，使 submit 内部提交失败

    def save_input(task_dir):
        path = task_dir / "input.jpg"
        path.write_bytes(b"fake-jpeg")
        return str(path)

    with pytest.raises(RuntimeError):
        manager.submit(stage="ocr", owner="session-a",
                       execute=BlockableJob(), input_saver=save_input)

    assert manager.task_count() == 0, "提交失败时必须撤销登记"
    assert not any(manager.tasks_root.iterdir()), "提交失败时必须清理任务目录与上传文件"


def test_purge_all_task_dirs_only_touches_task_root(tmp_path):
    manager = make_manager(tmp_path)
    stray = tmp_path / "other-data"
    stray.mkdir()
    (stray / "keep.txt").write_text("x")

    stale = manager.tasks_root / "old-task"
    stale.mkdir(parents=True)
    (stale / "sentence_0.mp3").write_bytes(b"old")
    (manager.tasks_root / "not-a-dir.txt").write_text("x")

    manager.purge_all_task_dirs()
    assert not stale.exists()
    assert (manager.tasks_root / "not-a-dir.txt").exists()
    assert (stray / "keep.txt").exists(), "只清理应用专属任务目录"


def test_cleanup_loop_thread_removes_expired(tmp_path):
    manager = make_manager(tmp_path, retention_seconds=0.1, cleanup_interval_seconds=0.05)
    manager.start_cleanup_loop()
    job = BlockableJob()
    snap = manager.submit(stage="tts", owner="session-a", execute=job)
    job.release.set()
    task_id = snap["task_id"]
    assert wait_until(lambda: manager.get(task_id, "session-a") is None, timeout=3), \
        "后台清理线程应按间隔移除过期任务"
    manager.close()
