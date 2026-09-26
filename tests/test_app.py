"""后端接口测试（NAS_MIGRATION 步骤 3）。

默认替换外部 API（OCR 与各 TTS 实现均为假实现），不消耗任何额度。
覆盖：会话与同源校验、输入上限、任务受理与状态查询、文件隔离、错误传播、
预算耗尽、音频服务与重启失效语义。
"""

import io
import json
import time

import pytest
from PIL import Image

import tasks as tasks_module
import app as app_module
from tasks import TaskManager


ORIGIN = {"Origin": "http://localhost"}  # Flask 测试客户端默认 Host 为 localhost


@pytest.fixture()
def manager(tmp_path):
    return TaskManager(tmp_path / "tasks")


@pytest.fixture()
def client(manager):
    flask_app = app_module.create_app(manager)
    return flask_app.test_client()


def make_png(width=10, height=10):
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def establish_session(client):
    """首次访问首页建立会话（签名 cookie 内含服务端生成的 sid）。"""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp


def wait_task(client, task_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        if data["status"] in ("succeeded", "failed"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


def submit_upload(client, png_bytes=make_png(), filename="list.png", headers=None):
    data = {"file": (io.BytesIO(png_bytes), filename)}
    return client.post("/upload", data=data, headers=headers or ORIGIN)


def submit_tts(client, sentences, engine="azure", language="sg-en",
              gender="female", speed=-15, headers=None):
    payload = {
        "sentences": sentences,
        "tts_engine": engine,
        "tts_language": language,
        "voice_gender": gender,
    }
    if speed is not None:
        payload["speed"] = speed
    return client.post("/generate-tts", data=json.dumps(payload),
                       content_type="application/json", headers=headers or ORIGIN)


SAMPLE_MARKDOWN = "Unit 1\nShe has a **beautiful** garden.\nHe runs **quickly**."


def fake_extract(image_path, mime_type, runtime_config):
    return app_module.parse_ocr_response(SAMPLE_MARKDOWN)


def fake_synth(text, tts_model, speed_percent, output_path):
    output_path.write_bytes(b"ID3,fake-mp3:" + text.encode("utf-8"))


# ---- 基础路由与会话 ----

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_index_sets_session_cookie(client):
    resp = establish_session(client)
    cookie = resp.headers.get("Set-Cookie", "")
    assert "endictation_session=" in cookie
    assert "HttpOnly" in cookie


def test_write_requires_session(client):
    data = {"file": (io.BytesIO(make_png()), "list.png")}
    resp = client.post("/upload", data=data, headers=ORIGIN)
    assert resp.status_code == 403
    assert resp.get_json()["error"]["code"] == "forbidden"


def test_write_requires_same_origin(client):
    establish_session(client)
    assert client.post("/upload", data={"file": (io.BytesIO(make_png()), "list.png")}).status_code == 403
    bad = dict(ORIGIN)
    bad["Origin"] = "http://evil.example"
    resp = client.post("/upload", data={"file": (io.BytesIO(make_png()), "list.png")}, headers=bad)
    assert resp.status_code == 403
    referer_only = {"Referer": "http://localhost/"}
    resp = client.post("/upload", data={"file": (io.BytesIO(make_png()), "list.png")}, headers=referer_only)
    assert resp.status_code == 202  # 仅 Referer 同源也可通过；有效图片直接受理


def test_upload_missing_file(client):
    establish_session(client)
    resp = client.post("/upload", data={}, headers=ORIGIN)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_input"


# ---- 输入校验 ----

def test_upload_accepts_any_content_without_local_validation(client, monkeypatch):
    """2026-09-25 用户决策：不做本地内容校验，任意字节直接受理，内容由提供方判定。"""
    monkeypatch.setattr(app_module, "extract_text_cloud", fake_extract)
    establish_session(client)
    resp = submit_upload(client,
                         png_bytes=b"\xff\xd8\xff\xe0MPO/JPEG-ish bytes",
                         filename="IMG_9888.jpeg")
    assert resp.status_code == 202
    done = wait_task(client, resp.get_json()["task_id"])
    assert done["status"] == "succeeded"


def test_upload_invalid_content_fails_via_provider(client, monkeypatch):
    """无效内容由提供方判定失败，任务级明确报错 ocr_failed。"""
    def reject_payload(image_path, mime_type, runtime_config):
        raise RuntimeError("Invalid image payload")

    monkeypatch.setattr(app_module, "extract_text_cloud", reject_payload)
    establish_session(client)
    resp = submit_upload(client, png_bytes=b"definitely not an image",
                         filename="note.txt")
    assert resp.status_code == 202
    done = wait_task(client, resp.get_json()["task_id"])
    assert done["status"] == "failed"
    assert done["error"]["code"] == "ocr_failed"
    assert "Invalid image payload" in done["error"]["message"]


def test_request_body_too_large(manager):
    flask_app = app_module.create_app(manager, max_content_length=1024)
    client = flask_app.test_client()
    establish_session(client)
    resp = client.post("/generate-tts", data="x" * 2048,
                       content_type="application/json", headers=ORIGIN)
    assert resp.status_code == 413
    assert resp.get_json()["error"]["code"] == "too_large"


def test_generate_tts_invalid_json(client):
    establish_session(client)
    resp = client.post("/generate-tts", data="{not json",
                      content_type="application/json", headers=ORIGIN)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_input"


def test_generate_tts_invalid_combo(client):
    establish_session(client)
    resp = submit_tts(client, [{"text": "Hello.", "is_title": False}],
                      engine="google", language="sg-en")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_tts_combo"


def test_generate_tts_sentence_limits(client):
    establish_session(client)
    # 超过 50 条非标题句子
    rows = [{"text": "Hello.", "is_title": False} for _ in range(51)]
    assert submit_tts(client, rows).status_code == 400
    # 单句超过 500 字符
    long_row = [{"text": "a" * 501, "is_title": False}]
    assert submit_tts(client, long_row).status_code == 400
    # 每句重点词超过 20 个
    many_words = [{"text": "Hello.", "is_title": False,
                   "bold_words": [{"word": f"w{i}"} for i in range(21)]}]
    assert submit_tts(client, many_words).status_code == 400
    # 重点词超过 100 字符
    long_word = [{"text": "Hello.", "is_title": False,
                  "bold_words": [{"word": "b" * 101}]}]
    assert submit_tts(client, long_word).status_code == 400
    # 合成总量超过 10000 字符
    big_rows = [{"text": "a" * 500, "is_title": False} for _ in range(21)]
    assert submit_tts(client, big_rows).status_code == 400
    # 只有标题，没有可合成句子
    assert submit_tts(client, [{"text": "Unit 1", "is_title": True}]).status_code == 400


def test_generate_tts_speed_validation(client):
    establish_session(client)
    base = [{"text": "Hello.", "is_title": False}]
    assert submit_tts(client, base, speed=999).status_code == 400
    resp = client.post("/generate-tts",
                       data='{"sentences":[{"text":"Hello.","is_title":false}],'
                            '"tts_engine":"azure","tts_language":"sg-en",'
                            '"voice_gender":"female","speed":NaN}',
                       content_type="application/json", headers=ORIGIN)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_input"


def test_generate_tts_gtts_ignores_speed(client, monkeypatch):
    """不支持速度的引擎收到非默认速度时忽略该值，不报错。"""
    establish_session(client)
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    resp = submit_tts(client, [{"text": "Hello.", "is_title": False}],
                      engine="gtts", language="uk-en", speed=30)
    assert resp.status_code == 202


def test_generate_tts_capacity_429(tmp_path):
    manager = TaskManager(tmp_path / "tasks", max_tasks=1)
    release = __import__("threading").Event()

    def blocking_execute(context):
        assert release.wait(5)
        return {}, None

    manager.submit(stage="tts", owner="someone", execute=blocking_execute)
    flask_app = app_module.create_app(manager)
    client = flask_app.test_client()
    establish_session(client)
    resp = submit_tts(client, [{"text": "Hello.", "is_title": False}])
    assert resp.status_code == 429
    body = resp.get_json()
    assert body["error"]["code"] == "capacity"
    assert "未被受理" in body["error"]["message"]
    release.set()


# ---- OCR 流程 ----

def test_upload_ocr_success_flow(client, monkeypatch, manager):
    monkeypatch.setattr(app_module, "extract_text_cloud", fake_extract)
    establish_session(client)
    resp = submit_upload(client)
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["status_url"] == f"/tasks/{body['task_id']}"

    done = wait_task(client, body["task_id"])
    assert done["status"] == "succeeded"
    assert done["stage"] == "ocr"
    assert done["current"] == 0 and done["total"] == 0 and done["progress"] == 0
    rows = done["result"]
    assert rows[0] == {"text": "Unit 1", "original_text": "Unit 1",
                       "is_title": True, "title": "Unit 1"}
    assert rows[1]["bold_words"] == [{"word": "beautiful"}]
    assert rows[2]["bold_words"] == [{"word": "quickly"}]
    assert all("audio_path" not in row for row in rows), "OCR 结果不应包含音频路径"

    task_dir = manager.task_dir(body["task_id"], manager_owner(manager, body["task_id"]))
    assert not list(task_dir.glob("input.*")), "OCR 原图应在任务结束后删除"


def manager_owner(manager, task_id):
    with manager._lock:
        return manager._tasks[task_id]["owner"]


def test_upload_ocr_failure_marks_ocr_failed(client, monkeypatch):
    def boom(image_path, mime_type, runtime_config):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(app_module, "extract_text_cloud", boom)
    establish_session(client)
    resp = submit_upload(client)
    assert resp.status_code == 202
    done = wait_task(client, resp.get_json()["task_id"])
    assert done["status"] == "failed"
    assert done["error"]["code"] == "ocr_failed"
    assert "quota exceeded" in done["error"]["message"]


def test_same_name_images_do_not_interfere(manager, monkeypatch):
    monkeypatch.setattr(app_module, "extract_text_cloud", fake_extract)
    flask_app = app_module.create_app(manager)
    client_a = flask_app.test_client()
    client_b = flask_app.test_client()
    establish_session(client_a)
    establish_session(client_b)

    resp_a = submit_upload(client_a, filename="list.png")
    resp_b = submit_upload(client_b, filename="list.png")
    assert resp_a.status_code == 202 and resp_b.status_code == 202
    task_a, task_b = resp_a.get_json()["task_id"], resp_b.get_json()["task_id"]
    assert task_a != task_b

    done_a = wait_task(client_a, task_a)
    done_b = wait_task(client_b, task_b)
    assert done_a["status"] == "succeeded" and done_b["status"] == "succeeded"


# ---- TTS 流程 ----

def test_generate_tts_success_structure_and_audio(client, monkeypatch, manager):
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    resp = submit_tts(client, rows)
    assert resp.status_code == 202
    task_id = resp.get_json()["task_id"]

    done = wait_task(client, task_id)
    assert done["status"] == "succeeded"
    assert done["current"] == done["total"] == 4  # 2 句 + 2 词
    assert done["progress"] == 100
    assert done["warnings"] is None

    result = done["result"]
    assert result[0] == {"text": "Unit 1", "original_text": "Unit 1",
                         "is_title": True, "title": "Unit 1"}
    assert result[1]["audio_path"] == f"/audio/{task_id}/sentence_1.mp3"
    assert result[1]["bold_words"] == [
        {"word": "beautiful", "audio_path": f"/audio/{task_id}/word_1_0.mp3"}]
    assert result[2]["audio_path"] == f"/audio/{task_id}/sentence_2.mp3"

    audio_resp = client.get(result[1]["audio_path"])
    assert audio_resp.status_code == 200
    assert audio_resp.data == b"ID3,fake-mp3:" + b"She has a beautiful garden."
    assert audio_resp.mimetype == "audio/mpeg"


def test_generate_tts_partial_failure(client, monkeypatch):
    def flaky_synth(text, tts_model, speed_percent, output_path):
        if text == "quickly":
            raise ValueError("Azure TTS取消")
        fake_synth(text, tts_model, speed_percent, output_path)

    monkeypatch.setattr(app_module, "synthesize_item", flaky_synth)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    resp = submit_tts(client, rows)
    task_id = resp.get_json()["task_id"]
    done = wait_task(client, task_id)

    assert done["status"] == "succeeded", "部分成功返回 succeeded + warnings"
    assert done["current"] == done["total"] == 4, "失败项也计入尝试进度"
    assert done["warnings"] == ["重点词 quickly 音频生成失败：Azure TTS取消"]
    result = done["result"]
    assert result[2]["audio_path"] == f"/audio/{task_id}/sentence_2.mp3"
    assert result[2]["bold_words"] == [{"word": "quickly"}], "失败项不暴露播放路径"
    assert client.get(f"/audio/{task_id}/word_2_0.mp3").status_code == 404


def test_generate_tts_all_failed(client, monkeypatch):
    def always_fail(text, tts_model, speed_percent, output_path):
        raise ValueError("凭据无效")

    monkeypatch.setattr(app_module, "synthesize_item", always_fail)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    resp = submit_tts(client, rows)
    done = wait_task(client, resp.get_json()["task_id"])

    assert done["status"] == "failed"
    assert done["error"]["code"] == "tts_failed"
    assert "凭据无效" in done["error"]["message"]


def test_generate_tts_budget_exhausted_partial(client, monkeypatch):
    clock = {"value": 0}
    monkeypatch.setattr(tasks_module.TaskContext, "elapsed_seconds",
                        lambda self: clock["value"])

    def one_then_exhaust(text, tts_model, speed_percent, output_path):
        fake_synth(text, tts_model, speed_percent, output_path)
        clock["value"] = app_module.TASK_BUDGET_SECONDS + 1

    monkeypatch.setattr(app_module, "synthesize_item", one_then_exhaust)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    resp = submit_tts(client, rows)
    task_id = resp.get_json()["task_id"]
    done = wait_task(client, task_id)

    assert done["status"] == "succeeded", "已有有效音频 → 部分成功"
    assert done["current"] == 1 and done["total"] == 4, "未尝试项不增加 current"
    assert any("未尝试" in warning and "预算" in warning for warning in done["warnings"])
    result = done["result"]
    assert result[1]["audio_path"] == f"/audio/{task_id}/sentence_1.mp3"
    assert "audio_path" not in result[2], "预算耗尽后不再生成"


def test_generate_tts_budget_exhausted_nothing_done(client, monkeypatch):
    monkeypatch.setattr(app_module, "TASK_BUDGET_SECONDS", 0)
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    resp = submit_tts(client, rows)
    done = wait_task(client, resp.get_json()["task_id"])
    assert done["status"] == "failed"
    assert done["error"]["code"] == "tts_failed"
    assert "预算已耗尽" in done["error"]["message"]
    assert done["current"] == 0


def test_repeated_tts_for_same_ocr(client, monkeypatch):
    """同一 OCR 结果重复生成：两次任务独立存在，互不覆盖。"""
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    first = submit_tts(client, rows).get_json()["task_id"]
    second = submit_tts(client, rows, speed=0).get_json()["task_id"]
    assert first != second
    assert wait_task(client, first)["status"] == "succeeded"
    assert wait_task(client, second)["status"] == "succeeded"


# ---- 隔离与失效语义 ----

def test_task_and_audio_isolated_between_sessions(manager, monkeypatch):
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    flask_app = app_module.create_app(manager)
    client_a = flask_app.test_client()
    client_b = flask_app.test_client()
    establish_session(client_a)
    establish_session(client_b)

    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    task_id = submit_tts(client_a, rows).get_json()["task_id"]
    done = wait_task(client_a, task_id)
    audio_path = done["result"][1]["audio_path"]

    # B 查询 A 的任务或音频：统一 404，不泄露任务是否存在
    assert client_b.get(f"/tasks/{task_id}").status_code == 404
    assert client_b.get(audio_path).status_code == 404
    # 无会话同样 404
    assert flask_app.test_client().get(f"/tasks/{task_id}").status_code == 404


def test_task_not_found_for_unknown_id(client):
    establish_session(client)
    resp = client.get("/tasks/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "task_not_found"


def test_audio_filename_restricted(client, monkeypatch):
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    establish_session(client)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    task_id = submit_tts(client, rows).get_json()["task_id"]
    wait_task(client, task_id)

    assert client.get(f"/audio/{task_id}/input.png").status_code == 404
    assert client.get(f"/audio/{task_id}/sentence_99.mp3").status_code == 404
    assert client.get(f"/audio/{task_id}/not_sentence_atall.mp3").status_code == 404


def test_restart_invalidates_old_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "synthesize_item", fake_synth)
    manager_one = TaskManager(tmp_path / "tasks")
    app_one = app_module.create_app(manager_one)
    client_one = app_one.test_client()
    establish_session(client_one)
    rows = app_module.parse_ocr_response(SAMPLE_MARKDOWN)
    task_id = submit_tts(client_one, rows).get_json()["task_id"]
    wait_task(client_one, task_id)

    # 模拟进程重启：遗留目录被清理，旧任务一律不存在
    manager_two = app_module.bootstrap_runtime(str(tmp_path))
    assert not (tmp_path / "tasks" / task_id).exists(), "启动时清理遗留任务目录"
    app_two = app_module.create_app(manager_two)
    client_two = app_two.test_client()
    establish_session(client_two)
    assert client_two.get(f"/tasks/{task_id}").status_code == 404
    manager_two.close()


def test_session_key_persisted_without_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "ignored-environment-key")
    manager = app_module.bootstrap_runtime(tmp_path)
    try:
        first = app_module.create_app(manager)
        second = app_module.create_app(manager)
        assert first.secret_key == second.secret_key
        assert len(first.secret_key) == 32
        assert (tmp_path / ".session-key").stat().st_mode & 0o777 == 0o600
    finally:
        manager.close()
