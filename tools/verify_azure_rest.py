#!/usr/bin/env python3
"""Azure TTS REST 合成路径验证脚本（NAS 迁移步骤 1）。

用法：
    .venv/bin/python tools/verify_azure_rest.py           # 受控故障验证（假凭据，只连本地故障服务）
    .venv/bin/python tools/verify_azure_rest.py --real    # 追加真实 API 合成（读取本地配置，消耗少量额度）

本脚本与生产实现保持同构：相同的 SSML 结构、请求头、超时元组 (connect, read) 与
「超时后由调用方捕获异常、释放连接、串行执行下一次调用」的执行模型。
受控故障测试全部指向 127.0.0.1 的本地服务，绝不把任何凭据（真实或假）发往真实 Azure 端点。

验证内容对应 NAS_MIGRATION.md §3.5 与步骤 1：
    A. 连接拒绝：错误明确传播（辅助用例，不替代连接建立超时）。
    B. 连接建立超时：本地塞满 backlog 的监听器，按选定 connect 超时值加容差判断。
    C. 服务端迟迟不返回数据：本地挂起服务，按选定 read 超时值加容差判断；
       并通过服务端观察到客户端 FIN 验证连接资源释放。
    D. 串行恢复：上一用例超时后，下一次调用可正常完成。
    E. 提供方拒绝（HTTP 401）：异常明确传播，不静默。
    R. 真实 API 合成（--real）：默认音色正常/调整语速各一条并落盘 MP3；无凭据时记为未验证。
"""

import argparse
import html
import os
import socket
import sys
import threading
import time
from pathlib import Path

import requests

# ---- 选定超时参数（与生产实现一致；受控用例按该值加容差判断） ----
# connect=5s：须小于本机 TCP SYN 重试放弃时间（macOS 约 7.8s），否则套接字超时永远不会触发；
# 10s 在 macOS 上会被 OS 层连接失败抢先（实测 7.83s），无法确定性验证套接字超时本身。
CONNECT_TIMEOUT = 5.0   # 秒
READ_TIMEOUT = 30.0     # 秒
TIMEOUT_TOLERANCE = 5.0  # 秒，容差

OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
DEFAULT_REGION = "southeastasia"

FAKE_KEY = "fake-key-not-a-real-credential"
REAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "azure_rest_verify_output")


def build_ssml(text, voice_name, speed):
    """与生产实现同构的 SSML：文本转义，prosody 控制语速。"""
    body = html.escape(text, quote=False)
    return f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{voice_name}'>
        <prosody rate='{speed}'>
            {body}
        </prosody>
    </voice>
</speak>
"""


def synthesize_azure_rest(text, *, voice_name, speed, key, region=DEFAULT_REGION,
                          output_path, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                          base_url=None):
    """与生产实现同构的 Azure REST 合成调用。

    base_url 仅供本脚本把调用指向本地故障服务；生产实现不暴露该参数。
    返回写入的文件路径；任何失败抛出 requests 异常或 ValueError，不静默降级。
    """
    ssml = build_ssml(text, voice_name, speed)
    url = f"{base_url or f'https://{region}.tts.speech.microsoft.com'}/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
    }
    response = requests.post(url, data=ssml.encode("utf-8"), headers=headers, timeout=timeout)
    response.raise_for_status()
    audio = response.content
    if not audio:
        raise ValueError("Azure TTS 返回空音频")
    with open(output_path, "wb") as f:
        f.write(audio)
    return output_path


# ---- 本地故障服务 ----

class RefusedEndpoint:
    """占用临时端口后立刻关闭，模拟连接拒绝。"""

    def __init__(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


class FullBacklogEndpoint:
    """listen 后塞满 accept 队列且从不 accept，模拟连接建立超时（TCP 握手无法完成）。"""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self._filler = socket.socket()
        self._filler.settimeout(5)
        try:
            self._filler.connect(("127.0.0.1", self.port))
        except OSError:
            # 队列已满时 filler 自身超时，同样达到占满目的
            pass

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        try:
            self._filler.close()
        finally:
            self.srv.close()


def drain_http_request(conn):
    """读取完整 HTTP 请求（头部 + Content-Length 指定的包体）。

    请求可能分多个 TCP 段到达（实测头部与包体常被 Nagle 拆开）；
    必须按 Content-Length 判定读完，不能把迟到的请求体当成客户端后续动作。
    客户端提前断开时返回 False。
    """
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return False
        buf += chunk
    header_blob, _, rest = buf.partition(b"\r\n\r\n")
    length = 0
    for line in header_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(rest) < length:
        chunk = conn.recv(65536)
        if not chunk:
            return False
        rest += chunk
    return True


class HangingEndpoint:
    """接受连接、读取完整请求后永不返回响应；并记录客户端是否主动关闭连接（资源释放证据）。"""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(5)
        self.port = self.srv.getsockname()[1]
        self.connections = 0
        self.accepted = threading.Event()
        self.client_closed = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections += 1
            self.accepted.set()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            conn.settimeout(15)
            if not drain_http_request(conn):
                return
            conn.settimeout(READ_TIMEOUT + TIMEOUT_TOLERANCE + 10)
            data = conn.recv(65536)  # 客户端关闭时返回 b''
            if data == b"":
                self.client_closed.set()
        except OSError:
            self.client_closed.set()
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.srv.close()


class MockTtsEndpoint:
    """本地模拟 TTS 服务：可配置状态码与响应体。"""

    def __init__(self, status=200, body=b"ID3,fake-mp3-for-local-verification-only"):
        self.status = status
        self.body = body
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(5)
        self.port = self.srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            conn.settimeout(15)
            if not drain_http_request(conn):
                return
            head = (
                f"HTTP/1.0 {self.status} {'OK' if self.status == 200 else 'Error'}\r\n"
                f"Content-Length: {len(self.body)}\r\n"
                "Content-Type: audio/mpeg\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            conn.sendall(head + self.body)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.srv.close()


# ---- 用例执行 ----

RESULTS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    RESULTS.append((name, ok))
    return ok


def run_fault_tests(tmp_dir):
    print(f"requests 版本: {requests.__version__}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"选定超时参数: connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s, 容差={TIMEOUT_TOLERANCE}s")
    print(f"输出格式: {OUTPUT_FORMAT}")
    print("-" * 60)

    out = os.path.join(tmp_dir, "out.mp3")

    # A. 连接拒绝（辅助用例）
    refused = RefusedEndpoint()
    t0 = time.monotonic()
    err = None
    try:
        synthesize_azure_rest("hello", voice_name="en-SG-LunaNeural", speed="-15%",
                              key=FAKE_KEY, output_path=out, base_url=refused.url)
    except Exception as e:  # noqa: BLE001 - 记录实际异常类型
        err = e
    elapsed = time.monotonic() - t0
    check("A 连接拒绝 → 明确错误传播",
          isinstance(err, requests.exceptions.ConnectionError)
          and not isinstance(err, requests.exceptions.ConnectTimeout)
          and elapsed < CONNECT_TIMEOUT,
          f"{type(err).__name__ if err else '无异常'}，耗时 {elapsed:.2f}s，文件未写入={not os.path.exists(out)}")

    # B. 连接建立超时
    backlog = FullBacklogEndpoint()
    t0 = time.monotonic()
    err = None
    try:
        synthesize_azure_rest("hello", voice_name="en-SG-LunaNeural", speed="-15%",
                              key=FAKE_KEY, output_path=out, base_url=backlog.url)
    except Exception as e:  # noqa: BLE001
        err = e
    elapsed = time.monotonic() - t0
    backlog.close()
    check("B 连接建立超时 → ConnectTimeout 且耗时不超容差",
          isinstance(err, requests.exceptions.ConnectTimeout)
          and CONNECT_TIMEOUT - 0.5 <= elapsed <= CONNECT_TIMEOUT + TIMEOUT_TOLERANCE,
          f"{repr(err)[:160] if err else '无异常'}，耗时 {elapsed:.2f}s（期望约 {CONNECT_TIMEOUT}s）")

    # C. 服务端迟迟不返回数据（读超时）+ 连接资源释放
    hanging = HangingEndpoint()
    t0 = time.monotonic()
    err = None
    try:
        synthesize_azure_rest("hello", voice_name="en-SG-LunaNeural", speed="-15%",
                              key=FAKE_KEY, output_path=out,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), base_url=hanging.url)
    except Exception as e:  # noqa: BLE001
        err = e
    elapsed = time.monotonic() - t0
    accepted = hanging.accepted.wait(2)
    released = hanging.client_closed.wait(READ_TIMEOUT + TIMEOUT_TOLERANCE)
    hanging.close()
    check("C 服务端挂起 → ReadTimeout 且耗时不超容差",
          isinstance(err, requests.exceptions.ReadTimeout)
          and READ_TIMEOUT - 0.5 <= elapsed <= READ_TIMEOUT + TIMEOUT_TOLERANCE,
          f"{repr(err)[:160] if err else '无异常'}，耗时 {elapsed:.2f}s（期望约 {READ_TIMEOUT}s），服务端已接受连接={accepted}")
    check("C2 超时后客户端关闭连接（资源释放）", released,
          f"服务端在等待窗口内观察到 FIN={released}，文件未写入={not os.path.exists(out)}")

    # D. 串行恢复：上一用例超时后，下一次调用可正常完成
    mock_ok = MockTtsEndpoint(status=200)
    if os.path.exists(out):
        os.remove(out)
    t0 = time.monotonic()
    err = None
    path = None
    try:
        path = synthesize_azure_rest("hello", voice_name="en-SG-LunaNeural", speed="-15%",
                                     key=FAKE_KEY, output_path=out, base_url=mock_ok.url)
    except Exception as e:  # noqa: BLE001
        err = e
    elapsed = time.monotonic() - t0
    mock_ok.close()
    content_ok = path and os.path.isfile(path) and open(path, "rb").read() == mock_ok.body
    check("D 超时后串行恢复 → 下一次调用成功", not err and bool(content_ok) and elapsed < 10,
          f"{'异常: ' + type(err).__name__ if err else '成功'}，耗时 {elapsed:.2f}s，内容一致={bool(content_ok)}")

    # E. 提供方拒绝（HTTP 401）
    mock_deny = MockTtsEndpoint(status=401, body=b'{"error":{"code":"401"}}')
    t0 = time.monotonic()
    err = None
    try:
        synthesize_azure_rest("hello", voice_name="en-SG-LunaNeural", speed="-15%",
                              key=FAKE_KEY, output_path=out + ".deny", base_url=mock_deny.url)
    except Exception as e:  # noqa: BLE001
        err = e
    elapsed = time.monotonic() - t0
    mock_deny.close()
    check("E 提供方拒绝 → HTTPError 明确传播",
          isinstance(err, requests.exceptions.HTTPError)
          and getattr(err, "response", None) is not None and err.response.status_code == 401
          and elapsed < 10,
          f"{type(err).__name__ if err else '无异常'}，耗时 {elapsed:.2f}s")


def run_real_test():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import ConfigStore, DATA_DIR
    secrets = ConfigStore(DATA_DIR / "config.json").current.secrets
    key = secrets.get("azure_speech_key")
    region = secrets.get("azure_speech_region")
    if not key or not region:
        check("R 真实 API 合成（未验证：缺少凭据）", False,
              "本地配置缺少 Azure 密钥或区域，按计划记为未验证，不当作候选技术失败")
        return
    os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)
    cases = [
        ("default_speed", "en-SG-LunaNeural", "-15%", "This is a short sentence for verification."),
        ("adjusted_speed", "en-SG-LunaNeural", "+0%", "This is a short sentence at a different speed."),
    ]
    all_ok = True
    details = []
    for name, voice, speed, text in cases:
        out = os.path.join(REAL_OUTPUT_DIR, f"{name}.mp3")
        try:
            t0 = time.monotonic()
            synthesize_azure_rest(text, voice_name=voice, speed=speed, key=key, region=region, output_path=out)
            elapsed = time.monotonic() - t0
            size = os.path.getsize(out)
            ok = size > 0
            details.append(f"{name}: {size} 字节, {elapsed:.2f}s")
        except Exception as e:  # noqa: BLE001
            ok = False
            details.append(f"{name}: {type(e).__name__}: {e}")
        all_ok = all_ok and ok
    check("R 真实 API 合成（默认/调整语速，输出 MP3 落盘）", all_ok,
          "; ".join(details) + f"；输出目录 {REAL_OUTPUT_DIR}，需人工播放确认")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="追加真实 API 合成验证（消耗少量额度）")
    args = parser.parse_args()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_fault_tests(tmp_dir)
    if args.real:
        print("-" * 60)
        run_real_test()

    print("-" * 60)
    failed = [name for name, ok in RESULTS if not ok]
    unverified = [name for name, ok in RESULTS if not ok and "未验证" in name]
    print(f"共 {len(RESULTS)} 项：通过 {len(RESULTS) - len(failed)}，失败 {len(failed) - len(unverified)}，未验证 {len(unverified)}")
    if failed and not all("未验证" in name for name in failed):
        sys.exit(1)


if __name__ == "__main__":
    main()
