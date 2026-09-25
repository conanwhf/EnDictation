# EnDictation 项目文档

## 项目简介

EnDictation 是一个 Web 听写练习应用。用户上传听写列表图片，应用自动 OCR 识别文本和重点单词，再按用户选择为每个句子和单词生成 TTS 音频。OCR 与 TTS 均为后台任务：提交接口立即返回任务 ID，前端轮询任务状态。

---

## 系统架构

```
┌─────────────┐  POST /upload (202)   ┌──────────────────────────┐
│   浏览器      │ ───────────────────► │  Gunicorn (1w + gthread4) │
│ (index.html) │ ◄─────────────────── │  Flask app.py             │
└──────┬───────┘  task_id/status_url  │   └─ TaskManager(tasks.py)│
       │                              │      └─ 串行 FIFO 执行器    │
       │  POST /generate-tts (202)    │        (ThreadPoolExecutor│
       │───────────────────────────► │         max_workers=1)    │
       │  GET /tasks/<task_id> (轮询)  │ └──────────────────────────┘
       │───────────────────────────► │           │      │      │
       │  GET /audio/<task_id>/<file> │           ▼      ▼      ▼
       │───────────────────────────► │        Gemini  Azure   gTTS /
       │  GET /health                 │        (OCR)   REST    Google Cloud TTS
       │───────────────────────────► │
       ▼                              ▼
```

- 单进程 Gunicorn（`--workers 1 --worker-class gthread --threads 4`），不使用 `--preload`：任务执行器与状态都在 worker 进程内初始化。
- 全局最多一个生成任务在执行；其余有效任务进入进程内 FIFO 队列，提交立即返回 `202`，结束后自动执行下一项。
- 没有数据库或外部队列服务；任务状态在内存，任务文件在容器卷。

---

## 目录结构

```
EnDictation/
├── app.py               # Flask 应用：路由、会话、校验、OCR/TTS 任务执行
├── tasks.py             # 任务模型：字典+锁、串行 FIFO、容量、owner 校验、过期清理
├── templates/
│   └── index.html       # 前端单页面（轮询 + DOM 渲染）
├── tests/               # 任务模型与接口测试（外部 API 为假实现）
├── conftest.py          # 测试环境（固定假 SECRET_KEY、临时 DATA_DIR）
├── tools/
│   └── verify_azure_rest.py   # Azure REST 受控故障验证脚本（步骤 1）
├── requirements.txt / requirements-dev.txt
├── Dockerfile / .dockerignore / compose.nas.yml
├── startup.sh           # 带配置与权限检查的 Gunicorn 启动脚本
└── docs/                 # 本文档与 NAS_MIGRATION.md
```

---

## 任务生命周期（tasks.py）

- `TaskManager.submit(stage, owner, execute, input_saver=None)` 在同一把锁内原子完成：容量检查 → 任务登记 → 输入保存 → 执行器提交；提交失败时撤销登记并清理任务目录（含上传文件）。
- 状态机：`queued` → `running` → `succeeded` / `failed`。快照字段：`task_id`、`stage`（`ocr`/`tts`）、`status`、`current`、`total`、`progress`、`message`、`warnings`、`result`、`error`。
- 容量：内存最多 100 条任务记录（含排队/执行/完成）。达到上限且无过期项可清理时抛 `CapacityError`，接口返回 `429` 与错误类型 `capacity`，明确说明未受理。
- 过期：完成任务按完成时间保留 24 小时；排队与执行中任务不参与过期清理。清理时同时移除记录与任务目录；启动时与每分钟检查一次（清理线程只在 worker 内启动）。
- OCR 原图在任务执行结束（成功或失败）后的 `finally` 中删除。
- 任务执行边界兜底捕获未预期异常：记录异常栈并按阶段标记 `ocr_failed` / `tts_failed`。

### 目录与文件布局

```
<DATA_DIR>/tasks/<task_id>/     # DATA_DIR 由环境变量指定（容器内 /data）
  input.<按MIME的扩展名>         # OCR 临时输入（服务端命名，结束后删除）
  sentence_<index>.mp3          # TTS 整句音频
  word_<index>_<index>.mp3       # TTS 重点词音频
```

上传文件名只作显示信息，不参与目录定位；所有路径由服务端构造。

---

## 会话与隔离

- Flask 签名 session cookie（`SECRET_KEY` 必需，无回退默认值；启动检查在 `app.bootstrap_runtime()`，Gunicorn worker 导入与 `python app.py` 共用）。cookie `HttpOnly`、`SameSite=Lax`、生存期 7 天，未启用 `Secure`（本阶段为本机/受信 LAN HTTP 验证）。
- 首次访问首页生成随机会话 ID（`sid`），任务登记 owner；状态与音频读取都校验 owner，不匹配时与「任务不存在」统一返回 `404`，不泄露任务是否曾经存在。
- 写接口（`/upload`、`/generate-tts`）要求同源：拒绝不匹配的 `Origin`；无 `Origin` 时按 `Referer` 校验；都没有则拒绝（`403`）。
- 同一浏览器的多个标签页共享会话；独立浏览器/无痕窗口相互隔离。

### 重启行为

内存任务状态随进程重启丢失；磁盘文件是否保留取决于容器重建与卷挂载，两者不能混为一谈。进程重启后：旧任务（含排队项）一律失效，状态与音频接口返回 `404`；worker 启动时删除 `<DATA_DIR>/tasks` 下遗留的任务目录（只清理应用专属目录）。

---

## 请求流程

### 提交任务（`POST /upload` / `POST /generate-tts`）

1. 同源与会话校验（失败 `403`）
2. 入口校验（见「输入限制」），错误返回对应 4xx 与稳定错误类型
3. 任务登记进串行队列，立即返回 `202` + `{"task_id", "status_url"}`

### 状态查询（`GET /tasks/<task_id>`）

返回任务快照；轮询返回 `200` 不代表任务成功，由 `status` 表达结果；失败任务的 `error` 含稳定错误类型（`invalid_input`、`invalid_tts_combo`、`too_large`、`capacity`、`ocr_failed`、`tts_failed`）与可读说明；`404` 统一提示任务不存在或已过期。

- OCR 阶段 `current`/`total`/`progress` 固定为 0，前端只显示「识别中」不渲染进度条。
- TTS 阶段 `total` = 非标题句子数 + 这些句子的重点词数之和；每次合成尝试结束（无论成功失败）`current` 加一；预算耗尽时未尝试项不计数、保持实际进度。

### OCR 执行（app.run_ocr_task）

一次 Gemini 调用（`parse_ocr_response` 输出结构化句子数组：`text`、`original_text`、`is_title`、`title`，非标题行含 `bold_words` 对象数组）；调用失败或超时 → 任务 `failed` + `ocr_failed`；提取不到文本同样失败，不返回空结果。

### TTS 执行（app.run_tts_task）

按句子顺序串行合成整句与重点词；单条失败记录原因并继续（不重试、不生成空文件）；部分成功 → `succeeded` + 可播放结果 + `warnings` 逐条说明；全部失败 → `failed` + `tts_failed`。任务级 10 分钟执行预算从实际开始执行起计，在连续合成调用之间检查；预算耗尽不再尝试剩余项，已有有效音频则按部分成功处理。

### 音频服务（`GET /audio/<task_id>/<filename>`）

文件名必须匹配 `^(sentence|word)_[0-9]+(_[0-9]+)?\.mp3$`；owner 校验失败、任务过期或文件清理竞争时统一 `404`；用 `send_from_directory` 拒绝路径穿越。

### 健康检查（`GET /health`）

返回 `{"status":"ok"}`，只证明进程可响应；不调用收费 API，不宣称凭据有效。Docker unhealthy 不会自动重启容器。

---

## 输入限制（初始设计值常量）

上传图片**不做本地内容校验**（2026-09-25 用户决策，恢复旧版直接上传）：MIME 按上传声明映射、默认 `image/jpeg`，内容有效性由提供方判定，无效内容以任务级 `ocr_failed` 明确报错。`create_app(max_content_length=...)` 参数保留，部署方显式设置后 413 处理继续生效。

| 项目 | 限制 |
|------|------|
| 非标题句子数 | ≤ 50 条 |
| 单句长度 | ≤ 500 字符 |
| 每句重点词 | ≤ 20 个，每个 ≤ 100 字符 |
| 单次合成总量 | 句子与重点词累计 ≤ 10000 字符 |
| 语速 | -50 到 50，非有限数值或越界 400；gTTS 等不支持速度的引擎忽略该值 |
| 单任务执行预算 | 10 分钟（不含排队） |
| 任务记录 | 最多 100 条；完成任务保留 24 小时 |
| 状态轮询 | 前端 1 秒间隔；网络错误重试 1s/2s/4s 后停止并提供重新查询入口 |

---

## 外部服务与超时

所有外部调用显式配置有限超时，不自动重试；提供方失败按任务规则传播，不用空音频或默认音色掩盖。

| 服务 | 实现 | 超时 |
|------|------|------|
| Gemini OCR | google-genai（`Client(http_options={timeout, retry_options: attempts=1})`） | 120 秒（`HttpOptions.timeout` 单位为毫秒，传 `120000`；真实测试证实误传秒值会在约 1 秒触发 read timeout） |
| Azure TTS | 文本转语音 REST `POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1`，`X-Microsoft-OutputFormat: audio-24khz-48kbitrate-mono-mp3`，requests | `(connect 5s, read 30s)` |
| gTTS | gTTS 2.5.4 构造器 `timeout=`（默认无限等待，必须显式传入） | 30 秒 |
| Google Cloud TTS | `synthesize_speech(..., retry=None, timeout=)` | 30 秒 |

Azure REST 路径的受控故障验证结论（连接拒绝、连接建立超时 5.00s、读超时 30.00s、超时后连接释放与串行恢复、401 传播）见 [NAS_MIGRATION.md §3.5](NAS_MIGRATION.md)；真实合成验证脚本 `tools/verify_azure_rest.py --real`。

### 可用音色

引擎/语言/音色组合由 `tts_voice_matrix` 定义（Azure 5 语言 × 男/女、Google Cloud TTS 与 gTTS 各 4 语言）；无效组合返回 `invalid_tts_combo`，不回退。gTTS 男/女声映射到同一配置。

---

## 前端（index.html）

- `uploadFile()` / `generateTts()`：提交后轮询 `status_url`（单条可停止定时链，1 秒间隔），仅任务成功时渲染；提交只发送契约字段（`sentences`/`tts_engine`/`tts_language`/`voice_gender`/`speed`）。
- `queued` 只显示「排队中」继续轮询，不提示重试；OCR 阶段不渲染进度条；TTS 按尝试进度渲染。
- 渲染全部使用 DOM API：标题/正文以 `textContent` 设置，正文按普通文本节点与 `<strong>` 节点拼装，`**加粗**` 前端格式化保留；播放按钮按固定模板构建，仅含有效 `audio_path` 的句子/重点词创建，其余保留加粗文字；无服务端 HTML 注入。
- 失败与部分失败：任务失败显示稳定错误说明；`warnings` 逐条可见；容量 `429` 明确提示未受理并保留 OCR 内容与选择。
- `currentOcrData` 保留原始 OCR 数据，重复 TTS 不触发新 OCR。

---

## 部署

### 环境变量

见 [README](../README.md) 的环境变量表。`SECRET_KEY` 缺失时启动直接失败；凭据只通过进程环境或只读文件挂载传入，日志不打印密钥与凭据 JSON。

### 容器（compose.nas.yml）

```bash
SECRET_KEY=... docker compose -f compose.nas.yml up -d
```

- 镜像 `linux/amd64`、Debian `python:3.11-slim`、非 root 运行；空 named volume 首挂载继承 `/data/tasks` 属主。
- 端口仅绑定 `127.0.0.1:15901:5001`；`restart: unless-stopped`（重启意味着未完成任务丢失，不表示续跑）。
- 本阶段不设 CPU/内存硬限制（本机空转实测约 99MiB / 0.12% CPU，供 NAS 部署值参考）。

### 本地运行

```bash
export SECRET_KEY="..." GOOGLE_API_KEY="..."
python app.py          # 开发入口
./startup.sh           # 带检查的 Gunicorn 启动
```

### 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## 已知限制

- **单 worker 串行**：底层调用持续不返回会阻塞后续任务（挂起风险由连接/读取超时缓解，但不构成严格总时限）；`/health` 仍可响应，恢复需人工重启。
- **无持久化**：任务状态只在内存，进程重启后旧任务失效；容器卷只保证文件在容器重建间的保留，不承诺任务续跑。
- **无跨设备同步**：会话 cookie 仅用于隔离，不是登录认证；不能据此开放给不可信网络。
- **免费额度**：Gemini OCR 与 gTTS 受提供方频率限制影响。
