# EnDictation 项目文档

## 项目简介

EnDictation 是一个 Web 听写练习应用。用户上传听写列表图片，应用自动 OCR 识别文本和重点单词，并为每个句子和单词生成 TTS 音频。

---

## 系统架构

```
┌─────────────┐     POST /upload      ┌──────────────┐
│   浏览器      │ ──────────────────►   │   Flask App   │
│  (index.html)│ ◄──────────────────  │   (app.py)    │
└──────┬───────┘     JSON 结果         └──────┬───────┘
       │                                      │
       │  GET /audio/*.mp3                    │
       │──────────────────────────────────────┘
       │
       │  GET /status (轮询进度)
       │──────────────────────────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                        Gemini API      Azure TTS       Google TTS
                        (OCR 识别)      (语音合成)       (语音合成)
```

应用为单进程 Flask 服务，没有数据库，所有中间数据存放在内存和本地文件系统中。

---

## 目录结构

```
EnDictation/
├── app.py                  # 主应用，包含所有后端逻辑
├── requirements.txt        # Python 依赖清单
├── templates/
│   └── index.html          # 前端单页面
├── uploads/                # 用户上传的图片（gitignore）
├── audio/                  # 生成的 MP3 音频文件（gitignore）
├── startup.sh              # Azure App Service 启动脚本
├── web.config              # IIS 服务器配置（Azure Windows 部署）
├── .deployment             # Azure 部署配置
├── azure.yaml              # Azure CLI 配置
├── .github/workflows/
│   └── main_endictaion.yml # GitHub Actions CI/CD
├── .gitignore
└── README.md
```

---

## 请求流程详解

### 图片上传处理 (`POST /upload`)

```
1. 接收图片文件 + TTS 模型选择
2. safe_filename() 清理文件名，保存到 uploads/
3. clean_audio_folder() 清理上一次生成的音频
4. 调用 Gemini API 进行 OCR
   └─ 使用 google-genai SDK 发送 OCR_PROMPT + 图片字节到 Gemini
5. parse_ocr_response() 解析返回的 markdown
   ├─ 移除代码块标记
   ├─ 按行分割，提取 **加粗单词**
   └─ 首行作为标题
6. 遍历每个句子：
   ├─ generate_audio() 生成整句音频
   └─ process_bold_words() 为每个加粗单词生成音频
7. 返回 JSON 数组给前端
```

### 进度查询 (`GET /status`)

前端每 500ms 轮询一次，读取全局 `processing_status` 字典：

```python
{
    "status": "idle" | "processing" | "done",
    "message": "正在处理第 3/10 个句子",
    "current": 3,
    "total": 10,
    "progress": 30
}
```

### 音频服务 (`GET /audio/<filename>`)

直接从 `audio/` 目录返回 MP3 文件。文件名经过 `safe_filename()` 清理，防止路径遍历。

---

## 后端模块详解 (app.py)

### 配置与常量

| 名称 | 说明 |
|------|------|
| `OCR_MODEL` | OCR 模型名称，当前为 `gemini-3-flash-preview` |
| `tts_models` | TTS 服务配置字典，包含 Azure 和 Google 两种类型共 9 个选项 |
| `OCR_PROMPT` | 发送给 Gemini 的 OCR 提示词 |
| `UPLOAD_FOLDER` | 上传目录路径，默认 `uploads/` |
| `AUDIO_FOLDER` | 音频目录路径，默认 `audio/` |

### OCR 相关函数

| 函数 | 说明 |
|------|------|
| `extract_text_cloud(image_path)` | 使用 google-genai SDK 调用 Gemini 进行 OCR：读取图片 → API 调用 → 解析结果 |
| `parse_ocr_response(text)` | 解析 Gemini 返回的 markdown，提取句子列表和 bold_words |

### TTS 相关函数

| 函数 | 说明 |
|------|------|
| `generate_audio(text, filename, tts_model)` | 路由函数，根据 `tts_model["type"]` 分发到具体实现 |
| `generate_audio_gtts(text, filename, tts_model)` | Google TTS，通过 gTTS 库调用，免费无需密钥 |
| `generate_audio_azure(text, filename, tts_model)` | Azure TTS，使用 SSML 格式控制语速和音色 |

所有 TTS 函数失败时均返回空文件（`create_empty_audio`），不中断整体流程。

### 安全函数

| 函数 | 说明 |
|------|------|
| `safe_filename(filename)` | 取 basename → Unicode 标准化 → 剔除特殊字符 → 空则用默认名 |
| `sanitize_html(text)` | 转义 `<` `>` 为 HTML 实体，保留 `**...**` 转为 `<strong>` |
| `update_processing_status(**kwargs)` | 安全更新全局状态，避免局部变量覆盖 |

---

## 前端详解 (index.html)

### 页面布局

单页面，无框架依赖，使用 Bootstrap 5 布局：

- **上传区域**: 支持拖放和点击选择，含 TTS 模型下拉选择器
- **加载指示器**: spinner + 进度条，通过轮询 `/status` 更新
- **结果区域**: 动态渲染句子列表，每句含播放按钮和重点单词按钮

### JavaScript 核心逻辑

| 函数 | 说明 |
|------|------|
| `handleFiles(files)` | 验证文件类型，显示预览，触发上传 |
| `uploadFile(file)` | 构建 FormData（含 TTS 选择），POST 到 `/upload`，启动状态轮询 |
| `checkProcessStatus()` | 每 500ms 轮询 `/status`，更新进度条和状态文字 |
| `displayResults(data)` | 将 JSON 结果渲染为带播放按钮的句子列表 |
| `playAudio(audioPath)` | 创建 `<audio>` 元素播放音频，播完自动移除 |

### 播放器实现

每次播放动态创建 `<audio>` 元素，`onended` 时从 DOM 移除，避免内存泄漏。音频路径统一处理为 `/audio/` 前缀。

---

## 外部服务

### Google Gemini (OCR)

- **模型**: `gemini-3-flash-preview`
- **调用方式**: 使用 `google-genai` Python SDK
- **认证**: API Key，通过环境变量 `GOOGLE_API_KEY` 配置
- **输入**: 图片字节流 + OCR 提示词
- **输出**: markdown 文本，重点单词用 `**...**` 标记

### Microsoft Azure TTS

- **区域**: Southeast Asia
- **认证**: 订阅密钥，通过环境变量 `AZURE_API_KEY` 配置
- **调用方式**: Azure Speech SDK，SSML 格式控制语速
- **可用音色**:

| 配置名 | 语音 | 说明 |
|--------|------|------|
| SG-man | en-SG-WayneNeural | 新加坡英语男声 |
| SG-woman | en-SG-LunaNeural | 新加坡英语女声 |
| UK-man | en-GB-OllieMultilingualNeural | 英式英语男声（支持多语言） |
| UK-woman | en-GB-LibbyNeural | 英式英语女声 |
| CH-man | zh-CN-YunyangNeural | 中文男声 |

### Google TTS (gTTS)

- **调用方式**: gTTS Python 库，无需 API 密钥
- **可用音色**: 通过 `lang` + `tld` 参数控制口音

| 配置名 | lang | tld | 说明 |
|--------|------|-----|------|
| UK-Google | en | co.uk | 英式英语 |
| US-Google | en | com | 美式英语 |
| French-Google | fr | fr | 法语 |
| Chinese-Google | zh | com | 中文 |

---

## 部署

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `GOOGLE_API_KEY` | 是 | Gemini OCR |
| `AZURE_API_KEY` | 否 | Azure TTS（不用 Microsoft 音色则不需要） |
| `PORT` | 否 | 服务端口，默认 5001 |

### Azure App Service 部署

通过 GitHub Actions 自动部署（`.github/workflows/main_endictaion.yml`）：

1. push 到 `main` 分支触发
2. CI 构建：安装依赖 → 打包 zip → 上传 artifact
3. CD 部署：下载 artifact → Azure OIDC 登录 → 部署到 App Service
4. `startup.sh` 创建必要目录并启动 Flask

Azure 门户需配置应用设置：`GOOGLE_API_KEY`、`AZURE_API_KEY`。

### 本地运行

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
python app.py
# 访问 http://localhost:5001
```

---

## 已知限制

- **单进程**: Flask 单进程运行，并发上传时全局 `processing_status` 会互相覆盖
- **无持久化**: 音频和上传文件存在本地磁盘，重启后丢失
- **内存 OCR**: 图片整体加载到内存后发送，大图片可能导致内存压力
- **状态轮询**: 前端 500ms 轮询 `/status`，实时性有限
