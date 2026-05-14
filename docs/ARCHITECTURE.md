# EnDictation 项目文档

## 项目简介

EnDictation 是一个 Web 听写练习应用。用户上传听写列表图片，应用自动 OCR 识别文本和重点单词，再按用户选择为每个句子和单词生成 TTS 音频。

---

## 系统架构

```
┌─────────────┐     POST /upload      ┌──────────────┐
│   浏览器      │ ──────────────────►   │   Flask App   │
│  (index.html)│ ◄──────────────────  │   (app.py)    │
└──────┬───────┘     JSON 结果         └──────┬───────┘
       │                                      │
       │  POST /generate-tts                  │
       │──────────────────────────────────────►
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
                                                        Google Cloud TTS
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
1. 接收图片文件
2. safe_filename() 清理文件名，保存到 uploads/
3. clean_audio_folder() 清理上一次生成的音频
4. 调用 Gemini API 进行 OCR
   └─ 使用 google-genai SDK 发送 OCR_PROMPT + 图片字节到 Gemini
5. parse_ocr_response() 解析返回的 markdown
   ├─ 移除代码块标记
   ├─ 按行分割，提取 **加粗单词**
   └─ 首行作为标题
6. 返回 OCR JSON 数组给前端
```

### TTS 生成 (`POST /generate-tts`)

```
1. 接收前端保存的 OCR JSON + 引擎 + 语言 + 音色 + 速度
2. 根据模型能力决定是否使用速度参数
   ├─ Microsoft TTS: 速度滑条值转成 Azure SSML rate
   ├─ Google Cloud TTS: 速度滑条值转成 speaking_rate
   └─ Google gTTS: 忽略速度，前端禁用滑条
3. clean_audio_folder() 清理上一次生成的音频
4. 遍历非标题句子：
   ├─ generate_audio() 生成整句音频
   └─ process_bold_words() 为每个加粗单词生成音频
5. 返回带 audio_path 的 JSON 数组给前端，生成失败的条目不暴露播放路径
```

### 进度查询 (`GET /status`)

前端每 500ms 轮询一次，读取全局 `processing_status` 字典：

```python
{
    "status": "idle" | "processing" | "done",
    "stage": "idle" | "ocr" | "tts",
    "request_id": "本轮前端请求ID",
    "message": "正在处理第 3/10 个句子",
    "current": 3,
    "total": 10,
    "progress": 30
}
```

前端每次上传 OCR 或生成 TTS 都会附带新的 `request_id`；轮询 `/status` 时只采纳同一个 `request_id` 的状态，避免读到上一轮任务遗留的 “OCR识别完成”。

### 音频服务 (`GET /audio/<filename>`)

直接从 `audio/` 目录返回 MP3 文件。文件名经过 `safe_filename()` 清理，防止路径遍历。

---

## 后端模块详解 (app.py)

### 配置与常量

| 名称 | 说明 |
|------|------|
| `OCR_MODEL` | OCR 模型名称，当前为 `gemini-3-flash-preview` |
| `tts_models` | TTS 服务配置字典，包含 Azure、gTTS 和 Google Cloud TTS 选项 |
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
| `generate_audio(text, filename, tts_model, speed_percent)` | 路由函数，根据 `tts_model["type"]` 分发到具体实现 |
| `generate_audio_gtts(text, filename, tts_model)` | Google TTS，通过 gTTS 库调用，免费无需密钥 |
| `generate_audio_azure(text, filename, tts_model, speed_percent)` | Azure TTS，使用 SSML 格式控制语速和音色 |
| `generate_audio_google_cloud(text, filename, tts_model, speed_percent)` | Google Cloud TTS，使用 Chirp 3 HD 音色和 `speaking_rate` 控制语速 |
| `resolve_tts_key(data)` | 兼容旧的单一下拉，同时把引擎、语言和音色组合解析为具体 TTS 配置 |

TTS 生成失败时会写入空文件作为后备，但返回结果只暴露非空音频文件，避免前端展示不可播放按钮；如果整轮没有生成任何有效音频，则接口返回错误。

### 安全函数

| 函数 | 说明 |
|------|------|
| `get_env_first(*names, default)` | 按顺序读取环境变量别名，兼容本地和云端命名差异 |
| `safe_filename(filename)` | 取 basename → Unicode 标准化 → 剔除特殊字符 → 空则用默认名 |
| `sanitize_html(text)` | 转义 `<` `>` 为 HTML 实体，保留 `**...**` 转为 `<strong>` |
| `update_processing_status(**kwargs)` | 安全更新全局状态，避免局部变量覆盖 |

---

## 前端详解 (index.html)

### 页面布局

单页面，无框架依赖，使用 Bootstrap 5 布局：

- **上传区域**: 支持拖放和点击选择，只触发 OCR
- **上传提示**: 页面只显示一句限制说明：“免费接口可能出现使用频率限制，图片处理失败请等待3分钟后重试”
- **TTS 控件**: OCR 完成后显示，两行布局；第一行是引擎和语言，第二行是速度和音色，右侧保留生成按钮
- **默认 TTS 组合**: Azure + 新加坡英语 + 女声 + 0.85x
- **加载指示器**: spinner + 进度条，通过轮询 `/status` 更新
- **结果区域**: 动态渲染句子列表，每句含播放按钮和重点单词按钮

### JavaScript 核心逻辑

| 函数 | 说明 |
|------|------|
| `handleFiles(files)` | 验证文件类型，显示预览，触发上传 |
| `uploadFile(file)` | 构建 FormData，附带本轮 `request_id` 后 POST 到 `/upload`，启动状态轮询 |
| `generateTts()` | 将当前 OCR 结果、引擎、语言、音色、速度和本轮 `request_id` POST 到 `/generate-tts` |
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
- **认证**: API Key，通过环境变量 `GOOGLE_API_KEY` 配置，也兼容 `GEMINI_API_KEY`
- **输入**: 图片字节流 + OCR 提示词
- **输出**: markdown 文本，重点单词用 `**...**` 标记

### Microsoft Azure TTS

- **区域**: Southeast Asia
- **认证**: 订阅密钥，通过环境变量 `AZURE_API_KEY` 配置，也兼容 `AZURE_SPEECH_KEY`
- **区域变量**: 默认 `southeastasia`，也可通过 `AZURE_SPEECH_REGION` 或 `AZURE_SERVICE_REGION` 覆盖
- **调用方式**: Azure Speech SDK，SSML 格式控制语速
- **可用音色**:

| 配置名 | 语音 | 说明 |
|--------|------|------|
| SG-man | en-SG-WayneNeural | 新加坡英语男声 |
| SG-woman | en-SG-LunaNeural | 新加坡英语女声 |
| UK-man | en-GB-OllieMultilingualNeural | 英式英语男声（支持多语言） |
| UK-woman | en-GB-LibbyNeural | 英式英语女声 |
| US-Azure-man | en-US-GuyNeural | 美式英语男声 |
| US-Azure-woman | en-US-JennyNeural | 美式英语女声 |
| CH-man | zh-CN-YunyangNeural | 中文男声 |
| CH-woman | zh-CN-XiaoxiaoNeural | 中文女声 |
| French-Azure-man | fr-FR-HenriNeural | 法语男声 |
| French-Azure-woman | fr-FR-DeniseNeural | 法语女声 |

### Google TTS (gTTS)

- **调用方式**: gTTS Python 库，无需 API 密钥
- **可用音色**: 通过 `lang` + `tld` 参数控制口音

| 配置名 | lang | tld | 说明 |
|--------|------|-----|------|
| UK-Google | en | co.uk | 英式英语 |
| US-Google | en | com | 美式英语 |
| French-Google | fr | fr | 法语 |
| Chinese-Google | zh | com | 中文 |

### Google Cloud Text-to-Speech

- **调用方式**: `google-cloud-texttospeech` Python 客户端
- **认证**: 优先使用 `GOOGLE_CLOUD_TTS_CREDENTIALS_JSON`，否则使用 Google ADC（例如 `GOOGLE_APPLICATION_CREDENTIALS`）
- **输出格式**: MP3
- **语速**: 使用 `speaking_rate`，由前端速度滑条换算

| 配置名 | 语音 | 说明 |
|--------|------|------|
| UK-Chirp-man | en-GB-Chirp3-HD-Charon | 英式英语男声 |
| UK-Chirp-woman | en-GB-Chirp3-HD-Leda | 英式英语女声 |
| US-Chirp-man | en-US-Chirp3-HD-Charon | 美式英语男声 |
| US-Chirp-woman | en-US-Chirp3-HD-Leda | 美式英语女声 |
| Chinese-Chirp-man | cmn-CN-Chirp3-HD-Charon | 普通话男声 |
| Chinese-Chirp-woman | cmn-CN-Chirp3-HD-Leda | 普通话女声 |
| French-Chirp-man | fr-FR-Chirp3-HD-Charon | 法语男声 |
| French-Chirp-woman | fr-FR-Chirp3-HD-Leda | 法语女声 |

### 前端组合矩阵

| 引擎 | 可选语言 |
|------|----------|
| Azure | 英式英语、美式英语、新加坡英语、中文、法语 |
| Google | 英式英语、美式英语、中文、法语 |
| gTTS | 英式英语、美式英语、中文、法语 |

gTTS 不提供真实男/女音色，前端仍显示男声/女声，但两者映射到同一个 gTTS 配置。

---

## 部署

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `GOOGLE_API_KEY` | 是 | Gemini OCR |
| `AZURE_API_KEY` | 否 | Azure TTS（不用 Microsoft 音色则不需要） |
| `GOOGLE_APPLICATION_CREDENTIALS` | 否 | Google Cloud TTS 服务账号文件路径，适合本地或有持久文件系统的部署 |
| `GOOGLE_CLOUD_TTS_CREDENTIALS_JSON` | 否 | Google Cloud TTS 服务账号 JSON 单行字符串，Azure App Service 推荐使用 |
| `PORT` | 否 | 服务端口，默认 5001 |

环境变量别名：`GEMINI_API_KEY` 可替代 `GOOGLE_API_KEY`；`AZURE_SPEECH_KEY` 可替代 `AZURE_API_KEY`；`AZURE_SPEECH_REGION` 可覆盖 Azure 区域。Google Cloud TTS JSON 也兼容 `GOOGLE_CREDENTIALS_JSON`、`GOOGLE_SERVICE_ACCOUNT_JSON`、`GCP_SERVICE_ACCOUNT_JSON`，文件路径也兼容 `GOOGLE_CLOUD_TTS_CREDENTIALS_FILE`、`GOOGLE_SERVICE_ACCOUNT_FILE`、`GCP_SERVICE_ACCOUNT_FILE`。

### Azure App Service 部署

通过 GitHub Actions 自动部署（`.github/workflows/main_endictaion.yml`）：

1. push 到 `main` 分支触发
2. CI 构建：安装依赖 → 打包 zip → 上传 artifact
3. CD 部署：下载 artifact → Azure OIDC 登录 → 部署到 App Service
4. `startup.sh` 创建必要目录并启动 Flask

Azure 门户需配置应用设置：`GOOGLE_API_KEY`、`AZURE_API_KEY`。使用 Google Cloud TTS 时，额外配置 `GOOGLE_CLOUD_TTS_CREDENTIALS_JSON`，不要配置本机 JSON 文件路径。

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
